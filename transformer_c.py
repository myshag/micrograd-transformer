"""
Расширение C+BLAS-компиляции на ТРАНСФОРМЕР.

Компилируем forward одного энкодер-блока трансформера (multi-head attention +
LayerNorm + FFN + остаточные связи, pre-norm) в C с вызовами OpenBLAS и сверяем
с нашим Python nn.TransformerEncoderLayer.

Главная идея, делающая это простым: для одной последовательности (batch=1) всё
выражается через 2D-операции. Головы внимания — это срезы столбцов Q/K/V, а BLAS
умеет работать с подматрицами через ведущий размер (lda). Никаких 4D-тензоров и
перестановок осей: каждое матумножение — обычный cblas_sgemm, softmax и
LayerNorm — циклы.

    x = x + Attention(LayerNorm(x))          (pre-norm)
    x = x + FFN(LayerNorm(x))

Требуется: gcc, libopenblas-dev.
Запуск:  python3 transformer_c.py
"""

import os
import subprocess
import tempfile

import numpy as np

import nn
from tensor import Tensor, no_grad

BINFILE = "/tmp/transformer_c.bin"

C_TEMPLATE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

#define T   @T@
#define D   @D@
#define NH  @NH@
#define DH  (D/NH)
#define F   @F@
#define EPS 1e-5f

/* веса (порядок совпадает с записью из Python) */
static float g1[D], be1[D];                 /* LayerNorm 1 */
static float Wq[D*D],bq[D], Wk[D*D],bk[D], Wv[D*D],bv[D], Wo[D*D],bo[D];
static float g2[D], be2[D];                 /* LayerNorm 2 */
static float Wf1[F*D],bf1[F], Wf2[D*F],bf2[D];
static float x[T*D];

/* рабочие буферы */
static float nrm[T*D], Q[T*D], K[T*D], V[T*D], sc[T*T], ctx[T*D], ao[T*D];
static float h1[T*F], h2[T*D];

/* LayerNorm по последней оси: out = (x-mean)/sqrt(var+eps)*g + b */
static void layernorm(const float* in, float* out, const float* g, const float* b){
  for(int t=0;t<T;t++){
    const float* r=in+t*D; float* o=out+t*D;
    float m=0; for(int j=0;j<D;j++) m+=r[j]; m/=D;
    float v=0; for(int j=0;j<D;j++){ float d=r[j]-m; v+=d*d; } v/=D;
    float inv=1.0f/sqrtf(v+EPS);
    for(int j=0;j<D;j++) o[j]=(r[j]-m)*inv*g[j]+b[j];
  }
}

/* y = xin @ Wᵀ + bias ; W имеет форму (out,in) — как в nn.Linear */
static void linear(const float* xin,int rows,int din,int dout,
                   const float* W,const float* bias,float* y){
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,rows,dout,din,
              1.0f,xin,din,W,din,0.0f,y,dout);
  for(int i=0;i<rows*dout;i++) y[i]+=bias[i%dout];
}

static void softmax_rows(float* s,int rows,int cols){
  for(int t=0;t<rows;t++){
    float* r=s+t*cols;
    float mx=r[0]; for(int j=1;j<cols;j++) if(r[j]>mx) mx=r[j];
    float sm=0; for(int j=0;j<cols;j++){ r[j]=expf(r[j]-mx); sm+=r[j]; }
    for(int j=0;j<cols;j++) r[j]/=sm;
  }
}

static void attention(const float* in,float* out){
  linear(in,T,D,D,Wq,bq,Q);
  linear(in,T,D,D,Wk,bk,K);
  linear(in,T,D,D,Wv,bv,V);
  float scale=1.0f/sqrtf((float)DH);
  for(int hh=0;hh<NH;hh++){
    const float* Qh=Q+hh*DH; const float* Kh=K+hh*DH; const float* Vh=V+hh*DH;
    /* sc(T,T) = Qh(T,DH) @ Khᵀ(DH,T) * scale   (срезы через lda=D) */
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,T,T,DH,
                scale,Qh,D,Kh,D,0.0f,sc,T);
    softmax_rows(sc,T,T);
    /* ctx_h(T,DH) = sc(T,T) @ Vh(T,DH)  -> в столбцы hh*DH, ldc=D */
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,DH,T,
                1.0f,sc,T,Vh,D,0.0f,ctx+hh*DH,D);
  }
  linear(ctx,T,D,D,Wo,bo,out);   /* выходная проекция */
}

int main(void){
  FILE* f=fopen("@BIN@","rb");
#define RD(a,n) if(fread(a,sizeof(float),n,f)!=(size_t)(n)){return 1;}
  RD(g1,D) RD(be1,D)
  RD(Wq,D*D) RD(bq,D) RD(Wk,D*D) RD(bk,D) RD(Wv,D*D) RD(bv,D) RD(Wo,D*D) RD(bo,D)
  RD(g2,D) RD(be2,D)
  RD(Wf1,F*D) RD(bf1,F) RD(Wf2,D*F) RD(bf2,D)
  RD(x,T*D)
  fclose(f);

  /* x = x + Attention(LayerNorm(x)) */
  layernorm(x,nrm,g1,be1);
  attention(nrm,ao);
  for(int i=0;i<T*D;i++) x[i]+=ao[i];

  /* x = x + FFN(LayerNorm(x)) ;  FFN = Linear->relu->Linear */
  layernorm(x,nrm,g2,be2);
  linear(nrm,T,D,F,Wf1,bf1,h1);
  for(int i=0;i<T*F;i++) if(h1[i]<0) h1[i]=0;
  linear(h1,T,F,D,Wf2,bf2,h2);
  for(int i=0;i<T*D;i++) x[i]+=h2[i];

  for(int i=0;i<T*D;i++) printf("%.7g ", x[i]);
  printf("\n");
  return 0;
}
"""


def main():
    T, D, NH, F = 16, 32, 4, 64
    rng = np.random.default_rng(0)

    layer = nn.TransformerEncoderLayer(D, NH, F, seed=1)
    x_in = rng.standard_normal((1, T, D)).astype(np.float32)

    # эталон: наш Python-движок
    with no_grad():
        y_ref = layer(Tensor(x_in)).data[0]        # (T, D)

    # выгрузить веса в бинарник в том же порядке, что читает C
    a = layer.attn
    ff = layer.ff.layers
    order = [
        layer.norm1.weight, layer.norm1.bias,
        a.Wq.weight, a.Wq.bias, a.Wk.weight, a.Wk.bias,
        a.Wv.weight, a.Wv.bias, a.Wo.weight, a.Wo.bias,
        layer.norm2.weight, layer.norm2.bias,
        ff[0].weight, ff[0].bias, ff[2].weight, ff[2].bias,
    ]
    with open(BINFILE, "wb") as fh:
        for p in order:
            fh.write(np.ascontiguousarray(p.data, dtype=np.float32).tobytes())
        fh.write(np.ascontiguousarray(x_in[0], dtype=np.float32).tobytes())

    src = (C_TEMPLATE.replace("@T@", str(T)).replace("@D@", str(D))
           .replace("@NH@", str(NH)).replace("@F@", str(F)).replace("@BIN@", BINFILE))
    tmp = tempfile.mkdtemp()
    cpath, epath = os.path.join(tmp, "tb.c"), os.path.join(tmp, "tb")
    open(cpath, "w").write(src)
    subprocess.run(["gcc", "-O2", "-o", epath, cpath, "-lopenblas", "-lm"], check=True)
    out = subprocess.run([epath], capture_output=True, text=True, check=True).stdout
    y_c = np.array(out.split(), dtype=np.float32).reshape(T, D)

    print("=== Энкодер-блок трансформера, скомпилированный в C+BLAS ===")
    print(f"Размерности: T={T}, d_model={D}, heads={NH}, d_ff={F}")
    print(f"\nвыход Python-движка [0,:5] = {y_ref[0, :5]}")
    print(f"выход C+BLAS         [0,:5] = {y_c[0, :5]}")
    diff = np.abs(y_ref - y_c).max()
    print(f"\nmax|Python - C| = {diff:.2e}  ->",
          "OK ✓" if diff < 1e-3 else "РАСХОЖДЕНИЕ ✗")
    print("\nВесь блок (attention+LayerNorm+FFN) посчитан в C: matmul-и -> cblas_sgemm,\n"
          "головы внимания -> срезы через lda, softmax/LayerNorm -> циклы.")


if __name__ == "__main__":
    main()
