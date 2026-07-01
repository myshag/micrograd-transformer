"""
Backward энкодер-блока трансформера в C+BLAS.

Продолжение transformer_c.py: теперь компилируем и ОБРАТНЫЙ проход блока
(multi-head attention + LayerNorm + FFN + residual) и сверяем ВСЕ градиенты —
по входу и по всем весам — с нашим Python-autograd.

Правила те же, что в главе 1, просто на матрицах:
  - градиент matmul = matmul с транспонированием (cblas_sgemm + CblasTrans);
  - головы внимания = срезы столбцов (через lda);
  - softmax backward: gs = p·(gp − Σ gp·p);
  - LayerNorm backward: стандартная формула через xhat, inv, средние по строке.

Требуется: gcc, libopenblas-dev.
Запуск:  python3 transformer_c_backward.py
"""

import os
import subprocess
import tempfile

import numpy as np

import nn
from tensor import Tensor

BINFILE = "/tmp/transformer_cb.bin"

C_TEMPLATE = r"""
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

#define T   @T@
#define D   @D@
#define NH  @NH@
#define DH  (D/NH)
#define F   @F@
#define EPS 1e-5f

/* веса */
static float g1[D],be1[D], Wq[D*D],bq[D],Wk[D*D],bk[D],Wv[D*D],bv[D],Wo[D*D],bo[D];
static float g2[D],be2[D], Wf1[F*D],bf1[F], Wf2[D*F],bf2[D];
/* forward-активации (нужны для backward) */
static float x0[T*D], n1[T*D], Q[T*D], K[T*D], V[T*D], P[NH*T*T];
static float ctx[T*D], a[T*D], x2[T*D], n2[T*D], h1[T*F], h2[T*D], y[T*D];
/* градиенты */
static float gy[T*D],gx2[T*D],gh2[T*D],gh1[T*F],gz1[T*F],gn2[T*D];
static float ga[T*D],gctx[T*D],gQ[T*D],gK[T*D],gV[T*D],gn1[T*D],gx0[T*D];
static float gp[T*T],gsc[T*T];
static float gWq[D*D],gbq[D],gWk[D*D],gbk[D],gWv[D*D],gbv[D],gWo[D*D],gbo[D];
static float gWf1[F*D],gbf1[F],gWf2[D*F],gbf2[D];
static float gg1[D],gbe1[D],gg2[D],gbe2[D];

static void linear(const float*xin,int rows,int din,int dout,const float*W,const float*b,float*yy){
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,rows,dout,din,1.0f,xin,din,W,din,0.0f,yy,dout);
  for(int i=0;i<rows*dout;i++) yy[i]+=b[i%dout];
}
static void layernorm(const float*in,float*out,const float*g,const float*b){
  for(int t=0;t<T;t++){ const float*r=in+t*D; float*o=out+t*D;
    float m=0; for(int j=0;j<D;j++) m+=r[j]; m/=D;
    float v=0; for(int j=0;j<D;j++){float d=r[j]-m; v+=d*d;} v/=D;
    float inv=1.0f/sqrtf(v+EPS);
    for(int j=0;j<D;j++) o[j]=(r[j]-m)*inv*g[j]+b[j]; }
}
/* gin += dLN/dinp ;  gg,gb += */
static void ln_backward(const float*inp,const float*g,const float*go,float*gin,float*gg,float*gb){
  for(int t=0;t<T;t++){ const float*r=inp+t*D; const float*o=go+t*D; float*gi=gin+t*D;
    float m=0; for(int j=0;j<D;j++) m+=r[j]; m/=D;
    float v=0; for(int j=0;j<D;j++){float d=r[j]-m; v+=d*d;} v/=D;
    float inv=1.0f/sqrtf(v+EPS);
    float xhat[D],dxh[D],sd=0,sdx=0;
    for(int j=0;j<D;j++){ xhat[j]=(r[j]-m)*inv; dxh[j]=o[j]*g[j];
      gg[j]+=o[j]*xhat[j]; gb[j]+=o[j]; sd+=dxh[j]; sdx+=dxh[j]*xhat[j]; }
    float md=sd/D, mdx=sdx/D;
    for(int j=0;j<D;j++) gi[j]+=inv*(dxh[j]-md-xhat[j]*mdx); }
}
static void colsum(const float*grad,int rows,int cols,float*out){
  for(int j=0;j<cols;j++) out[j]=0;
  for(int i=0;i<rows*cols;i++) out[i%cols]+=grad[i];
}
static void softmax_rows(float*s){ for(int t=0;t<T;t++){ float*r=s+t*T;
  float mx=r[0]; for(int j=1;j<T;j++) if(r[j]>mx) mx=r[j];
  float sm=0; for(int j=0;j<T;j++){ r[j]=expf(r[j]-mx); sm+=r[j]; }
  for(int j=0;j<T;j++) r[j]/=sm; } }

int main(void){
  FILE*f=fopen("@BIN@","rb");
#define RD(A,N) if(fread(A,sizeof(float),N,f)!=(size_t)(N))return 1;
  RD(g1,D)RD(be1,D)RD(Wq,D*D)RD(bq,D)RD(Wk,D*D)RD(bk,D)RD(Wv,D*D)RD(bv,D)RD(Wo,D*D)RD(bo,D)
  RD(g2,D)RD(be2,D)RD(Wf1,F*D)RD(bf1,F)RD(Wf2,D*F)RD(bf2,D)RD(x0,T*D)
  fclose(f);
  float scale=1.0f/sqrtf((float)DH);

  /* ---------- forward (с сохранением активаций) ---------- */
  layernorm(x0,n1,g1,be1);
  linear(n1,T,D,D,Wq,bq,Q); linear(n1,T,D,D,Wk,bk,K); linear(n1,T,D,D,Wv,bv,V);
  for(int h=0;h<NH;h++){ float*p=P+h*T*T;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,T,T,DH,scale,Q+h*DH,D,K+h*DH,D,0.0f,p,T);
    softmax_rows(p);
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,DH,T,1.0f,p,T,V+h*DH,D,0.0f,ctx+h*DH,D); }
  linear(ctx,T,D,D,Wo,bo,a);
  for(int i=0;i<T*D;i++) x2[i]=x0[i]+a[i];
  layernorm(x2,n2,g2,be2);
  linear(n2,T,D,F,Wf1,bf1,h1); for(int i=0;i<T*F;i++) if(h1[i]<0) h1[i]=0;
  linear(h1,T,F,D,Wf2,bf2,h2);
  for(int i=0;i<T*D;i++) y[i]=x2[i]+h2[i];

  /* ---------- backward (seed градиента выхода = 1, как ones_like) ---------- */
  for(int i=0;i<T*D;i++) gy[i]=1.0f;
  /* y = x2 + h2 */
  for(int i=0;i<T*D;i++){ gx2[i]+=gy[i]; gh2[i]=gy[i]; }
  /* FFN: h2 = h1@Wf2^T+bf2 */
  colsum(gh2,T,D,gbf2);
  cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,D,F,T,1.0f,gh2,D,h1,F,0.0f,gWf2,F);
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,F,D,1.0f,gh2,D,Wf2,F,0.0f,gh1,F);
  for(int i=0;i<T*F;i++) gz1[i]=(h1[i]>0)?gh1[i]:0.0f;      /* relu backward */
  colsum(gz1,T,F,gbf1);
  cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,F,D,T,1.0f,gz1,F,n2,D,0.0f,gWf1,D);
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,D,F,1.0f,gz1,F,Wf1,D,0.0f,gn2,D);
  ln_backward(x2,g2,gn2,gx2,gg2,gbe2);
  /* x2 = x0 + a */
  for(int i=0;i<T*D;i++){ gx0[i]+=gx2[i]; ga[i]=gx2[i]; }
  /* attention out = ctx@Wo^T+bo */
  colsum(ga,T,D,gbo);
  cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,D,D,T,1.0f,ga,D,ctx,D,0.0f,gWo,D);
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,D,D,1.0f,ga,D,Wo,D,0.0f,gctx,D);
  for(int h=0;h<NH;h++){ float*p=P+h*T*T;
    /* ctx_h = p @ V_h */
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,T,T,DH,1.0f,gctx+h*DH,D,V+h*DH,D,0.0f,gp,T);
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,T,DH,T,1.0f,p,T,gctx+h*DH,D,0.0f,gV+h*DH,D);
    /* softmax backward */
    for(int i=0;i<T;i++){ float dot=0; for(int k=0;k<T;k++) dot+=gp[i*T+k]*p[i*T+k];
      for(int j=0;j<T;j++) gsc[i*T+j]=p[i*T+j]*(gp[i*T+j]-dot); }
    /* sc = scale*(Q_h@K_h^T) */
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,DH,T,scale,gsc,T,K+h*DH,D,0.0f,gQ+h*DH,D);
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,T,DH,T,scale,gsc,T,Q+h*DH,D,0.0f,gK+h*DH,D); }
  /* Q,K,V = n1@W^T+b ; gn1 накапливает три вклада */
  colsum(gQ,T,D,gbq); colsum(gK,T,D,gbk); colsum(gV,T,D,gbv);
  cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,D,D,T,1.0f,gQ,D,n1,D,0.0f,gWq,D);
  cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,D,D,T,1.0f,gK,D,n1,D,0.0f,gWk,D);
  cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,D,D,T,1.0f,gV,D,n1,D,0.0f,gWv,D);
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,D,D,1.0f,gQ,D,Wq,D,0.0f,gn1,D);
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,D,D,1.0f,gK,D,Wk,D,1.0f,gn1,D);
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,D,D,1.0f,gV,D,Wv,D,1.0f,gn1,D);
  ln_backward(x0,g1,gn1,gx0,gg1,gbe1);

  /* печать градиентов для сверки */
#define PR(name,A,N) { printf("%s",name); for(int i=0;i<N;i++) printf(" %.7g",A[i]); printf("\n"); }
  PR("gx",gx0,T*D) PR("gWq",gWq,D*D) PR("gWo",gWo,D*D) PR("gWf1",gWf1,F*D) PR("gWf2",gWf2,D*F)
  PR("gg1",gg1,D) PR("gbe1",gbe1,D) PR("gg2",gg2,D) PR("gbq",gbq,D) PR("gbf2",gbf2,D)
  return 0;
}
"""


def main():
    T, D, NH, F = 16, 32, 4, 64
    rng = np.random.default_rng(0)
    layer = nn.TransformerEncoderLayer(D, NH, F, seed=1)
    x_in = rng.standard_normal((1, T, D)).astype(np.float32)

    X = Tensor(x_in)
    y = layer(X)
    y.backward()                       # seed = ones_like(y), как в C

    a = layer.attn
    ff = layer.ff.layers
    order = [layer.norm1.weight, layer.norm1.bias,
             a.Wq.weight, a.Wq.bias, a.Wk.weight, a.Wk.bias,
             a.Wv.weight, a.Wv.bias, a.Wo.weight, a.Wo.bias,
             layer.norm2.weight, layer.norm2.bias,
             ff[0].weight, ff[0].bias, ff[2].weight, ff[2].bias]
    with open(BINFILE, "wb") as fh:
        for p in order:
            fh.write(np.ascontiguousarray(p.data, dtype=np.float32).tobytes())
        fh.write(np.ascontiguousarray(x_in[0], dtype=np.float32).tobytes())

    src = (C_TEMPLATE.replace("@T@", str(T)).replace("@D@", str(D))
           .replace("@NH@", str(NH)).replace("@F@", str(F)).replace("@BIN@", BINFILE))
    tmp = tempfile.mkdtemp()
    cpath, epath = os.path.join(tmp, "tb.c"), os.path.join(tmp, "tb")
    open(cpath, "w").write(src)
    subprocess.run(["gcc", "-O2", "-Wno-unused-result", "-o", epath, cpath,
                    "-lopenblas", "-lm"], check=True)
    out = subprocess.run([epath], capture_output=True, text=True, check=True).stdout

    c = {}
    for line in out.strip().splitlines():
        p = line.split()
        c[p[0]] = np.array(p[1:], dtype=np.float32)

    ref = {
        "gx": X.grad[0].ravel(), "gWq": a.Wq.weight.grad.ravel(),
        "gWo": a.Wo.weight.grad.ravel(), "gWf1": ff[0].weight.grad.ravel(),
        "gWf2": ff[2].weight.grad.ravel(), "gg1": layer.norm1.weight.grad.ravel(),
        "gbe1": layer.norm1.bias.grad.ravel(), "gg2": layer.norm2.weight.grad.ravel(),
        "gbq": a.Wq.bias.grad.ravel(), "gbf2": ff[2].bias.grad.ravel(),
    }
    print("=== Backward трансформер-блока в C+BLAS: сверка с Python autograd ===")
    print(f"T={T}, d_model={D}, heads={NH}, d_ff={F}\n")
    worst = 0.0
    for k in ref:
        d = np.abs(ref[k] - c[k]).max()
        scale = max(1e-6, np.abs(ref[k]).max())
        worst = max(worst, d / scale)
        print(f"  {k:5s}: max|py - C| = {d:.2e}  (отн. {d/scale:.1e})")
    print("\nИтог:", "OK ✓ backward совпал" if worst < 1e-3 else "РАСХОЖДЕНИЕ ✗",
          f"(худшее относительное расхождение {worst:.1e})")


if __name__ == "__main__":
    main()
