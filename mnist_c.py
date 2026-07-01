"""
Многослойный C-обучатель MNIST с softmax — весь цикл обучения в C+BLAS.

Классификатор 784 -> 128 (ReLU) -> 10 (softmax), обучаемый минибатчевым SGD.
Forward, softmax+кросс-энтропия, backprop через оба слоя и SGD-шаг —
всё выполняется в одном скомпилированном C-бинарнике с вызовами OpenBLAS.
Python лишь готовит данные и веса (пишет в бинарный файл) и запускает бинарник.

Математика backward — та же, что в нашем проверенном autograd:
  softmax+CE:  dZ2 = (probs - onehot) / B
  слой 2:      gW2 = A1ᵀ@dZ2,  dA1 = dZ2@W2ᵀ
  relu:        dZ1 = dA1 · (A1>0)
  слой 1:      gW1 = Xᵀ@dZ1

Требуется: gcc, libopenblas-dev, data/mnist.npz (см. download_mnist.py).
Запуск:  python3 mnist_c.py
"""

import os
import subprocess
import tempfile
import time

import numpy as np

DIN, DOUT = 784, 10
DATAFILE = "/tmp/mnist_c.bin"

C_TEMPLATE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

#define NTR @NTR@
#define NTE @NTE@
#define DIN 784
#define H   @H@
#define DOUT 10
#define B   @B@

static float Xtr[(long)NTR*DIN], Xte[(long)NTE*DIN];
static int   ytr[NTR], yte[NTE];
static float W1[DIN*H], b1[H], W2[H*DOUT], b2[DOUT];
/* батч и промежуточные буферы */
static float xb[B*DIN]; static int yb[B];
static float Z1[B*H], A1[B*H], Z2[B*DOUT], probs[B*DOUT];
static float dZ2[B*DOUT], dA1[B*H], dZ1[B*H];
static float gW1[DIN*H], gb1[H], gW2[H*DOUT], gb2[DOUT];

static unsigned long rs = 88172645463325252UL;   /* xorshift RNG */
static unsigned long xr(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                     return t.tv_sec+t.tv_nsec*1e-9; }

/* forward для n<=B строк из X (n×DIN): заполняет A1 (n×H) и probs (n×DOUT) */
static void forward(const float* X, int n){
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,H,DIN,
              1.0f,X,DIN,W1,H,0.0f,Z1,H);
  for(int i=0;i<n*H;i++){ float v=Z1[i]+b1[i%H]; A1[i]=v>0?v:0; }   /* +b1, relu */
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,DOUT,H,
              1.0f,A1,H,W2,DOUT,0.0f,Z2,DOUT);
  for(int i=0;i<n*DOUT;i++) Z2[i]+=b2[i%DOUT];
  for(int r=0;r<n;r++){                                             /* softmax по строке */
    float* z=Z2+r*DOUT; float* p=probs+r*DOUT;
    float m=z[0]; for(int j=1;j<DOUT;j++) if(z[j]>m) m=z[j];
    float s=0; for(int j=0;j<DOUT;j++){ p[j]=expf(z[j]-m); s+=p[j]; }
    for(int j=0;j<DOUT;j++) p[j]/=s;
  }
}

static float test_accuracy(){
  int correct=0;
  for(int off=0; off<NTE; off+=B){
    int n = (NTE-off < B) ? (NTE-off) : B;
    forward(Xte+(long)off*DIN, n);
    for(int r=0;r<n;r++){
      int arg=0; float best=probs[r*DOUT];
      for(int j=1;j<DOUT;j++) if(probs[r*DOUT+j]>best){best=probs[r*DOUT+j];arg=j;}
      if(arg==yte[off+r]) correct++;
    }
  }
  return (float)correct/NTE;
}

int main(void){
  FILE* f=fopen("@DATAFILE@","rb");
  fread(Xtr,sizeof(float),(long)NTR*DIN,f); fread(ytr,sizeof(int),NTR,f);
  fread(Xte,sizeof(float),(long)NTE*DIN,f); fread(yte,sizeof(int),NTE,f);
  fread(W1,sizeof(float),DIN*H,f); fread(b1,sizeof(float),H,f);
  fread(W2,sizeof(float),H*DOUT,f); fread(b2,sizeof(float),DOUT,f);
  fclose(f);

  float lr=@LR@f; double t0=now();
  for(int it=1; it<=@ITERS@; it++){
    for(int i=0;i<B;i++){ int idx=xr()%NTR;
      memcpy(xb+(long)i*DIN, Xtr+(long)idx*DIN, DIN*sizeof(float)); yb[i]=ytr[idx]; }
    forward(xb, B);

    /* градиент softmax+CE: dZ2 = (probs - onehot)/B */
    memcpy(dZ2, probs, B*DOUT*sizeof(float));
    for(int i=0;i<B;i++) dZ2[i*DOUT+yb[i]] -= 1.0f;
    for(int i=0;i<B*DOUT;i++) dZ2[i] /= B;

    /* слой 2 */
    for(int j=0;j<DOUT;j++) gb2[j]=0; for(int i=0;i<B*DOUT;i++) gb2[i%DOUT]+=dZ2[i];
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,H,DOUT,B,
                1.0f,A1,H,dZ2,DOUT,0.0f,gW2,DOUT);
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,B,H,DOUT,
                1.0f,dZ2,DOUT,W2,DOUT,0.0f,dA1,H);
    /* relu backward */
    for(int i=0;i<B*H;i++) dZ1[i] = A1[i]>0 ? dA1[i] : 0.0f;
    /* слой 1 */
    for(int j=0;j<H;j++) gb1[j]=0; for(int i=0;i<B*H;i++) gb1[i%H]+=dZ1[i];
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,DIN,H,B,
                1.0f,xb,DIN,dZ1,H,0.0f,gW1,H);

    /* SGD-шаг */
    for(int i=0;i<DIN*H;i++)  W1[i]-=lr*gW1[i];
    for(int j=0;j<H;j++)      b1[j]-=lr*gb1[j];
    for(int i=0;i<H*DOUT;i++) W2[i]-=lr*gW2[i];
    for(int j=0;j<DOUT;j++)   b2[j]-=lr*gb2[j];

    if(it%@LOG@==0){
      float L=0; for(int i=0;i<B;i++) L += -logf(probs[i*DOUT+yb[i]]+1e-9f);
      printf("iter %5d  loss %.4f  test_acc %.4f\n", it, L/B, test_accuracy());
    }
  }
  printf("TIME %.2f\n", now()-t0);
  printf("FINAL_TEST_ACC %.4f\n", test_accuracy());
  return 0;
}
"""


def build_dataset(n_train=60000, seed=0):
    d = np.load("data/mnist.npz")
    Xtr = (d["x_train"].astype(np.float32) / 255.0)[:n_train]
    ytr = d["y_train"].astype(np.int32)[:n_train]
    Xte = d["x_test"].astype(np.float32) / 255.0
    yte = d["y_test"].astype(np.int32)
    rng = np.random.default_rng(seed)
    # He-инициализация (как в нашей Python-модели)
    W1 = (rng.standard_normal((DIN, 128)) * np.sqrt(2 / DIN)).astype(np.float32)
    b1 = np.zeros(128, np.float32)
    W2 = (rng.standard_normal((128, DOUT)) * np.sqrt(2 / 128)).astype(np.float32)
    b2 = np.zeros(DOUT, np.float32)
    with open(DATAFILE, "wb") as f:
        for a in (Xtr, ytr, Xte, yte, W1, b1, W2, b2):
            f.write(np.ascontiguousarray(a).tobytes())
    return len(Xtr), len(Xte), 128


def main(iters=5000, batch=64, lr=0.1, log_every=500):
    if not os.path.exists("data/mnist.npz"):
        raise SystemExit("Нет data/mnist.npz — сначала: python3 download_mnist.py")
    ntr, nte, hidden = build_dataset()

    src = (C_TEMPLATE
           .replace("@NTR@", str(ntr)).replace("@NTE@", str(nte))
           .replace("@H@", str(hidden)).replace("@B@", str(batch))
           .replace("@ITERS@", str(iters)).replace("@LR@", str(lr))
           .replace("@LOG@", str(log_every)).replace("@DATAFILE@", DATAFILE))

    tmp = tempfile.mkdtemp()
    cpath, epath = os.path.join(tmp, "m.c"), os.path.join(tmp, "m")
    open(cpath, "w").write(src)
    print("Компилирую многослойный обучатель (gcc + OpenBLAS)...")
    subprocess.run(["gcc", "-O2", "-Wno-unused-result", "-o", epath, cpath,
                    "-lopenblas", "-lm"], check=True)

    print(f"Обучаю MNIST 784->{hidden}->10 ЦЕЛИКОМ в C "
          f"(batch={batch}, lr={lr}, iters={iters}):\n")
    out = subprocess.run([epath], capture_output=True, text=True, check=True).stdout
    ctime = None
    for line in out.splitlines():
        if line.startswith("TIME"):
            ctime = float(line.split()[1])
        elif line.startswith("FINAL_TEST_ACC"):
            print(f"\nИтоговая точность на тесте: {float(line.split()[1]):.4f}")
        else:
            print(line)
    if ctime is not None:
        print(f"Всё обучение заняло {ctime:.2f} c в скомпилированном C+BLAS.")


if __name__ == "__main__":
    main()
