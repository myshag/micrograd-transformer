// Настоящие CUDA C ядра (для RealCudaBackend в backends.py).
//
// Это обычный CUDA C: компилируется nvcc/NVRTC в PTX->SASS и исполняется на
// SM-ядрах GPU. Логика — та же, что в CUDA-кернелах симулятора Numba, только
// теперь это код под настоящее железо. extern "C" — чтобы имена не искажались
// и их можно было найти по имени из Python (CuPy RawModule).
//
// Здесь НЕ запускается (нет GPU/toolkit); предназначено для машины с NVIDIA GPU.

extern "C" __global__
void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // строка
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // столбец
    if (i < M && j < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += A[i * K + k] * B[k * N + j];
        C[i * N + j] = acc;
    }
}

extern "C" __global__
void ew_add(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}

extern "C" __global__
void ew_mul(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] * b[i];
}

extern "C" __global__
void ew_relu(const float* x, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; o[i] = v > 0.0f ? v : 0.0f; }
}

extern "C" __global__
void ew_tanh(const float* x, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = tanhf(x[i]);
}

extern "C" __global__
void ew_sigmoid(const float* x, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = 1.0f / (1.0f + expf(-x[i]));
}

// backward активаций: o = градиент по входу (по forward-значению и gy)
extern "C" __global__
void grad_relu(const float* x, const float* gy, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = x[i] > 0.0f ? gy[i] : 0.0f;
}

extern "C" __global__
void grad_tanh(const float* t, const float* gy, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (1.0f - t[i] * t[i]) * gy[i];
}

extern "C" __global__
void grad_sigmoid(const float* s, const float* gy, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = s[i] * (1.0f - s[i]) * gy[i];
}
