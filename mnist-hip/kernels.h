#include "layer.h"

// Constructor
Layer::Layer(int M, int N, int O)
{
  this->M = M;
  this->N = N;
  this->O = O;

  float h_bias[N];
  float h_weight[N][M];

  output = NULL;
  preact = NULL;
  bias   = NULL;
  weight = NULL;

  for (int i = 0; i < N; ++i) {
    h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

    for (int j = 0; j < M; ++j)
      h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
  }

  hipMalloc(&output, sizeof(float) * O);
  hipMalloc(&preact, sizeof(float) * O);

  hipMalloc(&bias, sizeof(float) * N);
  hipMalloc(&weight, sizeof(float) * M * N);

  hipMalloc(&d_output, sizeof(float) * O);
  hipMalloc(&d_preact, sizeof(float) * O);
  hipMalloc(&d_weight, sizeof(float) * M * N);

  hipMemcpy(bias, h_bias, sizeof(float) * N, hipMemcpyHostToDevice);
  hipMemcpy(weight, h_weight, sizeof(float) * M * N, hipMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
  hipFree(output);
  hipFree(preact);

  hipFree(bias);
  hipFree(weight);

  hipFree(d_output);
  hipFree(d_preact);
  hipFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
  hipMemcpy(output, data, sizeof(float) * O, hipMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
  hipMemset(output, 0x00, sizeof(float) * O);
  hipMemset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear()
{
  hipMemset(d_output, 0x00, sizeof(float) * O);
  hipMemset(d_preact, 0x00, sizeof(float) * O);
  hipMemset(d_weight, 0x00, sizeof(float) * M * N);
}


__device__ float step_function(float v)
{
  return 1.f / (1.f + expf(-v));
}

__global__ void apply_step_function(float *input, float *output, const int N)
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
    output[idx] = step_function(input[idx]);
  }
}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
    err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
  }
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;

  for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
    output[idx] += dt * grad[idx];
  }
}

__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 5*5*6*24*24;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 5);
    const int i2 = ((idx /= 5  ) % 5);
    const int i3 = ((idx /= 5  ) % 6);
    const int i4 = ((idx /= 6  ) % 24);
    const int i5 = ((idx /= 24  ) % 24);
    atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
  }
}

__global__ void fp_bias_c1(float preact[6][24][24], float bias[6])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6*24*24;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 24);
    const int i3 = ((idx /= 24  ) % 24);
    preact[i1][i2][i3] += bias[i1];
  }
}

__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 4*4*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 4);
    const int i2 = ((idx /= 4  ) % 4);
    const int i3 = ((idx /= 4  ) % 6);
    const int i4 = ((idx /= 6  ) % 6);
    const int i5 = ((idx /= 6  ) % 6);
    atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
  }
}

__global__ void fp_bias_s1(float preact[6][6][6], float bias[1])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    preact[i1][i2][i3] += bias[0];
  }
}

__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 10*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 10);
    const int i2 = ((idx /= 10  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    const int i4 = ((idx /= 6  ) % 6);
    atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
  }
}

__global__ void fp_bias_f(float preact[10], float bias[10])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 10;
  for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
    preact[idx] += bias[idx];
  }
}

__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 10*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 10);
    const int i2 = ((idx /= 10  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    const int i4 = ((idx /= 6  ) % 6);
    d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
  }
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 10;
  for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
    bias[idx] += dt * d_preact[idx];
  }
}

__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 10*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 10);
    const int i2 = ((idx /= 10  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    const int i4 = ((idx /= 6  ) % 6);
    atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
  }
}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    const float o = step_function(preact[i1][i2][i3]);
    d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
  }
}

__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 1*4*4*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 1);
    const int i2 = ((idx /= 1  ) % 4);
    const int i3 = ((idx /= 4  ) % 4);
    const int i4 = ((idx /= 4  ) % 6);
    const int i5 = ((idx /= 6  ) % 6);
    const int i6 = ((idx /= 6  ) % 6);
    atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
  }
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6*6*6;
  const float d = 216; //pow(6.0f, 3.0f);

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
  }
}

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 1*4*4*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 1);
    const int i2 = ((idx /= 1  ) % 4);
    const int i3 = ((idx /= 4  ) % 4);
    const int i4 = ((idx /= 4  ) % 6);
    const int i5 = ((idx /= 6  ) % 6);
    const int i6 = ((idx /= 6  ) % 6);
    atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
  }
}

__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6*24*24;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 24);
    const int i3 = ((idx /= 24  ) % 24);
    const float o = step_function(preact[i1][i2][i3]);
    d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
  }
}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6*5*5*24*24;
  const float d = 576; //pow(24.0f, 2.0f);

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 5);
    const int i3 = ((idx /= 5  ) % 5);
    const int i4 = ((idx /= 5  ) % 24);
    const int i5 = ((idx /= 24  ) % 24);
    atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
  }
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24])
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6*24*24;
  const float d = 576; //pow(24.0f, 2.0f);

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 24);
    const int i3 = ((idx /= 24  ) % 24);
    atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
  }
}
