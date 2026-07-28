// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Inspect the CUDA 12.9 16U4_ALIGN16B TensorMap load representation on SM100.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>

#include <array>
#include <cstdint>
#include <cstdio>

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status_)); \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

#define CHECK_DRIVER(call)                                                      \
  do {                                                                          \
    CUresult status_ = (call);                                                   \
    if (status_ != CUDA_SUCCESS) {                                               \
      const char* message_ = nullptr;                                            \
      cuGetErrorString(status_, &message_);                                      \
      std::fprintf(stderr, "%s failed: %s\n", #call, message_);                 \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

struct alignas(128) SharedStorage {
  alignas(8) uint64_t barrier;
  alignas(128) uint8_t tile[128];
};

__global__ void unpack4_kernel(
    const __grid_constant__ CUtensorMap tensor_map, uint8_t* output) {
  __shared__ SharedStorage storage;
  if (threadIdx.x == 0) {
    cutlass::arch::ClusterBarrier::init(&storage.barrier, 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        &storage.barrier, 64);
    cute::SM90_TMA_LOAD_2D::copy(
        &tensor_map, &storage.barrier, 0, storage.tile, 0, 0);
    cutlass::arch::ClusterBarrier::wait(&storage.barrier, 0);
  }
  __syncthreads();

  if (threadIdx.x < 128) {
    output[threadIdx.x] = storage.tile[threadIdx.x];
  }
}

int main() {
  constexpr uint32_t kValues = 128;
  constexpr uint32_t kPackedBytes = kValues / 2;
  std::array<uint8_t, kPackedBytes> input{};
  for (uint32_t i = 0; i < kPackedBytes; ++i) {
    input[i] = static_cast<uint8_t>((2 * i) % 16) |
               static_cast<uint8_t>(((2 * i + 1) % 16) << 4);
  }

  uint8_t* device_input = nullptr;
  uint8_t* device_output = nullptr;
  CHECK_CUDA(cudaMalloc(&device_input, kPackedBytes));
  CHECK_CUDA(cudaMalloc(&device_output, kValues));
  CHECK_CUDA(cudaMemcpy(
      device_input, input.data(), kPackedBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(device_output, 0xcd, kValues));

  CUtensorMap tensor_map{};
  const cuuint64_t global_dim[2] = {kValues, 1};
  const cuuint64_t global_stride[1] = {kPackedBytes};
  const cuuint32_t box_dim[2] = {kValues, 1};
  const cuuint32_t element_stride[2] = {1, 1};
  CHECK_DRIVER(cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 2, device_input,
      global_dim, global_stride, box_dim, element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  unpack4_kernel<<<1, 128>>>(tensor_map, device_output);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::array<uint8_t, kValues> output{};
  CHECK_CUDA(cudaMemcpy(
      output.data(), device_output, kValues, cudaMemcpyDeviceToHost));
  bool exact = true;
  for (uint32_t i = 0; i < kValues; ++i) {
    const uint8_t expected = static_cast<uint8_t>(i % 16);
    if (output[i] != expected) {
      exact = false;
    }
    std::printf("%02x%s", output[i], (i + 1) % 32 == 0 ? "\n" : " ");
  }
  std::printf("TMA_UNPACK4_EXACT=%s\n", exact ? "true" : "false");

  CHECK_CUDA(cudaFree(device_output));
  CHECK_CUDA(cudaFree(device_input));
  return exact ? 0 : 2;
}
