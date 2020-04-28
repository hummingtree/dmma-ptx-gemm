
/**
 * Copyright (c) 2021, NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

#include "cuda_helper.cuh"

#include "mma_m8n8k4_fp64_sm80.cuh"

#define MATRIX_M 2048
#define MATRIX_N 2048 // 192 * 64
#define MATRIX_K 2048

#include <cuda_pipeline.h>

using namespace nvcuda::experimental;

/*
 * @brief Load data from global memory (col major) to smem (col major)
 */
template <int row_dim, int col_dim, int block_y, int block_z, int ld_gmem, int ld_smem>
__device__ inline void g2s_nn(double *smem, double *gmem, int row_offset, int col_offset, pipeline &pipe)
{
#pragma unroll
  for (int col = threadIdx.z; col < col_dim; col += block_z) {
#pragma unroll
    for (int row = threadIdx.y; row < row_dim; row += block_y) {
      int smem_idx = col * ld_smem + row;
      int gmem_idx = (col + col_offset) * ld_gmem + row_offset + row;
      memcpy_async(smem[smem_idx], gmem[gmem_idx], pipe);
    }
  }
}

/*
 * @brief Load data from global memory (col major) to smem (row major)
 */
template <int row_dim, int col_dim, int block_y, int block_z, int ld_gmem, int ld_smem>
__device__ inline void g2s_nt(double *smem, double *gmem, int row_offset, int col_offset, pipeline &pipe)
{
#pragma unroll
  for (int col = threadIdx.z; col < col_dim; col += block_z) {
#pragma unroll
    for (int row = threadIdx.y; row < row_dim; row += block_y) {
      int smem_idx = row * ld_smem + col;
      int gmem_idx = (col + col_offset) * ld_gmem + row_offset + row;
      memcpy_async(smem[smem_idx], gmem[gmem_idx], pipe);
    }
  }
}

/**
  This kernel performs GEMM.
  Data needed for each threadblock is first loaded into shared memory, before the thread-block
  level GEMM is performed.
*/
template <int block_y, int block_z, int M, int N, int K, int bM, int bN, int bK, int lda, int ldb, int ldc>
__global__ void __launch_bounds__(block_y *block_z, 1) mma_kernel(double *ptr_a, double *ptr_b, double *ptr_c)
{
  // Declare shared memory.
  extern __shared__ int smem[];

  // Offsets of this thread-block in global memory
  int m_offset = blockIdx.y * bM;
  int n_offset = blockIdx.z * bN;

  // For shared memory layout we use col majob for operand A and row major for operand B with pads.
  constexpr int ld_smem_a = bM + pad;
  constexpr int ld_smem_b = bN + pad;

  // Declare two sets of shared memory for pipelining using the memcpy_async (LDGSTS) instruction
  double *smem_a = reinterpret_cast<double *>(smem);
  double *smem_b = &smem_a[ld_smem_a * bK];

  double *smem_a_dup = &smem_b[ld_smem_b * bK];
  double *smem_b_dup = &smem_a_dup[ld_smem_a * bK];

  constexpr int tile_row_dim = bM / MMA_M; // number of tiles in the col dimension
  constexpr int tile_col_dim = bN / MMA_N; // number of tiles in the row dimension
  constexpr int tile_acc_dim = bK / MMA_K; // number of tiles in the acc dimension

  constexpr int total_warp = block_y * block_z / 32;

  constexpr int total_tile = tile_row_dim * tile_col_dim;
  constexpr int warp_cycle = total_tile / total_warp;

  static_assert(total_tile % total_warp == 0, "Total number of tiles should be divisible by the number of warps.");

  MmaOperandC op_c[warp_cycle]; // initilized to zero

  const int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const int warp_id = thread_id / 32;
  WarpRegisterMapping wrm(thread_id);

  double *smem_a_compute = smem_a;
  double *smem_b_compute = smem_b;
  double *smem_a_memory = smem_a_dup;
  double *smem_b_memory = smem_b_dup;

  pipeline pipe;

  g2s_nn<bM, bK, block_y, block_z, lda, ld_smem_a>(smem_a_compute, ptr_a, m_offset, 0, pipe);
  g2s_nt<bK, bN, block_y, block_z, ldb, ld_smem_b>(smem_b_compute, ptr_b, 0, n_offset, pipe);
  pipe.commit();

  int buffer_count = 0;
  for (int k_offset = 0; k_offset < K; k_offset += bK) {
    if (k_offset + bK < K) {
      g2s_nn<bM, bK, block_y, block_z, lda, ld_smem_a>(smem_a_memory, ptr_a, m_offset, k_offset + bK, pipe);
      g2s_nt<bK, bN, block_y, block_z, ldb, ld_smem_b>(smem_b_memory, ptr_b, k_offset + bK, n_offset, pipe);
    }
    // We use the one set of smem for compute, and let data being loaded into the other set of smem
    // while doing computation.
    pipe.commit();
    pipe.wait_prior<1>();
    __syncthreads();

    // MMA!
#pragma unroll
    for (int c = 0; c < warp_cycle; c++) {
      // The logical warp assigned to each part of the matrix.
      const int logical_warp_index = warp_id * warp_cycle + c;
      const int tile_m = logical_warp_index / tile_col_dim;
      const int tile_n = logical_warp_index - tile_m * tile_col_dim;

      for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {

        MmaOperandA op_a;
        op_a.template load<ld_smem_a>(smem_a_compute, tile_k, tile_m, wrm);

        MmaOperandB op_b;
        op_b.template load<ld_smem_b>(smem_b_compute, tile_k, tile_n, wrm);

        mma(op_c[c], op_a, op_b);

      } // tile_k
    }   // c

    if (k_offset + bK < K) {
      // If we have the next iteration to do, switch the smem buffers:
      // compute -> memory, memory -> compute
      __syncthreads();

      buffer_count = (buffer_count == 0 ? 1 : 0);
      smem_a_compute = buffer_count ? smem_a_dup : smem_a;
      smem_b_compute = buffer_count ? smem_b_dup : smem_b;
      smem_a_memory = buffer_count ? smem_a : smem_a_dup;
      smem_b_memory = buffer_count ? smem_b : smem_b_dup;
    }
  } // k_offset

  // Store result to ptr_c.
#pragma unroll
  for (int c = 0; c < warp_cycle; c++) {
    // The logical warp assigned to each part of the matrix.
    const int logical_warp_index = warp_id * warp_cycle + c;
    const int warp_row = logical_warp_index / tile_col_dim;
    const int warp_col = logical_warp_index - warp_row * tile_col_dim;

    op_c[c].template store<ldc>(ptr_c, m_offset + warp_row * MMA_M, n_offset + warp_col * MMA_N, wrm);
  }
}

int main(int argc, char *argv[])
{
  using F = double;

  F *a_fp64;
  F *b_fp64;

  F *c;
  F *c_cublas;
  F *c_mma;

  F *c_host_cublas;
  F *c_host_mma;

  curandGenerator_t gen;
  cublasHandle_t cublas_handle;

  cudaEvent_t start_mma;
  cudaEvent_t stop_mma;

  cudaEvent_t start_cublas;
  cudaEvent_t stop_cublas;

  cudaErrCheck(cudaEventCreate(&start_mma));
  cudaErrCheck(cudaEventCreate(&stop_mma));

  cudaErrCheck(cudaEventCreate(&start_cublas));
  cudaErrCheck(cudaEventCreate(&stop_cublas));

  cublasErrCheck(cublasCreate(&cublas_handle));

  // Use tensor cores
  cublasErrCheck(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

  cudaErrCheck(cudaMalloc((void **)&a_fp64, MATRIX_M * MATRIX_K * sizeof(F)));
  cudaErrCheck(cudaMalloc((void **)&b_fp64, MATRIX_K * MATRIX_N * sizeof(F)));

  cudaErrCheck(cudaMalloc((void **)&c, MATRIX_M * MATRIX_N * sizeof(F)));
  cudaErrCheck(cudaMalloc((void **)&c_cublas, MATRIX_M * MATRIX_N * sizeof(F)));
  cudaErrCheck(cudaMalloc((void **)&c_mma, MATRIX_M * MATRIX_N * sizeof(F)));

  c_host_cublas = (F *)malloc(MATRIX_M * MATRIX_N * sizeof(F));
  c_host_mma = (F *)malloc(MATRIX_M * MATRIX_N * sizeof(F));

  curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  curandErrCheck(curandGenerateUniformDouble(gen, a_fp64, MATRIX_M * MATRIX_K));
  curandErrCheck(curandGenerateUniformDouble(gen, b_fp64, MATRIX_K * MATRIX_N));

  curandErrCheck(curandGenerateUniformDouble(gen, c, MATRIX_M * MATRIX_N));

  curandErrCheck(curandDestroyGenerator(gen));

  cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(F), cudaMemcpyDeviceToDevice));
  cudaErrCheck(cudaMemcpy(c_mma, c, MATRIX_M * MATRIX_N * sizeof(F), cudaMemcpyDeviceToDevice));

  using ElementA = F;
  using ElementB = F;
  using ElementC = F;

  ElementC alpha = 1.0;
  ElementC beta = 0.0;

  printf("\nMatrix M = %d, Matrix N = %d, Matrix K = %d.\n\n", MATRIX_M, MATRIX_N, MATRIX_K);

  // First: using MMA
  constexpr int bM = 64;
  constexpr int bN = 64;
  constexpr int bK = 32;

  constexpr int block_y = 8;
  constexpr int block_z = 16;

  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = 1;
  blockDim.y = block_y; // ThreadblockShape / WarpShape
  blockDim.z = block_z;

  gridDim.x = 1;
  gridDim.y = MATRIX_M / bM;
  gridDim.z = MATRIX_N / bN;

  int shared_memory_size = ((bM + pad) * bK * sizeof(ElementA) + bK * (bN + pad) * sizeof(ElementB)) * 2;

  printf("Running with MMA ...\n");

  printf("Shared memory size = %05d\n", shared_memory_size);

  int niter = 1;

  auto kernel = mma_kernel<block_y, block_z, MATRIX_M, MATRIX_N, MATRIX_K, bM, bN, bK, MATRIX_M, MATRIX_K, MATRIX_M>;
  cudaErrCheck(cudaFuncSetAttribute((const void *)kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                    (int)cudaSharedmemCarveoutMaxShared));
  cudaErrCheck(
    cudaFuncSetAttribute((const void *)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
  cudaErrCheck(cudaEventRecord(start_mma));
  for (int i = 0; i < niter; i++) {
    kernel<<<gridDim, blockDim, shared_memory_size>>>(
      reinterpret_cast<ElementA *>(a_fp64), reinterpret_cast<ElementB *>(b_fp64), reinterpret_cast<ElementC *>(c_mma));
  }

  cudaErrCheck(cudaEventRecord(stop_mma));
  cudaErrCheck(cudaPeekAtLastError());

  // Now using cuBLAS
  printf("Running with cuBLAS...\n");
  cudaErrCheck(cudaEventRecord(start_cublas));

  for (int i = 0; i < niter; i++) {
    cublasErrCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_M, MATRIX_N, MATRIX_K, &alpha, a_fp64,
                                CUDA_R_64F, MATRIX_M, b_fp64, CUDA_R_64F, MATRIX_K, &beta, c_cublas, CUDA_R_64F,
                                MATRIX_M, CUDA_R_64F, CUBLAS_GEMM_DFALT_TENSOR_OP));
  }
  cudaErrCheck(cudaEventRecord(stop_cublas));

  // Error checking
  printf("\nChecking results...\n");
  cudaErrCheck(cudaMemcpy(c_host_mma, c_mma, MATRIX_M * MATRIX_N * sizeof(F), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(F), cudaMemcpyDeviceToHost));

  int errors = 0;

  for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
    if (fabs(c_host_mma[i] - c_host_cublas[i]) > 1e-8) {
      errors++;
      if (i < 10) printf("%05d: %8.4f vs. %8.4f.\n", i, c_host_mma[i], c_host_cublas[i]);
    }
  }

  if (errors > 0) {
    printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
  } else {
    printf("Results verified: cublas and WMMA agree.\n\n");
    float mma_time;
    float cublas_time;
    cudaErrCheck(cudaEventSynchronize(stop_mma));
    cudaErrCheck(cudaEventSynchronize(stop_cublas));
    cudaErrCheck(cudaEventElapsedTime(&mma_time, start_mma, stop_mma));
    cudaErrCheck(cudaEventElapsedTime(&cublas_time, start_cublas, stop_cublas));
    printf("mma took %8.4f ms = %.1f %% cublas.\n", mma_time, 100.0 * cublas_time / mma_time);
    printf("cublas took %8.4f ms, %.1f TFLOPS.\n", cublas_time,
           2.0 * MATRIX_M * MATRIX_N * MATRIX_K * niter / (cublas_time * 1e-3) / 1e+12);
  }

  cudaErrCheck(cudaEventDestroy(start_mma));
  cudaErrCheck(cudaEventDestroy(stop_mma));

  cudaErrCheck(cudaEventDestroy(start_cublas));
  cudaErrCheck(cudaEventDestroy(stop_cublas));

  cudaErrCheck(cudaFree(a_fp64));
  cudaErrCheck(cudaFree(b_fp64));

  cudaErrCheck(cudaFree(c));
  cudaErrCheck(cudaFree(c_cublas));
  cudaErrCheck(cudaFree(c_mma));

  free(c_host_cublas);
  free(c_host_mma);

  cudaErrCheck(cudaDeviceReset());
  return 0;
}
