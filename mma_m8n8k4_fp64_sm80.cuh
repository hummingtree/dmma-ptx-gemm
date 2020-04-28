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

#pragma once

constexpr int pad = 4;

constexpr int WARP_M = 4;
constexpr int WARP_N = 4;

constexpr int INST_M = 8;
constexpr int INST_N = 8;
constexpr int INST_K = 4;

constexpr int MMA_M = INST_M * WARP_M;
constexpr int MMA_N = INST_N * WARP_N;
constexpr int MMA_K = INST_K;

struct WarpRegisterMapping {

  int lane_id;
  int group_id;
  int thread_id_in_group;

  __device__ WarpRegisterMapping(int thread_id) :
    lane_id(thread_id & 31),
    group_id(lane_id >> 2),         // = lane_id / 4
    thread_id_in_group(lane_id & 3) // = lane_id % 4
  {
  }
};

/*
 * For MmaOperandA and MmaOperandB data needs to be loaded from shared memory to the registers of the threads in a warp
 * according to how the matrix elements are distributed accross the threads in the (D)MMA instruciton.
 * The specific distribution is documented in
 *   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f64
 */
struct MmaOperandA {

  using reg_type = double;
  reg_type reg[WARP_M];

  template <int lda> __device__ inline void load(void *smem, int tile_k, int tile_m, const WarpRegisterMapping &wrm)
  { // Assuming col major smem layout

    reg_type *A = reinterpret_cast<reg_type *>(smem);
    // column
    int k = tile_k * MMA_K + wrm.thread_id_in_group;
#pragma unroll
    for (int i = 0; i < WARP_M / 2; i++) {
#pragma unroll
      for (int b = 0; b < 2; b++) {
        int m = tile_m * MMA_M + (i * 8 + wrm.group_id) * 2 + b;
        reg[i * 2 + b] = A[k * lda + m];
      }
    }
  }
};

struct MmaOperandB {

  using reg_type = double;
  reg_type reg[WARP_N];

  template <int ldb> __device__ inline void load(void *smem, int tile_k, int tile_n, const WarpRegisterMapping &wrm)
  { // Assuming col major smem layout

    reg_type *B = reinterpret_cast<reg_type *>(smem);
    // row
    int k = tile_k * MMA_K + wrm.thread_id_in_group;

#pragma unroll
    for (int i = 0; i < WARP_N / 2; i++) {
#pragma unroll
      for (int b = 0; b < 2; b++) {
        int n = tile_n * MMA_N + (i * 8 + wrm.group_id) * 2 + b;
        reg[i * 2 + b] = B[k * ldb + n];
      }
    }
  }
};

/*
 * For MmaOperandC data needs to be stored from registers of the threads in a warp to (global) memory
 * according to how the matrix elements are distributed accross the threads in the (D)MMA instruciton.
 * The specific distribution is documented in
 *   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f64
 */
struct MmaOperandC {

  using reg_type = double;
  reg_type reg[WARP_M * WARP_N * 2];

  __device__ MmaOperandC()
  {
#pragma unroll
    for (int i = 0; i < WARP_M * WARP_N * 2; i++) { reg[i] = 0; }
  }

  template <int ldc> __device__ void store(void *ptr, int m_offset, int n_offset, const WarpRegisterMapping &wrm)
  {
    reg_type *C = reinterpret_cast<reg_type *>(ptr);
#pragma unroll
    for (int c = 0; c < 2; c++) {
#pragma unroll
      for (int n_i = 0; n_i < WARP_N / 2; n_i++) {
#pragma unroll
        for (int m_i = 0; m_i < WARP_M / 2; m_i++) {
#pragma unroll
          for (int n_b = 0; n_b < 2; n_b++) {
            double tmp[2];
#pragma unroll
            for (int m_b = 0; m_b < 2; m_b++) {
              int n_iter = n_i * 2 + n_b;
              int m_iter = m_i * 2 + m_b;
              int c_iter = (n_iter * WARP_M + m_iter) * 2;

              tmp[m_b] = reg[c_iter + c];
              // C[gmem_m + gmem_n * ldc] = reg[c_iter + c];
            }
            int gmem_m = m_offset + (m_i * 8 + wrm.group_id) * 2;
            int gmem_n = n_offset + ((n_i * 4 + wrm.thread_id_in_group) * 2 + c) * 2 + n_b;
            asm("st.cs.global.v2.f64 [%0+0], {%1, %2};" ::"l"(&C[gmem_m + gmem_n * ldc]), "d"(tmp[0]), "d"(tmp[1]));
          }
        }
      }
    }
  }
};

/*
 * Actually calling the (D)MMA instruction.
 */
__device__ void mma(MmaOperandC &op_c, const MmaOperandA &op_a, const MmaOperandB &op_b)
{
#pragma unroll
  for (int m_iter = 0; m_iter < WARP_M; m_iter++) {
#pragma unroll
    for (int n_iter = 0; n_iter < WARP_N; n_iter++) {
      int c_iter = (n_iter * WARP_M + m_iter) * 2;
      asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                   : "+d"(op_c.reg[c_iter + 0]), "+d"(op_c.reg[c_iter + 1])
                   : "d"(op_a.reg[m_iter]), "d"(op_b.reg[n_iter]));
    }
  }
}
