#include <stdio.h>
#include <mma.h>
#include <cstdint>
#include <iostream>
#include <torch/extension.h>
#include "./spmm_utils/dense_tile.h"
#include "./spmm_utils/compute.h"
#include "./spmm_utils/output_tile.h"
#define mma_k = 8
#include <cuda_fp16.h>
#include <cuda_runtime.h>


template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const half* __restrict__ t_value,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    const int* t_atomic,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    uint32_t output_fragment_[2] = {0,0};
    half * output_fragment = reinterpret_cast<half *>(output_fragment_);
    //稀疏的块, 16*8
    // __shared__ at::Half sparse_[64];
    // half * sparse = reinterpret_cast<half *>(sparse_);
    // __shared__ int sparse_to_col[8];

    float sparse_fragment[1] = {0.0};
    at::Half dense_fragment1_[4] = {0.0, 0.0, 0.0, 0.0};
    half * sparse_fragment1 = reinterpret_cast<half *>(sparse_fragment);
    half * dense_fragment = reinterpret_cast<half *>(dense_fragment1_);
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    
    const int * t_column_ = t_column + t_win_offset*8 + ((warpin_id %4)*2);
    // uint32_t * t_output_fragment_ = reinterpret_cast<uint32_t*>(t_output_fragment);
    //读取稠密矩阵的行偏移

    const half2 * matrix_base_ = reinterpret_cast<const half2 *>(rhs_matrix + dimN_index);
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
        // __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        // int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        long binary = __ldg(t_binary + t_win_offset + i);

        long temp = (binary >> (warpin_id*2));
        long a= 1;
        long mask = (a << (warpin_id*2));
        int fifthBit = ((temp) & 1);
        int block_offset = -1;
        if(fifthBit == 1){
            block_offset = __popcll(binary & (mask-1));
            sparse_fragment1[0] = __ldg(t_value + value_offset + block_offset);
        }else{
            sparse_fragment1[0]=__float2half(0.0);
        }
        fifthBit = ((temp>>1) & 1);
        if(fifthBit == 1){
            if(block_offset==-1)
            {
                mask = (a << ((warpin_id*2)+1));
                block_offset = __popcll(binary & (mask-1));
            }
            sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset + 1);
        }else{
            sparse_fragment1[1]=__float2half(0.0);
        }
        //搬运稀疏数据
        // if(threadIdx.x<nnz_block)
        // {
        //     half v = __ldg(t_value + value_offset + threadIdx.x);
        //     int row = __ldg(t_row + value_offset + threadIdx.x);
        //     int col = __ldg(t_col + value_offset + threadIdx.x);
        //     *(sparse + row*8 + col) = v;
        // }
        //  __syncthreads();
        //搬运dense数据
        // int col =  __ldg(t_column_ + (threadIdx.x%4)*2);
        // int col1 =  __ldg(t_column_ + (threadIdx.x%4)*2 + 1);
        long col_temp[2];
        for(int k=0; k<2; k++)
            col_temp[k] = __ldg(t_column_ + k);
        t_column_ += 8;
        int col_offset =  (warp_id<<3) + (warpin_id/4);
        for(int i=0;i<2;i++)
        {
            if(col_temp[i] != -1){
                const long offset = (col_temp[i]*(nOri/2));
                half2 temp = __ldg(matrix_base_ + offset + col_offset);
                dense_fragment[i]= temp.x;
                dense_fragment[i + 2]= temp.y;
            }else{
                dense_fragment[i]= __float2half(0.0);
                dense_fragment[i + 2]= __float2half(0.0);
            }
        
        }

        __syncwarp();

            //MMA计算
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]),  "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );  
            
    }
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = __half2float(output_fragment[0]);
                if((col+1)<nOri)
                *(output_matrix_+1) =  __half2float(output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = __half2float(output_fragment[1]);
                    if((col+1)<nOri)
                    *(output_matrix_+1) = __half2float( output_fragment[3]);
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ ,__half2float(output_fragment[0]));
                if((col+1)<nOri)
                atomicAdd(output_matrix_+1, __half2float(output_fragment[2]));
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , __half2float(output_fragment[1]));
                    if((col+1)<nOri)
                    atomicAdd(output_matrix_+1 , __half2float(output_fragment[3]));
                }
            }
        }
}



static __device__ __forceinline__ half2 u32_to_half2(uint32_t x) {
    // half2 在寄存器里就是 32-bit packed
    return *reinterpret_cast<half2*>(&x);
}

static __device__ __forceinline__ uint32_t half2_to_u32(half2 x) {
    return *reinterpret_cast<uint32_t*>(&x);
}

// 用 CUDA cores 的 half2 FMA 模拟：D = A*B + C（寄存器粒度）
static __device__ __forceinline__ void mma_m16n8k8_cc_emulate(
    uint32_t &out0_u32, uint32_t &out1_u32,   // 对应 {%0,%1}
    const uint32_t a0_u32, const uint32_t a1_u32, // 对应 {%2,%3}
    const uint32_t b0_u32                      // 对应 {%4}
) {
    half2 out0 = u32_to_half2(out0_u32);
    half2 out1 = u32_to_half2(out1_u32);

    const half2 a0 = u32_to_half2(a0_u32);
    const half2 a1 = u32_to_half2(a1_u32);
    const half2 b0 = u32_to_half2(b0_u32);

    // CUDA cores：逐 lane 做 half2 FMA
    out0 = __hfma2(a0, b0, out0);
    out1 = __hfma2(a1, b0, out1);

    out0_u32 = half2_to_u32(out0);
    out1_u32 = half2_to_u32(out1);
}

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_cc(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const half* __restrict__ t_value,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    const int* t_atomic,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;

constexpr int WARPS_PER_BLOCK = 4;
    //判断执行tcu还是cuda
__shared__ half smem_A[WARPS_PER_BLOCK][32 * 2];
__shared__ half smem_B[WARPS_PER_BLOCK][32 * 4];
__shared__ half smem_ACC[WARPS_PER_BLOCK][32 * 4];

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    uint32_t output_fragment_[2] = {0,0};
    half * output_fragment = reinterpret_cast<half *>(output_fragment_);

    //稀疏的块, 16*8
    // __shared__ at::Half sparse_[64];
    // half * sparse = reinterpret_cast<half *>(sparse_);
    // __shared__ int sparse_to_col[8];

    float sparse_fragment[2] = {0.0, 0.0};
    at::Half dense_fragment1_[4] = {0.0, 0.0, 0.0, 0.0};
    half * sparse_fragment1 = reinterpret_cast<half *>(sparse_fragment);
    half * dense_fragment = reinterpret_cast<half *>(dense_fragment1_);
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    
    const int * t_column_ = t_column + t_win_offset*8 + ((warpin_id %4)*2);
    // uint32_t * t_output_fragment_ = reinterpret_cast<uint32_t*>(t_output_fragment);
    //读取稠密矩阵的行偏移

    const half2 * matrix_base_ = reinterpret_cast<const half2 *>(rhs_matrix + dimN_index);
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
        // __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        // int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        long binary = __ldg(t_binary + t_win_offset + i);

        long temp = (binary >> (warpin_id*2));
        long a= 1;
        long mask = (a << (warpin_id*2));
        int fifthBit = ((temp) & 1);
        int block_offset = -1;
        if(fifthBit == 1){
            block_offset = __popcll(binary & (mask-1));
            sparse_fragment1[0] = __ldg(t_value + value_offset + block_offset);
        }else{
            sparse_fragment1[0]=__float2half(0.0);
        }
        fifthBit = ((temp>>1) & 1);
        if(fifthBit == 1){
            if(block_offset==-1)
            {
                mask = (a << ((warpin_id*2)+1));
                block_offset = __popcll(binary & (mask-1));
            }
            sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset + 1);
        }else{
            sparse_fragment1[1]=__float2half(0.0);
        }
        //搬运稀疏数据
        // if(threadIdx.x<nnz_block)
        // {
        //     half v = __ldg(t_value + value_offset + threadIdx.x);
        //     int row = __ldg(t_row + value_offset + threadIdx.x);
        //     int col = __ldg(t_col + value_offset + threadIdx.x);
        //     *(sparse + row*8 + col) = v;
        // }
        //  __syncthreads();
        //搬运dense数据
        // int col =  __ldg(t_column_ + (threadIdx.x%4)*2);
        // int col1 =  __ldg(t_column_ + (threadIdx.x%4)*2 + 1);
        long col_temp[2];
        for(int k=0; k<2; k++)
            col_temp[k] = __ldg(t_column_ + k);
        t_column_ += 8;
        int col_offset =  (warp_id<<3) + (warpin_id/4);
        for(int i=0;i<2;i++)
        {
            if(col_temp[i] != -1){
                const long offset = (col_temp[i]*(nOri/2));
                half2 temp = __ldg(matrix_base_ + offset + col_offset);
                dense_fragment[i]= temp.x;
                dense_fragment[i + 2]= temp.y;
            }else{
                dense_fragment[i]= __float2half(0.0);
                dense_fragment[i + 2]= __float2half(0.0);
            }
        
        }

        __syncwarp();
        
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            smem_A[warp_id][warpin_id * 2 + t] = sparse_fragment1[t];
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            smem_B[warp_id][warpin_id * 4 + t] = dense_fragment[t];
        }

        __syncwarp();

            //MMA计算
        // asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
        //         "{%0,%1}, \t"
        //         "{%2,%3}, \t"
        //         "{%4}, \t"
        //         "{%0,%1}; ":
        //         "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
        //         "r"(dense_fragment_[0]),  "r"(dense_fragment_[1]),
        //         "r"(sparse_fragment_[0])
        // );  
        // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("%f\n", __half2float(output_fragment[0]));
        // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("%f\n", __half2float(output_fragment[1]));
        // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("%f\n", __half2float(output_fragment[2]));
        // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("%f\n", __half2float(output_fragment[3]));
        // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("\n");



        for(int mm=0; mm<4; mm++){

            // sparse_fragment1
            int peer_s = (warpin_id%4)*8 + mm; 
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                sparse_fragment1[t] = smem_A[warp_id][(peer_s) * 2 + t];
                sparse_fragment1[t+2] = smem_A[warp_id][(peer_s+4)  * 2 + t];
            }


            // dense_fragment peer_d+0,1,2,3
            int peer_d = (warpin_id/4)*4 + mm; 
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                dense_fragment[t] = smem_B[warp_id][peer_d * 4 + t];
                // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("%f\n", __half2float(dense_fragment[t]));

            }

            // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("")
            output_fragment[0] = __hadd(output_fragment[0], __hmul(sparse_fragment1[0], dense_fragment[0]));
            output_fragment[0] = __hadd(output_fragment[0], __hmul(sparse_fragment1[1], dense_fragment[1]));


            output_fragment[1] = __hadd(output_fragment[1], __hmul(sparse_fragment1[2], dense_fragment[0]));
            output_fragment[1] = __hadd(output_fragment[1], __hmul(sparse_fragment1[3], dense_fragment[1]));
            // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("%f %f\n", __half2float(dense_fragment[1]),__half2float(dense_fragment[3]));


            output_fragment[2] = __hadd(output_fragment[2], __hmul(sparse_fragment1[0], dense_fragment[2]));
            output_fragment[2] = __hadd(output_fragment[2], __hmul(sparse_fragment1[1], dense_fragment[3]));

            output_fragment[3] = __hadd(output_fragment[3], __hmul(sparse_fragment1[2], dense_fragment[2]));
            output_fragment[3] = __hadd(output_fragment[3], __hmul(sparse_fragment1[3], dense_fragment[3]));

            // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("%f\n", __half2float(output_fragment[0]));
            // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("%f\n", __half2float(output_fragment[1]));
            // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("%f\n", __half2float(output_fragment[2]));
            // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("%f\n", __half2float(output_fragment[3]));
            // if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf("\n");
        }

        __syncwarp();

            
    }
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = __half2float(output_fragment[0]);
                if((col+1)<nOri)
                *(output_matrix_+1) =  __half2float(output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = __half2float(output_fragment[1]);
                    if((col+1)<nOri)
                    *(output_matrix_+1) = __half2float( output_fragment[3]);
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ ,__half2float(output_fragment[0]));
                if((col+1)<nOri)
                atomicAdd(output_matrix_+1, __half2float(output_fragment[2]));
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , __half2float(output_fragment[1]));
                    if((col+1)<nOri)
                    atomicAdd(output_matrix_+1 , __half2float(output_fragment[3]));
                }
            }
        }
}



template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_nomap(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const half* __restrict__ t_value,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    const int* t_atomic,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    uint32_t output_fragment_[2] = {0,0};
    half * output_fragment = reinterpret_cast<half *>(output_fragment_);
    //稀疏的块, 16*8
    // __shared__ at::Half sparse_[64];
    // half * sparse = reinterpret_cast<half *>(sparse_);
    // __shared__ int sparse_to_col[8];

    float sparse_fragment[1] = {0.0};
    at::Half dense_fragment1_[4] = {0.0, 0.0, 0.0, 0.0};
    half * sparse_fragment1 = reinterpret_cast<half *>(sparse_fragment);
    half * dense_fragment = reinterpret_cast<half *>(dense_fragment1_);
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    
    const int * t_column_ = t_column + t_win_offset*8 + ((warpin_id %4)*2);
    // uint32_t * t_output_fragment_ = reinterpret_cast<uint32_t*>(t_output_fragment);
    //读取稠密矩阵的行偏移

    // const half2 * matrix_base_ = reinterpret_cast<const half2 *>(rhs_matrix + dimN_index);
    const half * matrix_base_ = rhs_matrix + dimN_index;
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
        // __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        // int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        long binary = __ldg(t_binary + t_win_offset + i);

        long temp = (binary >> (warpin_id*2));
        long a= 1;
        long mask = (a << (warpin_id*2));
        int fifthBit = ((temp) & 1);
        int block_offset = -1;
        if(fifthBit == 1){
            block_offset = __popcll(binary & (mask-1));
            sparse_fragment1[0] = __ldg(t_value + value_offset + block_offset);
        }else{
            sparse_fragment1[0]=__float2half(0.0);
        }
        fifthBit = ((temp>>1) & 1);
        if(fifthBit == 1){
            if(block_offset==-1)
            {
                mask = (a << ((warpin_id*2)+1));
                block_offset = __popcll(binary & (mask-1));
            }
            sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset + 1);
        }else{
            sparse_fragment1[1]=__float2half(0.0);
        }
        //搬运稀疏数据
        // if(threadIdx.x<nnz_block)
        // {
        //     half v = __ldg(t_value + value_offset + threadIdx.x);
        //     int row = __ldg(t_row + value_offset + threadIdx.x);
        //     int col = __ldg(t_col + value_offset + threadIdx.x);
        //     *(sparse + row*8 + col) = v;
        // }
        //  __syncthreads();
        //搬运dense数据
        // int col =  __ldg(t_column_ + (threadIdx.x%4)*2);
        // int col1 =  __ldg(t_column_ + (threadIdx.x%4)*2 + 1);
        long col_temp[2];
        for(int k=0; k<2; k++)
            col_temp[k] = __ldg(t_column_ + k);
        t_column_ += 8;
        int col_offset = (warp_id<<4) + (warpin_id/4);
        for(int i=0;i<2;i++)
        {
            if(col_temp[i] != -1){
                const long offset = (col_temp[i]*(nOri));
                dense_fragment[i]= __ldg(matrix_base_ + offset + col_offset);
                dense_fragment[i + 2]= __ldg(matrix_base_ + offset + col_offset + 8);
            }else{
                dense_fragment[i]= __float2half(0.0);
                dense_fragment[i + 2]= __float2half(0.0);
            }
        
        }

        __syncwarp();

            //MMA计算
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]),  "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );  
            
    }
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = __half2float(output_fragment[0]);
                if((col+1)<nOri)
                *(output_matrix_+8) =  __half2float(output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = __half2float(output_fragment[1]);
                    if((col+1)<nOri)
                    *(output_matrix_+8) = __half2float( output_fragment[3]);
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ ,__half2float(output_fragment[0]));
                if((col+1)<nOri)
                atomicAdd(output_matrix_+8, __half2float(output_fragment[2]));
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , __half2float(output_fragment[1]));
                    if((col+1)<nOri)
                    atomicAdd(output_matrix_+8 , __half2float(output_fragment[3]));
                }
            }
        }
}


template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_seq(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const half* __restrict__ t_value,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    // const int* t_atomic,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    uint32_t output_fragment_[2] = {0,0};
    half * output_fragment = reinterpret_cast<half *>(output_fragment_);
    //稀疏的块, 16*8
    // __shared__ at::Half sparse_[64];
    // half * sparse = reinterpret_cast<half *>(sparse_);
    // __shared__ int sparse_to_col[8];

    float sparse_fragment[1] = {0.0};
    at::Half dense_fragment1_[4] = {0.0, 0.0, 0.0, 0.0};
    half * sparse_fragment1 = reinterpret_cast<half *>(sparse_fragment);
    half * dense_fragment = reinterpret_cast<half *>(dense_fragment1_);
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    
    const int * t_column_ = t_column + t_win_offset*8 + ((warpin_id %4)*2);
    // uint32_t * t_output_fragment_ = reinterpret_cast<uint32_t*>(t_output_fragment);
    //读取稠密矩阵的行偏移

    const half2 * matrix_base_ = reinterpret_cast<const half2 *>(rhs_matrix + dimN_index);
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
        // __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        // int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        long binary = __ldg(t_binary + t_win_offset + i);

        long temp = (binary >> (warpin_id*2));
        long a= 1;
        long mask = (a << (warpin_id*2));
        int fifthBit = ((temp) & 1);
        int block_offset = -1;
        if(fifthBit == 1){
            block_offset = __popcll(binary & (mask-1));
            sparse_fragment1[0] = __ldg(t_value + value_offset + block_offset);
        }else{
            sparse_fragment1[0]=__float2half(0.0);
        }
        fifthBit = ((temp>>1) & 1);
        if(fifthBit == 1){
            if(block_offset==-1)
            {
                mask = (a << ((warpin_id*2)+1));
                block_offset = __popcll(binary & (mask-1));
            }
            sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset + 1);
        }else{
            sparse_fragment1[1]=__float2half(0.0);
        }
        //搬运稀疏数据
        // if(threadIdx.x<nnz_block)
        // {
        //     half v = __ldg(t_value + value_offset + threadIdx.x);
        //     int row = __ldg(t_row + value_offset + threadIdx.x);
        //     int col = __ldg(t_col + value_offset + threadIdx.x);
        //     *(sparse + row*8 + col) = v;
        // }
        //  __syncthreads();
        //搬运dense数据
        // int col =  __ldg(t_column_ + (threadIdx.x%4)*2);
        // int col1 =  __ldg(t_column_ + (threadIdx.x%4)*2 + 1);
        long col_temp[2];
        for(int k=0; k<2; k++)
            col_temp[k] = __ldg(t_column_ + k);
        t_column_ += 8;
        int col_offset =  (warp_id<<3) + (warpin_id/4);
        for(int i=0;i<2;i++)
        {
            if(col_temp[i] != -1){
                const long offset = (col_temp[i]*(nOri/2));
                half2 temp = __ldg(matrix_base_ + offset + col_offset);
                dense_fragment[i]= temp.x;
                dense_fragment[i + 2]= temp.y;
            }else{
                dense_fragment[i]= __float2half(0.0);
                dense_fragment[i + 2]= __float2half(0.0);
            }
        
        }

        __syncwarp();

            //MMA计算
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]),  "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );  
            
    }
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        // int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
                if(col<nOri)
                *(output_matrix_ ) = __half2float(output_fragment[0]);
                if((col+1)<nOri)
                *(output_matrix_+1) =  __half2float(output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = __half2float(output_fragment[1]);
                    if((col+1)<nOri)
                    *(output_matrix_+1) = __half2float( output_fragment[3]);
                }

        }
}

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_bcrs(
    const int* __restrict__ row_offsets,
    const float* __restrict__ t_value,
    const int2* __restrict__ t_column,
    const int* t_window_row,
    const int* t_atomic,
    const half2* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    // if((dimN_index+((lane_id/32+1)*16))>dimN)
    // return;
    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec*2));
    int nonzeros = __ldg(row_offsets + (m_index_vec*2) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_fp16_map dense_tile_loader(row_offset_vec, t_value, t_column,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    // mmaDenseTile_fp16_test dense_tile_loader(row_offset_vec, values, col_indices,
    //     nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    // );
    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    half * output_fragment_half = reinterpret_cast<half *>(output_fragment);
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int cur_t_atomic = __ldg(t_atomic + m_index_vec);
    int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

    if(row<mOri)
    {
        float * output_matrix_ = output_matrix +(row*nOri)+col;
        if(cur_t_atomic==0)
        {
            //if(col<nOri)
            *(output_matrix_ ) = __half2float(output_fragment_half[0]);
            //if((col+1)<nOri)
            *(output_matrix_+1) =  __half2float(output_fragment_half[2]);
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                // if(col<nOri)
                *(output_matrix_) = __half2float(output_fragment_half[1]);
                // if((col+1)<nOri)
                *(output_matrix_+1) = __half2float( output_fragment_half[3]);
            }
        }else{
            // if(col<nOri)
            atomicAdd(output_matrix_ ,__half2float(output_fragment_half[0]));
            // if((col+1)<nOri)
            atomicAdd(output_matrix_+1, __half2float(output_fragment_half[2]));
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                // if(col<nOri)
                atomicAdd(output_matrix_ , __half2float(output_fragment_half[1]));
                // if((col+1)<nOri)
                atomicAdd(output_matrix_+1 , __half2float(output_fragment_half[3]));
            }
        }
    }
}


template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_bcrs_seq(
    const int* __restrict__ row_offsets,
    const float* __restrict__ t_value,
    const int2* __restrict__ t_column,
    const int* t_window_row,
    const half2* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    // if((dimN_index+((lane_id/32+1)*16))>dimN)
    // return;
    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec*2));
    int nonzeros = __ldg(row_offsets + (m_index_vec*2) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_fp16_map dense_tile_loader(row_offset_vec, t_value, t_column,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    // mmaDenseTile_fp16_test dense_tile_loader(row_offset_vec, values, col_indices,
    //     nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    // );
    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    half * output_fragment_half = reinterpret_cast<half *>(output_fragment);
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

    if(row<mOri)
    {
        float * output_matrix_ = output_matrix +(row*nOri)+col;

            //if(col<nOri)
            *(output_matrix_ ) = __half2float(output_fragment_half[0]);
            //if((col+1)<nOri)
            *(output_matrix_+1) =  __half2float(output_fragment_half[2]);
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                // if(col<nOri)
                *(output_matrix_) = __half2float(output_fragment_half[1]);
                // if((col+1)<nOri)
                *(output_matrix_+1) = __half2float( output_fragment_half[3]);
            }
    }
}
// fp16
template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_cuda(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    // if((warp_id+1)*64 > (nOri + 32))
    // return;
    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;

        // if(blockIdx.y==1 && threadIdx.x==0) printf("%d, %d\n",out_row_offset, nonzeros);
        // __shared__ float c_part_result[dimN];
        extern __shared__ at::Half sparse_2[];
        //half * sparse = sparse_;
        half * sparse =reinterpret_cast<half *>(sparse_2);
        int * sparse_col = (int *) & sparse_2[partsize];
        // *(c_part_result) = 0.0;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            
            // __shared__ float sparse[16*4];      
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();

            cudaComputeUtils_fp16_trans compute(
            reinterpret_cast<const float2 *>(rhs_matrix),
            sparse_col,
            sparse,
            output_matrix + out_row_offset*nOri,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
}

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v1_kernel_cuda(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    if((warp_id+1)*64 > (nOri + 32))
    return;
    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // __shared__ float c_part_result[dimN];
        extern __shared__ at::Half sparse_2[];
        //half * sparse = sparse_;
        half * sparse =reinterpret_cast<half *>(sparse_2);
        int * sparse_col = (int *) & sparse_2[partsize];
        // *(c_part_result) = 0.0;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            
            // __shared__ float sparse[16*4];      
            //32个线程把数据搬运到shared, 前提是partSize小于等于32
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();
            // if(threadIdx.x==0 && c_part_offset_vec==0)
            // {printf("%f, %f, %f\n",sparse[0], sparse[1] , sparse[2]);
            // printf("%d, %d, %d\n",sparse_col[0], sparse_col[1] , sparse_col[2]);}
            cudaComputeUtils_fp16_trans_v1 compute(
            rhs_matrix,
            sparse_col,
            // c_column + c_row_offset_vec,
            sparse,
            // c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, dimN_index,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
}

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_cuda_short(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;
    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;

    // if(swizzle)
    // c_part_offset_vec = __ldg(c_part_offset + c_part_offset_vec);
    if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // if(m_index_vec==0 && threadIdx.x==0)
        // printf("%d\n", nonzeros);
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            cudaComputeUtils_fp16_trans_short compute(
            reinterpret_cast<const float2 *>(rhs_matrix),
            c_column + c_row_offset_vec,
            c_value + c_row_offset_vec,
            output_matrix + out_row_offset*nOri,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}


template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v1_kernel_cuda_short(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts)
    return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    if((warp_id+1)*64 > (nOri + 32))
    return;
    int c_part_offset_vec = m_index_vec; 
    // if(swizzle)
    // c_part_offset_vec = __ldg(c_part_offset + c_part_offset_vec);
    if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // if(m_index_vec==0 && threadIdx.x==0)
        // printf("%d\n", nonzeros);
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            cudaComputeUtils_tf32_trans_short_v1 compute(
            rhs_matrix,
            c_column + c_row_offset_vec,
            c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, dimN_index,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}

// fp16
template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_cuda_seq(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    // if((warp_id+1)*64 > (nOri + 32))
    // return;
    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;

        // if(blockIdx.y==1 && threadIdx.x==0) printf("%d, %d\n",out_row_offset, nonzeros);
        // __shared__ float c_part_result[dimN];
        extern __shared__ at::Half sparse_2[];
        //half * sparse = sparse_;
        half * sparse =reinterpret_cast<half *>(sparse_2);
        int * sparse_col = (int *) & sparse_2[partsize];
        // *(c_part_result) = 0.0;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            
            // __shared__ float sparse[16*4];      
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();

            cudaComputeUtils_fp16_trans_seq compute(
            reinterpret_cast<const float2 *>(rhs_matrix),
            sparse_col,
            sparse,
            output_matrix + out_row_offset*nOri,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros,
            0);    

        }
    } 
}

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_cuda_short_seq(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;
    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;

    // if(swizzle)
    // c_part_offset_vec = __ldg(c_part_offset + c_part_offset_vec);
    if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // if(m_index_vec==0 && threadIdx.x==0)
        // printf("%d\n", nonzeros);
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            cudaComputeUtils_fp16_trans_short_seq compute(
            reinterpret_cast<const float2 *>(rhs_matrix),
            c_column + c_row_offset_vec,
            c_value + c_row_offset_vec,
            output_matrix + out_row_offset*nOri,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros,
            0);    

        }
    } 
    
    
}
//tcu cuda 
float spmm_forward_fp16_tcu_cuda_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    long *  t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    half * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    half * c_value_short, 

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
        int dev; 
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int numSM = prop.multiProcessorCount;   // 应该是 128

    // 计算每个 SM 上能激活的最大 block 数
        int maxBlocksPerSM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            (const void*)spmm_forward_fp16_csr_v2_kernel_tcu<128>,  // 显式实例化模板
            128,  // block size
            0     // dynamic shared memory
        );
    // 最终 grid 大小
    int grid = numSM * maxBlocksPerSM;
    // printf("TCU: Device: %s, SMs=%d, maxBlocksPerSM=%d → grid=%d\n",
    //        prop.name, numSM, maxBlocksPerSM, grid);

    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    //每个block默认处理128
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim_c(32, 1, 1);
    dim3 block_dim(64, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));


    // // 计算每个 SM 上能激活的最大 block 数
    // int maxBlocksPerSM1 = 0;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &maxBlocksPerSM1,
    //     (const void*)spmm_forward_fp16_csr_v2_kernel_cuda<128>,  // 显式实例化模板
    //     64,  // block size
    //     sharedmemory     // dynamic shared memory
    // );
    // // 最终 grid 大小
    // int grid1 = numSM * maxBlocksPerSM1;
    // printf("CUDA-long: Device: %s, SMs=%d, maxBlocksPerSM=%d → grid=%d\n",
    //         prop.name, numSM, maxBlocksPerSM1, grid1);

    // // 计算每个 SM 上能激活的最大 block 数
    // int maxBlocksPerSM2 = 0;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &maxBlocksPerSM2,
    //     (const void*)spmm_forward_fp16_csr_v2_kernel_cuda<128>,  // 显式实例化模板
    //     32,  // block size
    //     0   // dynamic shared memory
    // );
    // // 最终 grid 大小
    // int grid2 = numSM * maxBlocksPerSM2;
    // printf("CUDA-short: Device: %s, SMs=%d, maxBlocksPerSM=%d → grid=%d\n",
    //         prop.name, numSM, maxBlocksPerSM2, grid2);

    // half2 * rhs_matrix_c = reinterpret_cast<half2 *>(rhs_matrix);
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }

    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


float spmm_forward_fp16_tcu_cuda_kernel_cc_only(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    long *  t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    half * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    half * c_value_short, 

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
        int dev; 
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int numSM = prop.multiProcessorCount;   // 应该是 128

    // 计算每个 SM 上能激活的最大 block 数
        int maxBlocksPerSM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            (const void*)spmm_forward_fp16_csr_v2_kernel_tcu<128>,  // 显式实例化模板
            128,  // block size
            0     // dynamic shared memory
        );
    // 最终 grid 大小
    int grid = numSM * maxBlocksPerSM;
    // printf("TCU: Device: %s, SMs=%d, maxBlocksPerSM=%d → grid=%d\n",
    //        prop.name, numSM, maxBlocksPerSM, grid);

    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    //每个block默认处理128
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim_c(32, 1, 1);
    dim3 block_dim(64, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));


    // // 计算每个 SM 上能激活的最大 block 数
    // int maxBlocksPerSM1 = 0;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &maxBlocksPerSM1,
    //     (const void*)spmm_forward_fp16_csr_v2_kernel_cuda<128>,  // 显式实例化模板
    //     64,  // block size
    //     sharedmemory     // dynamic shared memory
    // );
    // // 最终 grid 大小
    // int grid1 = numSM * maxBlocksPerSM1;
    // printf("CUDA-long: Device: %s, SMs=%d, maxBlocksPerSM=%d → grid=%d\n",
    //         prop.name, numSM, maxBlocksPerSM1, grid1);

    // // 计算每个 SM 上能激活的最大 block 数
    // int maxBlocksPerSM2 = 0;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &maxBlocksPerSM2,
    //     (const void*)spmm_forward_fp16_csr_v2_kernel_cuda<128>,  // 显式实例化模板
    //     32,  // block size
    //     0   // dynamic shared memory
    // );
    // // 最终 grid 大小
    // int grid2 = numSM * maxBlocksPerSM2;
    // printf("CUDA-short: Device: %s, SMs=%d, maxBlocksPerSM=%d → grid=%d\n",
    //         prop.name, numSM, maxBlocksPerSM2, grid2);

    // half2 * rhs_matrix_c = reinterpret_cast<half2 *>(rhs_matrix);
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_cc<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_cc<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }

    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


//tcu cuda 
float spmm_forward_fp16_tcu_cuda_kernel_seq(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    long *  t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    half * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    half * c_value_short, 

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    //每个block默认处理128
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim_c(32, 1, 1);
    dim3 block_dim(64, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));
    // half2 * rhs_matrix_c = reinterpret_cast<half2 *>(rhs_matrix);

    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }

    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


float spmm_forward_fp16_tcu_cuda_hybrid_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    long *  t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    half * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    half * c_value_short, 

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    //每个block默认处理128
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim_c(32, 1, 1);
    dim3 block_dim(64, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));
    // half2 * rhs_matrix_c = reinterpret_cast<half2 *>(rhs_matrix);
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_nomap<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v1_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v1_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_nomap<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
        spmm_forward_fp16_csr_v1_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v1_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }

    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}




float spmm_forward_fp16_tcu_cuda_nomap_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    long *  t_binary,

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_nomap<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_nomap<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
    }

    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

//tcu cuda 
float spmm_forward_fp16_tcu_cuda_hybrid_kernel_seq(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    long *  t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    half * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    half * c_value_short, 

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    //每个block默认处理128
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim_c(32, 1, 1);
    dim3 block_dim(64, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));
    // half2 * rhs_matrix_c = reinterpret_cast<half2 *>(rhs_matrix);

    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_nomap<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v1_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v1_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_nomap<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v1_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v1_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }

    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


float spmm_forward_fp16_tcu_cuda_bcrs_kernel(
    int * t_row_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,


    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    half * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    half * c_value_short, 

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    int grid_x_t = (n1_t/128)+1;
    if(n1_t%128==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(256, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    //每个block默认处理128
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim_c(32, 1, 1);
    dim3 block_dim(64, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));
    // half2 * rhs_matrix_c = reinterpret_cast<half2 *>(rhs_matrix);
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_bcrs<128><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    reinterpret_cast<float *>(t_value), 
                    reinterpret_cast<int2 *>(t_column), 
                    t_window_row,
                    t_atomic,
                    reinterpret_cast<half2 *>(rhs_matrix), 
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_bcrs<128><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    reinterpret_cast<float *>(t_value),  
                    reinterpret_cast<int2 *>(t_column), 
                    t_window_row,
                    t_atomic,
                    reinterpret_cast<half2 *>(rhs_matrix), 
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

float spmm_forward_fp16_tcu_cuda_bcrs_kernel_seq(
    int * t_row_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,


    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    half * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    half * c_value_short, 

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    int grid_x_t = (n1_t/128)+1;
    if(n1_t%128==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(256, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    //每个block默认处理128
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim_c(32, 1, 1);
    dim3 block_dim(64, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));
    // half2 * rhs_matrix_c = reinterpret_cast<half2 *>(rhs_matrix);
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_bcrs<128><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    reinterpret_cast<float *>(t_value), 
                    reinterpret_cast<int2 *>(t_column), 
                    t_window_row,
                    t_atomic,
                    reinterpret_cast<half2 *>(rhs_matrix), 
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_bcrs<128><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    reinterpret_cast<float *>(t_value), 
                    reinterpret_cast<int2 *>(t_column), 
                    t_window_row,
                    t_atomic,
                    reinterpret_cast<half2 *>(rhs_matrix), 
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

//cuda 长短行 fp16
float spmm_forward_fp16_cuda_kernel_split(

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    half * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    half * c_value_short, 

    half * rhs_matrix,
    float * output_matrix,

    int partsize,
    const int dimN,
    const int mOri,
    int epoches,
    int parts,
    int parts_short,
    bool swizzle)
{

    int n1=dimN;
    if((dimN%64)!=0) n1=((dimN/64)+1)*64;
    int grid_x = (n1/128)+1;
    if(n1%128==0) grid_x-=1;

    int windows =  parts;
    int splitk = 0;
    if(windows<500000) splitk=8;
    else splitk=((windows/1250000)+1)*20;

    int windows_short =  parts_short;
    int splitk_short = 0;
    if(windows_short<500000) splitk_short=8;
    else splitk_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x, splitk ,((windows/splitk)+1));
    dim3 grid_dim_c_short(grid_x, splitk_short ,((windows_short/splitk_short)+1));
    dim3 block_dim_c(32, 1, 1);
    // int warpSize = 32;
    int sharedmemory = partsize*(sizeof(float)+ sizeof(int));
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk, parts, partsize);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk_short, parts_short);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk, parts,partsize);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk_short, parts_short);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


float spmm_forward_fp16_cuda_kernel_split_v1(

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    half * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    half * c_value_short, 

    half * rhs_matrix,
    float * output_matrix,

    int partsize,
    const int dimN,
    const int mOri,
    int epoches,
    int parts,
    int parts_short,
    bool swizzle)
{

    int n1=dimN;
    if((dimN%64)!=0) n1=((dimN/64)+1)*64;
    int grid_x = (n1/128)+1;
    if(n1%128==0) grid_x-=1;

    int windows =  parts;
    int splitk = 0;
    if(windows<500000) splitk=8;
    else splitk=((windows/1250000)+1)*20;

    int windows_short =  parts_short;
    int splitk_short = 0;
    if(windows_short<500000) splitk_short=8;
    else splitk_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x, splitk ,((windows/splitk)+1));
    dim3 grid_dim_c_short(grid_x, splitk_short ,((windows_short/splitk_short)+1));
    dim3 block_dim_c(32, 1, 1);
    // int warpSize = 32;
    int sharedmemory = partsize*(sizeof(float)+ sizeof(int));
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v1_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk, parts,partsize);
        
        spmm_forward_fp16_csr_v1_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk_short, parts_short);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v1_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk, parts,partsize);
        
        spmm_forward_fp16_csr_v1_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk_short, parts_short);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

/*
TF32
*/
template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const float* __restrict__ t_value,
    const int* __restrict__ t_column,
    const int* __restrict__ t_binary,
    const int* t_window_row,
    const int* t_atomic,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

        int warp_id = threadIdx.x>>5;
        if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
        int warpin_id = threadIdx.x%32;
        //用于TCU计算的结果
        float t_output_fragment[4] = {0.0, 0.0, 0.0, 0.0}; 
        //稀疏的块, 16*8
        // __shared__ float sparse[32];
        // __shared__ int sparse_to_col[4];
        float sparse_fragment[1] = {0.0};
        float dense_fragment[2] = {0.0, 0.0};
        uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
        uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
        const int * t_column_ = t_column + t_win_offset*4;
        //读取稠密矩阵的行偏移
        // int col_offset = dimN_index + (warp_id<<4) + (warpin_id/4);
        // const float * matrix_base_ = rhs_matrix + col_offset;
        const float2 * matrix_base_ = (reinterpret_cast<const float2 *>(rhs_matrix + dimN_index));
        //循环遍历每个block
        for(int i=0; i<tcu_blocks; i++)
        {
            sparse_fragment[0]=0.0;
            //block内非零元的数量
            int value_offset = __ldg(t_block_offset + t_win_offset + i);
            //搬运稀疏数据
            // if(threadIdx.x < 32){
                //warp内的线程均读同一个值binary
                int binary = __ldg(t_binary + t_win_offset + i);
                //取线程对应的二进制位
                int fifthBit = (binary >> warpin_id) & 1;
                if(fifthBit == 1){
                    //记录块内偏移
                    int mask = (1 << warpin_id) -1;
                    int block_offset = __popc(binary & mask);
                    sparse_fragment[0] = __ldg(t_value + value_offset + block_offset);
                }
            // }
            //搬运dense数据
            int col =  __ldg(t_column_ + (threadIdx.x%4));
            t_column_ += 4;
            if(col>=0){
                const int global_offset = (warp_id<<3) + (warpin_id/4);
                const long offset = (col*nOri/2) + global_offset;
                float2 temp = __ldg(matrix_base_ + offset);
                dense_fragment[0] = temp.x;
                dense_fragment[1] = temp.y;
            }
            //读取稀疏数据

            // *(sparse_fragment) = *(sparse + warpin_id);
            __syncwarp();
            //MMA计算
            asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
                : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
            
        }
        // if(threadIdx.x==0 && m_index_vec==0)
        // printf("%f, %f, %f, %f\n", t_output_fragment[0], t_output_fragment[1], t_output_fragment[2], t_output_fragment[3]);
        //原子写入gloabl
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

        // if(row<mOri)
        // {
        //     float * output_matrix_ = output_matrix +(row*nOri)+col;
        //     if(cur_t_atomic==0)
        //     {
        //         // if(col<nOri)
        //         *(output_matrix_ ) = t_output_fragment[0];
        //         // if((col+1)<nOri)
        //         *(output_matrix_+1) =  t_output_fragment[2];
        //         if((row+1)<mOri)
        //         {
        //             output_matrix_ += nOri;
        //             // if(col<nOri)
        //             *(output_matrix_) = t_output_fragment[1];
        //             // if((col+1)<nOri)
        //             *(output_matrix_+1) =  t_output_fragment[3];
        //         }
        //     }else{
        //         if(col<nOri)
        //         atomicAdd(output_matrix_ , t_output_fragment[0]);
        //         if((col+1)<nOri)
        //         atomicAdd(output_matrix_+1, t_output_fragment[2]);
        //         if((row+1)<mOri)
        //         {
        //             output_matrix_ += nOri;
        //             // if(col<nOri)
        //             atomicAdd(output_matrix_ , t_output_fragment[1]);
        //             // if((col+1)<nOri)
        //             atomicAdd(output_matrix_+1 , t_output_fragment[3]);
        //         }
        //     }
        // }
        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;

                // if(col<nOri)
                *(output_matrix_ ) = t_output_fragment[0];
                // if((col+1)<nOri)
                *(output_matrix_+1) =  t_output_fragment[2];
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    // if(col<nOri)
                    *(output_matrix_) = t_output_fragment[1];
                    // if((col+1)<nOri)
                    *(output_matrix_+1) =  t_output_fragment[3];
                }
        }
}


template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_tcu_seq(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const float* __restrict__ t_value,
    const int* __restrict__ t_column,
    const int* __restrict__ t_binary,
    const int* t_window_row,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

        int warp_id = threadIdx.x>>5;
        if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
        int warpin_id = threadIdx.x%32;
        //用于TCU计算的结果
        float t_output_fragment[4] = {0.0, 0.0, 0.0, 0.0}; 
        //稀疏的块, 16*8
        // __shared__ float sparse[32];
        // __shared__ int sparse_to_col[4];
        float sparse_fragment[1] = {0.0};
        float dense_fragment[2] = {0.0, 0.0};
        uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
        uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
        const int * t_column_ = t_column + t_win_offset*4;
        //读取稠密矩阵的行偏移
        // int col_offset = dimN_index + (warp_id<<4) + (warpin_id/4);
        // const float * matrix_base_ = rhs_matrix + col_offset;
        const float2 * matrix_base_ = (reinterpret_cast<const float2 *>(rhs_matrix + dimN_index));
        //循环遍历每个block
        for(int i=0; i<tcu_blocks; i++)
        {
            sparse_fragment[0]=0.0;
            //block内非零元的数量
            int value_offset = __ldg(t_block_offset + t_win_offset + i);
            //搬运稀疏数据
            // if(threadIdx.x < 32){
                //warp内的线程均读同一个值binary
                int binary = __ldg(t_binary + t_win_offset + i);
                //取线程对应的二进制位
                int fifthBit = (binary >> warpin_id) & 1;
                if(fifthBit == 1){
                    //记录块内偏移
                    int mask = (1 << warpin_id) -1;
                    int block_offset = __popc(binary & mask);
                    sparse_fragment[0] = __ldg(t_value + value_offset + block_offset);
                }
            // }
            //搬运dense数据
            int col =  __ldg(t_column_ + (threadIdx.x%4));
            t_column_ += 4;
            if(col>=0){
                const int global_offset = (warp_id<<3) + (warpin_id/4);
                const long offset = (col*nOri/2) + global_offset;
                float2 temp = __ldg(matrix_base_ + offset);
                dense_fragment[0] = temp.x;
                dense_fragment[1] = temp.y;
            }
            //读取稀疏数据

            // *(sparse_fragment) = *(sparse + warpin_id);
            __syncwarp();
            //MMA计算
            asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
                : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
            
        }
        // if(threadIdx.x==0 && m_index_vec==0)
        // printf("%f, %f, %f, %f\n", t_output_fragment[0], t_output_fragment[1], t_output_fragment[2], t_output_fragment[3]);
        //原子写入gloabl
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
                // if(col<nOri)
                *(output_matrix_ ) = t_output_fragment[0];
                // if((col+1)<nOri)
                *(output_matrix_+1) =  t_output_fragment[2];
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    // if(col<nOri)
                    *(output_matrix_) = t_output_fragment[1];
                    // if((col+1)<nOri)
                    *(output_matrix_+1) =  t_output_fragment[3];
                }

        }
}
template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_cuda(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const float* __restrict__ c_value,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;
    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;

    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // __shared__ float c_part_result[dimN];
        extern __shared__ float sparse_[];
        float * sparse =sparse_;
        int * sparse_col = (int *) & sparse_[partsize];
        // *(c_part_result) = 0.0;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            // printf("11111");
            // __shared__ float sparse[16*4];      
            //32个线程把数据搬运到shared, 前提是partSize小于等于32
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();
            // if(threadIdx.x==0 && c_part_offset_vec==0)
            // {printf("%f, %f, %f\n",sparse[0], sparse[1] , sparse[2]);
            // printf("%d, %d, %d\n",sparse_col[0], sparse_col[1] , sparse_col[2]);}
            cudaComputeUtils_tf32_trans compute(
            reinterpret_cast<const float4 *>(rhs_matrix),
            sparse_col,
            sparse,
            output_matrix + out_row_offset*nOri,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}

template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_cuda_short(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const float* __restrict__ c_value,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
   
    if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // if(m_index_vec==0 && threadIdx.x==0)
        // printf("%d\n", nonzeros);
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            cudaComputeUtils_tf32_trans_short compute(
            reinterpret_cast<const float4 *>(rhs_matrix),
            c_column + c_row_offset_vec,
            c_value + c_row_offset_vec,
            output_matrix + out_row_offset*nOri,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, __ldg(c_atomic + c_part_offset_vec));      

        }
    } 
    
    
}


template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_cuda_seq(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const float* __restrict__ c_value,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;
    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;

    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // __shared__ float c_part_result[dimN];
        extern __shared__ float sparse_[];
        float * sparse =sparse_;
        int * sparse_col = (int *) & sparse_[partsize];
        // *(c_part_result) = 0.0;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            // printf("11111");
            // __shared__ float sparse[16*4];      
            //32个线程把数据搬运到shared, 前提是partSize小于等于32
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();
            // if(threadIdx.x==0 && c_part_offset_vec==0)
            // {printf("%f, %f, %f\n",sparse[0], sparse[1] , sparse[2]);
            // printf("%d, %d, %d\n",sparse_col[0], sparse_col[1] , sparse_col[2]);}
            cudaComputeUtils_tf32_trans_seq compute(
            reinterpret_cast<const float4 *>(rhs_matrix),
            sparse_col,
            sparse,
            output_matrix + out_row_offset*nOri,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, 0);    

        }
    } 
    
    
}

template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_cuda_short_seq(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const float* __restrict__ c_value,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int nOri,
    int mOri,
    int splitk,
    int parts)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
   
    if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // if(m_index_vec==0 && threadIdx.x==0)
        // printf("%d\n", nonzeros);
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            cudaComputeUtils_tf32_trans_short_seq compute(
            reinterpret_cast<const float4 *>(rhs_matrix),
            c_column + c_row_offset_vec,
            c_value + c_row_offset_vec,
            output_matrix + out_row_offset*nOri,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, 0);      

        }
    } 
    
    
}

float spmm_forward_tf32_cuda_kernel_split(

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    float * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    float * c_value_short, 

    float * rhs_matrix,
    float * output_matrix,

    int partsize,
    const int dimN,
    const int mOri,
    int epoches,
    int parts,
    int parts_short,
    bool swizzle)
{
    // int n1=dimN;
    // if((dimN%16)!=0) n1=((dimN/16)+1)*16;
    // int grid_x = (n1/64)+1;
    // if(n1%64==0) grid_x-=1;
    // int windows =  (parts/4) + 1;
    // if(parts%4==0) windows-=1;
    // int splitk = 0;
    // if(windows<500000) splitk=8;
    // else splitk=((windows/1250000)+1)*20;

    int n1=dimN;
    if((dimN%64)!=0) n1=((dimN/64)+1)*64;
    int grid_x = (n1/128)+1;
    if(n1%128==0) grid_x-=1;

    int windows =  parts;
    int splitk = 0;
    if(windows<500000) splitk=8;
    else splitk=((windows/1250000)+1)*20;

    int windows_short =  parts_short;
    int splitk_short = 0;
    if(windows_short<500000) splitk_short=8;
    else splitk_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim(grid_x, splitk ,((windows/splitk)+1));
    dim3 grid_dim_short(grid_x, splitk_short ,((windows_short/splitk_short)+1));
    dim3 block_dim(32, 1, 1);
    // int warpSize = 32;
    int sharedmemory = partsize*(sizeof(float)+ sizeof(int));
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim, block_dim, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk, parts, partsize);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_short, block_dim, 0 ,stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk_short, parts_short);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim, block_dim, sharedmemory, stream1>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk, parts, partsize);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_short, block_dim, 0 ,stream2>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1, dimN, mOri, splitk_short, parts_short);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;

    return spmm_ms_avg;
}




//tcu cuda
float spmm_forward_tf32_tcu_cuda_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    int *  t_binary,
    
    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    float * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    float * c_value_short, 

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim(32, 1, 1);
    int sharedmemory = partsize_c*(sizeof(float)+ sizeof(int));
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            t_binary,
            t_window_row,
            t_atomic,
            rhs_matrix,  
            output_matrix,
            n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c, partsize_c);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim, 0 ,stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;

    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    cudaDeviceSynchronize();
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            t_binary,
            t_window_row,
            t_atomic,
            rhs_matrix,  
            output_matrix,
            n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c, partsize_c);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim, 0 ,stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);

    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


//tcu cuda
float spmm_forward_tf32_tcu_cuda_kernel_seq(
    int * t_row_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    int *  t_binary,
    
    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    float * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    float * c_value_short, 

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim(32, 1, 1);
    int sharedmemory = partsize_c*(sizeof(float)+ sizeof(int));
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            t_binary,
            t_window_row,
            t_atomic,
            rhs_matrix,  
            output_matrix,
            n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c, partsize_c);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;

    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    cudaDeviceSynchronize();
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            t_binary,
            t_window_row,
            t_atomic,
            rhs_matrix,  
            output_matrix,
            n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c, partsize_c);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);

    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


//onlt TCU
//tcu cuda
float spmm_forward_fp16_tcu_bcrs_part_kernel(
    int * t_row_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    int grid_x_t = (n1_t/128)+1;
    if(n1_t%128==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(256, 1, 1);


    for(int iter=0; iter<10; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_bcrs<128><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    reinterpret_cast<float *>(t_value), 
                    reinterpret_cast<int2 *>(t_column), 
                    t_window_row,
                    t_atomic,
                    reinterpret_cast<half2 *>(rhs_matrix), 
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_fp16_csr_v2_kernel_tcu_bcrs<128><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    reinterpret_cast<float *>(t_value),  
                    reinterpret_cast<int2 *>(t_column), 
                    t_window_row,
                    t_atomic,
                    reinterpret_cast<half2 *>(rhs_matrix), 
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_tcu_tcf(
    const int* __restrict__ row_offsets,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_tcl,
    const float* __restrict__ t_value,
    const int* __restrict__ t_column,
    const int* t_window_row,
    const int* t_atomic,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    // if((dimN_index+((lane_id/32+1)*16))>dimN)
    // return;
    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;

    // Load the row offset and calculate the number of nonzeros in the row
    int t_win_offset = __ldg(row_offsets + (m_index_vec));
    int tcu_blocks = __ldg(row_offsets + (m_index_vec) + 1) - t_win_offset; 
    if(tcu_blocks==0) return;
  
    //用于TCU计算的结果
    float t_output_fragment[4] = {0.0, 0.0, 0.0, 0.0}; 
    //稀疏的块, 8x4
    __shared__ float sparse[32];
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    const int * t_column_ = t_column + t_win_offset*4;
    //读取稠密矩阵的行偏移
    const float2 * matrix_base_ = (reinterpret_cast<const float2 *>(rhs_matrix + dimN_index));
    for(int i=0; i<tcu_blocks; i++)
    {
         __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        //block中的所有warp一起把稀疏数据搬运到sparse, sparse_to_col
        //block内部的每个线程初始化sparse tile 为0
        if(threadIdx.x < 32){
            sparse[threadIdx.x] = 0.0;
        }
        __syncthreads();
        // 获取列索引
        //搬运稀疏数据
        if(threadIdx.x<nnz_block)
        {
            float v = __ldg(t_value + value_offset + threadIdx.x);
            int row = __ldg(t_tcl + value_offset + threadIdx.x);
            *(sparse + row) = v;
        }
         __syncthreads();
        //搬运dense数据
        int col =  __ldg(t_column_ + (threadIdx.x%4));
        t_column_ += 4;
        if(col>=0){
            const int global_offset = (warp_id<<3) + (warpin_id/4);
            const long offset = (col*nOri/2) + global_offset;
            float2 temp = __ldg(matrix_base_ + offset);
            dense_fragment[0] = temp.x;
            dense_fragment[1] = temp.y;
        }

        //读取稀疏数据

        sparse_fragment[0] = *(sparse + warpin_id);
        __syncwarp();

        //MMA计算
        asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
                : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
        
    }

    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int cur_t_atomic = __ldg(t_atomic + m_index_vec);
    int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;
    
    if(row<mOri)
    {
        float * output_matrix_ = output_matrix +(row*nOri)+col;
        if(cur_t_atomic==0)
        {
            //if(col<nOri)
            *(output_matrix_ ) = t_output_fragment[0];
            //if((col+1)<nOri)
            *(output_matrix_+1) =  t_output_fragment[2];
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                // if(col<nOri)
                *(output_matrix_) = t_output_fragment[1];
                // if((col+1)<nOri)
                *(output_matrix_+1) =  t_output_fragment[3];
            }
        }else{
            // if(col<nOri)
            atomicAdd(output_matrix_ ,t_output_fragment[0]);
            // if((col+1)<nOri)
            atomicAdd(output_matrix_+1, t_output_fragment[2]);
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                // if(col<nOri)
                atomicAdd(output_matrix_ , t_output_fragment[1]);
                // if((col+1)<nOri)
                atomicAdd(output_matrix_+1 , t_output_fragment[3]);
            }
        }
    }
}

float spmm_forward_tf32_tcu_tcf_part_kernel(
    int * t_row_offset,
    int * t_tco, 
    int * t_tcl,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);


    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu_tcf<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_tco,
                    t_tcl,
                    t_value,  
                    t_column, 
                    t_window_row,
                    t_atomic,
                    rhs_matrix, 
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu_tcf<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_tco,
                    t_tcl,
                    t_value,  
                    t_column, 
                    t_window_row,
                    t_atomic,
                    rhs_matrix, 
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

void gcn_forward_fp16_tcu_cuda_bcrs_kernel(
    int * t_row_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,


    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    half * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    half * c_value_short, 

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri)
{
    //tcu
    int n1_t=dimN;
    int grid_x_t = (n1_t/128)+1;
    if(n1_t%128==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(256, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    //每个block默认处理128
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim_c(32, 1, 1);
    dim3 block_dim(64, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));
    // half2 * rhs_matrix_c = reinterpret_cast<half2 *>(rhs_matrix);
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);


        spmm_forward_fp16_csr_v2_kernel_tcu_bcrs<128><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    reinterpret_cast<float *>(t_value),  
                    reinterpret_cast<int2 *>(t_column), 
                    t_window_row,
                    t_atomic,
                    reinterpret_cast<half2 *>(rhs_matrix), 
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
}


template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_tcu_bcrs(
    const int* __restrict__ row_offsets,
    const float* __restrict__ t_value,
    const int* __restrict__ t_column,
    const int* t_window_row,
    const int* t_atomic,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    
    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int row_offset_vec = __ldg(row_offsets + (m_index_vec*2));
    int nonzeros = __ldg(row_offsets + (m_index_vec*2) + 1) - row_offset_vec; 
    if(nonzeros==0) return;

        //用于TCU计算的结果
        float t_output_fragment[4] = {0.0, 0.0, 0.0, 0.0}; 
        //稀疏的块, 16*8
        // __shared__ float sparse[32];
        // __shared__ int sparse_to_col[4];
        float sparse_fragment[1] = {0.0};
        float dense_fragment[2] = {0.0, 0.0};
        mmaDenseTile_tf32 dense_tile_loader(row_offset_vec, t_value, t_column,
            nOri, dimN_index, threadIdx.x, rhs_matrix,  reinterpret_cast<float2*>(dense_fragment), sparse_fragment
        );

        uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
        uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);

        int steps = nonzeros/4;
        int residue = nonzeros%4;

        if(steps > 0){
            #pragma unroll
            for(int i = 0; i < steps; i++){
                dense_tile_loader.Fetch(nOri,dimN_index);
                __syncwarp();
                //MMA计算
                asm volatile(
                "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                    : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
                    : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
            }
        }
        if(residue > 0){
            // sparse_tile_loader.Residue();
            // __syncwarp();
            dense_tile_loader.ResidueLoad(nOri,dimN_index);
            __syncwarp();
            //MMA计算
            asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
                : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
        }
          

            
                
        // if(threadIdx.x==0 && m_index_vec==0)
        // printf("%f, %f, %f, %f\n", t_output_fragment[0], t_output_fragment[1], t_output_fragment[2], t_output_fragment[3]);
        //原子写入gloabl
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                // if(col<nOri)
                *(output_matrix_ ) = t_output_fragment[0];
                // if((col+1)<nOri)
                *(output_matrix_+1) =  t_output_fragment[2];
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    // if(col<nOri)
                    *(output_matrix_) = t_output_fragment[1];
                    // if((col+1)<nOri)
                    *(output_matrix_+1) =  t_output_fragment[3];
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ , t_output_fragment[0]);
                if((col+1)<nOri)
                atomicAdd(output_matrix_+1, t_output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    // if(col<nOri)
                    atomicAdd(output_matrix_ , t_output_fragment[1]);
                    // if((col+1)<nOri)
                    atomicAdd(output_matrix_+1 , t_output_fragment[3]);
                }
            }
        }

}


float spmm_forward_tf32_tcu_bcrs_part_kernel(
    int * t_row_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu_bcrs<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_value, 
                    t_column, 
                    t_window_row,
                    t_atomic,
                    rhs_matrix, 
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu_bcrs<64><<<grid_dim_t, block_dim_t>>>(
                    t_row_offset, 
                    t_value, 
                    t_column, 
                    t_window_row,
                    t_atomic,
                    rhs_matrix, 
                    output_matrix,
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}




float spmm_forward_tf32_tcu_cuda_bcrs_kernel(
    int * t_row_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    
    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    float * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    float * c_value_short, 

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim(32, 1, 1);
    int sharedmemory = partsize_c*(sizeof(float)+ sizeof(int));
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu_bcrs<64><<<grid_dim_t, block_dim_t, 0 ,stream1>>>(
            t_row_offset, 
            t_value, 
            t_column, 
            t_window_row,
            t_atomic,
            rhs_matrix,  
            output_matrix,
            n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c, partsize_c);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim, 0 ,stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;

    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    cudaDeviceSynchronize();
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu_bcrs<64><<<grid_dim_t, block_dim_t, 0 ,stream1>>>(
            t_row_offset, 
            t_value, 
            t_column, 
            t_window_row,
            t_atomic,
            rhs_matrix,  
            output_matrix,
            n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c, partsize_c);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim, 0 ,stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);

    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


float spmm_forward_tf32_tcu_cuda_bcrs_kernel_seq(
    int * t_row_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    
    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    float * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    float * c_value_short, 

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim(32, 1, 1);
    int sharedmemory = partsize_c*(sizeof(float)+ sizeof(int));
   
    for(int iter=0; iter<10; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu_bcrs<64><<<grid_dim_t, block_dim_t>>>(
            t_row_offset, 
            t_value, 
            t_column, 
            t_window_row,
            t_atomic,
            rhs_matrix,  
            output_matrix,
            n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c, partsize_c);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;

    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    cudaDeviceSynchronize();
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_tf32_csr_v2_kernel_tcu_bcrs<64><<<grid_dim_t, block_dim_t>>>(
            t_row_offset, 
            t_value, 
            t_column, 
            t_window_row,
            t_atomic,
            rhs_matrix,  
            output_matrix,
            n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c, parts_c, partsize_c);
        spmm_forward_tf32_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, dimN, mOri, splitk_c_short, parts_c_short);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);

    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}
