#include <cuda_fp16.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <stdio.h>

// 仅复用原版 Libra 的 CUDA fallback kernel (spmm_forward_fp16_csr_v2_kernel_cuda / _short)
// TC kernel 已重写为适配 16x16 的版本
#include "./mGCNkernel.cu"


static __device__ __forceinline__ uint32_t pack_half2_u32(half lo, half hi)
{
    return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
}


static __device__ __forceinline__ uint32_t pack_float2half_u32(float lo, float hi)
{
    return pack_half2_u32(__float2half(lo), __float2half(hi));
}

static bool env_enabled_spmm_online(const char* name, bool default_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0;
}


// ============================================================================
// 适配 window=16 的 Dense TC Kernel
//
// 问题背景:
//   原版 Libra TC kernel 为 window=8 设计, 使用 64-bit binary 编码 8x8 块,
//   并通过 m16n8k8 转置 trick 将稠密 A (8x8) 作为 RHS operand.
//   升级到 window=16 后, 16x16 的 TC 块包含 256 个 entry, 无法用单个 64-bit
//   binary 编码. 同时 A 矩阵变为 16x16, 原版 kernel 会因维度不匹配而崩溃.
//
// 解决方案:
//   在 CPU 端将每个 16x16 TC 块拆分为 4 个 8x8 子块 (TL/TR/BL/BR),
//   每个子块拥有独立的 64-bit binary、8 列索引和 packed values.
//   GPU 端遍历这 4 个子块, 每个执行一次 m16n8k8, 结果累加到输出.
//
// t_window_row 编码: window_id * 16 + row_offset
//   TL/TR: row_offset = 0  (对应 window 的 rows 0-7)
//   BL/BR: row_offset = 8  (对应 window 的 rows 8-15)
// ============================================================================

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu_online(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const half* __restrict__ t_value,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* __restrict__ t_window_row,
    const int* __restrict__ t_atomic,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
#if __CUDA_ARCH__ >= 800
    int m_index_vec = (blockIdx.z * splitk) + blockIdx.y;
    if (m_index_vec >= windows) return;
    int dimN_index = blockIdx.x * Tile_N;

    // 该 TCU part 包含的子块数量
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset;
    if (tcu_blocks == 0) return;

    int warp_id = threadIdx.x >> 5;
    if ((dimN_index + (((warp_id) + 1) * 8)) > dimN) return;
    int warpin_id = threadIdx.x % 32;

    // 用于 TCU 计算的结果 (fp16 累加)
    uint32_t output_fragment_[2] = {0, 0};
    half* output_fragment = reinterpret_cast<half*>(output_fragment_);

    float sparse_fragment[1] = {0.0};
    uint32_t dense_fragment_[2] = {0, 0};
    half* sparse_fragment1 = reinterpret_cast<half*>(sparse_fragment);
    half* dense_fragment = reinterpret_cast<half*>(dense_fragment_);
    uint32_t* sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);

    // 线程身份: groupID=0..7 (映射到 MMA M-rows groupID 和 groupID+8)
    // tid_in_group=0..3 (映射到 MMA N-cols tid_in_group*2 和 tid_in_group*2+1)
    int groupID = warpin_id >> 2;         // warpin_id / 4
    int tid_in_group = warpin_id & 3;     // warpin_id % 4

    // t_column 有 8 列/子块, 每个线程读取 2 列
    const int* t_column_base = t_column + t_win_offset * 8 + (tid_in_group * 2);
    const half2* matrix_base_ = reinterpret_cast<const half2*>(rhs_matrix + dimN_index);

    // 遍历该 TCU part 下的每个子块 (16×8, 直接匹配 m16n8k8)
    for (int i = 0; i < tcu_blocks; i++)
    {
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        int bin_idx = t_win_offset * 2 + i * 2;
        long binary_lo = __ldg(t_binary + bin_idx);      // sparse rows 0-7
        long binary_hi = __ldg(t_binary + bin_idx + 1);  // sparse rows 8-15

        // 加载列索引和稠密 A 矩阵 (两次 MMA 共用)
        long col_temp[2];
        const int* t_column_ = t_column_base + i * 8;
        for (int k = 0; k < 2; k++)
            col_temp[k] = __ldg(t_column_ + k);

        int col_offset = (warp_id << 2) + tid_in_group;  // 32 cols/block
        for (int j = 0; j < 2; j++) {
            if (col_temp[j] != -1) {
                const long offset = (col_temp[j] * (nOri / 2));
                half2 temp_h2 = __ldg(matrix_base_ + offset + col_offset);
                dense_fragment[j] = temp_h2.x;
                dense_fragment[j + 2] = temp_h2.y;
            } else {
                dense_fragment[j] = __float2half(0.0);
                dense_fragment[j + 2] = __float2half(0.0);
            }
        }

        // === MMA 1: binary_lo, sparse rows 0-7 ===
        {
            long a = 1;
            int bit_pos = groupID * 8 + tid_in_group * 2;
            long temp = (binary_lo >> bit_pos);
            long mask = (a << bit_pos);
            int block_offset = -1;
            if ((temp & 1) == 1) {
                block_offset = __popcll(binary_lo & (mask - 1));
                sparse_fragment1[0] = __ldg(t_value + value_offset + block_offset);
            } else {
                sparse_fragment1[0] = __float2half(0.0);
            }
            if (((temp >> 1) & 1) == 1) {
                if (block_offset == -1) {
                    mask = (a << (bit_pos + 1));
                    block_offset = __popcll(binary_lo & (mask - 1));
                    sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset);
                } else {
                    sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset + 1);
                }
            } else {
                sparse_fragment1[1] = __float2half(0.0);
            }

            asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]), "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );
        }

        // === MMA 2: binary_hi, sparse rows 8-15 ===
        {
            int base_offset = __popcll(binary_lo);
            long a = 1;
            int bit_pos = groupID * 8 + tid_in_group * 2;
            long temp = (binary_hi >> bit_pos);
            long mask = (a << bit_pos);
            int block_offset = -1;
            if ((temp & 1) == 1) {
                block_offset = __popcll(binary_hi & (mask - 1));
                sparse_fragment1[0] = __ldg(t_value + value_offset + base_offset + block_offset);
            } else {
                sparse_fragment1[0] = __float2half(0.0);
            }
            if (((temp >> 1) & 1) == 1) {
                if (block_offset == -1) {
                    mask = (a << (bit_pos + 1));
                    block_offset = __popcll(binary_hi & (mask - 1));
                    sparse_fragment1[1] = __ldg(t_value + value_offset + base_offset + block_offset);
                } else {
                    sparse_fragment1[1] = __ldg(t_value + value_offset + base_offset + block_offset + 1);
                }
            } else {
                sparse_fragment1[1] = __float2half(0.0);
            }

            asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]), "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );
        }
    }

    // 写回: 16 rows per sub-block, 32 cols/block (Tile_N=32)
    int window_row_code = __ldg(t_window_row + m_index_vec);
    int cur_t_atomic = __ldg(t_atomic + m_index_vec);
    int row = window_row_code + groupID * 2;
    int col = dimN_index + warp_id * 8 + tid_in_group * 2;

    if (row < mOri) {
        float* output_matrix_ = output_matrix + (row * nOri) + col;
        if (cur_t_atomic == 0) {
            if (col < nOri)
                *(output_matrix_) = __half2float(output_fragment[0]);
            if ((col + 1) < nOri)
                *(output_matrix_ + 1) = __half2float(output_fragment[2]);
            if ((row + 1) < mOri) {
                output_matrix_ += nOri;
                if (col < nOri)
                    *(output_matrix_) = __half2float(output_fragment[1]);
                if ((col + 1) < nOri)
                    *(output_matrix_ + 1) = __half2float(output_fragment[3]);
            }
        } else {
            if (col < nOri)
                atomicAdd(output_matrix_, __half2float(output_fragment[0]));
            if ((col + 1) < nOri)
                atomicAdd(output_matrix_ + 1, __half2float(output_fragment[2]));
            if ((row + 1) < mOri) {
                output_matrix_ += nOri;
                if (col < nOri)
                    atomicAdd(output_matrix_, __half2float(output_fragment[1]));
                if ((col + 1) < nOri)
                    atomicAdd(output_matrix_ + 1, __half2float(output_fragment[3]));
            }
        }
    }
#endif
}


// ============================================================================
// SPTC CUDA kernel (非 mma, row-by-row 标量计算, window=16 兼容)
// ============================================================================

template <int FeatureTile, int RowTile>

__global__ void spmm_forward_fp16_sptc_online_kernel_cuda(

    const int* __restrict__ s_column,

    const float* __restrict__ s_value,

    const int8_t* __restrict__ s_pos,

    const int* __restrict__ s_window,

    const half* __restrict__ rhs_matrix,

    float* __restrict__ output_matrix,

    int groups,

    int window,

    int dimN,

    int mOri,

    int kOri,

    int feature_tiles,

    int row_tiles)

{

    long long work_id = (long long)blockIdx.x;
    int feature_tile_id = (int)(work_id % feature_tiles);
    long long tmp = work_id / feature_tiles;
    int group_id = (int)(tmp % groups);
    int row_tile_id = (int)(tmp / groups);

    if (group_id >= groups || row_tile_id >= row_tiles) return;


    int local_row = row_tile_id * RowTile + threadIdx.y;

    if (local_row >= window) return;


    int n = feature_tile_id * FeatureTile + threadIdx.x;

    if (n >= dimN) return;


    int global_row = __ldg(s_window + group_id) * window + local_row;

    if (global_row >= mOri) return;


    int base = (group_id * window + local_row) * 2;

    float acc = 0.0f;

    // 向量化加载: float2 一次读 2 个 float (s_value[base] 和 s_value[base+1])
    // s_pos 两个 int8_t 合并为 16-bit 一次加载, 减少 2 次 Global Memory 事务
    float2 s_vals = __ldg(reinterpret_cast<const float2*>(s_value + base));
    short s_pos_packed = __ldg(reinterpret_cast<const short*>(s_pos + base));
    int pos0 = (int)((int8_t)(s_pos_packed & 0xFF));
    int pos1 = (int)((int8_t)(s_pos_packed >> 8));

    if (pos0 >= 0) {
        int col0 = __ldg(s_column + group_id * 4 + pos0);
        if (col0 >= 0 && col0 < kOri) {
            half rhs0 = __ldg(rhs_matrix + col0 * dimN + n);
            acc += s_vals.x * __half2float(rhs0);
        }
    }
    if (pos1 >= 0) {
        int col1 = __ldg(s_column + group_id * 4 + pos1);
        if (col1 >= 0 && col1 < kOri) {
            half rhs1 = __ldg(rhs_matrix + col1 * dimN + n);
            acc += s_vals.y * __half2float(rhs1);
        }
    }


    if (acc != 0.0f) {

        atomicAdd(output_matrix + global_row * dimN + n, acc);

    }

}


float spmm_forward_fp16_sptc_online_kernel(

    int* s_column,

    float* s_value,

    int8_t* s_pos,

    int* s_window,

    half* rhs_matrix,

    float* output_matrix,

    int groups,

    int window,

    int dimN,

    int mOri,

    int kOri,

    int epoches)

{

    cudaMemset(output_matrix, 0, (size_t)mOri * dimN * sizeof(float));

    if (groups <= 0 || window <= 0 || dimN <= 0 || mOri <= 0) return 0.0f;


    const int feature_tile = 32;

    const int row_tile = 8;

    dim3 block_dim(feature_tile, row_tile, 1);

    int feature_tiles = (dimN + feature_tile - 1) / feature_tile;
    int row_tiles = (window + row_tile - 1) / row_tile;
    long long total_blocks = (long long)feature_tiles * groups * row_tiles;
    if (total_blocks > 2147483647LL) {
        printf("SPTC CUDA grid is too large: %lld blocks\n", total_blocks);
        return 0.0f;
    }
    dim3 grid_dim((unsigned int)total_blocks, 1, 1);


    for (int iter = 0; iter < 10; ++iter) {

        spmm_forward_fp16_sptc_online_kernel_cuda<feature_tile, row_tile><<<grid_dim, block_dim>>>(

            s_column, s_value, s_pos, s_window, rhs_matrix, output_matrix,

            groups, window, dimN, mOri, kOri, feature_tiles, row_tiles);

    }

    cudaDeviceSynchronize();


    float spmm_ms_avg = 0.0f;

    float spmm_ms = 0.0f;

    cudaEvent_t spmm_start;

    cudaEvent_t spmm_end;

    cudaEventCreate(&spmm_start);

    cudaEventCreate(&spmm_end);

    cudaEventRecord(spmm_start);

    for (int iter = 0; iter < epoches; ++iter) {

        spmm_forward_fp16_sptc_online_kernel_cuda<feature_tile, row_tile><<<grid_dim, block_dim>>>(

            s_column, s_value, s_pos, s_window, rhs_matrix, output_matrix,

            groups, window, dimN, mOri, kOri, feature_tiles, row_tiles);

    }

    cudaEventRecord(spmm_end);

    cudaEventSynchronize(spmm_end);

    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);

    cudaEventDestroy(spmm_start);

    cudaEventDestroy(spmm_end);

    spmm_ms_avg = spmm_ms / (float)epoches;


    cudaMemset(output_matrix, 0, (size_t)mOri * dimN * sizeof(float));

    spmm_forward_fp16_sptc_online_kernel_cuda<feature_tile, row_tile><<<grid_dim, block_dim>>>(

        s_column, s_value, s_pos, s_window, rhs_matrix, output_matrix,

        groups, window, dimN, mOri, kOri, feature_tiles, row_tiles);

    cudaDeviceSynchronize();

    return spmm_ms_avg;

}


// ============================================================================
// mma.sp 数据加载辅助函数 (Device 端)
// ============================================================================

__device__ __forceinline__ void load_sptc_row_chunk(

    const float* __restrict__ s_value,

    const int8_t* __restrict__ s_pos,

    int group_id,

    int group_limit,

    int row,

    int window,

    float& v0,

    float& v1,

    int& p0,

    int& p1)

{

    v0 = 0.0f;

    v1 = 0.0f;

    p0 = 0;

    p1 = 1;

    if (group_id >= group_limit || row >= window) return;


    int base = (group_id * window + row) * 2;

    int pos0 = (int)__ldg(s_pos + base);

    int pos1 = (int)__ldg(s_pos + base + 1);

    float val0 = __ldg(s_value + base);

    float val1 = __ldg(s_value + base + 1);


    if (pos0 < 0 && pos1 < 0) return;

    if (pos0 >= 0 && pos1 >= 0) {

        if (pos0 <= pos1) {

            p0 = pos0;

            p1 = pos1;

            v0 = val0;

            v1 = val1;

        } else {

            p0 = pos1;

            p1 = pos0;

            v0 = val1;

            v1 = val0;

        }

        return;

    }


    int pos = pos0 >= 0 ? pos0 : pos1;

    float val = pos0 >= 0 ? val0 : val1;

    int dummy = pos == 0 ? 1 : 0;

    if (dummy < pos) {

        p0 = dummy;

        p1 = pos;

        v0 = 0.0f;

        v1 = val;

    } else {

        p0 = pos;

        p1 = dummy;

        v0 = val;

        v1 = 0.0f;

    }

}


// ============================================================================
// [已移除] mma.sp 非 packed kernel — 统一使用 packed v3 kernel
// ============================================================================

__global__ void _spmm_forward_fp16_sptc_mma_online_kernel_cuda_removed(

    const int* __restrict__ s_column,

    const float* __restrict__ s_value,

    const int8_t* __restrict__ s_pos,

    const int* __restrict__ s_offsets,

    const int* __restrict__ tile_window,

    const int* __restrict__ tile_group,

    const half* __restrict__ rhs_matrix,

    float* __restrict__ output_matrix,

    int num_tiles,

    int window,

    int dimN,

    int mOri,

    int kOri,

    int feature_tiles)

{

#if __CUDA_ARCH__ >= 800
    // =========================================================================
    // 借鉴 MP-SpMM_SC25 三大优化重写的非 packed SPTC kernel:
    //   Task 1: 向量化全局内存访问 — B 矩阵 uint64_t 一次加载 4 half
    //   Task 2: Shared Memory Staging — 128 线程协作填充 B_smem[512]
    //   Task 3: ldmatrix PTX 指令 — Shared→Register, 替代手动 pack_half2_u32
    //
    // B_smem 列优先布局 (与 packed kernel 相同, 满足 ldmatrix 要求):
    //   B_smem[warp_id * 128 + feature * 16 + col]
    //   feature=0..7 (warp 内行), col=0..15 (tile 列)
    //   ldmatrix.sync.aligned.m8n8.x2.shared.b16:
    //     组0: cols 0-7, rows 0-7; 组1: cols 8-15, rows 0-7
    // =========================================================================
    __shared__ __align__(16) half B_smem[512];  // 4 warps × 16 cols × 8 features

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int group_id_in_warp = lane >> 2;
    int thread_id_in_group = lane & 3;

    int tile_id = blockIdx.x / feature_tiles;
    if (tile_id >= num_tiles) return;

    int feature_tile_id = blockIdx.x - tile_id * feature_tiles;
    int feature_base = feature_tile_id * 32;

    int window_id = __ldg(tile_window + tile_id);
    int group_base = __ldg(tile_group + tile_id);
    int group_end = __ldg(s_offsets + window_id + 1);
    if (group_base >= group_end) return;

    // =========================================================================
    // 阶段1: 加载稀疏 A 值和 metadata (Global → Register)
    // 利用 float→half 打包和位运算, 与 packed kernel 的 uint64_t 路径对称
    // =========================================================================
    float ar0[8], ar1[8];
    int meta_pos0[4], meta_pos1[4], meta_pos2[4], meta_pos3[4];

    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        float v0, v1;
        int p0, p1;
        load_sptc_row_chunk(
            s_value, s_pos, group_base + chunk, group_end,
            group_id_in_warp, window, v0, v1, p0, p1);
        ar0[chunk * 2] = v0;
        ar0[chunk * 2 + 1] = v1;
        meta_pos0[chunk] = p0;
        meta_pos1[chunk] = p1;

        load_sptc_row_chunk(
            s_value, s_pos, group_base + chunk, group_end,
            group_id_in_warp + 8, window, v0, v1, p0, p1);
        ar1[chunk * 2] = v0;
        ar1[chunk * 2 + 1] = v1;
        meta_pos2[chunk] = p0;
        meta_pos3[chunk] = p1;
    }

    uint32_t a0 = pack_float2half_u32(ar0[thread_id_in_group * 2], ar0[thread_id_in_group * 2 + 1]);
    uint32_t a1 = pack_float2half_u32(ar1[thread_id_in_group * 2], ar1[thread_id_in_group * 2 + 1]);

    // 16 行 metadata → 32-bit: 低 16 位 rows 0-7, 高 16 位 rows 8-15
    uint32_t meta = 0;
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        uint32_t nibble0 = (uint32_t)(meta_pos0[chunk] & 3) | ((uint32_t)(meta_pos1[chunk] & 3) << 2);
        meta |= nibble0 << (chunk * 4);
        uint32_t nibble1 = (uint32_t)(meta_pos2[chunk] & 3) | ((uint32_t)(meta_pos3[chunk] & 3) << 2);
        meta |= nibble1 << (16 + chunk * 4);
    }

    // =========================================================================
    // 阶段2: 128 线程协作向量化加载 B 矩阵 Global → Shared Memory
    //
    // 线程映射: threadIdx.x → (col_of_tile, feat_group)
    //   col_of_tile = t / 8    → 0..15 (16 个 tile 列)
    //   feat_group  = t % 8    → 0..7
    //   feat_start  = feat_group * 4 → 0,4,8,12,16,20,24,28 (共 32 特征)
    //
    // 与旧版的关键区别:
    //   旧版每个线程 4 次标量 __ldg → 4 次 Global Memory 事务
    //   新版每个线程 1 次 uint64_t 加载 4 half + 散布到 Shared → 1 次事务
    //   带宽利用率提升 4×, 且数据在 Shared 中可被 ldmatrix 硬件高效读取
    // =========================================================================
    {
        int col_of_tile = threadIdx.x / 8;
        int feat_group = threadIdx.x % 8;
        int feat_start = feat_group * 4;

        // 列索引: s_column 以 group 为单位索引, 每个 group 4 列
        // col_of_tile → (group_within_tile, pos_within_group)
        int group_idx = group_base + col_of_tile / 4;
        int col_pos = col_of_tile % 4;
        int col_global = -1;
        if (group_idx < group_end)
            col_global = __ldg(s_column + group_idx * 4 + col_pos);

        if (col_global >= 0 && col_global < kOri &&
            feature_base + feat_start < dimN) {
            const half* src = rhs_matrix + col_global * dimN + feature_base + feat_start;

            if (feature_base + feat_start + 3 < dimN) {
                // 快路径: uint64_t 向量化加载 4 个 half
                uint64_t loaded = *reinterpret_cast<const uint64_t*>(src);
                half* vals = reinterpret_cast<half*>(&loaded);
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    B_smem[(feat_start + f) * 16 + col_of_tile] = vals[f];
                }
            } else {
                // 边界慢路径: 特征维度尾部不足 4 个
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    int feat = feat_start + f;
                    if (feature_base + feat < dimN)
                        B_smem[feat * 16 + col_of_tile] = __ldg(src + f);
                    else
                        B_smem[feat * 16 + col_of_tile] = __float2half(0.0f);
                }
            }
        } else {
            // 无效列或特征越界: 写零
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                B_smem[(feat_start + f) * 16 + col_of_tile] = __float2half(0.0f);
            }
        }
    }

    __syncthreads();  // B_smem 写入完成

    if (feature_base + warp_id * 8 >= dimN) return;

    // =========================================================================
    // 阶段3: ldmatrix PTX 指令 — Shared Memory → Register
    //
    // 替代旧版 4 次标量 __ldg + pack_half2_u32 的手动拼接.
    // ldmatrix 硬件将 Shared 中的 16×8 B 瓦片直接转换为 MMA 所需的
    // 寄存器碎片格式, 消除所有中间打包开销.
    //
    // B_smem 地址: warp_id * 128 偏移, 每个 warp 独占 128 half (8 feat × 16 col)
    // =========================================================================
    uint32_t b0, b1;
    {
        unsigned smem_addr = __cvta_generic_to_shared(B_smem + warp_id * 128);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b0), "=r"(b1)
            : "r"(smem_addr)
        );
    }

    // =========================================================================
    // 阶段4: Tensor Core MMA 计算
    // =========================================================================
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6, %7}, "
        "{%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));

    // =========================================================================
    // 阶段5: 输出写回
    // =========================================================================
    int row0 = window_id * window + group_id_in_warp;
    int row1 = row0 + 8;
    int out_col0 = feature_base + warp_id * 8 + thread_id_in_group * 2;
    int out_col1 = out_col0 + 1;

    if (group_id_in_warp < window && row0 < mOri) {
        if (out_col0 < dimN) atomicAdd(output_matrix + row0 * dimN + out_col0, c0);
        if (out_col1 < dimN) atomicAdd(output_matrix + row0 * dimN + out_col1, c1);
    }
    if (group_id_in_warp + 8 < window && row1 < mOri) {
        if (out_col0 < dimN) atomicAdd(output_matrix + row1 * dimN + out_col0, c2);
        if (out_col1 < dimN) atomicAdd(output_matrix + row1 * dimN + out_col1, c3);
    }
#endif
}


// ============================================================================
// SPTC Packed Kernel v4 — Per-Warp __syncwarp (消除 __syncthreads 开销)
//
// 相比 v3 的关键升级:
//   v3: 128 线程协作填充 B_smem[512] → 需要 __syncthreads (128线程屏障)
//   v4: 每个 warp 独立加载自己所需的 8 个 feature 行到独立 SMEM 区域,
//       使用 __syncwarp (32线程屏障) 替代 __syncthreads,
//       消除跨 warp 同步开销 (约 1-2μs 每次).
//
// 架构:
//   Grid:  (num_windows, feature_tiles_32)  2D
//   Block: (128, 1, 1)  4 warps × 32 lanes
//   B_smem: [512] half, 分 4 块独立区域 (每个 warp 128 half)
//
// 每 warp B 加载: 32 lanes 覆盖 16 cols × 8 features = 128 half
//   - col = lane / 2  (0..15)
//   - 偶数 lane: 加载 feature 0..3  / 奇数 lane: 加载 feature 4..7
//   - 每 lane 1 次 uint64_t 加载 (4 half)
//   - 无跨 warp 依赖 → 只需 __syncwarp
//
// ldmatrix 从 warp 的独立 SMEM 区域读取, 布局由 PTX ISA 保证.
// ============================================================================

__global__ void spmm_forward_fp16_sptc_mma_packed_v3_kernel_cuda(
    const int* __restrict__ s_row_ptr,          // [num_windows+1] CSR row pointer
    const int* __restrict__ s_tile_column,      // [num_tiles * 16]
    const uint32_t* __restrict__ s_packed_a,    // [num_tiles * 8 * 4 * 2]
    const uint32_t* __restrict__ s_packed_meta, // [num_tiles * 8]
    const half* __restrict__ rhs_matrix,        // [kOri * dimN]
    float* __restrict__ output_matrix,           // [mOri * dimN]
    int num_windows,
    int window,
    int dimN,
    int mOri,
    int kOri,
    int feature_tiles)
{
#if __CUDA_ARCH__ >= 800
    // B_smem 分为 4 个独立区域, 每 warp 独占 128 half (8 features × 16 cols)
    __shared__ __align__(16) half B_smem[512];

    int window_id = blockIdx.x;
    int feature_tile_id = blockIdx.y;
    if (window_id >= num_windows || feature_tile_id >= feature_tiles) return;

    int tile_start = __ldg(s_row_ptr + window_id);
    int tile_end   = __ldg(s_row_ptr + window_id + 1);
    if (tile_start >= tile_end) return;

    int feature_base = feature_tile_id * 32;
    if (feature_base >= dimN) return;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int group_id_in_warp = lane >> 2;
    int thread_id_in_group = lane & 3;

    // 每个 warp 的 feature 基地址和 SMEM 基地址 (SMEM 内偏移)
    int warp_feat_base = feature_base + warp_id * 8;
    half* warp_smem = B_smem + warp_id * 128;  // 128 half = 8 features × 16 cols

    // 每 warp 内 B 加载: col = lane/2 (0..15), feat_off = 0 (偶数lane) 或 4 (奇数lane)
    int col_of_tile = lane >> 1;          // 0..15
    int feat_off = (lane & 1) ? 4 : 0;   // 0 or 4

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    for (int t = tile_start; t < tile_end; ++t) {
        // --- Load A values (uint64_t vectorized, 保持不变) ---
        int a_base = ((t * 8 + group_id_in_warp) * 4 + thread_id_in_group) * 2;
        uint64_t a_packed = __ldg(reinterpret_cast<const uint64_t*>(s_packed_a + a_base));
        uint32_t a0 = (uint32_t)(a_packed & 0xFFFFFFFFull);
        uint32_t a1 = (uint32_t)(a_packed >> 32);
        uint32_t meta = __ldg(s_packed_meta + t * 8 + group_id_in_warp);

        // --- Per-Warp B loading (32 lanes 独立加载 8 features × 16 cols) ---
        const int* tile_col = s_tile_column + t * 16;
        {
            int col_global = __ldg(tile_col + col_of_tile);
            if (col_global >= 0 && col_global < kOri &&
                warp_feat_base + feat_off < dimN) {
                const half* src = rhs_matrix + col_global * dimN + warp_feat_base + feat_off;

                if (warp_feat_base + feat_off + 3 < dimN) {
                    uint64_t loaded = *reinterpret_cast<const uint64_t*>(src);
                    half* vals = reinterpret_cast<half*>(&loaded);
                    warp_smem[(feat_off + 0) * 16 + col_of_tile] = vals[0];
                    warp_smem[(feat_off + 1) * 16 + col_of_tile] = vals[1];
                    warp_smem[(feat_off + 2) * 16 + col_of_tile] = vals[2];
                    warp_smem[(feat_off + 3) * 16 + col_of_tile] = vals[3];
                } else {
                    #pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        int feat = feat_off + f;
                        if (warp_feat_base + feat < dimN)
                            warp_smem[feat * 16 + col_of_tile] = __ldg(src + f);
                        else
                            warp_smem[feat * 16 + col_of_tile] = __float2half(0.0f);
                    }
                }
            } else {
                #pragma unroll
                for (int f = 0; f < 4; ++f)
                    warp_smem[(feat_off + f) * 16 + col_of_tile] = __float2half(0.0f);
            }
        }

        __syncwarp();  // 仅 warp 内同步 (替代 v3 的 __syncthreads)

        // --- ldmatrix + mma.sp (one MMA per warp, 8 features per warp) ---
        if (warp_feat_base < dimN) {
            uint32_t b0, b1;
            unsigned smem_addr = __cvta_generic_to_shared(warp_smem);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(b0), "=r"(b1)
                : "r"(smem_addr)
            );

            asm volatile(
                "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x1;\n"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
        }

        __syncwarp();  // 确保当前 tile 的 MMA 完成再覆盖 B_smem
    }

    // --- Output write (no atomicAdd) ---
    int row0 = window_id * window + group_id_in_warp;
    int row1 = row0 + 8;
    int out_col0 = feature_base + warp_id * 8 + thread_id_in_group * 2;
    int out_col1 = out_col0 + 1;

    if (group_id_in_warp < window && row0 < mOri) {
        if (out_col0 < dimN) output_matrix[row0 * dimN + out_col0] = c0;
        if (out_col1 < dimN) output_matrix[row0 * dimN + out_col1] = c2;
    }
    if (group_id_in_warp + 8 < window && row1 < mOri) {
        if (out_col0 < dimN) output_matrix[row1 * dimN + out_col0] = c1;
        if (out_col1 < dimN) output_matrix[row1 * dimN + out_col1] = c3;
    }
#endif
}


// Packed mma.sp kernel (Native 16x16, 无 Warp Stacking)
//
// window=16: CPU 端已将所有 16 行的 A 值打包进 s_packed_a,
// rows 0-7 通过 a0, rows 8-15 通过 a1 加载.
// 一次 mma.sp.m16n8k16 即可完成整个 16x16 瓦片的计算, 无算力浪费.
// ============================================================================

__global__ void _spmm_forward_fp16_sptc_mma_packed_online_kernel_cuda_removed(
    const int* __restrict__ s_tile_column,
    const uint32_t* __restrict__ s_packed_a,
    const uint32_t* __restrict__ s_packed_meta,
    const int* __restrict__ s_tile_window,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int num_tiles,
    int window,
    int dimN,
    int mOri,
    int kOri,
    int feature_tiles)
{
#if __CUDA_ARCH__ >= 800
    // =========================================================================
    // 借鉴 MP-SpMM_SC25 (kernels.cu) 的三大硬件级优化:
    //   Task 1: 向量化全局内存访问 — A 值 uint64_t (4 half), B 值 uint64_t (4 half)
    //   Task 2: Shared Memory Staging — 128 线程协作填充 B_smem[512],
    //           并在 B 写入前预加载 A 值, 利用 ILP 重叠 Global Memory 延迟
    //   Task 3: ldmatrix PTX 指令 — Shared→Register 高效排布给 MMA
    //
    // B_smem 列优先 (column-major) 布局 (满足 ldmatrix 要求):
    //   B_smem[warp_id * 128 + feature * 16 + col]
    //   feature=0..7 为 warp 内行 (ldmatrix 的矩阵行),
    //   col=0..15 为 tile 列 (ldmatrix 的矩阵列, row stride=32 byte)
    //   ldmatrix.sync.aligned.m8n8.x2.shared.b16 读 2 组 8×8:
    //     组0: cols 0-7, rows 0-7; 组1: cols 8-15, rows 0-7
    //   每组 8×8=64 half, stride=32 byte 满足 16-byte 对齐
    // =========================================================================
    __shared__ __align__(16) half B_smem[512];  // 4 warps × 16 cols × 8 features = 512 half

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int group_id_in_warp = lane >> 2;
    int thread_id_in_group = lane & 3;

    int tile_id = blockIdx.x / feature_tiles;
    bool valid_tile = (tile_id < num_tiles);

    int feature_tile_id = blockIdx.x - tile_id * feature_tiles;
    int feature_base = feature_tile_id * 32;

    // =========================================================================
    // 阶段1: 预加载 A 值 (uint64_t 向量化) + 并行写 B → Shared Memory
    //
    // 借鉴 MP-SpMM Buint64_Cfloat4 kernel:
    //   A 值 uint64_t 合并加载 (a0+a1 共 4 half) 与 B_smem 写入在指令级交错,
    //   利用 ILP 隐藏 Global Memory 延迟. 这是 MP-SpMM Apipeline 思想的简化版,
    //   无需 cp.async 复杂性即可实现有效的 load-load 重叠.
    //
    // 线程映射 (B 加载): threadIdx.x → (col_of_tile, feat_group)
    //   col_of_tile = t / 8     → 0..15 (16 个 tile 列)
    //   feat_group  = t % 8     → 0..7
    //   feat_start  = feat_group * 4  → 0,4,8,12,16,20,24,28 (共 32 特征)
    // =========================================================================
    const int* tile_col = s_tile_column + (valid_tile ? tile_id : 0) * 16;

    // 预加载 A 值: uint64_t 一次加载 4 个 half (a0 低2 + a1 高2)
    // s_packed_a 布局: 每线程 2 个 uint32_t (各含 2 half), 地址连续
    // 在 B_smem 写入前发起此加载, Global Memory 延迟被后续 Shared 写操作部分吸收
    uint32_t a0 = 0, a1 = 0, meta = 0;
    if (valid_tile) {
        int a_base = (((tile_id * 8 + group_id_in_warp) * 4 + thread_id_in_group) * 2);
        // uint64_t 合并加载 a0 和 a1 → 1 次 64-bit 事务替代 2 次 32-bit __ldg
        uint64_t a_packed = __ldg(reinterpret_cast<const uint64_t*>(s_packed_a + a_base));
        a0 = (uint32_t)(a_packed & 0xFFFFFFFFull);
        a1 = (uint32_t)(a_packed >> 32);
        meta = __ldg(s_packed_meta + tile_id * 8 + group_id_in_warp);
    }

    // B 矩阵 Global → Shared Memory (与 A 值加载在 ILP 层面并行)
    {
        int col_of_tile = threadIdx.x / 8;
        int feat_group = threadIdx.x % 8;
        int feat_start = feat_group * 4;

        int col_global = -1;
        if (valid_tile)
            col_global = __ldg(tile_col + col_of_tile);

        if (col_global >= 0 && col_global < kOri &&
            feature_base + feat_start < dimN) {
            const half* src = rhs_matrix + col_global * dimN + feature_base + feat_start;

            if (feature_base + feat_start + 3 < dimN) {
                // 快路径: uint64_t 向量化加载 4 个 half → 1 次 64-bit 全局内存事务
                uint64_t loaded = *reinterpret_cast<const uint64_t*>(src);
                half* vals = reinterpret_cast<half*>(&loaded);
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    B_smem[(feat_start + f) * 16 + col_of_tile] = vals[f];
                }
            } else {
                // 边界慢路径: 标量加载 (特征维度尾部不足 4 个)
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    int feat = feat_start + f;
                    if (feature_base + feat < dimN)
                        B_smem[feat * 16 + col_of_tile] = __ldg(src + f);
                    else
                        B_smem[feat * 16 + col_of_tile] = __float2half(0.0f);
                }
            }
        } else {
            // 无效列或特征越界: 写零
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                B_smem[(feat_start + f) * 16 + col_of_tile] = __float2half(0.0f);
            }
        }
    }

    __syncthreads();  // B_smem 写入完成, ldmatrix 可安全读取

    // 无效 tile 或越界 block 提前退出
    if (!valid_tile) return;
    if (feature_base >= dimN) return;

    // =========================================================================
    // 阶段2: ldmatrix PTX 指令 — Shared Memory → Register (A 值在阶段1已预加载)
    //
    // 借鉴 MP-SpMM 的数据排布, 但进一步提升为 ldmatrix 硬件指令:
    //   ldmatrix.sync.aligned.m8n8.x2.shared.b16 一次性加载 2 组 8×8 矩阵,
    //   将 B_smem 中 16×8 的 B 瓦片高效排布为 MMA 所需的寄存器碎片格式.
    //
    // 地址计算: B_smem + warp_id * 128
    //   warp 0 → B_smem[0]     (特征 0..7)
    //   warp 1 → B_smem[128]   (特征 8..15)
    //   warp 2 → B_smem[256]   (特征 16..23)
    //   warp 3 → B_smem[384]   (特征 24..31)
    //
    // 输出映射 (以 lane_id 为例, groupID=lane/4, threadID=lane%4):
    //   b0 = {B[threadID*2][groupID], B[threadID*2+1][groupID]}
    //       即 tile 列 threadID*2 和 threadID*2+1 在特征 groupID 处的 B 值
    //   b1 = {B[threadID*2+8][groupID], B[threadID*2+9][groupID]}
    //       即 tile 列 threadID*2+8 和 threadID*2+9 在特征 groupID 处的 B 值
    //
    // 该排布与 mma.sp.sync.aligned.m16n8k16 的 B 操作数要求完全一致,
    //   消除了原先逐个标量 __ldg + pack_half2_u32 的手动拼接开销
    // =========================================================================
    uint32_t b0, b1;
    {
        unsigned smem_addr = __cvta_generic_to_shared(B_smem + warp_id * 128);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b0), "=r"(b1)
            : "r"(smem_addr)
        );
    }

    // =========================================================================
    // 阶段3: Tensor Core MMA 计算
    // =========================================================================
    float c0 = 0.0f;
    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;

    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6, %7}, "
        "{%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));

    // =========================================================================
    // 阶段4: 输出写回 (借鉴 MP-SpMM float2 向量化写入思路)
    // =========================================================================
    {
        int window_id = __ldg(s_tile_window + tile_id);
        int row0 = window_id * window + group_id_in_warp;
        int row1 = row0 + 8;
        int out_col0 = feature_base + warp_id * 8 + thread_id_in_group * 2;
        int out_col1 = out_col0 + 1;

        if (group_id_in_warp < window && row0 < mOri) {
            if (out_col0 < dimN) atomicAdd(output_matrix + row0 * dimN + out_col0, c0);
            if (out_col1 < dimN) atomicAdd(output_matrix + row0 * dimN + out_col1, c1);
        }
        if (group_id_in_warp + 8 < window && row1 < mOri) {
            if (out_col0 < dimN) atomicAdd(output_matrix + row1 * dimN + out_col0, c2);
            if (out_col1 < dimN) atomicAdd(output_matrix + row1 * dimN + out_col1, c3);
        }
    }
#endif
}


// ============================================================================
// Host launch wrappers
// ============================================================================

float spmm_forward_fp16_sptc_mma_online_kernel(

    int* s_column,

    float* s_value,

    int8_t* s_pos,

    int* s_offsets,

    int* tile_window,

    int* tile_group,

    half* rhs_matrix,

    float* output_matrix,

    int num_tiles,

    int window,

    int dimN,

    int mOri,

    int kOri,

    int epoches)

{

    cudaMemset(output_matrix, 0, (size_t)mOri * dimN * sizeof(float));

    if (num_tiles <= 0 || window <= 0 || dimN <= 0 || mOri <= 0) return 0.0f;


    dim3 block_dim(128, 1, 1);

    int feature_tiles = (dimN + 31) / 32;
    long long total_blocks = (long long)feature_tiles * num_tiles;
    if (total_blocks > 2147483647LL) {
        printf("SPTC MMA grid is too large: %lld blocks\n", total_blocks);
        return 0.0f;
    }
    dim3 grid_dim((unsigned int)total_blocks, 1, 1);


    for (int iter = 0; iter < 10; ++iter) {

        _spmm_forward_fp16_sptc_mma_online_kernel_cuda_removed<<<grid_dim, block_dim>>>(

            s_column, s_value, s_pos, s_offsets, tile_window, tile_group, rhs_matrix, output_matrix,

            num_tiles, window, dimN, mOri, kOri, feature_tiles);

    }

    cudaDeviceSynchronize();


    float spmm_ms_avg = 0.0f;

    float spmm_ms = 0.0f;

    cudaEvent_t spmm_start;

    cudaEvent_t spmm_end;

    cudaEventCreate(&spmm_start);

    cudaEventCreate(&spmm_end);

    cudaEventRecord(spmm_start);

    for (int iter = 0; iter < epoches; ++iter) {

        _spmm_forward_fp16_sptc_mma_online_kernel_cuda_removed<<<grid_dim, block_dim>>>(

            s_column, s_value, s_pos, s_offsets, tile_window, tile_group, rhs_matrix, output_matrix,

            num_tiles, window, dimN, mOri, kOri, feature_tiles);

    }

    cudaEventRecord(spmm_end);

    cudaEventSynchronize(spmm_end);

    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);

    cudaEventDestroy(spmm_start);

    cudaEventDestroy(spmm_end);

    spmm_ms_avg = spmm_ms / (float)epoches;


    cudaMemset(output_matrix, 0, (size_t)mOri * dimN * sizeof(float));

    _spmm_forward_fp16_sptc_mma_online_kernel_cuda_removed<<<grid_dim, block_dim>>>(

        s_column, s_value, s_pos, s_offsets, tile_window, tile_group, rhs_matrix, output_matrix,

        num_tiles, window, dimN, mOri, kOri, feature_tiles);

    cudaDeviceSynchronize();

    return spmm_ms_avg;

}

float spmm_forward_fp16_sptc_mma_packed_online_kernel(
    int* s_row_ptr,
    int* s_tile_column,
    uint32_t* s_packed_a,
    uint32_t* s_packed_meta,
    half* rhs_matrix,
    float* output_matrix,
    int num_tiles,
    int num_windows,
    int window,
    int dimN,
    int mOri,
    int kOri,
    int epoches)
{
    cudaMemset(output_matrix, 0, (size_t)mOri * dimN * sizeof(float));
    if (num_tiles <= 0 || num_windows <= 0 || window <= 0 || dimN <= 0 || mOri <= 0) return 0.0f;

    int feature_tiles = (dimN + 31) / 32;
    dim3 block_dim(128, 1, 1);
    dim3 grid_dim(num_windows, feature_tiles, 1);

    for (int iter = 0; iter < 10; ++iter) {
        spmm_forward_fp16_sptc_mma_packed_v3_kernel_cuda<<<grid_dim, block_dim>>>(
            s_row_ptr, s_tile_column, s_packed_a, s_packed_meta,
            rhs_matrix, output_matrix,
            num_windows, window, dimN, mOri, kOri, feature_tiles);
    }
    cudaDeviceSynchronize();

    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for (int iter = 0; iter < epoches; ++iter) {
        spmm_forward_fp16_sptc_mma_packed_v3_kernel_cuda<<<grid_dim, block_dim>>>(
            s_row_ptr, s_tile_column, s_packed_a, s_packed_meta,
            rhs_matrix, output_matrix,
            num_windows, window, dimN, mOri, kOri, feature_tiles);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    spmm_ms_avg = spmm_ms / (float)epoches;

    cudaMemset(output_matrix, 0, (size_t)mOri * dimN * sizeof(float));
    spmm_forward_fp16_sptc_mma_packed_v3_kernel_cuda<<<grid_dim, block_dim>>>(
        s_row_ptr, s_tile_column, s_packed_a, s_packed_meta,
        rhs_matrix, output_matrix,
        num_windows, window, dimN, mOri, kOri, feature_tiles);
    cudaDeviceSynchronize();
    return spmm_ms_avg;
}


// ============================================================================
// 辅助 kernel: 矩阵加法 (用于 SPTC 结果合并到主输出)
// ============================================================================

__global__ void add_float_matrix_online_kernel(
    float* __restrict__ output_matrix,
    const float* __restrict__ sptc_output_matrix,
    long long elements)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < elements; idx += stride) {
        output_matrix[idx] += sptc_output_matrix[idx];
    }
}


// ============================================================================
// Launch helpers
// ============================================================================

// 使用新的 TC kernel (适配 16x16 窗口)
static void launch_fp16_tcu_cuda_base_online(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value,
    int * t_column,
    int* t_window_row,
    int * t_atomic,
    long * t_binary,

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
    cudaStream_t stream_t,
    cudaStream_t stream_c,
    cudaStream_t stream_c_short)
{
    int n1_t = dimN;
    if ((dimN % 16) != 0) n1_t = ((dimN / 16) + 1) * 16;
    int grid_x_t = (n1_t / 32) + 1;
    if (n1_t % 32 == 0) grid_x_t -= 1;
    int splitk_t = 0;
    if (parts_t < 500000) splitk_t = 8;
    else splitk_t = ((parts_t / 1250000) + 1) * 20;
    dim3 grid_dim_t(grid_x_t, splitk_t, ((parts_t / splitk_t) + 1));
    dim3 block_dim_t(128, 1, 1);

    int n1_c = dimN;
    if ((dimN % 64) != 0) n1_c = ((dimN / 64) + 1) * 64;
    int grid_x_c = (n1_c / 128) + 1;
    if (n1_c % 128 == 0) grid_x_c -= 1;

    int windows = parts_c;
    int splitk_c = 0;
    if (windows < 500000) splitk_c = 8;
    else splitk_c = ((windows / 1250000) + 1) * 20;

    int windows_short = parts_c_short;
    int splitk_c_short = 0;
    if (windows_short < 500000) splitk_c_short = 8;
    else splitk_c_short = ((windows_short / 1250000) + 1) * 20;

    dim3 grid_dim_c(grid_x_c, splitk_c, ((windows / splitk_c) + 1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short, ((windows_short / splitk_c_short) + 1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = partsize_c * (sizeof(half) + sizeof(int));

    // 使用重写后的 TC kernel (适配 16x16 子块拆分)
    // 设置 LIBRA_SKIP_TC_KERNEL=1 可跳过 TC kernel (用于隔离 crash 来源)
    if (!env_enabled_spmm_online("LIBRA_SKIP_TC_KERNEL", false) && parts_t > 0) {
    spmm_forward_fp16_csr_v2_kernel_tcu_online<32><<<grid_dim_t, block_dim_t, 0, stream_t>>>(
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
    } // LIBRA_SKIP_TC_KERNEL

    spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream_c>>>(
        c_row_offset,
        c_row,
        c_atomic,
        c_column,
        c_value,
        rhs_matrix,
        output_matrix,
        n1_c, dimN, mOri, splitk_c, parts_c, partsize_c);

    spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim_c, 0, stream_c_short>>>(
        c_row_offset_short,
        c_row_short,
        c_atomic_short,
        c_column_short,
        c_value_short,
        rhs_matrix,
        output_matrix,
        n1_c, dimN, mOri, splitk_c_short, parts_c_short);
}

static void _launch_sptc_mma_online_removed(
    int* s_column,
    float* s_value,
    int8_t* s_pos,
    int* s_offsets,
    int* tile_window,
    int* tile_group,
    half* rhs_matrix,
    float* output_matrix,
    int num_tiles,
    int window,
    int dimN,
    int mOri,
    int kOri,
    cudaStream_t stream_s)
{
    if (num_tiles <= 0 || window <= 0 || dimN <= 0 || mOri <= 0) return;
    dim3 block_dim(128, 1, 1);
    int feature_tiles = (dimN + 31) / 32;
    long long total_blocks = (long long)feature_tiles * num_tiles;
    if (total_blocks > 2147483647LL) {
        printf("SPTC stream MMA grid is too large: %lld blocks\n", total_blocks);
        return;
    }
    dim3 grid_dim((unsigned int)total_blocks, 1, 1);
    _spmm_forward_fp16_sptc_mma_online_kernel_cuda_removed<<<grid_dim, block_dim, 0, stream_s>>>(
        s_column, s_value, s_pos, s_offsets, tile_window, tile_group, rhs_matrix, output_matrix,
        num_tiles, window, dimN, mOri, kOri, feature_tiles);
}

static void _launch_sptc_mma_packed_online_removed(
    int* s_tile_column,
    uint32_t* s_packed_a,
    uint32_t* s_packed_meta,
    int* s_tile_window,
    half* rhs_matrix,
    float* output_matrix,
    int num_tiles,
    int window,
    int dimN,
    int mOri,
    int kOri,
    cudaStream_t stream_s)
{
    if (num_tiles <= 0 || window <= 0 || dimN <= 0 || mOri <= 0) return;
    dim3 block_dim(128, 1, 1);
    int feature_tiles = (dimN + 31) / 32;
    long long total_blocks = (long long)feature_tiles * num_tiles;
    if (total_blocks > 2147483647LL) {
        printf("SPTC stream packed MMA grid is too large: %lld blocks\n", total_blocks);
        return;
    }
    dim3 grid_dim((unsigned int)total_blocks, 1, 1);
    _spmm_forward_fp16_sptc_mma_packed_online_kernel_cuda_removed<<<grid_dim, block_dim, 0, stream_s>>>(
        s_tile_column, s_packed_a, s_packed_meta, s_tile_window, rhs_matrix, output_matrix,
        num_tiles, window, dimN, mOri, kOri, feature_tiles);
}

// v3 launch: 2D Grid (num_windows, feature_tiles_32), K-loop 在 kernel 内部, 无 atomicAdd
//   每个 block 独占处理 16 rows × 32 features 的输出区域
//   使用 Shared Memory + ldmatrix (成熟可靠模式)
static void launch_sptc_mma_packed_v3_online(
    int* s_row_ptr,
    int* s_tile_column,
    uint32_t* s_packed_a,
    uint32_t* s_packed_meta,
    half* rhs_matrix,
    float* output_matrix,
    int num_tiles,
    int num_windows,
    int window,
    int dimN,
    int mOri,
    int kOri,
    cudaStream_t stream_s)
{
    if (num_tiles <= 0 || num_windows <= 0 || window <= 0 || dimN <= 0 || mOri <= 0) return;

    // 2D Grid: (num_windows, feature_tiles)
    //   block.x = window_id  → 0..num_windows-1
    //   block.y = feature_tile_id → 0..feature_tiles-1
    int feature_tiles = (dimN + 31) / 32;  // 32 features per block (4 warps × 8 features)
    dim3 block_dim(128, 1, 1);
    dim3 grid_dim(num_windows, feature_tiles, 1);

    spmm_forward_fp16_sptc_mma_packed_v3_kernel_cuda<<<grid_dim, block_dim, 0, stream_s>>>(
        s_row_ptr, s_tile_column, s_packed_a, s_packed_meta,
        rhs_matrix, output_matrix,
        num_windows, window, dimN, mOri, kOri, feature_tiles);
}


// ============================================================================
// 并行主入口: TC + CUDA + SPTC 多流并行 (无 Stacking)
// ============================================================================

float spmm_forward_fp16_tcu_cuda_sptc_mma_parallel_online_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value,
    int * t_column,
    int* t_window_row,
    int * t_atomic,
    long * t_binary,

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

    int* s_column,
    float* s_value,
    int8_t* s_pos,
    int* s_offsets,
    int* tile_window,
    int* tile_group,
    int* s_tile_column,
    uint32_t* s_packed_a,
    uint32_t* s_packed_meta,
    int* s_tile_window,

    int* s_window_tile_offset,   // v3 kernel: s_row_ptr [num_windows+1]

    half * rhs_matrix,
    float * output_matrix,
    float * sptc_output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int num_tiles,
    const int num_windows,        // v3 kernel
    const int window,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches,
    float* tc_ms_avg_out,
    float* cuda_long_ms_avg_out,
    float* cuda_short_ms_avg_out,
    float* sptc_ms_avg_out)
{
    if (tc_ms_avg_out != nullptr) *tc_ms_avg_out = 0.0f;
    if (cuda_long_ms_avg_out != nullptr) *cuda_long_ms_avg_out = 0.0f;
    if (cuda_short_ms_avg_out != nullptr) *cuda_short_ms_avg_out = 0.0f;
    if (sptc_ms_avg_out != nullptr) *sptc_ms_avg_out = 0.0f;

    long long elements = (long long)mOri * dimN;
    cudaMemset(output_matrix, 0, (size_t)elements * sizeof(float));
    cudaMemset(sptc_output_matrix, 0, (size_t)elements * sizeof(float));
    if (epoches <= 0 || dimN <= 0 || mOri <= 0) return 0.0f;

    cudaStream_t stream_t;
    cudaStream_t stream_c;
    cudaStream_t stream_c_short;
    cudaStream_t stream_s;
    cudaStream_t stream_timer;
    cudaStreamCreateWithFlags(&stream_t, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_c, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_c_short, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_s, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_timer, cudaStreamNonBlocking);

    for (int iter = 0; iter < 10; ++iter) {
        launch_fp16_tcu_cuda_base_online(
            t_row_offset, t_blockNew_offset, t_value, t_column, t_window_row, t_atomic, t_binary,
            c_row_offset, c_row, c_atomic, c_column, c_value,
            c_row_offset_short, c_row_short, c_atomic_short, c_column_short, c_value_short,
            rhs_matrix, output_matrix,
            parts_t, parts_c, partsize_c, parts_c_short, dimN, mOri,
            stream_t, stream_c, stream_c_short);

        launch_sptc_mma_packed_v3_online(
            s_window_tile_offset, s_tile_column, s_packed_a, s_packed_meta,
            rhs_matrix, sptc_output_matrix, num_tiles, num_windows, window, dimN, mOri, kOri, stream_s);
    }
    cudaDeviceSynchronize();
    // 检查 warmup 中是否有 GPU 错误, 避免 sticky error 导致后续操作异常
    {
        cudaError_t warmup_err = cudaGetLastError();
        if (warmup_err != cudaSuccess) {
            printf("SPTC parallel warmup GPU error: %s\n", cudaGetErrorString(warmup_err));
        }
    }
    cudaMemset(output_matrix, 0, (size_t)elements * sizeof(float));
    cudaMemset(sptc_output_matrix, 0, (size_t)elements * sizeof(float));
    cudaDeviceSynchronize();

    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEvent_t done_t;
    cudaEvent_t done_c;
    cudaEvent_t done_c_short;
    cudaEvent_t done_s;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventCreate(&done_t);
    cudaEventCreate(&done_c);
    cudaEventCreate(&done_c_short);
    cudaEventCreate(&done_s);

    cudaEventRecord(spmm_start, stream_timer);
    cudaStreamWaitEvent(stream_t, spmm_start, 0);
    cudaStreamWaitEvent(stream_c, spmm_start, 0);
    cudaStreamWaitEvent(stream_c_short, spmm_start, 0);
    cudaStreamWaitEvent(stream_s, spmm_start, 0);

    for (int iter = 0; iter < epoches; ++iter) {
        launch_fp16_tcu_cuda_base_online(
            t_row_offset, t_blockNew_offset, t_value, t_column, t_window_row, t_atomic, t_binary,
            c_row_offset, c_row, c_atomic, c_column, c_value,
            c_row_offset_short, c_row_short, c_atomic_short, c_column_short, c_value_short,
            rhs_matrix, output_matrix,
            parts_t, parts_c, partsize_c, parts_c_short, dimN, mOri,
            stream_t, stream_c, stream_c_short);

        launch_sptc_mma_packed_v3_online(
            s_window_tile_offset, s_tile_column, s_packed_a, s_packed_meta,
            rhs_matrix, sptc_output_matrix, num_tiles, num_windows, window, dimN, mOri, kOri, stream_s);
    }

    cudaEventRecord(done_t, stream_t);
    cudaEventRecord(done_c, stream_c);
    cudaEventRecord(done_c_short, stream_c_short);
    cudaEventRecord(done_s, stream_s);
    cudaStreamWaitEvent(stream_timer, done_t, 0);
    cudaStreamWaitEvent(stream_timer, done_c, 0);
    cudaStreamWaitEvent(stream_timer, done_c_short, 0);
    cudaStreamWaitEvent(stream_timer, done_s, 0);
    cudaEventRecord(spmm_end, stream_timer);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    float spmm_ms_avg = spmm_ms / (float)epoches;

    float tc_ms = 0.0f;
    float cuda_long_ms = 0.0f;
    float cuda_short_ms = 0.0f;
    float sptc_ms = 0.0f;
    cudaEventElapsedTime(&tc_ms, spmm_start, done_t);
    cudaEventElapsedTime(&cuda_long_ms, spmm_start, done_c);
    cudaEventElapsedTime(&cuda_short_ms, spmm_start, done_c_short);
    cudaEventElapsedTime(&sptc_ms, spmm_start, done_s);
    if (tc_ms_avg_out != nullptr) *tc_ms_avg_out = tc_ms / (float)epoches;
    if (cuda_long_ms_avg_out != nullptr) *cuda_long_ms_avg_out = cuda_long_ms / (float)epoches;
    if (cuda_short_ms_avg_out != nullptr) *cuda_short_ms_avg_out = cuda_short_ms / (float)epoches;
    if (sptc_ms_avg_out != nullptr) *sptc_ms_avg_out = sptc_ms / (float)epoches;

    cudaMemset(output_matrix, 0, (size_t)elements * sizeof(float));
    cudaMemset(sptc_output_matrix, 0, (size_t)elements * sizeof(float));
    cudaDeviceSynchronize();

    launch_fp16_tcu_cuda_base_online(
        t_row_offset, t_blockNew_offset, t_value, t_column, t_window_row, t_atomic, t_binary,
        c_row_offset, c_row, c_atomic, c_column, c_value,
        c_row_offset_short, c_row_short, c_atomic_short, c_column_short, c_value_short,
        rhs_matrix, output_matrix,
        parts_t, parts_c, partsize_c, parts_c_short, dimN, mOri,
        stream_t, stream_c, stream_c_short);

    launch_sptc_mma_packed_v3_online(
        s_window_tile_offset, s_tile_column, s_packed_a, s_packed_meta,
        rhs_matrix, sptc_output_matrix, num_tiles, num_windows, window, dimN, mOri, kOri, stream_s);
    cudaDeviceSynchronize();

    int threads = 256;
    int blocks = (int)((elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (elements > 0) {
        add_float_matrix_online_kernel<<<blocks, threads>>>(output_matrix, sptc_output_matrix, elements);
    }
    cudaDeviceSynchronize();

    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaEventDestroy(done_t);
    cudaEventDestroy(done_c);
    cudaEventDestroy(done_c_short);
    cudaEventDestroy(done_s);
    cudaStreamDestroy(stream_t);
    cudaStreamDestroy(stream_c);
    cudaStreamDestroy(stream_c_short);
    cudaStreamDestroy(stream_s);
    cudaStreamDestroy(stream_timer);

    return spmm_ms_avg;
}
