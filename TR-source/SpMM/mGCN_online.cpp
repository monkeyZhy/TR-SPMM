#include <torch/extension.h>

#include <cuda_fp16.h>

#include <cuda_runtime.h>

#include <assert.h>

#include <algorithm>

#include <cstdint>

#include <vector>



#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")

// 强制内存连续: 对于非连续 tensor 自动调用 .contiguous() 修复,
// 避免因 view/transpose 等操作导致 cudaMemcpy 读到非连续内存而 segfault
#define ENSURE_CONTIGUOUS(x) \
    if (!(x).is_contiguous()) { \
        (x) = (x).contiguous(); \
    }

#define CHECK_DTYPE(x, expected_dtype) TORCH_CHECK(x.dtype() == expected_dtype, #x " has an unexpected dtype")



inline cudaError_t checkCuda(cudaError_t result) {

    if (result != cudaSuccess) {

        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));

        assert(result == cudaSuccess);

    }

    return result;

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

    int epoches);



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

    int epoches);

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
    int epoches);

// 并行 kernel: TC + CUDA + SPTC (无 Warp Stacking, window=16 native)
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

    int* s_window_tile_offset,   // v2 kernel: 窗口→tile范围 [num_windows+1]

    half * rhs_matrix,
    float * output_matrix,
    float * sptc_output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int num_tiles,
    const int num_windows,        // v2 kernel: 窗口总数
    const int window,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches,
    float* tc_ms_avg_out,
    float* cuda_long_ms_avg_out,
    float* cuda_short_ms_avg_out,
    float* sptc_ms_avg_out);



template <typename T>

static void cuda_malloc_or_dummy(T** ptr, int64_t elements) {

    int64_t safe_elements = elements > 0 ? elements : 1;

    checkCuda(cudaMalloc(ptr, safe_elements * (int64_t)sizeof(T)));

}



template <typename T>

static void cuda_copy_h2d_if_needed(T* dst, const T* src, int64_t elements) {

    if (elements > 0) {

        checkCuda(cudaMemcpy(dst, src, elements * (int64_t)sizeof(T), cudaMemcpyHostToDevice));

    }

}



std::vector<torch::Tensor> spmm_forward_fp16_sptc_online(

    torch::Tensor s_column,

    torch::Tensor s_value,

    torch::Tensor s_pos,

    torch::Tensor s_window,

    torch::Tensor rhs_matrix,

    int window,

    const int dimN,

    const int mOri,

    const int kOri,

    int epoches)

{

    CHECK_CPU(s_column);

    CHECK_CPU(s_value);

    CHECK_CPU(s_pos);

    CHECK_CPU(s_window);

    CHECK_CPU(rhs_matrix);

    ENSURE_CONTIGUOUS(s_column);

    ENSURE_CONTIGUOUS(s_value);

    ENSURE_CONTIGUOUS(s_pos);

    ENSURE_CONTIGUOUS(s_window);

    ENSURE_CONTIGUOUS(rhs_matrix);

    CHECK_DTYPE(s_column, torch::kInt32);

    CHECK_DTYPE(s_value, torch::kFloat32);

    CHECK_DTYPE(s_pos, torch::kInt8);

    CHECK_DTYPE(s_window, torch::kInt32);

    CHECK_DTYPE(rhs_matrix, torch::kFloat16);

    TORCH_CHECK(window > 0, "window must be positive");

    TORCH_CHECK(dimN > 0, "dimN must be positive");

    TORCH_CHECK(mOri > 0, "mOri must be positive");

    TORCH_CHECK(kOri > 0, "kOri must be positive");

    TORCH_CHECK(epoches > 0, "epoches must be positive");

    TORCH_CHECK(s_column.dim() == 2 && s_column.size(1) == 4, "s_column must have shape [groups, 4]");

    TORCH_CHECK(s_value.dim() == 3 && s_value.size(1) == window && s_value.size(2) == 2,

                "s_value must have shape [groups, window, 2]");

    TORCH_CHECK(s_pos.dim() == 3 && s_pos.size(1) == window && s_pos.size(2) == 2,

                "s_pos must have shape [groups, window, 2]");

    TORCH_CHECK(s_window.dim() == 1, "s_window must be 1-D");

    TORCH_CHECK(rhs_matrix.numel() == (int64_t)kOri * dimN, "rhs_matrix must have kOri * dimN elements");



    int groups = s_window.size(0);

    TORCH_CHECK(s_column.size(0) == groups, "s_column group count mismatch");

    TORCH_CHECK(s_value.size(0) == groups, "s_value group count mismatch");

    TORCH_CHECK(s_pos.size(0) == groups, "s_pos group count mismatch");



    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);



    int* s_column_ = s_column.data_ptr<int>();

    float* s_value_ = s_value.data_ptr<float>();

    int8_t* s_pos_ = s_pos.data_ptr<int8_t>();

    int* s_window_ = s_window.data_ptr<int>();

    half* rhs_matrix_ = reinterpret_cast<half*>(rhs_matrix.data_ptr<at::Half>());

    float* output_matrix_ = output_matrix.data_ptr<float>();



    int* d_s_column;

    float* d_s_value;

    int8_t* d_s_pos;

    int* d_s_window;

    half* d_rhs_matrix;

    float* d_output_matrix;



    cuda_malloc_or_dummy(&d_s_column, s_column.numel());

    cuda_malloc_or_dummy(&d_s_value, s_value.numel());

    cuda_malloc_or_dummy(&d_s_pos, s_pos.numel());

    cuda_malloc_or_dummy(&d_s_window, s_window.numel());

    cuda_malloc_or_dummy(&d_rhs_matrix, (int64_t)kOri * dimN);

    cuda_malloc_or_dummy(&d_output_matrix, (int64_t)mOri * dimN);



    cuda_copy_h2d_if_needed(d_s_column, s_column_, s_column.numel());

    cuda_copy_h2d_if_needed(d_s_value, s_value_, s_value.numel());

    cuda_copy_h2d_if_needed(d_s_pos, s_pos_, s_pos.numel());

    cuda_copy_h2d_if_needed(d_s_window, s_window_, s_window.numel());

    cuda_copy_h2d_if_needed(d_rhs_matrix, rhs_matrix_, (int64_t)kOri * dimN);



    float spmm_ms_avg = spmm_forward_fp16_sptc_online_kernel(

        d_s_column,

        d_s_value,

        d_s_pos,

        d_s_window,

        d_rhs_matrix,

        d_output_matrix,

        groups,

        window,

        dimN,

        mOri,

        kOri,

        epoches);



    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (int64_t)mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));



    cudaFree(d_s_column);

    cudaFree(d_s_value);

    cudaFree(d_s_pos);

    cudaFree(d_s_window);

    cudaFree(d_rhs_matrix);

    cudaFree(d_output_matrix);

    cudaDeviceSynchronize();



    return {output_matrix, torch::tensor(spmm_ms_avg)};

}



std::vector<torch::Tensor> spmm_forward_fp16_sptc_mma_online(

    torch::Tensor s_column,

    torch::Tensor s_value,

    torch::Tensor s_pos,

    torch::Tensor s_offsets,

    torch::Tensor rhs_matrix,

    int window,

    const int dimN,

    const int mOri,

    const int kOri,

    int epoches)

{

    CHECK_CPU(s_column);

    CHECK_CPU(s_value);

    CHECK_CPU(s_pos);

    CHECK_CPU(s_offsets);

    CHECK_CPU(rhs_matrix);

    ENSURE_CONTIGUOUS(s_column);

    ENSURE_CONTIGUOUS(s_value);

    ENSURE_CONTIGUOUS(s_pos);

    ENSURE_CONTIGUOUS(s_offsets);

    ENSURE_CONTIGUOUS(rhs_matrix);

    CHECK_DTYPE(s_column, torch::kInt32);

    CHECK_DTYPE(s_value, torch::kFloat32);

    CHECK_DTYPE(s_pos, torch::kInt8);

    CHECK_DTYPE(s_offsets, torch::kInt32);

    CHECK_DTYPE(rhs_matrix, torch::kFloat16);

    TORCH_CHECK(window > 0 && window <= 16, "mma path requires 0 < window <= 16");

    TORCH_CHECK(dimN > 0, "dimN must be positive");

    TORCH_CHECK(mOri > 0, "mOri must be positive");

    TORCH_CHECK(kOri > 0, "kOri must be positive");

    TORCH_CHECK(epoches > 0, "epoches must be positive");

    TORCH_CHECK(s_column.dim() == 2 && s_column.size(1) == 4, "s_column must have shape [groups, 4]");

    TORCH_CHECK(s_value.dim() == 3 && s_value.size(1) == window && s_value.size(2) == 2,

                "s_value must have shape [groups, window, 2]");

    TORCH_CHECK(s_pos.dim() == 3 && s_pos.size(1) == window && s_pos.size(2) == 2,

                "s_pos must have shape [groups, window, 2]");

    TORCH_CHECK(s_offsets.dim() == 1 && s_offsets.size(0) >= 1, "s_offsets must be 1-D with at least one item");

    TORCH_CHECK(rhs_matrix.numel() == (int64_t)kOri * dimN, "rhs_matrix must have kOri * dimN elements");



    int groups = s_column.size(0);

    TORCH_CHECK(s_value.size(0) == groups, "s_value group count mismatch");

    TORCH_CHECK(s_pos.size(0) == groups, "s_pos group count mismatch");

    int* s_offsets_cpu = s_offsets.data_ptr<int>();

    int num_windows = s_offsets.size(0) - 1;

    TORCH_CHECK(s_offsets_cpu[0] == 0, "s_offsets must start with 0");

    TORCH_CHECK(s_offsets_cpu[num_windows] == groups, "s_offsets end must equal group count");

    std::vector<int> tile_window_host;

    std::vector<int> tile_group_host;

    tile_window_host.reserve((groups + 3) / 4);

    tile_group_host.reserve((groups + 3) / 4);

    for (int win = 0; win < num_windows; ++win) {

        int group_count = s_offsets_cpu[win + 1] - s_offsets_cpu[win];

        TORCH_CHECK(group_count >= 0, "s_offsets must be non-decreasing");

        for (int group = s_offsets_cpu[win]; group < s_offsets_cpu[win + 1]; group += 4) {

            tile_window_host.push_back(win);

            tile_group_host.push_back(group);

        }

    }

    int num_tiles = (int)tile_window_host.size();



    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);



    int* s_column_ = s_column.data_ptr<int>();

    float* s_value_ = s_value.data_ptr<float>();

    int8_t* s_pos_ = s_pos.data_ptr<int8_t>();

    half* rhs_matrix_ = reinterpret_cast<half*>(rhs_matrix.data_ptr<at::Half>());

    float* output_matrix_ = output_matrix.data_ptr<float>();



    int* d_s_column;

    float* d_s_value;

    int8_t* d_s_pos;

    int* d_s_offsets;

    int* d_tile_window;

    int* d_tile_group;

    half* d_rhs_matrix;

    float* d_output_matrix;



    cuda_malloc_or_dummy(&d_s_column, s_column.numel());

    cuda_malloc_or_dummy(&d_s_value, s_value.numel());

    cuda_malloc_or_dummy(&d_s_pos, s_pos.numel());

    cuda_malloc_or_dummy(&d_s_offsets, s_offsets.numel());

    cuda_malloc_or_dummy(&d_tile_window, tile_window_host.size());

    cuda_malloc_or_dummy(&d_tile_group, tile_group_host.size());

    cuda_malloc_or_dummy(&d_rhs_matrix, (int64_t)kOri * dimN);

    cuda_malloc_or_dummy(&d_output_matrix, (int64_t)mOri * dimN);



    cuda_copy_h2d_if_needed(d_s_column, s_column_, s_column.numel());

    cuda_copy_h2d_if_needed(d_s_value, s_value_, s_value.numel());

    cuda_copy_h2d_if_needed(d_s_pos, s_pos_, s_pos.numel());

    cuda_copy_h2d_if_needed(d_s_offsets, s_offsets_cpu, s_offsets.numel());

    cuda_copy_h2d_if_needed(d_tile_window, tile_window_host.data(), tile_window_host.size());

    cuda_copy_h2d_if_needed(d_tile_group, tile_group_host.data(), tile_group_host.size());

    cuda_copy_h2d_if_needed(d_rhs_matrix, rhs_matrix_, (int64_t)kOri * dimN);



    float spmm_ms_avg = spmm_forward_fp16_sptc_mma_online_kernel(

        d_s_column,

        d_s_value,

        d_s_pos,

        d_s_offsets,

        d_tile_window,

        d_tile_group,

        d_rhs_matrix,

        d_output_matrix,

        num_tiles,

        window,

        dimN,

        mOri,

        kOri,

        epoches);



    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (int64_t)mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));



    cudaFree(d_s_column);

    cudaFree(d_s_value);

    cudaFree(d_s_pos);

    cudaFree(d_s_offsets);

    cudaFree(d_tile_window);

    cudaFree(d_tile_group);

    cudaFree(d_rhs_matrix);

    cudaFree(d_output_matrix);

    cudaDeviceSynchronize();



    return {output_matrix, torch::tensor(spmm_ms_avg)};

}

std::vector<torch::Tensor> spmm_forward_fp16_sptc_mma_packed_online(
    torch::Tensor s_tile_column,
    torch::Tensor s_packed_a,
    torch::Tensor s_packed_meta,
    torch::Tensor s_window_tile_offset,
    torch::Tensor rhs_matrix,
    int window,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    CHECK_CPU(s_tile_column);
    CHECK_CPU(s_packed_a);
    CHECK_CPU(s_packed_meta);
    CHECK_CPU(s_window_tile_offset);
    CHECK_CPU(rhs_matrix);
    ENSURE_CONTIGUOUS(s_tile_column);
    ENSURE_CONTIGUOUS(s_packed_a);
    ENSURE_CONTIGUOUS(s_packed_meta);
    ENSURE_CONTIGUOUS(s_window_tile_offset);
    ENSURE_CONTIGUOUS(rhs_matrix);
    CHECK_DTYPE(s_tile_column, torch::kInt32);
    CHECK_DTYPE(s_packed_a, torch::kInt32);
    CHECK_DTYPE(s_packed_meta, torch::kInt32);
    CHECK_DTYPE(s_window_tile_offset, torch::kInt32);
    CHECK_DTYPE(rhs_matrix, torch::kFloat16);
    TORCH_CHECK(window > 0 && window <= 16, "packed mma path requires 0 < window <= 16");
    TORCH_CHECK(dimN > 0, "dimN must be positive");
    TORCH_CHECK(mOri > 0, "mOri must be positive");
    TORCH_CHECK(kOri > 0, "kOri must be positive");
    TORCH_CHECK(epoches > 0, "epoches must be positive");
    TORCH_CHECK(s_tile_column.dim() == 2 && s_tile_column.size(1) == 16,
                "s_tile_column must have shape [tiles, 16]");
    TORCH_CHECK(s_packed_a.dim() == 4 && s_packed_a.size(1) == 8 &&
                s_packed_a.size(2) == 4 && s_packed_a.size(3) == 2,
                "s_packed_a must have shape [tiles, 8, 4, 2]");
    TORCH_CHECK(s_packed_meta.dim() == 2 && s_packed_meta.size(1) == 8,
                "s_packed_meta must have shape [tiles, 8]");
    TORCH_CHECK(s_window_tile_offset.dim() == 1 && s_window_tile_offset.size(0) >= 2,
                "s_window_tile_offset must be 1-D with at least 2 elements");
    TORCH_CHECK(rhs_matrix.numel() == (int64_t)kOri * dimN, "rhs_matrix must have kOri * dimN elements");

    int num_tiles = s_tile_column.size(0);
    int num_windows = s_window_tile_offset.size(0) - 1;
    TORCH_CHECK(s_packed_a.size(0) == num_tiles, "s_packed_a tile count mismatch");
    TORCH_CHECK(s_packed_meta.size(0) == num_tiles, "s_packed_meta tile count mismatch");

    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

    int* s_tile_column_ = s_tile_column.data_ptr<int>();
    uint32_t* s_packed_a_ = reinterpret_cast<uint32_t*>(s_packed_a.data_ptr<int>());
    uint32_t* s_packed_meta_ = reinterpret_cast<uint32_t*>(s_packed_meta.data_ptr<int>());
    int* s_window_tile_offset_ = s_window_tile_offset.data_ptr<int>();
    half* rhs_matrix_ = reinterpret_cast<half*>(rhs_matrix.data_ptr<at::Half>());
    float* output_matrix_ = output_matrix.data_ptr<float>();

    int* d_s_tile_column;
    uint32_t* d_s_packed_a;
    uint32_t* d_s_packed_meta;
    int* d_s_window_tile_offset;
    half* d_rhs_matrix;
    float* d_output_matrix;

    cuda_malloc_or_dummy(&d_s_tile_column, s_tile_column.numel());
    cuda_malloc_or_dummy(&d_s_packed_a, s_packed_a.numel());
    cuda_malloc_or_dummy(&d_s_packed_meta, s_packed_meta.numel());
    cuda_malloc_or_dummy(&d_s_window_tile_offset, s_window_tile_offset.numel());
    cuda_malloc_or_dummy(&d_rhs_matrix, (int64_t)kOri * dimN);
    cuda_malloc_or_dummy(&d_output_matrix, (int64_t)mOri * dimN);

    cuda_copy_h2d_if_needed(d_s_tile_column, s_tile_column_, s_tile_column.numel());
    cuda_copy_h2d_if_needed(d_s_packed_a, s_packed_a_, s_packed_a.numel());
    cuda_copy_h2d_if_needed(d_s_packed_meta, s_packed_meta_, s_packed_meta.numel());
    cuda_copy_h2d_if_needed(d_s_window_tile_offset, s_window_tile_offset_, s_window_tile_offset.numel());
    cuda_copy_h2d_if_needed(d_rhs_matrix, rhs_matrix_, (int64_t)kOri * dimN);

    float spmm_ms_avg = spmm_forward_fp16_sptc_mma_packed_online_kernel(
        d_s_window_tile_offset,
        d_s_tile_column,
        d_s_packed_a,
        d_s_packed_meta,
        d_rhs_matrix,
        d_output_matrix,
        num_tiles,
        num_windows,
        window,
        dimN,
        mOri,
        kOri,
        epoches);

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (int64_t)mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_s_tile_column);
    cudaFree(d_s_packed_a);
    cudaFree(d_s_packed_meta);
    cudaFree(d_s_window_tile_offset);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize();

    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

// 并行入口: TC + CUDA + SPTC (无 Warp Stacking, window=16 native)
std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda_sptc_mma_parallel_online(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset,
    torch::Tensor t_column,
    torch::Tensor t_value,
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor t_binary,

    torch::Tensor c_row_offsets,
    torch::Tensor c_row,
    torch::Tensor c_atomic,
    torch::Tensor c_column,
    torch::Tensor c_value,

    torch::Tensor c_row_offsets_short,
    torch::Tensor c_row_short,
    torch::Tensor c_atomic_short,
    torch::Tensor c_column_short,
    torch::Tensor c_value_short,

    torch::Tensor s_column,
    torch::Tensor s_value,
    torch::Tensor s_pos,
    torch::Tensor s_offsets,
    torch::Tensor s_tile_window,
    torch::Tensor s_tile_column,
    torch::Tensor s_packed_a,
    torch::Tensor s_packed_meta,
    torch::Tensor s_window_tile_offset,  // v2 kernel
    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    int window,
    const int dimN,
    const int mOri,
    const int kOri,
    int epoches)
{
    CHECK_CPU(t_row_offset);
    CHECK_CPU(t_blockNew_offset);
    CHECK_CPU(t_column);
    CHECK_CPU(t_value);
    CHECK_CPU(t_window_row);
    CHECK_CPU(t_atomic);
    CHECK_CPU(t_binary);
    CHECK_CPU(c_row_offsets);
    CHECK_CPU(c_row);
    CHECK_CPU(c_atomic);
    CHECK_CPU(c_column);
    CHECK_CPU(c_value);
    CHECK_CPU(c_row_offsets_short);
    CHECK_CPU(c_row_short);
    CHECK_CPU(c_atomic_short);
    CHECK_CPU(c_column_short);
    CHECK_CPU(c_value_short);
    CHECK_CPU(s_column);
    CHECK_CPU(s_value);
    CHECK_CPU(s_pos);
    CHECK_CPU(s_offsets);
    CHECK_CPU(s_tile_window);
    CHECK_CPU(s_tile_column);
    CHECK_CPU(s_packed_a);
    CHECK_CPU(s_packed_meta);
    CHECK_CPU(s_window_tile_offset);
    CHECK_CPU(rhs_matrix);

    ENSURE_CONTIGUOUS(t_row_offset);
    ENSURE_CONTIGUOUS(t_blockNew_offset);
    ENSURE_CONTIGUOUS(t_column);
    ENSURE_CONTIGUOUS(t_value);
    ENSURE_CONTIGUOUS(t_window_row);
    ENSURE_CONTIGUOUS(t_atomic);
    ENSURE_CONTIGUOUS(t_binary);
    ENSURE_CONTIGUOUS(c_row_offsets);
    ENSURE_CONTIGUOUS(c_row);
    ENSURE_CONTIGUOUS(c_atomic);
    ENSURE_CONTIGUOUS(c_column);
    ENSURE_CONTIGUOUS(c_value);
    ENSURE_CONTIGUOUS(c_row_offsets_short);
    ENSURE_CONTIGUOUS(c_row_short);
    ENSURE_CONTIGUOUS(c_atomic_short);
    ENSURE_CONTIGUOUS(c_column_short);
    ENSURE_CONTIGUOUS(c_value_short);
    ENSURE_CONTIGUOUS(s_column);
    ENSURE_CONTIGUOUS(s_value);
    ENSURE_CONTIGUOUS(s_pos);
    ENSURE_CONTIGUOUS(s_offsets);
    ENSURE_CONTIGUOUS(s_tile_window);
    ENSURE_CONTIGUOUS(s_tile_column);
    ENSURE_CONTIGUOUS(s_packed_a);
    ENSURE_CONTIGUOUS(s_packed_meta);
    ENSURE_CONTIGUOUS(s_window_tile_offset);
    ENSURE_CONTIGUOUS(rhs_matrix);

    CHECK_DTYPE(t_row_offset, torch::kInt32);
    CHECK_DTYPE(t_blockNew_offset, torch::kInt32);
    CHECK_DTYPE(t_column, torch::kInt32);
    CHECK_DTYPE(t_value, torch::kFloat16);
    CHECK_DTYPE(t_window_row, torch::kInt32);
    CHECK_DTYPE(t_atomic, torch::kInt32);
    CHECK_DTYPE(t_binary, torch::kInt64);
    CHECK_DTYPE(c_row_offsets, torch::kInt32);
    CHECK_DTYPE(c_row, torch::kInt32);
    CHECK_DTYPE(c_atomic, torch::kInt32);
    CHECK_DTYPE(c_column, torch::kInt32);
    CHECK_DTYPE(c_value, torch::kFloat16);
    CHECK_DTYPE(c_row_offsets_short, torch::kInt32);
    CHECK_DTYPE(c_row_short, torch::kInt32);
    CHECK_DTYPE(c_atomic_short, torch::kInt32);
    CHECK_DTYPE(c_column_short, torch::kInt32);
    CHECK_DTYPE(c_value_short, torch::kFloat16);
    CHECK_DTYPE(s_column, torch::kInt32);
    CHECK_DTYPE(s_value, torch::kFloat32);
    CHECK_DTYPE(s_pos, torch::kInt8);
    CHECK_DTYPE(s_offsets, torch::kInt32);
    CHECK_DTYPE(s_tile_window, torch::kInt32);
    CHECK_DTYPE(s_tile_column, torch::kInt32);
    CHECK_DTYPE(s_packed_a, torch::kInt32);
    CHECK_DTYPE(s_packed_meta, torch::kInt32);
    CHECK_DTYPE(rhs_matrix, torch::kFloat16);

    TORCH_CHECK(window > 0 && window <= 16, "parallel mma path requires 0 < window <= 16");
    TORCH_CHECK(dimN > 0, "dimN must be positive");
    TORCH_CHECK(mOri > 0, "mOri must be positive");
    TORCH_CHECK(kOri > 0, "kOri must be positive");
    TORCH_CHECK(epoches > 0, "epoches must be positive");
    TORCH_CHECK(rhs_matrix.numel() == (int64_t)kOri * dimN, "rhs_matrix must have kOri * dimN elements");
    TORCH_CHECK(s_column.dim() == 2 && s_column.size(1) == 4, "s_column must have shape [groups, 4]");
    TORCH_CHECK(s_value.dim() == 3 && s_value.size(1) == window && s_value.size(2) == 2,
                "s_value must have shape [groups, window, 2]");
    TORCH_CHECK(s_pos.dim() == 3 && s_pos.size(1) == window && s_pos.size(2) == 2,
                "s_pos must have shape [groups, window, 2]");
    TORCH_CHECK(s_offsets.dim() == 1 && s_offsets.size(0) >= 1, "s_offsets must be 1-D with at least one item");
    TORCH_CHECK(s_tile_window.dim() == 1, "s_tile_window must be 1-D");
    TORCH_CHECK(s_tile_column.dim() == 2 && s_tile_column.size(1) == 16,
                "s_tile_column must have shape [tiles, 16]");
    TORCH_CHECK(s_packed_a.dim() == 4 && s_packed_a.size(1) == 8 &&
                s_packed_a.size(2) == 4 && s_packed_a.size(3) == 2,
                "s_packed_a must have shape [tiles, 8, 4, 2]");
    TORCH_CHECK(s_packed_meta.dim() == 2 && s_packed_meta.size(1) == 8,
                "s_packed_meta must have shape [tiles, 8]");

    int groups = s_column.size(0);
    TORCH_CHECK(s_value.size(0) == groups, "s_value group count mismatch");
    TORCH_CHECK(s_pos.size(0) == groups, "s_pos group count mismatch");
    int* s_offsets_cpu = s_offsets.data_ptr<int>();
    int num_windows = s_offsets.size(0) - 1;
    TORCH_CHECK(s_offsets_cpu[0] == 0, "s_offsets must start with 0");
    TORCH_CHECK(s_offsets_cpu[num_windows] == groups, "s_offsets end must equal group count");

    std::vector<int> tile_window_host;
    std::vector<int> tile_group_host;
    tile_window_host.reserve((groups + 3) / 4);
    tile_group_host.reserve((groups + 3) / 4);
    for (int win = 0; win < num_windows; ++win) {
        int group_count = s_offsets_cpu[win + 1] - s_offsets_cpu[win];
        TORCH_CHECK(group_count >= 0, "s_offsets must be non-decreasing");
        for (int group = s_offsets_cpu[win]; group < s_offsets_cpu[win + 1]; group += 4) {
            tile_window_host.push_back(win);
            tile_group_host.push_back(group);
        }
    }
    int num_tiles = (int)tile_window_host.size();
    int packed_num_tiles = s_tile_window.size(0);
    TORCH_CHECK(s_tile_column.size(0) == packed_num_tiles, "s_tile_column tile count mismatch");
    TORCH_CHECK(s_packed_a.size(0) == packed_num_tiles, "s_packed_a tile count mismatch");
    TORCH_CHECK(s_packed_meta.size(0) == packed_num_tiles, "s_packed_meta tile count mismatch");
    TORCH_CHECK(packed_num_tiles == num_tiles,
                "packed tile count must match offsets-derived tile count");

    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    float* output_matrix_ = output_matrix.data_ptr<float>();

    int* t_row_offset_ = t_row_offset.data_ptr<int>();
    int* t_blockNew_offset_ = t_blockNew_offset.data_ptr<int>();
    int* t_column_ = t_column.data_ptr<int>();
    half* t_value_ = reinterpret_cast<half*>(t_value.data_ptr<at::Half>());
    int* t_window_row_ = t_window_row.data_ptr<int>();
    int* t_atomic_ = t_atomic.data_ptr<int>();
    long* t_binary_ = t_binary.data_ptr<long>();

    int* c_row_offsets_ = c_row_offsets.data_ptr<int>();
    int* c_row_ = c_row.data_ptr<int>();
    int* c_atomic_ = c_atomic.data_ptr<int>();
    int* c_column_ = c_column.data_ptr<int>();
    half* c_value_ = reinterpret_cast<half*>(c_value.data_ptr<at::Half>());

    int* c_row_offsets_short_ = c_row_offsets_short.data_ptr<int>();
    int* c_row_short_ = c_row_short.data_ptr<int>();
    int* c_atomic_short_ = c_atomic_short.data_ptr<int>();
    int* c_column_short_ = c_column_short.data_ptr<int>();
    half* c_value_short_ = reinterpret_cast<half*>(c_value_short.data_ptr<at::Half>());

    int* s_column_ = s_column.data_ptr<int>();
    float* s_value_ = s_value.data_ptr<float>();
    int8_t* s_pos_ = s_pos.data_ptr<int8_t>();
    int* s_tile_window_ = s_tile_window.data_ptr<int>();
    int* s_tile_column_ = s_tile_column.data_ptr<int>();
    uint32_t* s_packed_a_ = reinterpret_cast<uint32_t*>(s_packed_a.data_ptr<int>());
    uint32_t* s_packed_meta_ = reinterpret_cast<uint32_t*>(s_packed_meta.data_ptr<int>());
    int* s_window_tile_offset_ = s_window_tile_offset.data_ptr<int>();
    half* rhs_matrix_ = reinterpret_cast<half*>(rhs_matrix.data_ptr<at::Half>());

    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column, *d_t_window_row, *d_t_atomic;
    long *d_t_binary;
    half *d_t_value;
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    half *d_c_value, *d_c_value_short;
    int *d_s_column, *d_s_offsets, *d_tile_window, *d_tile_group;
    int *d_s_tile_window, *d_s_tile_column;
    int *d_s_window_tile_offset;
    uint32_t *d_s_packed_a, *d_s_packed_meta;
    float *d_s_value;
    int8_t *d_s_pos;
    half *d_rhs_matrix;
    float *d_output_matrix, *d_sptc_output_matrix;

    cuda_malloc_or_dummy(&d_t_row_offset, t_row_offset.numel());
    cuda_malloc_or_dummy(&d_t_blockNew_offset, t_blockNew_offset.numel());
    cuda_malloc_or_dummy(&d_t_column, t_column.numel());
    cuda_malloc_or_dummy(&d_t_value, t_value.numel());
    cuda_malloc_or_dummy(&d_t_window_row, t_window_row.numel());
    cuda_malloc_or_dummy(&d_t_atomic, t_atomic.numel());
    cuda_malloc_or_dummy(&d_t_binary, t_binary.numel());
    cuda_malloc_or_dummy(&d_c_row_offsets, c_row_offsets.numel());
    cuda_malloc_or_dummy(&d_c_row, c_row.numel());
    cuda_malloc_or_dummy(&d_c_atomic, c_atomic.numel());
    cuda_malloc_or_dummy(&d_c_column, c_column.numel());
    cuda_malloc_or_dummy(&d_c_value, c_value.numel());
    cuda_malloc_or_dummy(&d_c_row_offsets_short, c_row_offsets_short.numel());
    cuda_malloc_or_dummy(&d_c_row_short, c_row_short.numel());
    cuda_malloc_or_dummy(&d_c_atomic_short, c_atomic_short.numel());
    cuda_malloc_or_dummy(&d_c_column_short, c_column_short.numel());
    cuda_malloc_or_dummy(&d_c_value_short, c_value_short.numel());
    cuda_malloc_or_dummy(&d_s_column, s_column.numel());
    cuda_malloc_or_dummy(&d_s_value, s_value.numel());
    cuda_malloc_or_dummy(&d_s_pos, s_pos.numel());
    cuda_malloc_or_dummy(&d_s_offsets, s_offsets.numel());
    cuda_malloc_or_dummy(&d_tile_window, tile_window_host.size());
    cuda_malloc_or_dummy(&d_tile_group, tile_group_host.size());
    cuda_malloc_or_dummy(&d_s_tile_window, s_tile_window.numel());
    cuda_malloc_or_dummy(&d_s_tile_column, s_tile_column.numel());
    cuda_malloc_or_dummy(&d_s_packed_a, s_packed_a.numel());
    cuda_malloc_or_dummy(&d_s_packed_meta, s_packed_meta.numel());
    cuda_malloc_or_dummy(&d_s_window_tile_offset, s_window_tile_offset.numel());
    cuda_malloc_or_dummy(&d_rhs_matrix, (int64_t)kOri * dimN);
    cuda_malloc_or_dummy(&d_output_matrix, (int64_t)mOri * dimN);
    cuda_malloc_or_dummy(&d_sptc_output_matrix, (int64_t)mOri * dimN);

    cuda_copy_h2d_if_needed(d_t_row_offset, t_row_offset_, t_row_offset.numel());
    cuda_copy_h2d_if_needed(d_t_blockNew_offset, t_blockNew_offset_, t_blockNew_offset.numel());
    cuda_copy_h2d_if_needed(d_t_column, t_column_, t_column.numel());
    cuda_copy_h2d_if_needed(d_t_value, t_value_, t_value.numel());
    cuda_copy_h2d_if_needed(d_t_window_row, t_window_row_, t_window_row.numel());
    cuda_copy_h2d_if_needed(d_t_atomic, t_atomic_, t_atomic.numel());
    cuda_copy_h2d_if_needed(d_t_binary, t_binary_, t_binary.numel());
    cuda_copy_h2d_if_needed(d_c_row_offsets, c_row_offsets_, c_row_offsets.numel());
    cuda_copy_h2d_if_needed(d_c_row, c_row_, c_row.numel());
    cuda_copy_h2d_if_needed(d_c_atomic, c_atomic_, c_atomic.numel());
    cuda_copy_h2d_if_needed(d_c_column, c_column_, c_column.numel());
    cuda_copy_h2d_if_needed(d_c_value, c_value_, c_value.numel());
    cuda_copy_h2d_if_needed(d_c_row_offsets_short, c_row_offsets_short_, c_row_offsets_short.numel());
    cuda_copy_h2d_if_needed(d_c_row_short, c_row_short_, c_row_short.numel());
    cuda_copy_h2d_if_needed(d_c_atomic_short, c_atomic_short_, c_atomic_short.numel());
    cuda_copy_h2d_if_needed(d_c_column_short, c_column_short_, c_column_short.numel());
    cuda_copy_h2d_if_needed(d_c_value_short, c_value_short_, c_value_short.numel());
    cuda_copy_h2d_if_needed(d_s_column, s_column_, s_column.numel());
    cuda_copy_h2d_if_needed(d_s_value, s_value_, s_value.numel());
    cuda_copy_h2d_if_needed(d_s_pos, s_pos_, s_pos.numel());
    cuda_copy_h2d_if_needed(d_s_offsets, s_offsets_cpu, s_offsets.numel());
    cuda_copy_h2d_if_needed(d_tile_window, tile_window_host.data(), tile_window_host.size());
    cuda_copy_h2d_if_needed(d_tile_group, tile_group_host.data(), tile_group_host.size());
    cuda_copy_h2d_if_needed(d_s_tile_window, s_tile_window_, s_tile_window.numel());
    cuda_copy_h2d_if_needed(d_s_tile_column, s_tile_column_, s_tile_column.numel());
    cuda_copy_h2d_if_needed(d_s_packed_a, s_packed_a_, s_packed_a.numel());
    cuda_copy_h2d_if_needed(d_s_packed_meta, s_packed_meta_, s_packed_meta.numel());
    cuda_copy_h2d_if_needed(d_s_window_tile_offset, s_window_tile_offset_, s_window_tile_offset.numel());
    cuda_copy_h2d_if_needed(d_rhs_matrix, rhs_matrix_, (int64_t)kOri * dimN);

    float tc_ms_avg = 0.0f;
    float cuda_long_ms_avg = 0.0f;
    float cuda_short_ms_avg = 0.0f;
    float sptc_ms_avg = 0.0f;
    float spmm_ms_avg = spmm_forward_fp16_tcu_cuda_sptc_mma_parallel_online_kernel(
        d_t_row_offset,
        d_t_blockNew_offset,
        d_t_value,
        d_t_column,
        d_t_window_row,
        d_t_atomic,
        d_t_binary,
        d_c_row_offsets,
        d_c_row,
        d_c_atomic,
        d_c_column,
        d_c_value,
        d_c_row_offsets_short,
        d_c_row_short,
        d_c_atomic_short,
        d_c_column_short,
        d_c_value_short,
        d_s_column,
        d_s_value,
        d_s_pos,
        d_s_offsets,
        d_tile_window,
        d_tile_group,
        d_s_tile_column,
        d_s_packed_a,
        d_s_packed_meta,
        d_s_tile_window,
        d_s_window_tile_offset,
        d_rhs_matrix,
        d_output_matrix,
        d_sptc_output_matrix,
        parts_t,
        parts_c,
        partsize_c,
        parts_c_short,
        num_tiles,
        num_windows,
        window,
        dimN,
        mOri,
        kOri,
        epoches,
        &tc_ms_avg,
        &cuda_long_ms_avg,
        &cuda_short_ms_avg,
        &sptc_ms_avg);

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (int64_t)mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_column);
    cudaFree(d_t_value);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);
    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);
    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);
    cudaFree(d_s_column);
    cudaFree(d_s_value);
    cudaFree(d_s_pos);
    cudaFree(d_s_offsets);
    cudaFree(d_tile_window);
    cudaFree(d_tile_group);
    cudaFree(d_s_tile_window);
    cudaFree(d_s_tile_column);
    cudaFree(d_s_packed_a);
    cudaFree(d_s_packed_meta);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaFree(d_sptc_output_matrix);
    cudaDeviceSynchronize();

    return {
        output_matrix,
        torch::tensor(spmm_ms_avg),
        torch::tensor(tc_ms_avg),
        torch::tensor(cuda_long_ms_avg),
        torch::tensor(cuda_short_ms_avg),
        torch::tensor(sptc_ms_avg)
    };
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("forward_fp16_sptc_online", &spmm_forward_fp16_sptc_online, "Libra online SPTC 2:4 SpMM (window=16)");

    m.def("forward_fp16_sptc_mma_online", &spmm_forward_fp16_sptc_mma_online, "Libra online SPTC 2:4 SpMM with mma.sp (window=16)");

    m.def("forward_fp16_sptc_mma_packed_online", &spmm_forward_fp16_sptc_mma_packed_online, "Libra online packed SPTC 2:4 SpMM with mma.sp (window=16 native)");

    m.def("forward_fp16_tcu_cuda_sptc_mma_parallel_online", &spmm_forward_fp16_tcu_cuda_sptc_mma_parallel_online, "Libra online TC/CUDA + SPTC mma.sp parallel SpMM (window=16)");

}
