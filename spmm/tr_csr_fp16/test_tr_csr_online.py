import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'spmm'))
sys.path.insert(0, str(PROJECT_ROOT / 'TR-source'))
sys.path.insert(0, str(PROJECT_ROOT / 'TR-source/SpMM'))

import Libra6SpMMOnline
import torch

from tr_csr_fp16.mdataset2_online import GCN_dataset_tcu_sptc_cuda_online


def _zero_sptc_like(inputInfo):
    groups = inputInfo.s_columnTensor[:0].contiguous()
    values = inputInfo.s_valueTensor[:0].contiguous()
    pos = inputInfo.s_posTensor[:0].contiguous()
    offsets = torch.zeros_like(inputInfo.s_offsetTensor)
    tile_window = inputInfo.s_tile_windowTensor[:0].contiguous()
    tile_column = inputInfo.s_tile_columnTensor[:0].contiguous()
    packed_a = inputInfo.s_packed_aTensor[:0].contiguous()
    packed_meta = inputInfo.s_packed_metaTensor[:0].contiguous()
    window_tile_offset = torch.zeros_like(inputInfo.s_window_tile_offsetTensor)
    return groups, values, pos, offsets, tile_window, tile_column, packed_a, packed_meta, window_tile_offset


def _base_tcu_cuda_profile_online(inputInfo, epoches):
    (empty_s_column, empty_s_value, empty_s_pos, empty_s_offsets,
     empty_s_tile_window, empty_s_tile_column, empty_s_packed_a,
     empty_s_packed_meta, empty_s_window_tile_offset) = _zero_sptc_like(inputInfo)
    result = Libra6SpMMOnline.forward_fp16_tcu_cuda_sptc_mma_parallel_online(
        inputInfo.t_rowNew_offsetTensor,
        inputInfo.t_blockTensor,
        inputInfo.t_columnTensor,
        inputInfo.t_valueTensor,
        inputInfo.t_window_rowTensor,
        inputInfo.t_atomicTensor,
        inputInfo.t_binaryTensor,

        inputInfo.c_row_offsetTensor,
        inputInfo.c_rowTensor,
        inputInfo.c_atomicTensor,
        inputInfo.c_colTensor,
        inputInfo.c_valueTensor,

        inputInfo.c_row_offsetTensor_short,
        inputInfo.c_rowTensor_short,
        inputInfo.c_atomicTensor_short,
        inputInfo.c_colTensor_short,
        inputInfo.c_valueTensor_short,

        empty_s_column,
        empty_s_value,
        empty_s_pos,
        empty_s_offsets,
        empty_s_tile_window,
        empty_s_tile_column,
        empty_s_packed_a,
        empty_s_packed_meta,
        empty_s_window_tile_offset,
        inputInfo.x,

        inputInfo.parts_t,
        inputInfo.parts_c,
        inputInfo.partsize_c,
        inputInfo.parts_c_short,
        inputInfo.window,
        inputInfo.x.size(1),
        inputInfo.num_nodes_ori,
        inputInfo.num_nodes_dst,
        epoches)

    if len(result) == 2:
        x_base, ms_base = result
        zero = torch.tensor(0.0)
        return x_base, ms_base, zero, zero, zero
    x_base, ms_base, ms_tc, ms_cuda_long, ms_cuda_short, _ = result
    return x_base, ms_base, ms_tc, ms_cuda_long, ms_cuda_short


def kernel_tcu_sptc_cuda_online(inputInfo, epoches):
    sptc_kernel = os.environ.get("LIBRA_SPTC_KERNEL", "mma").lower()
    sptc_parallel = os.environ.get("LIBRA_SPTC_PARALLEL", "1") == "1"
    if sptc_parallel and sptc_kernel != "cuda":
        result = Libra6SpMMOnline.forward_fp16_tcu_cuda_sptc_mma_parallel_online(
            inputInfo.t_rowNew_offsetTensor,
            inputInfo.t_blockTensor,
            inputInfo.t_columnTensor,
            inputInfo.t_valueTensor,
            inputInfo.t_window_rowTensor,
            inputInfo.t_atomicTensor,
            inputInfo.t_binaryTensor,

            inputInfo.c_row_offsetTensor,
            inputInfo.c_rowTensor,
            inputInfo.c_atomicTensor,
            inputInfo.c_colTensor,
            inputInfo.c_valueTensor,

            inputInfo.c_row_offsetTensor_short,
            inputInfo.c_rowTensor_short,
            inputInfo.c_atomicTensor_short,
            inputInfo.c_colTensor_short,
            inputInfo.c_valueTensor_short,

            inputInfo.s_columnTensor,
            inputInfo.s_valueTensor,
            inputInfo.s_posTensor,
            inputInfo.s_offsetTensor,
            inputInfo.s_tile_windowTensor,
            inputInfo.s_tile_columnTensor,
            inputInfo.s_packed_aTensor,
            inputInfo.s_packed_metaTensor,
            inputInfo.s_window_tile_offsetTensor,
            inputInfo.x,

            inputInfo.parts_t,
            inputInfo.parts_c,
            inputInfo.partsize_c,
            inputInfo.parts_c_short,
            inputInfo.window,
            inputInfo.x.size(1),
            inputInfo.num_nodes_ori,
            inputInfo.num_nodes_dst,
            epoches)
        if len(result) == 2:
            x_parallel, spmm_ms_parallel = result
            zero = torch.tensor(0.0)
            ms_tc = zero
            ms_cuda_long = zero
            ms_cuda_short = zero
            ms_sptc_parallel = zero
        else:
            x_parallel, spmm_ms_parallel, ms_tc, ms_cuda_long, ms_cuda_short, ms_sptc_parallel = result
        inputInfo.profile_times = {
            "base_tc_cuda": spmm_ms_parallel,
            "tc": ms_tc,
            "cuda_long": ms_cuda_long,
            "cuda_short": ms_cuda_short,
            "sptc": ms_sptc_parallel,
        }
        return x_parallel, spmm_ms_parallel, torch.tensor(0.0), "parallel_mma"

    x_base, spmm_ms_base, ms_tc, ms_cuda_long, ms_cuda_short = _base_tcu_cuda_profile_online(inputInfo, epoches)

    if sptc_kernel == "cuda":
        x_sptc, spmm_ms_sptc = Libra6SpMMOnline.forward_fp16_sptc_online(
            inputInfo.s_columnTensor,
            inputInfo.s_valueTensor,
            inputInfo.s_posTensor,
            inputInfo.s_windowTensor,
            inputInfo.x,
            inputInfo.window,
            inputInfo.x.size(1),
            inputInfo.num_nodes_ori,
            inputInfo.num_nodes_dst,
            epoches)
    else:
        sptc_kernel = "mma_packed"
        x_sptc, spmm_ms_sptc = Libra6SpMMOnline.forward_fp16_sptc_mma_packed_online(
            inputInfo.s_tile_columnTensor,
            inputInfo.s_packed_aTensor,
            inputInfo.s_packed_metaTensor,
            inputInfo.s_window_tile_offsetTensor,
            inputInfo.x,
            inputInfo.window,
            inputInfo.x.size(1),
            inputInfo.num_nodes_ori,
            inputInfo.num_nodes_dst,
            epoches)

    inputInfo.profile_times = {
        "base_tc_cuda": spmm_ms_base,
        "tc": ms_tc,
        "cuda_long": ms_cuda_long,
        "cuda_short": ms_cuda_short,
        "sptc": spmm_ms_sptc,
    }
    return x_base + x_sptc, spmm_ms_base, spmm_ms_sptc, sptc_kernel


def test_tcu_sptc_cuda_online(
    data,
    epoches,
    dimN,
    density,
    partsize_t,
    partsize_c,
    shortsize,
    data_path,
    window,
    wide,
    tc_min_util=0.0,
    sptc_min_util=0.50,
    sptc_threshold=12,
):
    inputInfo = GCN_dataset_tcu_sptc_cuda_online(
        data,
        dimN,
        density,
        partsize_t,
        partsize_c,
        shortsize,
        data_path,
        window,
        wide,
        tc_min_util,
        sptc_min_util,
        sptc_threshold,
    )

    _, spmm_ms_base, spmm_ms_sptc, sptc_kernel = kernel_tcu_sptc_cuda_online(inputInfo, epoches)
    execution_time = round(spmm_ms_base.item() + spmm_ms_sptc.item(), 4)
    print(str(dimN) + '-' + data + ' online-tcu-sptc-cuda-' + str(density) + '-' + str(execution_time))
    if sptc_kernel == "parallel_mma":
        print('parallel_tc_cuda_sptc_mma=' + str(round(spmm_ms_base.item(), 4)))
    else:
        print('base_tc_cuda=' + str(round(spmm_ms_base.item(), 4)) + ', sptc_' + sptc_kernel + '=' + str(round(spmm_ms_sptc.item(), 4)))
    profile = getattr(inputInfo, "profile_times", None)
    if profile is not None:
        cuda_ms = max(profile["cuda_long"].item(), profile["cuda_short"].item())
        print('kernel_breakdown: tc=' + str(round(profile["tc"].item(), 4)) +
              ', cuda=' + str(round(cuda_ms, 4)) +
              ', cuda_long=' + str(round(profile["cuda_long"].item(), 4)) +
              ', cuda_short=' + str(round(profile["cuda_short"].item(), 4)) +
              ', sptc=' + str(round(profile["sptc"].item(), 4)))
    return execution_time, inputInfo.duration
