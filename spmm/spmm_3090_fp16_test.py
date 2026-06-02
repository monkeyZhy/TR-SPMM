import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'spmm'))
sys.path.insert(0, str(PROJECT_ROOT / 'TR-source'))
sys.path.insert(0, str(PROJECT_ROOT / 'TR-source/SpMM'))

from tr_csr_fp16 import test_tr_csr_online  # noqa: E402

if "LIBRA_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LIBRA_CUDA_VISIBLE_DEVICES"]
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")


if __name__ == "__main__":
    gpu_device = torch.cuda.current_device()
    print(torch.cuda.get_device_name(gpu_device))

    dimN = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    dataset = sys.argv[2] if len(sys.argv) > 2 else "2D_27628_bjtcai"
    epoches = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # 强制 16x16 维度对齐: window=16, wide=16
    # density: window=16 时只有密度极高的列才送给 TC, 测试范围 [10, 12, 14, 15]
    density = int(os.environ.get("LIBRA_DENSITY", "8"))
    partsize_t = 32
    partsize_c = 32
    shortsize = 3
    data_path = "sp_matrix"
    window = 16
    wide = 16
    tc_min_util = float(os.environ.get("LIBRA_TC_MIN_UTIL", "0.0"))
    sptc_min_util = float(os.environ.get("LIBRA_SPTC_MIN_UTIL", "0.5"))
    # SPTC 绝对 nnz 阈值: 16x16 块中总非零元 < sptc_threshold 则退回 CUDA
    sptc_threshold = int(os.environ.get("LIBRA_SPTC_THRESHOLD", "12"))

    test_tr_csr_online.test_tcu_sptc_cuda_online(
        dataset,
        epoches,
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
