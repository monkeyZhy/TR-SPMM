#!/usr/bin/env python3
from pathlib import Path
import sys

import os

import numpy as np
import torch
from scipy.sparse import coo_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'TR-source'))

try:
    import Rabbit
except ImportError:
    class _RabbitFallback:
        @staticmethod
        def reorder(edge_index, num_nodes):
            return edge_index, torch.arange(num_nodes, dtype=torch.int32)
    Rabbit = _RabbitFallback()

import Libra5BlockOnline

from tr_csr_fp16.tr_paths import load_graph_npz  # noqa: E402
from tr_csr_fp16.tr_tca_reorder import apply_tca_reorder  # noqa: E402


class GCN_dataset_tcu_sptc_cuda_online(torch.nn.Module):
    """
    tr SpMM online split:
      1. tr-native dense-ish TC blocks,
      2. strict SPTC 2:4 groups,
      3. tr CUDA fallback rows.
    """
    def __init__(
        self,
        data,
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
        super(GCN_dataset_tcu_sptc_cuda_online, self).__init__()
        self.graph = load_graph_npz(data_path, data)
        self.num_features = dimN
        self.window = window
        self.init_edges(density, partsize_t, partsize_c, shortsize, window, wide, tc_min_util, sptc_min_util, sptc_threshold)
        self.init_embedding()

    def init_edges(self, density, partsize_t, partsize_c, shortsize, window, wide, tc_min_util, sptc_min_util, sptc_threshold):
        self.num_nodes_ori = self.graph['num_nodes_src'] - 0
        self.num_nodes_dst = self.graph['num_nodes_dst'] - 0
        self.num_nodes = self.num_nodes_ori
        if self.num_nodes_ori % window != 0:
            self.num_nodes = self.num_nodes_ori + window - self.num_nodes_ori % window

        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        self.edge_index = np.stack([src_li, dst_li])
        val = [1] * len(src_li)
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()

        # ---- TCA Reorder: 聚集稠密块以提升 Dense TC 利用率 ----
        if os.environ.get("LIBRA_TCA_REORDER", "0") == "1":
            reordered_ptr, reordered_idx, _ = apply_tca_reorder(
                adj.indptr.astype(np.int32),
                adj.indices.astype(np.int32),
                self.num_nodes_ori,
                thres=16,
                verbose=True,
            )
            # 从重排后的 CSR 重建 edge_index
            new_src = np.zeros(len(reordered_idx), dtype=np.int64)
            for row in range(self.num_nodes_ori):
                for j in range(reordered_ptr[row], reordered_ptr[row + 1]):
                    new_src[j] = row
            self.edge_index = np.stack([new_src, reordered_idx])
            adj = coo_matrix(
                (val, (new_src, reordered_idx)),
                shape=(self.num_nodes, self.num_nodes_dst)
            ).tocsr()
        else:
            print("TCA reorder disabled (LIBRA_TCA_REORDER=0)")

        self.edge_index, _ = Rabbit.reorder(torch.IntTensor(self.edge_index), self.num_nodes)
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.num_edges = self.column_index.numel()
        self.degrees = torch.randn(self.num_edges)

        tensors = Libra5BlockOnline.block_sptc_2to4_online(
            self.row_pointers,
            self.column_index,
            self.degrees,
            partsize_t,
            partsize_c,
            shortsize,
            density,
            window,
            wide,
            float(tc_min_util),
            float(sptc_min_util),
            int(sptc_threshold),
        )

        self.t_rowNew_offsetTensor, \
        self.t_blockTensor, \
        self.t_columnTensor, \
        self.t_valueTensor, \
        self.t_window_rowTensor, \
        self.t_atomicTensor, \
        self.t_binaryTensor, \
        self.c_row_offsetTensor, \
        self.c_rowTensor, \
        self.c_atomicTensor, \
        self.c_colTensor, \
        self.c_valueTensor, \
        self.c_row_offsetTensor_short, \
        self.c_rowTensor_short, \
        self.c_atomicTensor_short, \
        self.c_colTensor_short, \
        self.c_valueTensor_short, \
        self.s_columnTensor, \
        self.s_maskTensor, \
        self.s_valueTensor, \
        self.s_posTensor, \
        self.s_windowTensor, \
        self.s_offsetTensor, \
        self.s_tile_windowTensor, \
        self.s_tile_groupTensor, \
        self.s_tile_columnTensor, \
        self.s_packed_aTensor, \
        self.s_packed_metaTensor, \
        self.s_window_tile_offsetTensor, \
        self.duration = tensors

        self.parts_t = self.t_atomicTensor.shape[0]
        self.parts_c = self.c_atomicTensor.shape[0]
        self.parts_c_short = self.c_atomicTensor_short.shape[0]
        self.partsize_c = partsize_c

        self.t_valueTensor = self.t_valueTensor.half()
        self.c_valueTensor = self.c_valueTensor.half()
        self.c_valueTensor_short = self.c_valueTensor_short.half()

        tcu_nnz = self.t_valueTensor.numel()
        sptc_nnz = int((self.s_posTensor >= 0).sum().item())
        cuda_nnz = self.c_valueTensor.numel() + self.c_valueTensor_short.numel()
        print("tcu: " + str(tcu_nnz) + ";   sptc: " + str(sptc_nnz) + ";   cuda: " + str(cuda_nnz))
        print("parts_t: " + str(self.parts_t) +
              ";   parts_c: " + str(self.parts_c) +
              ";   parts_c_short: " + str(self.parts_c_short) +
              ";   sptc_groups: " + str(self.s_columnTensor.shape[0]) +
              ";   sptc_tiles: " + str(self.s_tile_windowTensor.shape[0]))

    def init_embedding(self):
        self.x = torch.randn(self.num_nodes_dst, self.num_features)
        self.x = self.x.half()
