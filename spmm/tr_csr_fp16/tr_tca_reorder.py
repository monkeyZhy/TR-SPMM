#!/usr/bin/env python3
"""
TCA (TCU-Aware) Row Reordering for Libra Dense TC Pipeline.
方案 B — 纯 CPU 实现, 仅依赖 datasketch (MinHashLSH).

基于 DTC-SpMM_ASPLOS24 的 TCA_reorder.py 核心算法:
  1. MinHashLSH 行签名 + 相似行查询
  2. 纯 Python Jaccard 相似度 (集合并交)
  3. 优先队列 + 并查集聚类 (max cluster size ≤ 16)
  4. 两级聚类: 行级 (thres=16) → 簇级 (thres=128)
"""

import time
import numpy as np
from datasketch import MinHash, MinHashLSH


def _setup_seed(seed=2022):
    import os
    import random
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _jaccard_sets(s1, s2):
    """预计算 set 版本的 Jaccard, 用于快速相似度比较."""
    if not s1 or not s2:
        return 0.0
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return float(inter) / union if union > 0 else 0.0


def _jaccard_lists(l1, l2):
    """从列表计算 Jaccard (用于 union-find fallback)."""
    if len(l1) == 0 or len(l2) == 0:
        return 0.0
    inter = len(set(l1).intersection(set(l2)))
    union = len(set(l1).union(set(l2)))
    return float(inter) / union if union > 0 else 0.0


def apply_tca_reorder(row_ptr, col_ind, num_nodes, thres=16, verbose=True):
    """
    对 CSR 格式的稀疏矩阵应用 TCA 行重排, 聚集稠密块以提升 TC 利用率.
    方案 B: 纯 CPU 实现 (仅依赖 datasketch).

    Args:
        row_ptr:  np.ndarray [num_nodes+1], CSR indptr
        col_ind:  np.ndarray [num_edges], CSR indices
        num_nodes: int, 源节点数
        thres:     int, 最大聚类大小 (默认 16, 对应 16×16 TC 窗口)
        verbose:   bool, 是否输出详细日志

    Returns:
        new_row_ptr:  np.ndarray [num_nodes+1], 重排后的 CSR indptr
        new_col_ind:  np.ndarray [num_edges], 重排后的 CSR indices
        reorder_id:   np.ndarray [num_nodes], 节点映射表 (new_id → old_id)
    """
    _setup_seed(2022)

    num_nnz = len(col_ind)
    t_start = time.time()
    per = 128
    lsh_thres = 0.2

    # =========================================================================
    # Step 0: 预计算每行的邻居集合 (用于快速 Jaccard)
    # =========================================================================
    t0 = time.time()
    row_sets = [None] * num_nodes   # set of neighbor column indices
    row_lists = [None] * num_nodes  # list version for union-find fallback
    for i in range(num_nodes):
        neighbors = col_ind[row_ptr[i]:row_ptr[i + 1]].tolist()
        row_lists[i] = neighbors
        row_sets[i] = set(neighbors)
    if verbose:
        print(f"TCA: precomputed {num_nodes} row neighbor sets, "
              f"time = {time.time() - t0:.2f}s")

    # =========================================================================
    # Step 1: 构建 MinHash LSH 索引 (纯 CPU)
    # =========================================================================
    t1 = time.time()
    lsh = MinHashLSH(threshold=lsh_thres, num_perm=per)
    allver = []
    for i in range(num_nodes):
        m = MinHash(num_perm=per)
        for j in range(row_ptr[i], row_ptr[i + 1]):
            m.update(str(col_ind[j]).encode('utf-8'))
        lsh.insert(str(i), m)
        allver.append(m)
    if verbose:
        print(f"TCA: LSH init time = {time.time() - t1:.2f}s")

    # =========================================================================
    # Step 2: 查询 LSH + 计算 Jaccard, 入优先队列
    # =========================================================================
    import queue as Q

    def _make_pair_key(a, b, n):
        if a > b:
            a, b = b, a
        return a * n + b

    class Pair:
        __slots__ = ('p1', 'p2', 'simi')
        def __init__(self, p1, p2, simi):
            self.p1 = p1
            self.p2 = p2
            self.simi = simi
        def __lt__(self, other):
            return self.simi > other.simi

    sset = set()
    que = Q.PriorityQueue()

    t2 = time.time()
    for i in range(num_nodes):
        if row_ptr[i] == row_ptr[i + 1]:
            continue
        res = lsh.query(allver[i])
        for item_str in res:
            item = int(item_str)
            if item == i or _make_pair_key(i, item, num_nodes) in sset:
                continue
            sim = _jaccard_sets(row_sets[i], row_sets[item])
            que.put(Pair(i, item, sim))
            sset.add(_make_pair_key(i, item, num_nodes))
        if verbose and i % 5000 == 0 and i > 0:
            print(f"TCA: queried LSH row {i}/{num_nodes}, "
                  f"queue size = {que.qsize()}")

    t3 = time.time()
    if verbose:
        print(f"TCA: LSH query + Jaccard time = {t3 - t2:.2f}s, "
              f"queue size = {que.qsize()}")

    # =========================================================================
    # Step 3: 优先队列 + 并查集聚类 (Level 1: 行级, max size = thres)
    # =========================================================================
    def _root(i, cluster_id):
        while i != cluster_id[i]:
            cluster_id[i] = cluster_id[cluster_id[i]]
            i = cluster_id[i]
        return i

    cluster_id = list(range(num_nodes))
    cluster_sz = [1] * num_nodes
    deleted = [0] * num_nodes
    num_cluster = num_nodes

    t4 = time.time()
    while (not que.empty()) and num_cluster > 0:
        item = que.get()
        p1, p2 = item.p1, item.p2
        sset.discard(_make_pair_key(p1, p2, num_nodes))

        if p1 == cluster_id[p1] and p2 == cluster_id[p2]:
            if deleted[p1] or deleted[p2]:
                continue
            if cluster_sz[p1] < cluster_sz[p2]:
                cluster_id[p1] = p2
                num_cluster -= 1
                cluster_sz[p2] += cluster_sz[p1]
                if cluster_sz[p2] >= thres:
                    deleted[p2] = 1
                    num_cluster -= 1
            else:
                cluster_id[p2] = p1
                num_cluster -= 1
                cluster_sz[p1] += cluster_sz[p2]
                if cluster_sz[p1] >= thres:
                    deleted[p1] = 1
                    num_cluster -= 1
        else:
            r1 = _root(p1, cluster_id)
            r2 = _root(p2, cluster_id)
            if deleted[r1] or deleted[r2]:
                continue
            if r1 != r2 and _make_pair_key(r1, r2, num_nodes) not in sset:
                que.put(Pair(r1, r2, _jaccard_lists(row_lists[r1], row_lists[r2])))
                sset.add(_make_pair_key(r1, r2, num_nodes))

    t5 = time.time()
    if verbose:
        print(f"TCA: row-level clustering time = {t5 - t4:.2f}s")

    # =========================================================================
    # Step 4: 构建初始簇
    # =========================================================================
    clusters = {}
    for i in range(num_nodes):
        ro = _root(i, cluster_id)
        if ro in clusters:
            clusters[ro].append(i)
        else:
            clusters[ro] = [i]
    cluster_keys = list(clusters.keys())
    cluster_num = len(cluster_keys)
    if verbose:
        print(f"TCA: row-level clusters = {cluster_num}")

    # =========================================================================
    # Step 5: 两级聚类 Level 2 — 簇级聚类
    # =========================================================================
    cluster_thres_val = 128
    per_c = 128
    lsh_c = MinHashLSH(threshold=0.2, num_perm=per_c)
    allver_c = []
    lists_c = [[] for _ in range(cluster_num)]

    for cnt, key in enumerate(cluster_keys):
        m = MinHash(num_perm=per_c)
        list_cluster_i = []
        for node in clusters[key]:
            list_cluster_i.extend(row_lists[node])
        list_cluster_i = list(set(list_cluster_i))
        lists_c[cnt] = list_cluster_i
        for idx in list_cluster_i:
            m.update(str(idx).encode('utf-8'))
        lsh_c.insert(str(cnt), m)
        allver_c.append(m)

    que_c = Q.PriorityQueue()
    sset_c = set()

    def _make_pair_key_c(a, b):
        if a > b:
            a, b = b, a
        return a * cluster_num + b

    for i in range(cluster_num):
        if len(lists_c[i]) == 0:
            continue
        res = lsh_c.query(allver_c[i])
        for item in res:
            item_i = int(item)
            if item_i == i or _make_pair_key_c(i, item_i) in sset_c:
                continue
            if len(lists_c[item_i]) == 0:
                continue
            que_c.put(Pair(i, item_i, _jaccard_lists(lists_c[i], lists_c[item_i])))
            sset_c.add(_make_pair_key_c(i, item_i))

    if verbose:
        print(f"TCA: cluster-level queue size = {que_c.qsize()}")

    cluster_id_c = list(range(cluster_num))
    cluster_sz_c = [1] * cluster_num
    deleted_c = [0] * cluster_num
    num_cluster_c = cluster_num

    def _root_c(i):
        while i != cluster_id_c[i]:
            cluster_id_c[i] = cluster_id_c[cluster_id_c[i]]
            i = cluster_id_c[i]
        return i

    while (not que_c.empty()) and num_cluster_c > 0:
        item = que_c.get()
        p1, p2 = item.p1, item.p2
        sset_c.discard(_make_pair_key_c(p1, p2))

        if p1 == cluster_id_c[p1] and p2 == cluster_id_c[p2]:
            if deleted_c[p1] or deleted_c[p2]:
                continue
            if cluster_sz_c[p1] < cluster_sz_c[p2]:
                cluster_id_c[p1] = p2
                num_cluster_c -= 1
                cluster_sz_c[p2] += cluster_sz_c[p1]
                if cluster_sz_c[p2] >= cluster_thres_val:
                    deleted_c[p2] = 1
                    num_cluster_c -= 1
            else:
                cluster_id_c[p2] = p1
                num_cluster_c -= 1
                cluster_sz_c[p1] += cluster_sz_c[p2]
                if cluster_sz_c[p1] >= cluster_thres_val:
                    deleted_c[p1] = 1
                    num_cluster_c -= 1
        else:
            r1 = _root_c(p1)
            r2 = _root_c(p2)
            if deleted_c[r1] or deleted_c[r2]:
                continue
            if r1 != r2 and _make_pair_key_c(r1, r2) not in sset_c:
                que_c.put(Pair(r1, r2, _jaccard_lists(lists_c[r1], lists_c[r2])))
                sset_c.add(_make_pair_key_c(r1, r2))

    # =========================================================================
    # Step 6: 构建最终的重排映射
    # =========================================================================
    clusters_c = {}
    for i in range(cluster_num):
        ro = _root_c(i)
        if ro in clusters_c:
            clusters_c[ro].append(i)
        else:
            clusters_c[ro] = [i]

    reorder_map = []
    for j in clusters_c:
        for k in clusters_c[j]:
            for item in clusters[cluster_keys[k]]:
                reorder_map.append(item)

    all_reordered = set(reorder_map)
    for i in range(num_nodes):
        if i not in all_reordered:
            reorder_map.append(i)

    assert len(reorder_map) == num_nodes, \
        f"reorder_map size {len(reorder_map)} != {num_nodes}"

    # =========================================================================
    # Step 7: 应用重排 (仅重排行, 列索引保持不变), 重建 CSR
    # =========================================================================
    old_to_new = {old: new for new, old in enumerate(reorder_map)}

    new_src_all = np.zeros(num_nnz, dtype=np.int32)
    new_dst_all = np.zeros(num_nnz, dtype=np.int32)

    edge_idx = 0
    for i in range(num_nodes):
        for j in range(row_ptr[i], row_ptr[i + 1]):
            new_src_all[edge_idx] = old_to_new[i]
            new_dst_all[edge_idx] = col_ind[j]  # 列索引保持不变
            edge_idx += 1

    sort_idx = np.argsort(new_src_all)
    new_src_sorted = new_src_all[sort_idx]
    new_dst_sorted = new_dst_all[sort_idx]

    new_row_ptr = np.zeros(num_nodes + 1, dtype=np.int32)
    new_col_ind = np.zeros(num_nnz, dtype=np.int32)

    for i in range(num_nnz):
        new_col_ind[i] = new_dst_sorted[i]

    cur_row = 0
    for i in range(num_nnz):
        while cur_row < new_src_sorted[i]:
            new_row_ptr[cur_row + 1] = new_row_ptr[cur_row]
            cur_row += 1
        new_row_ptr[cur_row + 1] = i + 1
    while cur_row < num_nodes:
        new_row_ptr[cur_row + 1] = new_row_ptr[cur_row]
        cur_row += 1

    t_end = time.time()
    if verbose:
        print(f"TCA: total reorder time = {t_end - t_start:.2f}s")

    reorder_id = np.array(reorder_map, dtype=np.int32)
    return new_row_ptr.astype(np.int32), new_col_ind.astype(np.int32), reorder_id
