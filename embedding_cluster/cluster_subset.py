# -*- coding: utf-8 -*-
"""
PCQM4Mv2 子集蒸馏：embedding -> (可选 PCA) -> UMAP/降维 -> HDBSCAN -> 选代表点

后端自适应:
    --backend cuml    : RAPIDS cuML (GPU), 服务器用; 大数据量走 采样拟合 + approximate_predict
    --backend sklearn : sklearn.cluster.HDBSCAN (CPU), 本地验证用; 全量直接拟合
    --backend auto    : 优先 cuml, 不可用则 sklearn (默认)
降维:
    cuml 后端  -> cuML UMAP(15维)
    sklearn    -> 若装了 umap-learn 用 UMAP(15维), 否则 PCA 到 15 维

输入:
    --emb   : embedding .npy, shape (N, D) 维度任意; 多个文件用逗号分隔
    --meta  : 与 embedding 行对齐的 CSV (含 smiles 列), 用于回溯每个分子
输出 (每个 emb 一套):
    <name>_subset.csv      : 选中的行 + cluster / role / strength
    <name>_labels_all.npy  : 全库簇标签
    <name>_run_config.json : 超参与统计, 保证可复现
    <name>_picked_idx.npy  : 选中行号 (下游按行号切原始数据用)
    <name>_umap_all.npy    : 全库低维坐标 (N,15) float32, 供体检/画图

选点目标 (--quota-power / --within / --noise-order):
    默认 = 按全库分布做分层随机抽样 (正比配额 + 簇内随机 + 噪声层按实测噪声率)。
    适用于"子集预训练 -> 在全量上推理复现全量效果": 判据是分布匹配, 实测该配置
    的能量距离与随机基线同量级, 而 sqrt 配额 + FPS 会让"全库到子集的最近邻距离"
    比随机抽样差 54%~120%。追求稀有化学空间覆盖时才切到 --quota-power 0.5 --within fps。
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 0. 后端选择
# ----------------------------------------------------------------------
def resolve_backend(requested: str) -> str:
    if requested in ("cuml", "auto"):
        try:
            import cupy  # noqa: F401
            import cuml  # noqa: F401
            return "cuml"
        except ImportError:
            if requested == "cuml":
                raise
    return "sklearn"


# ----------------------------------------------------------------------
# 1. 数据加载 (流式: 绝不把整份 float32 拷进内存/显存)
# ----------------------------------------------------------------------
ROW_CHUNK = 200_000        # 分块行数; 2048 维 float32 一块约 1.6 GB


def load_embeddings(path: str) -> np.ndarray:
    """-> 原始 (N, D) 数组, 只读, 大文件走 mmap。L2 归一化推迟到消费处。

    不能像以前那样 `np.load(path).astype(float32)` 再除范数: 全量 Morgan 是
    (3.45M, 2048) uint8 = 6.9 GB, astype 出 28 GB, `X / norms` 再复制一份
    28 GB, 峰值 ~63 GB, 在 62 GB 内存的机器上必炸。
    """
    return np.load(path, mmap_mode="r")


def _norm_rows(Xc: np.ndarray) -> np.ndarray:
    """就地 L2 归一化一个 float32 块 (欧氏距离 ~ 余弦距离)。"""
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Xc /= n
    return Xc


def l2_normalize(X, chunk: int = ROW_CHUNK) -> np.ndarray:
    """分块归一化 -> 新的 (N, D) float32 数组。仅在跳过 PCA 时才会调用。"""
    out = np.empty(X.shape, dtype=np.float32)
    for a in range(0, len(X), chunk):
        b = min(a + chunk, len(X))
        out[a:b] = _norm_rows(np.asarray(X[a:b], dtype=np.float32))
    return out


# ----------------------------------------------------------------------
# 2. 可选 PCA 预降维 (确定性 Gram-PCA, cuml / sklearn 共用一套实现)
# ----------------------------------------------------------------------
class GramPCA:
    """走 D×D 协方差特征分解的分块 PCA, fit / transform 都不需要整份数据驻留。

    为什么不用 cuml.PCA 或 sklearn.PCA:
      * cuml 26.10 的 `PCA.__init__` **没有** `random_state` 参数, 老代码传了
        就直接 TypeError。这个 bug 一直没暴露, 是因为交接文档里唯一那次 cuml
        验证用的是 20k×64 维输入, 64 <= pca_dim(100), PCA 整段被跳过了。
      * 就算不传该参数也不行: cuml.PCA 要 `cp.asarray(X)` 把整份 float32 送上
        卡, 全量 Morgan 28 GB > 24 GB 显存。
    协方差只有 D×D (2048² float64 = 33 MB), 分块累加即可; 对称特征分解没有
    随机性, 所以同输入必得同输出, 天然可复现, 不需要 seed。
    """

    def __init__(self, n_components: int, backend: str = "cuml",
                 chunk: int = ROW_CHUNK, normalize: bool = True):
        self.n_components = int(n_components)
        self.backend = backend
        self.chunk = int(chunk)
        self.normalize = normalize
        self.mean_ = None
        self.components_ = None            # (k, D) float32
        self.explained_variance_ = None

    def _xp(self):
        if self.backend == "cuml":
            import cupy as cp
            return cp
        return np

    def fit(self, X):
        xp = self._xp()
        n, d = X.shape
        k = int(min(self.n_components, d))
        total = xp.zeros(d, dtype=xp.float64)
        gram = xp.zeros((d, d), dtype=xp.float64)
        for a in range(0, n, self.chunk):
            b = min(a + self.chunk, n)
            Xc = np.asarray(X[a:b], dtype=np.float32)
            if self.normalize:
                _norm_rows(Xc)
            Xg = xp.asarray(Xc)
            total += Xg.sum(axis=0, dtype=xp.float64)
            # float32 矩阵乘 (走 tensor core) + float64 累加, 精度足够定位主轴
            gram += (Xg.T @ Xg).astype(xp.float64, copy=False)
            del Xc, Xg
        mean = (total / n).astype(xp.float64)
        cov = gram / n - xp.outer(mean, mean)
        del gram, total
        w, V = xp.linalg.eigh(cov)                     # 升序, 确定性
        pick = xp.argsort(w)[::-1][:k]
        self.mean_ = xp.asnumpy(mean).astype(np.float32)
        self.components_ = np.ascontiguousarray(
            xp.asnumpy(V[:, pick].T), dtype=np.float32)
        self.explained_variance_ = np.clip(xp.asnumpy(w[pick]), 0, None)
        ev = float(self.explained_variance_.sum())
        print(f"[pca] Gram-PCA {d} -> {k} dims, top-{k} 方差占比 "
              f"{100 * ev / max(float(xp.asnumpy(cov).trace()), 1e-12):.2f}%")
        return self

    def transform(self, X, chunk: int = None) -> np.ndarray:
        """分块投影 -> (N, k) float32。输入若是原始 embedding 会先归一化。"""
        chunk = int(chunk or self.chunk)
        C = self.components_
        out = np.empty((len(X), C.shape[0]), dtype=np.float32)
        for a in range(0, len(X), chunk):
            b = min(a + chunk, len(X))
            Xc = np.asarray(X[a:b], dtype=np.float32)
            if self.normalize:
                _norm_rows(Xc)
            out[a:b] = (Xc - self.mean_) @ C.T
        return out

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


def maybe_pca(X, pca_dim: int, seed: int, backend: str):
    """维度高于 pca_dim 时先 PCA。返回 (X_used, pca_model 或 None)。

    pca_model 只用于对 *原始维度* 数据做 transform; seed 参数保留是为了兼容
    旧调用签名, GramPCA 本身是确定性的, 用不到它。
    """
    if pca_dim <= 0 or X.shape[1] <= pca_dim:
        # 不降维: 这里必须真的把归一化结果物化出来, 后面要反复按下标取行
        return l2_normalize(X), None
    pca = GramPCA(pca_dim, backend=backend).fit(X)
    return pca.transform(X), pca


# ----------------------------------------------------------------------
# 3. 降维器 (UMAP 或 PCA-15)
# ----------------------------------------------------------------------
class Reducer:
    """统一 fit / transform 接口, 屏蔽 cuml-UMAP / umap-learn / sklearn-PCA 差异"""

    def __init__(self, backend: str, seed: int, n_components: int = 15,
                 n_neighbors: int = 50):
        self.backend = backend
        self.seed = seed
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.kind = None
        self.model = None

    def fit(self, X):
        d = min(self.n_components, X.shape[1])
        if self.backend == "cuml":
            import cupy as cp
            from cuml import UMAP
            self.kind = "cuml-umap"
            self.model = UMAP(n_neighbors=self.n_neighbors, min_dist=0.0,
                              n_components=d, metric="euclidean",
                              random_state=self.seed, verbose=True)
            self.model.fit(cp.asarray(X))
            self.embedding_ = cp.asnumpy(self.model.embedding_).astype(np.float32)
            return self
        try:
            import umap  # umap-learn
            self.kind = "umap-learn"
            self.model = umap.UMAP(n_neighbors=self.n_neighbors, min_dist=0.0,
                                   n_components=d, metric="euclidean",
                                   random_state=self.seed)
            self.embedding_ = self.model.fit_transform(X).astype(np.float32)
            return self
        except ImportError:
            from sklearn.decomposition import PCA
            self.kind = "sklearn-pca"
            self.model = PCA(n_components=d, random_state=self.seed)
            self.embedding_ = self.model.fit_transform(X).astype(np.float32)
            return self

    def transform(self, X):
        if self.backend == "cuml":
            import cupy as cp
            return cp.asnumpy(self.model.transform(cp.asarray(X))).astype(np.float32)
        return self.model.transform(X).astype(np.float32)


# ----------------------------------------------------------------------
# 4. HDBSCAN
# ----------------------------------------------------------------------
class Clusterer:
    """统一 fit / predict 接口。labels: -1 = 噪声; strength: 越大越典型"""

    def __init__(self, backend: str, min_cluster_size: int, min_samples: int,
                 seed: int):
        self.backend = backend
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.seed = seed
        self.model = None
        self.labels_ = None
        self.strength_ = None

    def fit(self, emb: np.ndarray):
        if self.backend == "cuml":
            import cupy as cp
            from cuml.cluster import HDBSCAN
            self.model = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
                gen_min_span_tree=False,
                prediction_data=True,               # approximate_predict 依赖
                verbose=True,
            )
            self.model.fit(cp.asarray(emb))
            self.labels_ = cp.asnumpy(self.model.labels_).astype(np.int32)
            # cuml: probabilities_ 越大越典型
            self.strength_ = cp.asnumpy(self.model.probabilities_).astype(np.float32)
        else:
            from sklearn.cluster import HDBSCAN
            self.model = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
                copy=True,
            )
            self.model.fit_predict(emb)
            self.labels_ = self.model.labels_.astype(np.int32)
            # sklearn probabilities_: 越大越典型, 与 cuml 语义一致
            if hasattr(self.model, "probabilities_"):
                self.strength_ = self.model.probabilities_.astype(np.float32)
            elif hasattr(self.model, "outliers_scores_"):
                self.strength_ = self._calibrate_strength(
                    self.model.outliers_scores_.astype(np.float32))
            else:
                self.strength_ = np.ones(len(self.labels_), dtype=np.float32)
        return self

    def _calibrate_strength(self, scores: np.ndarray) -> np.ndarray:
        """sklearn 的 outliers_scores_ 方向随版本/实现而异, 用噪声点均值自动定向:
        保证 strength 越大越典型 (与 cuml probabilities_ 语义一致)"""
        noise = scores[self.labels_ == -1]
        inlier = scores[self.labels_ >= 0]
        lo, hi = float(scores.min()), float(scores.max())
        rng = max(hi - lo, 1e-12)
        higher_is_anomalous = (len(noise) == 0 or len(inlier) == 0
                               or noise.mean() > inlier.mean())
        if higher_is_anomalous:
            return ((hi - scores) / rng).astype(np.float32)
        return ((scores - lo) / rng).astype(np.float32)

    def predict(self, emb: np.ndarray):
        """把未参与拟合的样本归簇, 返回 (labels, strength)"""
        if self.backend == "cuml":
            import cupy as cp
            from cuml.cluster.hdbscan import approximate_predict
            lab, prob = approximate_predict(self.model, cp.asarray(emb))
            return cp.asnumpy(lab).astype(np.int32), cp.asnumpy(prob).astype(np.float32)
        # sklearn 无 approximate_predict: 退化为最近簇质心归属
        centers = {}
        for c in np.unique(self.labels_[self.labels_ >= 0]):
            centers[c] = self.emb_fit_[self.labels_ == c].mean(0)
        ids = np.array(list(centers.keys()))
        C = np.stack([centers[c] for c in ids])
        d = ((emb[:, None, :] - C[None, :, :]) ** 2).sum(-1) if emb.shape[0] <= 50_000 \
            else self._chunked_dist(emb, C)
        nearest = d.argmin(1)
        return ids[nearest].astype(np.int32), np.zeros(len(emb), dtype=np.float32)

    def _chunked_dist(self, emb, C, chunk=20_000):
        out = np.empty((len(emb), len(C)), dtype=np.float32)
        for s in range(0, len(emb), chunk):
            e = emb[s:s + chunk]
            out[s:s + chunk] = ((e ** 2).sum(1)[:, None]
                                - 2 * e @ C.T + (C ** 2).sum(1)[None, :])
        return out


# ----------------------------------------------------------------------
# 5. 配额分配 (power 可调, 严格对齐 budget)
# ----------------------------------------------------------------------
def cluster_quotas(counts: np.ndarray, budget: int, power: float = 1.0) -> np.ndarray:
    """按 counts**power 分配配额, 小簇保底 1, 总和严格 == min(budget, counts.sum())。

    power=1.0 (正比): 每簇采样率一致 -> 子集分布 == 全库分布。目标是"子集预训练
        后在全量上推理复现全量效果"时, 这才是正确选择。
    power=0.5 (sqrt): 抬高稀有簇占比, **刻意**偏离全库分布, 只在追求稀有化学空间
        覆盖 / OOD 鲁棒时才该用。morgan 对照组实测: 相比正比配额, 全库到子集的
        最近邻距离差 16%, 能量距离差两个数量级 (0.319 vs 0.0012, 随机基线 0.0005)。
    """
    budget = int(min(budget, counts.sum()))
    w = np.power(counts.astype(np.float64), float(power))
    raw = budget * w / w.sum()
    q = np.floor(raw).astype(np.int64)
    q = np.clip(q, 1, counts)                 # 保底 1, 不超过簇容量
    # 余量按小数部分从大到小补齐
    rem = budget - int(q.sum())
    if rem > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        i = 0
        while rem > 0 and i < 10 * len(order):
            j = order[i % len(order)]
            if q[j] < counts[j]:
                q[j] += 1
                rem -= 1
            i += 1
    elif rem < 0:                             # 保底 1 导致超发: 从大簇回收
        order = np.argsort(-counts)
        i = 0
        while rem < 0 and i < 10 * len(order):
            j = order[i % len(order)]
            if q[j] > 1:
                q[j] -= 1
                rem += 1
            i += 1
    return q


# ----------------------------------------------------------------------
# 6. 典型性分数 (替代退化的 probabilities_) 与簇内选点策略
# ----------------------------------------------------------------------
def cluster_centroids(V_all, uniq, members) -> np.ndarray:
    C = np.empty((len(uniq), V_all.shape[1]), dtype=np.float32)
    for i, c in enumerate(uniq):
        C[i] = np.asarray(V_all[members[int(c)]], dtype=np.float32).mean(0)
    return C


def typicality_scores(V_all, labels, uniq, members, need_noise_dist=False):
    """-> (N,) float32, **越大越典型**。用来替掉 cuml HDBSCAN 的 probabilities_。

    为什么必须替: 实测 cuml 26.10 的 probabilities_ 在全库上只有 2 个取值
    (噪声=0, 簇内=1.0, 占 95%)。于是旧代码里两处依赖 strength 排序的逻辑全部
    退化成"按行号取前 M 个"——簇内 top-M 候选池形同虚设, "挑最离群的噪声点"
    也只是取前 250 行。设计意图整段落空, 而且 FPS 因此是在一个任意子样本上
    挑外围点, 实测比随机还差。

    改用 UMAP 空间到所属簇质心的距离: 簇内越小越典型; 噪声点(可选)用到最近簇
    质心的距离, 越远越离群。统一映射到 [0,1]。
    need_noise_dist=False 时跳过噪声的最近质心搜索 —— 默认策略下噪声是随机抽的,
    用不到这个分数, 全量时能省掉一次 噪声数×簇数 的距离计算。
    """
    N = len(labels)
    dist = np.full(N, np.inf, dtype=np.float32)
    if len(uniq) == 0:
        return np.zeros(N, dtype=np.float32)
    C = cluster_centroids(V_all, uniq, members)
    for i, c in enumerate(uniq):
        idx = members[int(c)]
        X = np.asarray(V_all[idx], dtype=np.float32)
        dist[idx] = np.linalg.norm(X - C[i], axis=1)
    noise_idx = np.nonzero(labels == -1)[0]
    if need_noise_dist and len(noise_idx) and len(C):
        c2 = (C ** 2).sum(1)
        for a in range(0, len(noise_idx), 4096):
            idx = noise_idx[a:a + 4096]
            X = np.asarray(V_all[idx], dtype=np.float32)
            dd = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + c2[None, :]
            dist[idx] = np.sqrt(np.maximum(dd.min(1), 0))
    fin = dist[np.isfinite(dist)]
    if len(fin) == 0:
        return np.zeros(N, dtype=np.float32)
    hi, lo = float(fin.max()), float(fin.min())
    out = np.zeros(N, dtype=np.float32)
    m = np.isfinite(dist)
    out[m] = (hi - dist[m]) / max(hi - lo, 1e-12)   # 距离越小 -> 分数越高
    return out


def pick_medoids(V_all: np.ndarray, idx: np.ndarray, strengths: np.ndarray,
                 k: int, top_m: int = 100) -> np.ndarray:
    k = int(min(k, len(idx)))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    pool = int(max(top_m, k))                  # 配额超过 top_m 时放大候选池
    order = np.argsort(-strengths[idx])[:min(pool, len(idx))]
    cand = idx[order]
    V = V_all[cand].astype(np.float32)
    D = np.sqrt(np.maximum(
        (V ** 2).sum(1)[:, None] - 2 * V @ V.T + (V ** 2).sum(1)[None, :], 0))
    first = int(D.sum(1).argmin())               # medoid: 到候选集总距离最小
    chosen = [first]
    dmin = D[:, first].copy()
    while len(chosen) < k:
        dmin[chosen] = -1.0
        nxt = int(dmin.argmax())
        chosen.append(nxt)
        dmin = np.minimum(dmin, D[:, nxt])       # farthest-point: 离已选最远
    return cand[chosen]


def pick_within(V_all, idx, strengths, k, mode="random", rng=None, top_m=100):
    """簇内选 k 个点。mode:
      random : 均匀随机 -> 保持簇内密度分布, 分布匹配最优 (默认)
      medoid : 取典型性最高的 k 个 -> 全挤在簇心, 实测最差
      fps    : strength top-M 候选池 + 最远点采样 -> 追求分散, 分布匹配差
    """
    k = int(min(k, len(idx)))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if mode == "fps":
        return pick_medoids(V_all, idx, strengths, k, top_m)
    if mode == "medoid":
        return idx[np.argsort(-strengths[idx])[:k]]
    r = rng if rng is not None else np.random.default_rng(0)
    return r.choice(idx, size=k, replace=False)


def cluster_members(labels: np.ndarray, uniq: np.ndarray) -> dict:
    """{簇号: 该簇所有行下标}。一次排序 + searchsorted, O(N log N)。

    旧实现在选点循环里对每个簇做一次 `np.where(labels == c)`, 复杂度是
    O(簇数 × N): 全量 3.45M 行、上万簇时就是上百亿次比较, 光选点能跑几小时。
    """
    inl = np.nonzero(labels >= 0)[0]
    if len(inl) == 0 or len(uniq) == 0:
        return {}
    lab = labels[inl]
    order = np.argsort(lab, kind="stable")
    rows = inl[order]
    lab_sorted = lab[order]
    lo = np.searchsorted(lab_sorted, uniq, side="left")
    hi = np.searchsorted(lab_sorted, uniq, side="right")
    return {int(c): rows[a:b] for c, a, b in zip(uniq, lo, hi)}


def select_subset(V_all: np.ndarray, labels: np.ndarray, strengths: np.ndarray,
                  target: int, noise_frac: float = -1.0, top_m: int = 100,
                  power: float = 1.0, within: str = "random",
                  noise_order: str = "random", rng=None):
    """返回 (picked_indices, meta[(cluster, role)])，len(picked) <= min(target, N)

    role 只表示来源层: "cluster"(来自某个簇) / "noise"(噪声层), 与簇内算法无关。

    默认参数 = **按全库分布做分层随机抽样**: 正比配额 + 簇内随机 + 噪声层按实测
    噪声率给名额再随机抽。这是"子集预训练 -> 全量推理复现全量效果"这个目标下
    分布匹配的最优解, 同时比 iid 随机多一层保证: 每个簇都必然有代表点。

    noise_frac < 0 表示自动取实测噪声率。写死 0.05 两头都错: molformer 噪声
    22.6% 会被低配 4.5 倍, morgan 噪声 0.6% 反而被高配 8 倍。
    """
    N = len(labels)
    target = int(min(target, N))
    rng = rng if rng is not None else np.random.default_rng(0)
    uniq = np.unique(labels[labels >= 0])
    counts = np.bincount(labels[labels >= 0], minlength=int(uniq.max()) + 1)[uniq] \
        if len(uniq) else np.empty(0, dtype=np.int64)
    if noise_frac < 0:
        noise_frac = float((labels == -1).sum()) / max(N, 1)
    noise_frac = float(min(max(noise_frac, 0.0), 1.0))
    budget_c = int(round(target * (1 - noise_frac)))
    quotas = cluster_quotas(counts, budget_c, power) if len(uniq) \
        else np.empty(0, dtype=np.int64)
    members = cluster_members(labels, uniq)

    taken = np.zeros(N, dtype=bool)
    picked, meta = [], []

    def add(indices, cluster_id: int, role: str, limit: int = None) -> int:
        """按下标顺序写入未选过的点, 返回实际新增数"""
        n_new = 0
        for s in indices:
            if limit is not None and n_new >= limit:
                break
            s = int(s)
            if taken[s]:
                continue
            taken[s] = True
            picked.append(s)
            meta.append((cluster_id, role))
            n_new += 1
        return n_new

    for c, q in zip(uniq, quotas):
        idx = members[int(c)]
        sel = pick_within(V_all, idx, strengths, int(q), within, rng, top_m)
        add(sel, int(c), "cluster")

    # 噪声层: random = 均匀抽(匹配全库分布); outlier = 最离群优先(覆盖边界样本)
    n_noise = target - len(picked)
    noise_idx = np.nonzero(labels == -1)[0]
    if n_noise > 0 and len(noise_idx) > 0:
        take = min(n_noise, len(noise_idx))
        if noise_order == "outlier":
            sel_noise = noise_idx[np.argsort(strengths[noise_idx])[:take]]
        else:
            sel_noise = rng.choice(noise_idx, size=take, replace=False)
        add(sel_noise, -1, "noise")

    # 仍有缺口 (噪声不够) -> 大簇优先逐个补足; 一整轮无新增即认为容量耗尽
    short = target - len(picked)
    if short > 0 and len(uniq):
        used = {}
        for c, _ in meta:
            used[c] = used.get(c, 0) + 1
        order = np.argsort(-counts)              # 大簇优先
        ncl, fails, i = len(order), 0, 0
        while short > 0 and fails <= ncl:
            c = int(uniq[order[i % ncl]])
            i += 1
            idx = members[c]
            k = used.get(c, 0) + 1
            if k > len(idx):
                fails += 1
                continue
            sel = pick_within(V_all, idx, strengths, k, within, rng, top_m)
            n_new = add(sel, c, "cluster", limit=short)
            if n_new == 0:
                fails += 1
                continue
            fails = 0
            used[c] = used.get(c, 0) + n_new
            short -= n_new
        if short > 0:
            print(f"[select] 警告: 簇内容量耗尽, 实得 {len(picked)} < target {target}")

    return np.array(picked, dtype=np.int64), meta


# ----------------------------------------------------------------------
# 7. 单模型完整流程
# ----------------------------------------------------------------------
def run_one(emb_path: str, meta: pd.DataFrame, args, rng) -> dict:
    name = os.path.splitext(os.path.basename(emb_path))[0]
    backend = args.backend
    X_raw = load_embeddings(emb_path)            # 原始维度, 只读不改
    assert len(meta) == len(X_raw), f"{name}: emb 与 meta 行数不一致"
    N, dim = X_raw.shape
    print(f"\n===== [{name}] N={N}, dim={dim}, backend={backend} =====")

    X_used, pca = maybe_pca(X_raw, args.pca_dim, args.seed, backend)
    if pca is not None:
        print(f"[pca] {dim} -> {X_used.shape[1]}")
    del X_raw                                    # 后面只用 X_used, 尽早放掉 mmap 句柄

    # cuml 大数据量: 采样拟合 + 其余 approximate_predict; sklearn: 全量拟合
    use_sampling = (backend == "cuml" and args.fit_n < N)
    if use_sampling:
        fit_idx = np.sort(rng.choice(N, size=args.fit_n, replace=False))
        rest_idx = np.setdiff1d(np.arange(N), fit_idx)
    else:
        if backend == "sklearn" and args.fit_n < N:
            print(f"[warn] sklearn 后端不支持增量归簇, 忽略 --fit-n 用全量 {N} 拟合")
        fit_idx = np.arange(N)
        rest_idx = np.empty(0, dtype=np.int64)

    print(f"[reduce] fit on {len(fit_idx)} ...")
    reducer = Reducer(backend, args.seed, n_components=args.n_components,
                      n_neighbors=args.n_neighbors).fit(X_used[fit_idx])
    emb_fit = reducer.embedding_
    print(f"[reduce] {reducer.kind}: {X_used.shape[1]} -> {emb_fit.shape[1]}")

    print("[hdbscan] fit ...")
    clus = Clusterer(backend, args.min_cluster_size, args.min_samples, args.seed)
    clus.fit(emb_fit)
    clus.emb_fit_ = emb_fit                      # sklearn 增量归属需要

    labels = np.full(N, -1, dtype=np.int32)
    strength = np.zeros(N, dtype=np.float32)
    labels[fit_idx] = clus.labels_
    strength[fit_idx] = clus.strength_

    # 全库低维坐标 (medoid 距离用), 分批 transform 控显存/内存
    V_all = np.empty((N, emb_fit.shape[1]), dtype=np.float32)
    V_all[fit_idx] = emb_fit
    chunk = args.chunk
    for s in range(0, len(rest_idx), chunk):
        idx = rest_idx[s:s + chunk]
        # 注意: transform 的输入必须是 X_used (PCA 后), 不能再套一次 pca
        Xi = X_used[idx]
        V_all[idx] = reducer.transform(Xi)
        if backend == "cuml":
            lab, st = clus.predict(V_all[idx])
            labels[idx] = lab
            strength[idx] = st

    n_clusters = len(np.unique(labels[labels >= 0]))
    noise_rate = float((labels == -1).mean())
    print(f"[hdbscan] clusters={n_clusters}, noise={noise_rate:.2%}")

    if use_sampling:
        # 采样拟合的前提: approximate_predict 的簇号必须和 fit 的编号一致。
        # 一致率过低说明两批标签根本不在同一套编号下, 混在一起会产出垃圾子集,
        # 这里宁可当场报错也不要静默跑完全量。
        probe = fit_idx[:min(20_000, len(fit_idx))]
        plab, _ = clus.predict(V_all[probe])
        agree = float((plab == labels[probe]).mean())
        print(f"[check] approximate_predict vs fit 标签一致率 = {agree:.4f} "
              f"(抽样 {len(probe)})")
        if agree < 0.90:
            raise RuntimeError(
                f"approximate_predict 簇号与 fit 不一致 (agree={agree:.3f}); "
                f"全库标签不可信, 请改用 --fit-n {N} 全量拟合")

    base = os.path.join(args.outdir, name)
    np.save(base + "_labels_all.npy", labels)

    # cuml 的 probabilities_ 实测退化(全库只有 2 个取值), 直接用会让簇内 top-M
    # 候选池和"最离群噪声"两处排序全部失效。这里用 UMAP 空间到簇质心的距离重建
    # 一个真正的典型性分数, 覆盖掉它。
    n_raw_uniq = len(np.unique(np.round(strength, 6)))
    uniq_c = np.unique(labels[labels >= 0])
    memb_c = cluster_members(labels, uniq_c)
    strength = typicality_scores(V_all, labels, uniq_c, memb_c,
                                 need_noise_dist=(args.noise_order == "outlier"))
    print(f"[strength] 后端原始分数 unique={n_raw_uniq}"
          f"{' (退化, 已用质心距离重建)' if n_raw_uniq <= 8 else ''}"
          f" -> 典型性分数 median={float(np.median(strength)):.4f} "
          f"unique={len(np.unique(np.round(strength, 4)))}")

    print(f"[select] 选代表点 (target={args.target}) ...")
    picked, pmeta = select_subset(V_all, labels, strength, args.target,
                                  noise_frac=args.noise_frac, top_m=args.top_m,
                                  power=args.quota_power, within=args.within,
                                  noise_order=args.noise_order, rng=rng)

    # 落盘选中行号: subset.csv 只有 smiles, 下游按行号切原始数据/建训练集时
    # 得能精确回溯, 靠 smiles 反查既慢又怕重复。
    np.save(base + "_picked_idx.npy", picked)
    # 落盘全库低维坐标 (N,15) float32: 全量 3.45M 也才 207 MB。
    # 用来算子集覆盖度、画图、复核簇结构, 免得为了体检重跑一遍 UMAP。
    np.save(base + "_umap_all.npy", V_all)

    out = meta.iloc[picked].copy()
    out["cluster"] = [m[0] for m in pmeta]
    out["role"] = [m[1] for m in pmeta]
    out["strength"] = strength[picked]
    out.to_csv(base + "_subset.csv", index=False)

    cfg = dict(vars(args))                       # 独立快照, 避免多模型互相覆盖
    cfg.update({"backend": backend, "reducer": reducer.kind,
                "emb": emb_path, "N": int(N), "dim": int(dim),
                "n_clusters": int(n_clusters), "noise_rate": noise_rate,
                "picked": int(len(picked)),
                "n_noise": int(sum(1 for m in pmeta if m[1] == "noise")),
                "strength_backend_unique": int(n_raw_uniq)})
    with open(base + "_run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)
    print(f"[done] {name}: picked={len(picked)} -> {base}_subset.csv")
    return cfg


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True,
                    help="embedding .npy, (N,D) D 任意; 多个文件逗号分隔")
    ap.add_argument("--meta", required=True, help="CSV, 与 emb 行对齐, 含 smiles 列")
    ap.add_argument("--outdir", default="subsets")
    ap.add_argument("--backend", default="auto", choices=["auto", "cuml", "sklearn"])
    ap.add_argument("--target", type=int, default=50_000, help="目标子集大小")
    ap.add_argument("--fit-n", type=int, default=500_000,
                    help="cuml 后端: UMAP/HDBSCAN 拟合采样数")
    ap.add_argument("--chunk", type=int, default=200_000, help="分批 transform 大小")
    ap.add_argument("--pca-dim", type=int, default=100,
                    help="维度高于此值先 PCA; 0=关闭")
    ap.add_argument("--n-components", type=int, default=15, help="UMAP/降维目标维度")
    ap.add_argument("--n-neighbors", type=int, default=50, help="UMAP n_neighbors")
    ap.add_argument("--min-cluster-size", type=int, default=300)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--quota-power", type=float, default=1.0,
                    help="跨簇配额指数: 1.0=正比(匹配全库分布, 默认), 0.5=sqrt(抬高稀有簇)")
    ap.add_argument("--within", default="random", choices=["random", "medoid", "fps"],
                    help="簇内选点: random=保持密度(默认), medoid=取最典型, fps=最远点采样")
    ap.add_argument("--noise-order", default="random", choices=["random", "outlier"],
                    help="噪声层抽样: random=均匀(默认), outlier=最离群优先")
    ap.add_argument("--noise-frac", type=float, default=-1.0,
                    help="噪声层名额占比; <0 表示自动取实测噪声率(默认)")
    ap.add_argument("--top-m", type=int, default=100, help="--within fps 时的候选池")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.backend = resolve_backend(args.backend)
    os.makedirs(args.outdir, exist_ok=True)
    meta = pd.read_csv(args.meta)

    for emb_path in args.emb.split(","):
        # 每个模型用独立的 rng: 否则第二个模型的抽样结果会依赖第一个模型消耗了
        # 多少随机数, 单独重跑某一个模型就没法复现。
        run_one(emb_path.strip(), meta, args, np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
