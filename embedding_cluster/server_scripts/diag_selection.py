# -*- coding: utf-8 -*-
"""选点策略分解诊断: 为什么 medoid+FPS 的覆盖度反而不如随机?

QA 里量到 "全库到子集的 UMAP 最近邻距离" 当前策略是随机子集的 ~2 倍(更差)。
这个指标同时受两个旋钮影响, 必须拆开看才知道该改哪个:

  旋钮 A  跨簇配额  sqrt(簇大小) vs 正比(簇大小)
          sqrt 会**故意**抬高稀有簇的占比 -> 子集不再匹配全库密度分布。
  旋钮 B  簇内选点  FPS(最远点) vs 随机 vs medoid优先
          FPS 故意挑边界/外围点 -> 稠密核心反而没人代表。

2x2(+纯随机) 五种组合跑同一份 labels/坐标, 用同一批查询点比, 差异就归因清楚了。
另外顺手查 cuml HDBSCAN 的 probabilities_ 是不是退化成常数 —— 若几乎全是 1.0,
那 "strength top-M 候选池" 这层筛选等于没生效, 簇内选点就退化成按行号取前 M 个。
"""
import os

import numpy as np
import pandas as pd

EMB = "/mnt/data/yangsensen/embedding"
SUB = f"{EMB}/pilot_100k"


def quotas_sqrt(counts, budget):
    w = np.sqrt(counts.astype(np.float64))
    q = np.floor(budget * w / w.sum()).astype(np.int64)
    q = np.clip(q, 1, counts)
    rem = budget - int(q.sum())
    order = np.argsort(-(budget * w / w.sum() - np.floor(budget * w / w.sum())))
    i = 0
    while rem > 0 and i < 10 * len(order):
        j = order[i % len(order)]
        if q[j] < counts[j]:
            q[j] += 1
            rem -= 1
        i += 1
    while rem < 0 and i < 20 * len(order):
        j = order[i % len(order)]
        if q[j] > 1:
            q[j] -= 1
            rem += 1
        i += 1
    return q


def quotas_prop(counts, budget):
    q = np.floor(budget * counts / counts.sum()).astype(np.int64)
    q = np.clip(q, 0, counts)
    rem = budget - int(q.sum())
    order = np.argsort(-(budget * counts / counts.sum()))
    i = 0
    while rem > 0 and i < 20 * len(order):
        j = order[i % len(order)]
        if q[j] < counts[j]:
            q[j] += 1
            rem -= 1
        i += 1
    return q


def members_of(labels, uniq):
    inl = np.nonzero(labels >= 0)[0]
    lab = labels[inl]
    o = np.argsort(lab, kind="stable")
    rows, ls = inl[o], lab[o]
    lo = np.searchsorted(ls, uniq, "left")
    hi = np.searchsorted(ls, uniq, "right")
    return {int(c): rows[a:b] for c, a, b in zip(uniq, lo, hi)}


def fps(V, cand, k):
    """最远点采样 (与 cluster_subset.pick_medoids 同逻辑, 但候选池由外部给)"""
    k = int(min(k, len(cand)))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    X = V[cand].astype(np.float32)
    D = np.sqrt(np.maximum((X ** 2).sum(1)[:, None] - 2 * X @ X.T
                           + (X ** 2).sum(1)[None, :], 0))
    first = int(D.sum(1).argmin())
    chosen = [first]
    dmin = D[:, first].copy()
    while len(chosen) < k:
        dmin[chosen] = -1.0
        nxt = int(dmin.argmax())
        chosen.append(nxt)
        dmin = np.minimum(dmin, D[:, nxt])
    return cand[chosen]


def build(V, labels, strength, uniq, counts, memb, q, mode, rng, top_m=100):
    out = []
    for c, k in zip(uniq, q):
        k = int(k)
        if k <= 0:
            continue
        idx = memb[int(c)]
        if mode == "random":
            out.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
        elif mode == "fps":                       # 现行: strength top-M 池 + FPS
            pool = int(max(top_m, k))
            cand = idx[np.argsort(-strength[idx])[:min(pool, len(idx))]]
            out.append(fps(V, cand, k))
        elif mode == "fps_all":                   # 全簇成员做池 + FPS (无 strength 筛选)
            out.append(fps(V, idx, k))
        elif mode == "dense":                     # 反方向: 挑最靠簇中心的
            X = V[idx].astype(np.float32)
            cen = X.mean(0)
            d = ((X - cen) ** 2).sum(1)
            out.append(idx[np.argsort(d)[:k]])
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def nn_stats(V, q_idx, sel, xp):
    S = xp.asarray(np.asarray(V[sel], dtype=np.float32))
    s2 = (S ** 2).sum(1)
    chunk = int(max(256, min(8192, 4e7 // max(len(sel), 1))))
    d = np.empty(len(q_idx), dtype=np.float32)
    for a in range(0, len(q_idx), chunk):
        b = min(a + chunk, len(q_idx))
        Q = xp.asarray(np.asarray(V[q_idx[a:b]], dtype=np.float32))
        dd = (Q ** 2).sum(1)[:, None] - 2 * (Q @ S.T) + s2[None, :]
        d[a:b] = xp.asnumpy(dd.min(axis=1))
    d = np.sqrt(np.maximum(d, 0))
    p = np.percentile(d, [50, 95])
    return float(d.mean()), float(p[0]), float(p[1])


def energy_distance(V, a_idx, b_idx, xp, cap=4000):
    """两样本 energy distance: 越小说明两个点集分布越像。
    随机子集 vs 随机子集 = 基线(≈采样噪声), 用它当尺子。"""
    rng = np.random.default_rng(0)
    A = xp.asarray(np.asarray(V[rng.choice(a_idx, size=min(cap, len(a_idx)), replace=False)],
                            dtype=np.float32))
    B = xp.asarray(np.asarray(V[rng.choice(b_idx, size=min(cap, len(b_idx)), replace=False)],
                            dtype=np.float32))

    def pd_(X, Y):
        return xp.sqrt(xp.maximum((X ** 2).sum(1)[:, None] - 2 * X @ Y.T
                                  + (Y ** 2).sum(1)[None, :], 0))
    dab = float(pd_(A, B).mean())
    daa = float(pd_(A, A).mean())
    dbb = float(pd_(B, B).mean())
    return 2 * dab - daa - dbb


def main():
    import cupy as xp
    for name, stem in [("molformer", "molformer_emb_merged"),
                       ("chemberta2", "chemberta2_emb_merged"),
                       ("morgan", "morgan_emb")]:
        print(f"\n{'=' * 74}\n[{name}]\n{'=' * 74}")
        V = np.load(f"{SUB}/{name}/{stem}_umap_all.npy", mmap_mode="r")
        labels = np.load(f"{SUB}/{name}/{stem}_labels_all.npy")
        picked = np.load(f"{SUB}/{name}/{stem}_picked_idx.npy")
        sub = pd.read_csv(f"{SUB}/{name}/{stem}_subset.csv")
        N = len(labels)
        target = len(picked)
        rng = np.random.default_rng(42)

        # --- probabilities_ 是否退化
        s = sub["strength"].to_numpy()
        print(f"子集 strength: ==1.0 占 {100*float((s>=0.999).mean()):.1f}%, "
              f"unique={len(np.unique(np.round(s,4)))}, median={np.median(s):.4f}")

        inl = labels[labels >= 0]
        uniq, counts = np.unique(inl, return_counts=True)
        memb = members_of(labels, uniq)
        budget = int(round(target * 0.95))
        q_idx = np.sort(rng.choice(N, size=min(20000, N), replace=False))
        rnd_ref = np.sort(rng.choice(N, size=target, replace=False))

        # 全库 strength 没有落盘(只有子集那份), 这里用常数代替 —— 若 cuml 的
        # probabilities_ 确实退化成 ~1.0, 常数池与真实池行为等价, 正好复现现场。
        flat_strength = np.ones(N, np.float32)
        variants = {}
        for qname, qf in [("sqrt配额", quotas_sqrt), ("正比配额", quotas_prop)]:
            q = qf(counts, budget)
            for mname, mode in [("FPS(现行)", "fps"), ("随机", "random"),
                                ("最密", "dense")]:
                variants[f"{qname}+{mname}"] = build(
                    V, labels, flat_strength, uniq, counts, memb, q, mode, rng)
        variants["纯随机"] = rnd_ref

        base_mean = None
        print(f"{'策略':22s} {'n':>5s} {'NN距离mean':>11s} {'p50':>8s} {'p95':>8s} "
              f"{'vs随机':>8s} {'能量距离':>9s}")
        for k, sel in variants.items():
            m, p50, p95 = nn_stats(V, q_idx, sel, xp)
            ed = energy_distance(V, sel, rnd_ref, xp)
            if k == "纯随机":
                base_mean = m
            rel = f"{100*(base_mean-m)/base_mean:+.1f}%" if base_mean else "-"
            print(f"{k:22s} {len(sel):5d} {m:11.4f} {p50:8.4f} {p95:8.4f} "
                  f"{rel:>8s} {ed:9.5f}")

        ed_base = energy_distance(V, rnd_ref,
                                  np.sort(rng.choice(N, size=target, replace=False)), xp)
        print(f"能量距离基线(随机 vs 随机) = {ed_base:.5f}  <- 低于此量级即视为'与全库同分布'")


if __name__ == "__main__":
    main()
