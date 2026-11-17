# -*- coding: utf-8 -*-
"""验证归因: 覆盖度变差是不是因为"噪声点没被代表"。

diag_selection.py 里各策略的 NN 距离 p50 只比随机差 ~25%, p95 却差 2 倍以上 ——
典型的"少数查询点被完全漏掉"。而 molformer 有 22.6% 的库被判为 HDBSCAN 噪声,
现行 noise_frac=0.05 只给它们 250/5000 个名额, 相当于 4.5 倍的低配。
这里把查询点按 噪声/簇内 分开量, 若差距全落在噪声组即归因成立。
同时用**真实产出的 picked**(含 250 个噪声名额, n=5000)对比, 排除样本数不等的干扰。
"""
import numpy as np
import pandas as pd

EMB = "/mnt/data/yangsensen/embedding"
SUB = f"{EMB}/pilot_100k"


def nn_dist(V, q_idx, sel, xp):
    S = xp.asarray(np.asarray(V[sel], dtype=np.float32))
    s2 = (S ** 2).sum(1)
    chunk = int(max(256, min(8192, 4e7 // max(len(sel), 1))))
    d = np.empty(len(q_idx), dtype=np.float32)
    for a in range(0, len(q_idx), chunk):
        b = min(a + chunk, len(q_idx))
        Q = xp.asarray(np.asarray(V[q_idx[a:b]], dtype=np.float32))
        dd = (Q ** 2).sum(1)[:, None] - 2 * (Q @ S.T) + s2[None, :]
        d[a:b] = xp.asnumpy(dd.min(axis=1))
    return np.sqrt(np.maximum(d, 0))


def main():
    import cupy as xp
    for name, stem in [("molformer", "molformer_emb_merged"),
                       ("chemberta2", "chemberta2_emb_merged"),
                       ("morgan", "morgan_emb")]:
        V = np.load(f"{SUB}/{name}/{stem}_umap_all.npy", mmap_mode="r")
        labels = np.load(f"{SUB}/{name}/{stem}_labels_all.npy")
        picked = np.load(f"{SUB}/{name}/{stem}_picked_idx.npy")
        sub = pd.read_csv(f"{SUB}/{name}/{stem}_subset.csv")
        N = len(labels)
        rng = np.random.default_rng(42)
        rnd = np.sort(rng.choice(N, size=len(picked), replace=False))
        q = np.sort(rng.choice(N, size=20000, replace=False))

        is_noise_q = labels[q] == -1
        noise_rate = float((labels == -1).mean())
        n_noise_sel = int((sub["role"] == "outlier").sum())
        sel_noise_rate = n_noise_sel / len(picked)

        d_sel = nn_dist(V, q, picked, xp)
        d_rnd = nn_dist(V, q, rnd, xp)

        print(f"\n{'=' * 70}\n[{name}]  库噪声率={noise_rate:.2%}  "
              f"子集噪声名额={n_noise_sel}/{len(picked)}={sel_noise_rate:.2%}  "
              f"低配 {noise_rate / max(sel_noise_rate, 1e-9):.1f}x")
        print(f"{'查询点类型':14s} {'占比':>7s} {'聚类子集 mean':>14s} "
              f"{'p95':>8s} {'随机子集 mean':>14s} {'p95':>8s} {'倍数':>7s}")
        for tag, m in [("簇内", ~is_noise_q), ("噪声", is_noise_q), ("全部", np.ones_like(is_noise_q))]:
            if m.sum() == 0:
                continue
            a, b = d_sel[m], d_rnd[m]
            print(f"{tag:14s} {100 * m.mean():6.1f}% {a.mean():14.4f} "
                  f"{np.percentile(a, 95):8.4f} {b.mean():14.4f} "
                  f"{np.percentile(b, 95):8.4f} {a.mean() / max(b.mean(), 1e-9):6.2f}x")


if __name__ == "__main__":
    main()
