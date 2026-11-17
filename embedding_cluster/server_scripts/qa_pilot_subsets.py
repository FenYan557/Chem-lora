# -*- coding: utf-8 -*-
"""中试子集体检: 结构不变量 + 覆盖度 + 跨模型重合度。

analyze_subset.py 依赖合成数据的 true_cluster, 真实 PCQM4Mv2 没有真值, 用不了。
这里改成三个不依赖真值的角度:

  A. 不变量  —— 子集大小/无重复/行号与 csv 对齐/每簇都有代表点/role 与簇号自洽/
                配额正比于簇大小 (簇质量分布的总变差距离 TV 要接近 0);
  B. 覆盖度  —— 从全库抽样, 量每个分子到"最近子集成员"的距离, 与同规模
                **随机子集**比。目标是"子集预训练 -> 全量推理复现全量效果",
                即分布匹配, 所以判据是**与随机打平**(gain≈0), 而不是比随机更远。
                旧版 sqrt 配额 + 簇内 FPS 在这里实测 -54% ~ -120%, 是明确的回归。
                在 UMAP 空间看簇结构覆盖, 在原始 embedding 空间看化学语义覆盖。
                均值反映整体, p95 反映尾部有没有被漏掉。
  C. 重合度  —— 三套 embedding 选出的分子有多少是同一批。重合越低说明三种
                表征捕捉的信息越互补, 也越值得做下游对比。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

EMB = "/mnt/data/yangsensen/embedding"
DEFAULT_ROOT = f"{EMB}/pilot_100k"
MODELS = {
    "molformer": dict(emb=f"{EMB}/molformer/output/molformer_emb_merged.npy",
                      meta=f"{EMB}/molformer/output/molformer_smiles_merged.csv",
                      sub="molformer"),
    "chemberta2": dict(emb=f"{EMB}/chemberta2/output/chemberta2_emb_merged.npy",
                       meta=f"{EMB}/chemberta2/output/chemberta2_smiles_merged.csv",
                       sub="chemberta2"),
    "morgan": dict(emb=f"{EMB}/morgan_fp/output/morgan_emb.npy",
                   meta=f"{EMB}/morgan_fp/output/morgan_smiles.csv",
                   sub="morgan"),
}

N_FAIL = 0


def check(cond, msg):
    global N_FAIL
    if cond:
        return True
    N_FAIL += 1
    print(f"  [FAIL] {msg}")
    return False


def get_xp():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:
        return np


# ------------------------------------------------------------------ B. 覆盖度
def nn_dist_umap(V, q_idx, sel_idx, xp):
    """UMAP 空间: 每个查询点到最近子集成员的欧氏距离。

    走 |q|²-2q·s+|s|² 展开而不是广播减法: 后者要开 chunk×|子集|×15 的临时张量,
    全量子集 5 万时轻松上 GB。chunk 按子集大小自适应, 把临时矩阵压在 ~160 MB。
    """
    S = xp.asarray(np.asarray(V[sel_idx], dtype=np.float32))
    s2 = (S ** 2).sum(1)
    chunk = int(max(256, min(8192, 4e7 // max(len(sel_idx), 1))))
    out = np.empty(len(q_idx), dtype=np.float32)
    for a in range(0, len(q_idx), chunk):
        b = min(a + chunk, len(q_idx))
        Q = xp.asarray(np.asarray(V[q_idx[a:b]], dtype=np.float32))
        d = (Q ** 2).sum(1)[:, None] - 2 * (Q @ S.T) + s2[None, :]
        out[a:b] = xp.asnumpy(d.min(axis=1))
        del Q, d
    return np.sqrt(np.maximum(out, 0))


def nn_sim_native(X, q_idx, sel_idx, kind, xp, chunk=4096):
    """原始 embedding 空间: 每个查询点到子集的最大相似度 (越大越好)。
    kind='cos' 走 L2 归一化余弦; kind='tanimoto' 走二值指纹 Jaccard。"""
    S = np.asarray(X[sel_idx])
    if kind == "tanimoto":
        Sb = (S != 0)
        # 计数也要放到 xp 上: 与 inter(cupy) 混算会 TypeError
        s_cnt = xp.asarray(Sb.sum(1, keepdims=True).astype(np.float32))
        Sg = xp.asarray(Sb.astype(np.float32))
    else:
        Sg = xp.asarray(_l2(S.astype(np.float32)))
    out = np.empty(len(q_idx), dtype=np.float32)
    for a in range(0, len(q_idx), chunk):
        b = min(a + chunk, len(q_idx))
        Q = np.asarray(X[q_idx[a:b]])
        if kind == "tanimoto":
            Qb = (Q != 0)
            q_cnt = xp.asarray(Qb.sum(1, keepdims=True).astype(np.float32))
            Qg = xp.asarray(Qb.astype(np.float32))
            inter = Qg @ Sg.T
            union = q_cnt + s_cnt.reshape(1, -1) - inter
            sim = inter / xp.maximum(union, 1)
        else:
            sim = xp.asarray(_l2(Q.astype(np.float32))) @ Sg.T
        out[a:b] = xp.asnumpy(sim.max(axis=1))
        del Q, sim
    return out


def _l2(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n


def summarize(d, higher_better=False):
    q = np.percentile(d, [50, 95, 99])
    return dict(mean=float(np.mean(d)), p50=float(q[0]),
                p95=float(q[1]), p99=float(q[2]))


# ------------------------------------------------------------------ 主流程
def qa_model(name, paths, args, xp, rng):
    print(f"\n{'=' * 66}\n[{name}]\n{'=' * 66}")
    stem = None
    for f in sorted(os.listdir(paths["sub"])):
        if f.endswith("_subset.csv"):
            stem = f[: -len("_subset.csv")]
    if stem is None:
        check(False, f"{paths['sub']} 下没有 *_subset.csv")
        return None
    base = os.path.join(paths["sub"], stem)

    sub = pd.read_csv(base + "_subset.csv")
    picked = np.load(base + "_picked_idx.npy")
    labels = np.load(base + "_labels_all.npy")
    V = np.load(base + "_umap_all.npy", mmap_mode="r")
    cfg = json.load(open(base + "_run_config.json", encoding="utf-8"))
    meta = pd.read_csv(paths["meta"])
    X = np.load(paths["emb"], mmap_mode="r")
    N = len(meta)

    # ---- A. 不变量
    print("-- 不变量")
    check(len(sub) == cfg["target"] == len(picked),
          f"子集大小不符: csv={len(sub)} cfg={cfg['target']} idx={len(picked)}")
    check(len(set(picked.tolist())) == len(picked), "picked 有重复行号")
    check(int(picked.min()) >= 0 and int(picked.max()) < N, "picked 行号越界")
    # 源表本身就有重复 SMILES (100k 里 324 行, 同一分子多条记录), 所以子集里
    # 偶尔出现重复是数据属性而非选点 bug; 只在重复率超过源表水平时才报警。
    n_dup_sub = len(sub) - sub["smiles"].nunique()
    n_dup_meta = len(meta) - meta["smiles"].nunique()
    rate_sub = n_dup_sub / max(len(sub), 1)
    rate_meta = n_dup_meta / max(len(meta), 1)
    print(f"  重复 SMILES: 子集 {n_dup_sub} ({rate_sub:.3%}) vs 源表 "
          f"{n_dup_meta} ({rate_meta:.3%})")
    check(rate_sub <= 3 * rate_meta + 1e-6,
          f"子集重复率 {rate_sub:.3%} 远高于源表 {rate_meta:.3%}, 选点可能有问题")
    # 行号与 csv 必须指同一批分子 —— 证明落盘的 picked_idx 没错位
    exp_smiles = meta.iloc[picked]["smiles"].to_numpy()
    check(bool((exp_smiles == sub["smiles"].to_numpy()).all()),
          "subset.csv 的 smiles 与 meta.iloc[picked] 不一致 (行号错位!)")
    check(len(V) == N and len(labels) == N, "umap/labels 行数与 meta 不符")
    check(int((labels >= 0).sum()) > 0, "全库没有任何非噪声点")
    print(f"  N={N}  target={cfg['target']}  picked={len(picked)}  "
          f"clusters={cfg['n_clusters']}  noise={cfg['noise_rate']:.2%}")

    # ---- 配额与簇覆盖
    print("-- 簇覆盖与配额")
    inl = labels[labels >= 0]
    uniq, counts = np.unique(inl, return_counts=True)
    per = sub[sub["cluster"] >= 0].groupby("cluster").size()
    check(len(per) == len(uniq),
          f"有簇没被选到: 覆盖 {len(per)}/{len(uniq)}")
    quota_ok = all(int(per.get(int(c), 0)) <= int(n) for c, n in zip(uniq, counts))
    check(quota_ok, "某簇选中数超过了该簇总容量")
    q = per.reindex(uniq).fillna(0).to_numpy().astype(float)
    # 正比配额: 选中数应与簇大小强相关 (spearman 只看秩, 与 sqrt/线性无关)
    rho = float(pd.Series(q).corr(pd.Series(counts.astype(float)), method="spearman"))
    check(rho > 0.9, f"配额与簇大小相关性过低: rho={rho:.3f}")
    # 分布匹配的硬指标: 子集的"簇质量分布"应与全库一致。
    # TV = 0.5*sum|p_sub - p_full|, 正比配额下应 <0.02, sqrt 配额会到 0.2~0.4。
    p_sub = q / max(q.sum(), 1)
    p_full = counts / counts.sum()
    tv = float(0.5 * np.abs(p_sub - p_full).sum())
    check(tv < 0.05, f"簇质量分布偏离全库过多: TV={tv:.4f} (应 <0.05)")
    rate = q / counts
    print(f"  覆盖簇 {len(per)}/{len(uniq)}  配额-簇大小 spearman={rho:.4f}  "
          f"簇质量 TV={tv:.4f}")
    print(f"  每簇采样率: 中位 {np.median(rate):.3%}  "
          f"[p5 {np.percentile(rate,5):.3%} ~ p95 {np.percentile(rate,95):.3%}]  "
          f"全库簇内比例 {counts.sum()/N:.3%}")
    print(f"  最大簇 {counts.max()} -> 选中 {int(q[counts.argmax()])}; "
          f"最小簇 {counts.min()} -> 选中 {int(q[counts.argmin()])}")
    roles = set(sub["role"].astype(str).unique())
    check(roles <= {"cluster", "noise"}, f"role 出现未知取值: {sorted(roles)}")
    check(bool(((sub["cluster"] == -1) == (sub["role"] == "noise")).all()),
          "role 与 cluster==-1 不自洽 (噪声层标记错位)")
    n_noise_sub = int((sub["role"] == "noise").sum())
    exp_noise = cfg["target"] * cfg["noise_rate"]
    check(abs(n_noise_sub - exp_noise) <= 0.02 * cfg["target"] + 50,
          f"噪声层名额 {n_noise_sub} 偏离实测噪声率应得的 {exp_noise:.0f}")
    print(f"  role 分布: {sub['role'].value_counts().to_dict()}  "
          f"(噪声层期望 {exp_noise:.0f}, 全库噪声率 {cfg['noise_rate']:.2%})")
    print(f"  strength: min={sub['strength'].min():.3f} "
          f"median={sub['strength'].median():.3f} max={sub['strength'].max():.3f}")

    # ---- B. 覆盖度
    print(f"-- 覆盖度 (查询 {args.queries} 个, 对照 = 同规模随机子集)")
    nq = min(args.queries, N)
    q_idx = np.sort(rng.choice(N, size=nq, replace=False))
    base_idx = np.sort(rng.choice(N, size=len(picked), replace=False))

    d_sel = nn_dist_umap(V, q_idx, picked, xp)
    d_rnd = nn_dist_umap(V, q_idx, base_idx, xp)
    s_sel, s_rnd = summarize(d_sel), summarize(d_rnd)
    gain = 100 * (s_rnd["mean"] - s_sel["mean"]) / max(s_rnd["mean"], 1e-9)
    print(f"  UMAP  距离  聚类子集: mean={s_sel['mean']:.4f} p95={s_sel['p95']:.4f}"
          f" | 随机: mean={s_rnd['mean']:.4f} p95={s_rnd['p95']:.4f}")
    print(f"  UMAP  平均最近邻距离变化 {gain:+.1f}% (分布匹配目标: ≈0 即打平随机; "
          f"旧版 sqrt+FPS 实测 -94.8/-119.7/-54.0%)")
    check(gain > -5.0,
          f"{name}: 聚类子集 UMAP 覆盖度明显差于随机子集 ({gain:+.1f}%)")

    kind = "tanimoto" if name == "morgan" else "cos"
    sim_sel = nn_sim_native(X, q_idx, picked, kind, xp)
    sim_rnd = nn_sim_native(X, q_idx, base_idx, kind, xp)
    n_sel, n_rnd = summarize(sim_sel), summarize(sim_rnd)
    unit = "Tanimoto" if kind == "tanimoto" else "cosine"
    print(f"  原生 {unit:9s} 聚类子集: mean={n_sel['mean']:.4f} "
          f"p5={np.percentile(sim_sel,5):.4f} | 随机: mean={n_rnd['mean']:.4f} "
          f"p5={np.percentile(sim_rnd,5):.4f}")
    dgain = 100 * (n_sel["mean"] - n_rnd["mean"]) / max(abs(n_rnd["mean"]), 1e-9)
    print(f"  原生 平均最近邻相似度变化 {dgain:+.1f}% (目标: ≈0 或略正)")
    check(dgain > -5.0,
          f"{name}: 聚类子集原生 {unit} 覆盖度明显差于随机子集 ({dgain:+.1f}%)")

    return dict(picked=picked, umap=dict(sel=s_sel, rnd=s_rnd, gain_pct=gain),
                native=dict(kind=kind, sel=n_sel, rnd=n_rnd, gain_pct=dgain),
                quota=dict(spearman=rho, mass_tv=tv,
                           median_rate=float(np.median(rate))),
                n_clusters=int(cfg["n_clusters"]), noise_rate=cfg["noise_rate"],
                picked_n=int(len(picked)), n_noise=int(n_noise_sub))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="中试输出根目录 (含 molformer/ chemberta2/ morgan/ 三个子目录)")
    ap.add_argument("--out", default="",
                    help="qa_summary.json 落盘路径; 默认写到 <root>/qa_summary.json")
    args = ap.parse_args()
    xp = get_xp()
    print(f"backend for coverage math: {xp.__name__}")
    print(f"subset root: {args.root}")
    rng = np.random.default_rng(args.seed)

    res = {}
    for i, (name, p) in enumerate(MODELS.items()):
        paths = dict(p, sub=os.path.join(args.root, p["sub"]))
        r = qa_model(name, paths, args, xp, rng)
        if r:
            res[name] = r

    # ---- C. 跨模型重合度
    print(f"\n{'=' * 66}\n[跨模型重合度]\n{'=' * 66}")
    names = list(res)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = set(res[names[i]]["picked"].tolist()), set(res[names[j]]["picked"].tolist())
            inter = len(a & b)
            jac = inter / max(len(a | b), 1)
            print(f"  {names[i]:10s} vs {names[j]:10s}: 交集 {inter:5d} "
                  f"({100*inter/max(len(a),1):.2f}% of {len(a)})  Jaccard={jac:.4f}")

    print(f"\n{'=' * 66}")
    if N_FAIL == 0:
        print("QA ALL PASS")
    else:
        print(f"QA FAILED: {N_FAIL} 项不通过")
    out = args.out or os.path.join(args.root, "qa_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "picked"}
                   for k, v in res.items()}, f, ensure_ascii=False, indent=2)
    print(f"summary -> {out}")
    raise SystemExit(1 if N_FAIL else 0)


if __name__ == "__main__":
    main()
