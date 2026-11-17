# -*- coding: utf-8 -*-
"""按簇查看: 每个簇多大、抽了多少、里面的分子长什么样。

只做"摆出来给人眼看"这一件事:
  1. <model>_cluster_table.csv  —— 全部簇的 大小 / 抽中数 / 抽样率 / 示例 SMILES
  2. <model>/C*.png             —— 前 K 大簇 + 噪声层, 每簇一张结构图网格
  3. <model>_overview.png       —— 前 K 大簇 + 噪声层拼成一张长图, 便于快速扫

刻意不做的事: 不给簇起化学类别名、不做官能团/骨架归类统计。
要看"这组是什么"由人眼判断; 需要自动归类时再单独加。
"""
import argparse
import io
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

RDLogger.DisableLog("rdApp.*")

EMB = "/mnt/data/yangsensen/embedding"
SUB_SUFFIX = "_subset.csv"


def find_stem(d):
    hits = [f for f in sorted(os.listdir(d)) if f.endswith(SUB_SUFFIX)]
    if not hits:
        raise SystemExit(f"[profile] {d} 下找不到 *{SUB_SUFFIX}")
    return hits[0][: -len(SUB_SUFFIX)]


def safe_mols(smiles_list):
    """SMILES -> (mol, 原序号) 列表, 解析失败的丢掉并计数。"""
    out, bad = [], 0
    for i, s in enumerate(smiles_list):
        try:
            m = Chem.MolFromSmiles(str(s))
        except Exception:
            m = None
        if m is None or m.GetNumAtoms() == 0:
            bad += 1
            continue
        out.append((m, i))
    return out, bad


def make_grid(mols, legends, per_row, cell):
    """自己拼网格: rdkit 2026.09 的 Draw.MolsToGridImage 内部调用与 C++ 签名不匹配,
    但底层 MolDraw2DCairo.DrawMolecules 正常, 所以直接用底层多面板绘制器。"""
    cw, ch = int(cell[0]), int(cell[1])
    per_row = max(1, int(per_row))
    rows = max(1, -(-len(mols) // per_row))
    d = rdMolDraw2D.MolDraw2DCairo(cw * per_row, ch * rows, cw, ch)
    opt = d.drawOptions()
    opt.padding = 8
    opt.legendFontSize = 15
    d.DrawMolecules(list(mols), legends=[(x if x else " ") for x in legends])
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def profile_model(name, sub_dir, args, rng):
    stem = find_stem(sub_dir)
    base = os.path.join(sub_dir, stem)
    print(f"\n{'=' * 70}\n[{name}]  {base}\n{'=' * 70}")

    sub = pd.read_csv(base + SUB_SUFFIX)
    labels = np.load(base + "_labels_all.npy")
    picked = np.load(base + "_picked_idx.npy")
    cfg = json.load(open(base + "_run_config.json", encoding="utf-8"))
    N = len(labels)

    # ---- 全库簇规模 (labels 是 3.45M 行的标签, -1 = 噪声层)
    uniq, counts = np.unique(labels, return_counts=True)
    noise_i = int(np.where(uniq == -1)[0][0]) if (uniq == -1).any() else -1
    n_noise = int(counts[noise_i]) if noise_i >= 0 else 0
    cl_ids = np.array([c for c in uniq if c >= 0])
    cl_sz = np.array([int(counts[i]) for i, c in enumerate(uniq) if c >= 0])
    order = np.argsort(-cl_sz)
    cl_ids, cl_sz = cl_ids[order], cl_sz[order]

    print(f"全库 N={N}  簇数={len(cl_ids)}  噪声点={n_noise} ({n_noise / N:.2%})")
    print(f"簇大小: 中位 {int(np.median(cl_sz))}  均值 {cl_sz.mean():.0f}  "
          f"p25 {int(np.percentile(cl_sz, 25))}  p75 {int(np.percentile(cl_sz, 75))}  "
          f"最大 {cl_sz[0]}  最小 {cl_sz[-1]}")
    tot = max(int(cl_sz.sum()), 1)
    cum = np.cumsum(cl_sz) / tot
    for k in (10, 20, 50, 100):
        if k <= len(cum):
            print(f"  前 {k:3d} 大簇 占非噪声分子 {cum[k - 1]:.1%}")
    for th in (100000, 10000, 1000, 100):
        print(f"  规模 >= {th:6d} 的簇: {int((cl_sz >= th).sum()):4d} 个")
    print(f"子集 picked={len(picked)}  target={cfg['target']}  "
          f"噪声层名额={int((sub['role'] == 'noise').sum())}")

    # ---- 逐簇表格 (全部簇, 不只是前 K)
    per_pick = sub[sub["cluster"] >= 0].groupby("cluster").size()
    rows, examples = [], []
    for cid, size in zip(cl_ids, cl_sz):
        g = sub[sub["cluster"] == int(cid)]
        npk = int(len(g))
        ex = g["smiles"].head(args.examples_csv).tolist()
        rows.append(dict(cluster=int(cid), n_total=int(size), n_picked=npk,
                         rate_pct=round(100.0 * npk / int(size), 3),
                         strength_median=round(float(g["strength"].median()), 4)
                         if npk else float("nan"),
                         example_smiles=" | ".join(map(str, ex))))
    g_noise = sub[sub["role"] == "noise"]
    rows.append(dict(cluster=-1, n_total=n_noise, n_picked=int(len(g_noise)),
                     rate_pct=round(100.0 * len(g_noise) / max(n_noise, 1), 3),
                     strength_median=round(float(g_noise["strength"].median()), 4)
                     if len(g_noise) else float("nan"),
                     example_smiles=" | ".join(
                         map(str, g_noise["smiles"].head(args.examples_csv)))))
    tab = pd.DataFrame(rows)
    tab_path = os.path.join(args.outdir, f"{name}_cluster_table.csv")
    tab.to_csv(tab_path, index=False, encoding="utf-8-sig")
    print(f"[out] 全部 {len(tab)} 行簇表 -> {tab_path}")

    # ---- 结构图: 前 K 大簇 + 噪声层
    img_dir = os.path.join(args.outdir, name)
    os.makedirs(img_dir, exist_ok=True)
    top = list(zip(cl_ids[:args.top], cl_sz[:args.top]))
    blocks = []
    for rank, (cid, size) in enumerate(top, 1):
        g = sub[sub["cluster"] == int(cid)]
        blocks.append((f"C{int(cid)}", int(size), int(len(g)), g["smiles"].tolist(),
                       f"rank{rank:03d}_C{int(cid):05d}_n{int(size)}"))
    blocks.append(("NOISE", n_noise, int(len(g_noise)),
                   g_noise["smiles"].tolist(), "rank999_NOISE"))

    overview_mols, overview_legends = [], []
    n_bad_total = 0
    for tag, size, npk, smiles_pool, fname in blocks:
        if len(smiles_pool) == 0:
            continue
        take = min(args.examples, len(smiles_pool))
        idx = rng.choice(len(smiles_pool), size=take, replace=False) \
            if len(smiles_pool) > take else np.arange(len(smiles_pool))
        idx = np.sort(idx)
        picked_smiles = [smiles_pool[i] for i in idx]
        mols, bad = safe_mols(picked_smiles)
        n_bad_total += bad
        if not mols:
            print(f"  [warn] {tag} 的示例 SMILES 全部解析失败, 跳过出图")
            continue
        legends = [f"{tag} n={size} picked={npk}"] + [""] * (len(mols) - 1)
        p = os.path.join(img_dir, f"{fname}.png")
        try:
            grid = make_grid([m for m, _ in mols], legends, len(mols), args.cell)
            grid.save(p)
        except Exception as e:
            print(f"  [warn] {tag} 出图失败: {type(e).__name__}: {str(e)[:120]}")
            continue
        overview_mols.extend(m for m, _ in mols)
        overview_legends.extend(legends)
        print(f"  [img] {tag:10s} 全库 {size:8d}  抽中 {npk:5d} "
              f"({100.0 * npk / max(size, 1):.2f}%)  -> {os.path.basename(p)}")
    if overview_mols:
        per_row = args.examples
        ov_path = os.path.join(args.outdir, f"{name}_overview.png")
        try:
            ov = make_grid(overview_mols, overview_legends, per_row, args.cell)
            ov.save(ov_path)
            print(f"[out] 总览长图 ({len(overview_mols)} 个分子, "
                  f"{ov.size[1] // args.cell[1]} 行 x {per_row} 列, "
                  f"{ov.size[0]}x{ov.size[1]} px) -> {ov_path}")
        except Exception as e:
            print(f"  [warn] {name} 总览图失败: {type(e).__name__}: {str(e)[:120]}")
    if n_bad_total:
        print(f"  [note] 共 {n_bad_total} 个示例 SMILES RDKit 解析失败已跳过")

    return dict(name=name, n_clusters=int(len(cl_ids)), noise=n_noise,
                table=tab_path, img_dir=img_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{EMB}/full_3.4m")
    ap.add_argument("--models", default="molformer,chemberta2,morgan")
    ap.add_argument("--outdir", default=f"{EMB}/full_3.4m/profile")
    ap.add_argument("--top", type=int, default=30, help="出图的簇数 (按规模取前 K)")
    ap.add_argument("--examples", type=int, default=8, help="每簇画几个分子")
    ap.add_argument("--examples-csv", type=int, default=12, help="表格里每簇写几条 SMILES")
    ap.add_argument("--cell", type=int, nargs=2, default=(240, 200))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.cell = tuple(args.cell)
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    done = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        sub_dir = os.path.join(args.root, name)
        if not os.path.isdir(sub_dir):
            print(f"[skip] {sub_dir} 不存在")
            continue
        done.append(profile_model(name, sub_dir, args, rng))

    print(f"\n{'=' * 70}\n[profile] DONE  输出目录 {args.outdir}")
    for d in done:
        print(f"  {d['name']:11s} 簇数={d['n_clusters']:4d} 噪声={d['noise']:8d}  "
              f"表={os.path.basename(d['table'])}")


if __name__ == "__main__":
    main()
