# -*- coding: utf-8 -*-
"""单分子卡片: 一张图只画一个分子, 图上标注它来自哪个簇。

每张卡片内容:
  标题栏  —— <模型名> ｜ 第 X 簇 (按规模排第 N)
  第二行  —— 该簇全库有多少分子 / 占全库比例 / 本次抽中多少个
  第三行  —— 这个分子是怎么挑出来的 (最典型 or 随机) + 典型性分数
  中间    —— RDKit 2D 结构 (单分子, 大画布)
  底部    —— SMILES 原文

卡片高度按文字实际行数自适应, 不会溢出。
默认每个模型出 8 张: 规模最大的 6 个簇 + 中位规模簇 + 噪声层。
"""
import argparse
import io
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

EMB = "/mnt/data/yangsensen/embedding"
FONT_CANDIDATES = [
    "/usr/share/fonts/simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def pick_font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size), p
            except Exception:
                continue
    return ImageFont.load_default(), "PIL-default"


def find_stem(d):
    hits = [f for f in sorted(os.listdir(d)) if f.endswith("_subset.csv")]
    if not hits:
        raise SystemExit(f"[cards] {d} 下找不到 *_subset.csv")
    return hits[0][: -len("_subset.csv")]


# ------------------------------------------------------------ 文字排版
def wrap_to_width(dr, text, font, max_width):
    """按像素宽度贪心折行 (中文没有空格, 必须逐字符量)。"""
    out, cur = [], ""
    for ch in str(text):
        if ch == "\n":
            out.append(cur)
            cur = ""
            continue
        if dr.textlength(cur + ch, font=font) <= max_width or not cur:
            cur += ch
        else:
            out.append(cur)
            cur = ch
    out.append(cur)
    return [ln for ln in out if ln != ""] or [""]


def line_h(dr, font):
    b = dr.textbbox((0, 0), "中文Ag0()%｜", font=font)
    return int(b[3] - b[1]) + 2


def draw_lines(dr, x, y, lines, font, fill, leading):
    for ln in lines:
        dr.text((x, y), ln, font=font, fill=fill)
        y += line_h(dr, font) + leading
    return y


# ------------------------------------------------------------ 出图
def render_mol(smiles, size):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opt = d.drawOptions()
    opt.padding = 25
    opt.bondLineWidth = 2.6
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB")


def make_card(out_path, title, l2, l3, smiles, mol_img, fonts, card_w, mol_h):
    f_title, f_body, f_smi = fonts
    scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    maxw = card_w - 40
    tl = wrap_to_width(scratch, title, f_title, maxw)
    bl2 = wrap_to_width(scratch, l2, f_body, maxw)
    bl3 = wrap_to_width(scratch, l3, f_body, maxw)
    sl = wrap_to_width(scratch, f"SMILES:  {smiles}", f_smi, maxw)
    pad = 14
    head_h = (pad + len(tl) * (line_h(scratch, f_title) + 4) + 8
              + len(bl2) * (line_h(scratch, f_body) + 3) + 4
              + len(bl3) * (line_h(scratch, f_body) + 3) + pad)
    foot_h = pad + len(sl) * (line_h(scratch, f_smi) + 4) + pad
    H = head_h + mol_h + foot_h

    card = Image.new("RGB", (card_w, H), "white")
    dr = ImageDraw.Draw(card)
    dr.rectangle([0, 0, card_w, head_h], fill=(236, 242, 250))
    dr.line([0, head_h, card_w, head_h], fill=(140, 160, 190), width=2)
    y = pad
    y = draw_lines(dr, 20, y, tl, f_title, (12, 40, 85), 4)
    y = draw_lines(dr, 20, y + 4, bl2, f_body, (25, 25, 25), 3)
    draw_lines(dr, 20, y + 2, bl3, f_body, (105, 105, 105), 3)

    if mol_img is not None:
        card.paste(mol_img, ((card_w - mol_img.size[0]) // 2, head_h))
    else:
        dr.text((20, head_h + 40), "该 SMILES RDKit 无法解析, 未出结构图",
                font=f_body, fill=(185, 30, 30))

    fy = head_h + mol_h
    dr.line([0, fy, card_w, fy], fill=(200, 200, 200), width=1)
    draw_lines(dr, 20, fy + pad, sl, f_smi, (30, 30, 30), 4)
    dr.rectangle([0, 0, card_w - 1, H - 1], outline=(165, 165, 165), width=2)
    card.save(out_path)
    return card.size


# ------------------------------------------------------------ 主流程
def cards_for_model(name, sub_dir, args, rng, fonts):
    stem = find_stem(sub_dir)
    base = os.path.join(sub_dir, stem)
    sub = pd.read_csv(base + "_subset.csv")
    labels = np.load(base + "_labels_all.npy")
    N = len(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    cmap = {int(c): int(n) for c, n in zip(uniq, counts)}
    n_noise = cmap.get(-1, 0)
    cl_ids = np.array([c for c in uniq if c >= 0])
    cl_sz = np.array([cmap[int(c)] for c in cl_ids])
    order = np.argsort(-cl_sz)
    cl_ids, cl_sz = cl_ids[order], cl_sz[order]

    out_dir = os.path.join(args.outdir, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[cards] === {name} ===  簇数={len(cl_ids)}  "
          f"噪声点={n_noise:,} ({n_noise / N:.2%})  字体={fonts[3]}")

    chosen = [(int(cl_ids[i]), int(cl_sz[i]), f"按规模排第 {i + 1} 大")
              for i in range(min(args.top, len(cl_ids)))]
    mid = len(cl_ids) // 2
    chosen.append((int(cl_ids[mid]), int(cl_sz[mid]),
                   f"中位规模, 第 {mid + 1}/{len(cl_ids)}"))
    chosen.append((-1, n_noise, "噪声层, 不属于任何簇"))

    made = []
    for i, (cid, size, rank_txt) in enumerate(chosen, 1):
        if cid == -1:
            g = sub[sub["role"] == "noise"]
            tag = "NOISE"
            title = f"{name}  ｜  噪声层 (cluster = -1)"
        else:
            g = sub[sub["cluster"] == cid]
            tag = f"C{cid:05d}"
            title = f"{name}  ｜  第 {cid} 簇  （{rank_txt}）"
        if len(g) == 0:
            print(f"  [skip] {tag}: 子集里没有成员")
            continue
        if args.which == "typical":
            row = g.loc[g["strength"].idxmax()]
            how = "该簇最典型的成员（典型性分数最高）"
        else:
            row = g.iloc[int(rng.integers(len(g)))]
            how = "该簇内随机抽的一个成员"
        l2 = (f"全库该簇 {size:,} 个分子（占全库 {100 * size / N:.2f}%）"
              f"　本次抽中 {len(g):,} 个（{100 * len(g) / max(size, 1):.2f}%）")
        l3 = f"{how}　典型性分数 {float(row['strength']):.3f}　split={row['split']}"
        p = os.path.join(out_dir, f"card{i:02d}_{tag}.png")
        mol_img = render_mol(str(row["smiles"]), (args.card_w - 40, args.mol_h))
        sz = make_card(p, title, l2, l3, str(row["smiles"]), mol_img,
                       fonts[:3], args.card_w, args.mol_h)
        print(f"  [img] {tag:8s} n={size:>9,} picked={len(g):>6,} "
              f"-> {os.path.basename(p)}  {sz[0]}x{sz[1]}px")
        made.append(p)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{EMB}/full_3.4m")
    ap.add_argument("--models", default="molformer,chemberta2,morgan")
    ap.add_argument("--outdir", default=f"{EMB}/full_3.4m/cards")
    ap.add_argument("--top", type=int, default=6, help="每个模型出图的大簇个数")
    ap.add_argument("--which", default="typical", choices=["typical", "random"])
    ap.add_argument("--card-w", type=int, default=820)
    ap.add_argument("--mol-h", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fonts = (pick_font(25)[0], pick_font(19)[0], pick_font(16)[0], pick_font(19)[1])
    total = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        d = os.path.join(args.root, name)
        if not os.path.isdir(d):
            print(f"[skip] {d} 不存在")
            continue
        total += cards_for_model(name, d, args, rng, fonts)
    print(f"\n[cards] DONE  共 {len(total)} 张 -> {args.outdir}")


if __name__ == "__main__":
    main()
