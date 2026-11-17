# -*- coding: utf-8 -*-
"""定位: 为什么单分子卡片中间的结构图是空白的。"""
import io

import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
SUB = ("/mnt/data/yangsensen/embedding/full_3.4m/molformer/"
       "molformer_emb_merged_subset.csv")


def dark(im):
    a = np.asarray(im.convert("L"))
    return int((a < 200).sum()), im.size, im.mode


def try_draw(m, tag, w=780, h=600, opts=None):
    try:
        d = rdMolDraw2D.MolDraw2DCairo(w, h)
        o = d.drawOptions()
        o.padding = 25
        if opts:
            for k, v in opts.items():
                setattr(o, k, v)
        d.DrawMolecule(m)
        d.FinishDrawing()
        png = d.GetDrawingText()
        im = Image.open(io.BytesIO(png))
        n, size, mode = dark(im)
        print(f"    {tag:28s} png={len(png):7d}B mode={mode} size={size} "
              f"dark={n}")
        return n
    except Exception as e:
        print(f"    {tag:28s} EXC {type(e).__name__}: {str(e)[:150]}")
        return -1


sub = pd.read_csv(SUB)
for cid in [50, 40, 6, 0, 5, 19, 86]:
    g = sub[sub.cluster == cid]
    row = g.loc[g["strength"].idxmax()]
    smi = str(row["smiles"])
    m = Chem.MolFromSmiles(smi)
    print(f"\n=== C{cid}  strength={row['strength']:.4f}  n_pick={len(g)}")
    print(f"  SMILES: {smi}")
    print(f"  parsed={m is not None}", end="")
    if m is None:
        m2 = Chem.MolFromSmiles(smi, sanitize=False)
        print(f"  sanitize=False parsed={m2 is not None}")
        if m2 is not None:
            try:
                m2.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(m2, Chem.SanitizeFlags.SANITIZE_ALL ^
                                 Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                                 catchErrors=True)
                print(f"  relaxed-sanitize atoms={m2.GetNumAtoms()}")
                try_draw(m2, "relaxed -> cairo")
            except Exception as e:
                print(f"  relaxed sanitize EXC {type(e).__name__}: {e}")
        continue
    print(f"  atoms={m.GetNumAtoms()}  confs={m.GetNumConformers()}  "
          f"mw={Chem.Descriptors.MolWt(m):.1f}")
    try:
        rdDepictor.Compute2DCoords(m)
        print("  Compute2DCoords: OK")
    except Exception as e:
        print(f"  Compute2DCoords EXC {type(e).__name__}: {str(e)[:150]}")
    try_draw(m, "A 原样(不预生成坐标)")
    try_draw(m, "B 预算坐标后")
    try_draw(m, "C prepareMols=False", opts={"prepareMols": False})
    try_draw(m, "D bondLineWidth默认")
