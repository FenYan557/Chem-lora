# -*- coding: utf-8 -*-
"""隔离测试: 这台机器上 rdkit 哪条出图路径能用。"""
import sys
import traceback

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog("rdApp.*")
ms = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1C(=O)O")]


def t(tag, fn):
    try:
        r = fn()
        print(f"[OK]   {tag}: {type(r).__name__} {getattr(r, 'size', '')}")
    except Exception as e:
        print(f"[FAIL] {tag}: {type(e).__name__}: {str(e)[:200]}")


t("MolsToGridImage legends=None",
  lambda: Draw.MolsToGridImage(mols=ms, molsPerRow=2, subimageSize=(200, 200),
                               useSVG=False))
t("MolsToGridImage legends=list",
  lambda: Draw.MolsToGridImage(mols=ms, molsPerRow=2, subimageSize=(200, 200),
                               legends=["a", "b"], useSVG=False))
t("MolsToGridImage tuple",
  lambda: Draw.MolsToGridImage(mols=tuple(ms), molsPerRow=2,
                               subimageSize=(200, 200), useSVG=False))
t("MolToImage single", lambda: Draw.MolToImage(ms[0], size=(200, 200)))
t("MolToImage legend", lambda: Draw.MolToImage(ms[0], size=(200, 200), legend="x"))


def cairo_direct():
    from rdkit.Chem.Draw import rdMolDraw2D
    d = rdMolDraw2D.MolDraw2DCairo(240, 200)
    d.DrawMolecule(ms[0])
    d.FinishDrawing()
    png = d.GetDrawingText()
    import io
    from PIL import Image
    return Image.open(io.BytesIO(png))


t("MolDraw2DCairo direct", cairo_direct)


def cairo_drawmols():
    from rdkit.Chem.Draw import rdMolDraw2D
    d = rdMolDraw2D.MolDraw2DCairo(480, 200)
    d.DrawMolecules(list(ms), legends=["a", "b"])
    d.FinishDrawing()
    import io
    from PIL import Image
    return Image.open(io.BytesIO(d.GetDrawingText()))


t("MolDraw2DCairo.DrawMolecules", cairo_drawmols)
print("python", sys.version.split()[0])
