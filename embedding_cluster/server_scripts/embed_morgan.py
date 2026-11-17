# -*- coding: utf-8 -*-
"""Morgan fingerprint (ECFP4, r=2, 2048 bit) for PCQM4Mv2 SMILES. CPU, multiprocess.

Input : /mnt/data/yangsensen/data/*.jsonl.gz  (or smiles_all.parquet cache)
Output: ../output/morgan_emb.npy          (N, 2048) uint8   (dense bits)
        ../output/morgan_emb_packed.npy   (N, 256)  uint8   (bit-packed, 8x smaller)
        ../output/morgan_smiles.csv       smiles + split of the KEPT rows
Usage : python embed_morgan.py --n-jobs 32
        python embed_morgan.py --limit 2000      # smoke test

Notes:
  * uses rdFingerprintGenerator (the AllChem.GetMorganFingerprintAsBitVect API is
    deprecated in modern RDKit);
  * streams results with imap into a preallocated array instead of building a
    3.37 M-element Python list (~7 GB of small objects).
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(BASE, "..", "..")))   # mol_io.py lives in embedding/

from mol_io import load_smiles   # noqa: E402

from rdkit import Chem, RDLogger                     # noqa: E402
from rdkit.Chem import rdFingerprintGenerator        # noqa: E402
from rdkit.DataStructs import ConvertToNumpyArray    # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUTDIR = os.path.normpath(os.path.join(BASE, "..", "output"))
N_BITS = 2048
RADIUS = 2
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def fp_of(smi):
    """-> uint8 array of N_BITS, or None if the SMILES cannot be parsed."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    arr = np.zeros(N_BITS, dtype=np.uint8)
    ConvertToNumpyArray(GEN.GetFingerprint(mol), arr)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only first N molecules (smoke test)")
    ap.add_argument("--no-packed", action="store_true", help="skip the bit-packed copy")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    df = load_smiles()
    if args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
    n = len(df)
    print("[morgan] %d molecules, radius=%d, bits=%d, n_jobs=%d"
          % (n, RADIUS, N_BITS, args.n_jobs), flush=True)

    emb = np.zeros((n, N_BITS), dtype=np.uint8)
    valid = np.zeros(n, dtype=bool)
    smiles = df["smiles"].tolist()

    t0 = time.time()
    with Pool(args.n_jobs) as pool:
        for idx, vec in enumerate(pool.imap(fp_of, smiles, chunksize=4096)):
            if vec is not None:
                emb[idx] = vec
                valid[idx] = True
            if idx and idx % 200000 == 0:
                el = time.time() - t0
                rate = idx / max(el, 1e-9)
                print("  %d/%d  %.0f mol/s  elapsed %.0fs  eta %.0fs"
                      % (idx, n, rate, el, (n - idx) / max(rate, 1e-9)), flush=True)

    dropped = int((~valid).sum())
    if dropped:
        print("[morgan] dropped %d unparsable SMILES (%.3f%%)" % (dropped, 100.0 * dropped / n), flush=True)
    emb = emb[valid]
    df = df[valid].reset_index(drop=True)

    np.save(os.path.join(OUTDIR, "morgan_emb.npy"), emb)
    df.to_csv(os.path.join(OUTDIR, "morgan_smiles.csv"), index=False)
    print("[morgan] saved dense %s uint8 (%.2f GB) -> %s"
          % (emb.shape, emb.nbytes / 1024**3, OUTDIR), flush=True)
    if not args.no_packed:
        packed = np.packbits(emb, axis=1)
        np.save(os.path.join(OUTDIR, "morgan_emb_packed.npy"), packed)
        print("[morgan] saved packed %s uint8 (%.2f GB)"
              % (packed.shape, packed.nbytes / 1024**3), flush=True)
    print("[morgan] mean on-bits/molecule: %.1f  (%.0f mol/s total)"
          % (emb.sum(axis=1).mean(), n / max(time.time() - t0, 1e-9)), flush=True)


if __name__ == "__main__":
    main()
