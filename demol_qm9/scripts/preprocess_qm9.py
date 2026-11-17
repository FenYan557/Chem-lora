#!/usr/bin/env python
"""Featurise QM9 and write the memory-mapped cache used for training.

Usage::

    python scripts/preprocess_qm9.py --raw-dir data/qm9/dsgdb9nsd.xyz \
                                     --cache-dir data/qm9_cache

Produces one ``.npy`` per array (atoms, bonds, SPD matrices, line-graph SPD,
shared-atom index, targets) plus ``cache_stats.json`` and ``split.npy`` with the
paper's split: 10,000 validation / 10,831 test / the rest for training
(Appendix D.4, following Tholke & De Fabritiis 2022).

Bond orders and aromaticity come from rdkit when it is importable, otherwise
from the pure-numpy geometry back-end (Algorithm 1 + bond-length ratios).  The
chosen back-end is recorded in ``cache_stats.json``; a cache built with one
back-end should not be mixed with the other.

Peak memory for the full dataset is roughly 1.5 GB (all arrays are held in RAM
before ``np.save``).  Use ``--limit`` for a quick partial cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demol import chem  # noqa: E402
from demol.features import rdkit_available  # noqa: E402
from demol.qm9 import (  # noqa: E402
    N_TARGETS,
    N_TEST,
    N_VAL,
    QM9_TARGETS,
    TARGET_NAMES,
    iter_qm9_dir,
    make_split,
    write_cache,
)


def summarise(records: list[dict]) -> dict:
    """Distribution checks that catch a mis-parsed archive early."""
    n_atoms = np.array([r["feats"].atomic_numbers.shape[0] for r in records])
    n_bonds = np.array([r["feats"].bond_pairs.shape[0] for r in records])
    targets = np.stack([r["targets"] for r in records])
    elements = np.unique(np.concatenate([r["feats"].atomic_numbers for r in records]))
    gap = targets[:, TARGET_NAMES.index("gap")]
    homo = targets[:, TARGET_NAMES.index("homo")]
    lumo = targets[:, TARGET_NAMES.index("lumo")]
    return {
        "molecules": len(records),
        "elements": [int(z) for z in elements],
        "atoms_per_molecule": {
            "min": int(n_atoms.min()), "max": int(n_atoms.max()),
            "mean": round(float(n_atoms.mean()), 2),
        },
        "bonds_per_molecule": {
            "min": int(n_bonds.min()), "max": int(n_bonds.max()),
            "mean": round(float(n_bonds.mean()), 2),
        },
        "target_stats": {
            name: {"mean": round(float(targets[:, k].mean()), 5),
                   "std": round(float(targets[:, k].std()), 5)}
            for k, name in enumerate(TARGET_NAMES)
        },
        "gap_consistency_max_abs_err_meV": float(
            np.abs((lumo - homo) - gap).max()
        ),
        "index_range": [int(records[0]["index"]), int(records[-1]["index"])],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, help="directory of QM9 .xyz files")
    ap.add_argument("--cache-dir", required=True, help="output directory for the .npy cache")
    ap.add_argument("--backend", choices=("auto", "rdkit", "geometry"), default="auto")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N molecules")
    ap.add_argument("--seed", type=int, default=42, help="split RNG seed")
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--no-split", action="store_true",
                    help="skip writing split.npy (e.g. when caching a partial subset)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.raw_dir):
        print(f"[err ] raw dir not found: {args.raw_dir}", file=sys.stderr)
        return 2

    use_rdkit = args.backend == "rdkit" or (args.backend == "auto" and rdkit_available())
    if args.backend == "rdkit" and not rdkit_available():
        print("[err ] --backend rdkit requested but rdkit is not importable", file=sys.stderr)
        return 2
    print(f"[info] featurisation backend: {'rdkit' if use_rdkit else 'geometry (pure numpy)'}")
    print(f"[info] bond perception: covalent radii, alpha = {chem.BOND_THRESHOLD_ALPHA}")

    started = time.time()
    records: list[dict] = []
    for rec in iter_qm9_dir(args.raw_dir, prefer_rdkit=use_rdkit, limit=args.limit):
        records.append(rec)
        if len(records) % 5000 == 0:
            rate = len(records) / max(1e-9, time.time() - started)
            print(f"       {len(records):7d} molecules  ({rate:6.0f}/s)")
    if not records:
        print("[err ] no molecules parsed", file=sys.stderr)
        return 1
    print(f"[info] featurised {len(records)} molecules in {time.time() - started:.0f} s")

    summary = summarise(records)
    gap_err = summary["gap_consistency_max_abs_err_meV"]
    if gap_err > 1.0:
        print(f"[warn] HOMO-LUMO gap inconsistency up to {gap_err:.3f} meV; "
              "the target column layout may differ from the assumption in qm9.py")

    os.makedirs(args.cache_dir, exist_ok=True)
    stats = write_cache(records, args.cache_dir)
    stats.update(summary)
    stats["backend"] = "rdkit" if use_rdkit else "geometry"
    stats["bond_threshold_alpha"] = chem.BOND_THRESHOLD_ALPHA
    stats["n_targets"] = N_TARGETS
    stats["targets"] = [
        {"name": t.name, "unit": t.unit, "atomised": t.atomize} for t in QM9_TARGETS
    ]
    with open(os.path.join(args.cache_dir, "cache_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    if not args.no_split:
        n = len(records)
        if n <= args.n_val + args.n_test:
            print(f"[warn] only {n} molecules cached; writing a proportional split instead "
                  f"of {args.n_val}/{args.n_test}")
            n_val = max(1, int(0.08 * n))
            n_test = max(1, int(0.08 * n))
        else:
            n_val, n_test = args.n_val, args.n_test
        split = make_split(n, seed=args.seed, n_val=n_val, n_test=n_test)
        np.save(os.path.join(args.cache_dir, "split.npy"), split)
        counts = np.bincount(split, minlength=3)
        print(f"[ok  ] split (seed {args.seed}): train={counts[0]} val={counts[1]} test={counts[2]}")

    print(f"[ok  ] cache written to {args.cache_dir}")
    print(json.dumps({k: stats[k] for k in ("n_molecules", "n_atoms", "n_bonds",
                                            "max_atoms", "max_bonds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
