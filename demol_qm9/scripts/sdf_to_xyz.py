#!/usr/bin/env python
"""Convert the DeepChem QM9 archive (SDF + CSV) into per-molecule .xyz files.

The DeepChem / moldata redistributions of QM9 ship as one big ``.sdf``
(3D geometries) plus a ``.sdf.csv`` (the 20 scalar properties).  This script
joins the two and writes one standard-layout ``.xyz`` per molecule::

    <n_atoms>
    gdb <index> <A> <B> <C> <mu> <alpha> <homo> <lumo> <gap> <r2> <zpve>
        <u0> <u298> <h298> <g298> <cv> <u0_atom> <u298_atom> <h298_atom> <g298_atom>
    <element> <x> <y> <z>
    ...

which ``demol.qm9.parse_qm9_file`` (and hence ``scripts/preprocess_qm9.py``)
reads directly.  No rdkit needed: the SDF V2000 record is parsed with a
fixed-column reader and cross-checked against the CSV.

Usage::

    python scripts/sdf_to_xyz.py --sdf data/qm9/qm9.sdf \
                                 --csv data/qm9/qm9.sdf.csv \
                                 --out-dir data/qm9/dsgdb9nsd.xyz
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPECTED_ELEMENTS = {"H", "C", "N", "O", "F"}


def load_properties(csv_path: str) -> dict[int, str]:
    """mol_id ("gdb_1") -> the 20 property values as one space-joined string."""
    props: dict[int, str] = {}
    with open(csv_path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        if header[0] != "mol_id" or len(header) != 21:
            raise ValueError(f"unexpected csv header: {header}")
        for lineno, line in enumerate(fh, start=2):
            parts = line.rstrip("\n").split(",")
            if len(parts) != 21:
                raise ValueError(f"{csv_path}:{lineno}: expected 21 columns, got {len(parts)}")
            mol_id = parts[0]
            idx = int(mol_id.split("_")[1])
            props[idx] = " ".join(parts[1:])
    return props


def iter_sdf_records(sdf_path: str):
    """Yield (title, [atom_lines]) for each record of an SDF V2000 file."""
    with open(sdf_path, "r", encoding="utf-8", errors="replace") as fh:
        record: list[str] = []
        for line in fh:
            if line.startswith("$$$$"):
                if record:
                    yield record
                record = []
            else:
                record.append(line.rstrip("\n"))
        if record:
            yield record


def parse_record(lines: list[str]) -> tuple[str, list[tuple[str, float, float, float]]]:
    """Return (title, [(element, x, y, z)]) from one SDF record."""
    title = lines[0].strip()
    counts = lines[3]
    try:
        n_atoms = int(counts[0:3])
    except ValueError:
        n_atoms = int(counts.split()[0])
    atoms: list[tuple[str, float, float, float]] = []
    for line in lines[4 : 4 + n_atoms]:
        try:
            x = float(line[0:10])
            y = float(line[10:20])
            z = float(line[20:30])
            el = line[31:34].strip()
        except (ValueError, IndexError):
            parts = line.split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            el = parts[3]
        if el not in EXPECTED_ELEMENTS:
            raise ValueError(f"unexpected element {el!r} in record {title!r}")
        atoms.append((el, x, y, z))
    if len(atoms) != n_atoms:
        raise ValueError(f"record {title!r}: counts line says {n_atoms} atoms, read {len(atoms)}")
    return title, atoms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdf", required=True, help="path to qm9.sdf / gdb9.sdf")
    ap.add_argument("--csv", required=True, help="path to qm9.sdf.csv / gdb9.sdf.csv")
    ap.add_argument("--out-dir", required=True, help="directory for per-molecule .xyz files")
    ap.add_argument("--limit", type=int, default=None, help="only convert the first N records")
    args = ap.parse_args(argv)

    props = load_properties(args.csv)
    print(f"[csv ] {len(props)} property rows from {args.csv}")
    os.makedirs(args.out_dir, exist_ok=True)

    n_written = 0
    n_skipped = 0
    seen: set[int] = set()
    for record in iter_sdf_records(args.sdf):
        title, atoms = parse_record(record)
        # titles look like "gdb 1"; fall back to sequential numbering
        parts = title.replace("_", " ").split()
        idx = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else n_written + n_skipped + 1
        prop_str = props.get(idx)
        if prop_str is None:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"[skip] no csv properties for {title!r}")
            continue
        seen.add(idx)
        out_path = os.path.join(args.out_dir, f"dsgdb9nsd_{idx:06d}.xyz")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(f"{len(atoms)}\n")
            out.write(f"gdb {idx} {prop_str}\n")
            for el, x, y, z in atoms:
                out.write(f"{el} {x:.10g} {y:.10g} {z:.10g}\n")
        n_written += 1
        if n_written % 10000 == 0:
            print(f"[..  ] {n_written} molecules written")
        if args.limit is not None and n_written >= args.limit:
            break

    print(f"[done] {n_written} xyz files -> {args.out_dir} "
          f"({n_skipped} records without csv match)")
    missing = len(props) - len(seen)
    if args.limit is None and missing:
        print(f"[warn] {missing} csv rows had no matching SDF record")
    print("\nnext: python scripts/preprocess_qm9.py --raw-dir "
          f'"{args.out_dir}" --cache-dir data/qm9_cache')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
