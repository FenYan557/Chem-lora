"""QM9 parsing, caching, splitting and dense batching.

Data flow::

    dsgdb9nsd.xyz/*.xyz  --parse_qm9-->  per-molecule MolFeatures
                       --preprocess-->  cache/*.npy  (memory mapped)
                       --QM9Dataset-->  ragged numpy dicts
                       --qm9_collate--> dense padded torch tensors

Split follows the DeMol paper's QM9 protocol (Appendix D.4), which itself
follows Tholke & De Fabritiis (2022): 10,000 validation / 10,831 test
molecules drawn at random, the remaining ~113k used for training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from . import chem
from .features import MolFeatures, featurize_molecule

# --- physical constants -----------------------------------------------------
HA_TO_MEV = 27211.386245988
HA_TO_KCAL = 627.5094740631

# Atomisation reference energies (Hartree) from the QM9/GDB-9 atomref table.
ATOMREF_HA = {1: -0.500273, 6: -37.84677, 7: -54.56800, 8: -75.03622, 9: -99.71900}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    column: int            # index into the 22 raw values of xyz line 2
    atom_column: int       # pre-atomised column, -1 when absent
    unit: str
    scale: float           # raw (Hartree) -> reported unit
    atomize: bool          # subtract per-element reference energies


# Column order of QM9 xyz line 2 (after "gdb <index>"):
#   0 A, 1 B, 2 C, 3 mu, 4 alpha, 5 HOMO, 6 LUMO, 7 gap, 8 R2, 9 ZPVE,
#   10 U0, 11 U298, 12 H298, 13 G298, 14 Cv,
#   15 U0_atom, 16 U298_atom, 17 H298_atom, 18 G298_atom, 19 w1, 20 w2, 21 w3
QM9_TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec("mu", 3, -1, "D", 1.0, False),
    TargetSpec("alpha", 4, -1, "Bohr^3", 1.0, False),
    TargetSpec("homo", 5, -1, "meV", HA_TO_MEV, False),
    TargetSpec("lumo", 6, -1, "meV", HA_TO_MEV, False),
    TargetSpec("gap", 7, -1, "meV", HA_TO_MEV, False),
    TargetSpec("r2", 8, -1, "Bohr^2", 1.0, False),
    TargetSpec("zpve", 9, -1, "meV", HA_TO_MEV, False),
    TargetSpec("u0", 10, 15, "kcal/mol", HA_TO_KCAL, True),
    TargetSpec("u298", 11, 16, "kcal/mol", HA_TO_KCAL, True),
    TargetSpec("h298", 12, 17, "kcal/mol", HA_TO_KCAL, True),
    TargetSpec("g298", 13, 18, "kcal/mol", HA_TO_KCAL, True),
    TargetSpec("cv", 14, -1, "cal/mol*K", 1.0, False),
)
TARGET_NAMES = tuple(t.name for t in QM9_TARGETS)
N_TARGETS = len(QM9_TARGETS)

N_VAL = 10_000
N_TEST = 10_831


# --------------------------------------------------------------------------
# raw xyz parsing
# --------------------------------------------------------------------------
def _is_atom_line(line: str) -> bool:
    """True when the line looks like ``<element> <x> <y> <z> [charge]``."""
    parts = line.split()
    if len(parts) < 4 or parts[0] not in chem.SYMBOL_TO_Z:
        return False
    try:
        float(parts[1])
        float(parts[2])
        float(parts[3])
    except ValueError:
        return False
    return True


def _all_floats(line: str) -> bool:
    try:
        [float(t) for t in line.split()]
        return True
    except ValueError:
        return False


def parse_qm9_file(path: str) -> tuple[int, np.ndarray, np.ndarray, list[str], list[str]]:
    """Parse one QM9 ``.xyz`` record.

    Standard figshare layout (sdata201422)::

        line 0            number of atoms
        line 1            "gdb <index>" followed by 15-21 scalar properties
        lines 2..2+n      element, x, y, z [, Mulliken charge]
        line 2+n          harmonic vibrational frequencies (optional)
        line 2+n+1        SMILES (optional)
        line 2+n+2        InChI (optional)

    The atom block is located by scanning for the first atom-looking line,
    so variants with extra comment/SMILES lines before the coordinates also
    parse correctly.  Returns (index, atomic_numbers, pos, raw_values,
    [smiles, inchi]); SMILES/InChI are best-effort metadata only.
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    n_atoms = int(lines[0].strip())
    header = lines[1].split()
    if len(header) >= 2 and not header[0].lstrip("+-").isdigit():
        index = int(header[1])
        raw = np.array([float(v) for v in header[2:]], dtype=np.float64)
    else:  # header without the leading "gdb" tag
        index = int(header[0])
        raw = np.array([float(v) for v in header[1:]], dtype=np.float64)
    start = None
    for i in range(2, len(lines)):
        if _is_atom_line(lines[i]):
            start = i
            break
    if start is None:
        raise ValueError(f"{path}: no atom block found")
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[start : start + n_atoms]:
        parts = line.split()
        if not _is_atom_line(line):
            raise ValueError(f"{path}: atom block truncated or misaligned")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    # best-effort SMILES / InChI after the atom block (skip frequencies)
    rest = [ln.strip() for ln in lines[start + n_atoms :] if ln.strip()]
    if rest and _all_floats(rest[0]):
        rest = rest[1:]
    smiles = rest[0] if rest else ""
    inchi = rest[1] if len(rest) > 1 else ""
    z = np.array([chem.SYMBOL_TO_Z.get(s, 0) for s in symbols], dtype=np.int64)
    if z.shape[0] != n_atoms or (z == 0).any():
        raise ValueError(f"{path}: unsupported element or malformed coordinate block")
    return index, z, np.array(coords, dtype=np.float32), raw, [smiles, inchi]


def target_values(raw: np.ndarray, atomic_numbers: np.ndarray) -> np.ndarray:
    """Extract the 12 benchmark targets, converted to the paper's units.

    Energies U0/U/H/G are reported as *atomisation* energies (the standard QM9
    protocol used by EGNN / TorchMD-NET / PaiNN / DeMol): the pre-atomised
    columns are used when the file provides them, otherwise per-element
    reference energies are subtracted.
    """
    n_raw = raw.shape[0]
    out = np.zeros(N_TARGETS, dtype=np.float64)
    for k, spec in enumerate(QM9_TARGETS):
        if spec.atomize and spec.atom_column >= 0 and n_raw > spec.atom_column:
            out[k] = raw[spec.atom_column] * spec.scale
            continue
        value = raw[spec.column]
        if spec.atomize:
            ref = sum(ATOMREF_HA.get(int(z), 0.0) for z in atomic_numbers)
            value = value - ref
        out[k] = value * spec.scale
    return out.astype(np.float32)


def verify_target_columns(raw: np.ndarray) -> str | None:
    """Sanity check of the assumed column layout; returns an error string."""
    if raw.shape[0] < 15:
        return f"only {raw.shape[0]} raw values (expected >= 15)"
    homo, lumo, gap = raw[5], raw[6], raw[7]
    if abs((lumo - homo) - gap) > 1e-3:
        return f"gap {gap:.6f} != lumo-homo {lumo - homo:.6f}"
    return None


def iter_qm9_dir(raw_dir: str, prefer_rdkit: bool = True, limit: int | None = None):
    """Yield MolFeatures + targets for every .xyz file in `raw_dir`."""
    names = sorted(f for f in os.listdir(raw_dir) if f.endswith(".xyz"))
    if limit is not None:
        names = names[:limit]
    for name in names:
        path = os.path.join(raw_dir, name)
        index, z, pos, raw, _ = parse_qm9_file(path)
        problem = verify_target_columns(raw)
        if problem is not None:
            raise ValueError(f"{name}: {problem}")
        feats = featurize_molecule(pos, z, prefer_rdkit=prefer_rdkit)
        yield {
            "index": index,
            "feats": feats,
            "targets": target_values(raw, z),
            "raw": raw,
        }


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
CACHE_ARRAYS = (
    "z", "atom_class", "atom_fields", "pos", "atom_offsets",
    "bond_pairs", "bond_fields", "bond_class", "bond_offsets",
    "spd", "spd_offsets", "spd_line", "spd_line_offsets",
    "line_shared", "line_shared_offsets", "targets", "index",
)


def build_line_graph_arrays(feats: MolFeatures):
    """Precompute the bond-centric graph arrays for one molecule."""
    pairs = feats.bond_pairs
    m = pairs.shape[0]
    shared = chem.shared_atom_matrix(pairs)
    line_adj = shared >= 0
    spd_line = chem.shortest_path_distances(line_adj).astype(np.int8) if m else np.zeros((0, 0), np.int8)
    return shared.astype(np.int16), spd_line


def write_cache(records, out_dir: str) -> dict:
    """Materialise featurised molecules into memory-mappable .npy arrays."""
    os.makedirs(out_dir, exist_ok=True)
    z, atom_class, atom_fields, pos = [], [], [], []
    bond_pairs, bond_fields, bond_class = [], [], []
    spd, spd_line, line_shared = [], [], []
    targets, index = [], []
    atom_off = [0]
    bond_off = [0]
    spd_off = [0]
    spd_line_off = [0]
    line_off = [0]
    for rec in records:
        feats: MolFeatures = rec["feats"]
        n = feats.atomic_numbers.shape[0]
        m = feats.bond_pairs.shape[0]
        z.append(feats.atomic_numbers.astype(np.int16))
        atom_class.append(feats.atom_class.astype(np.int8))
        atom_fields.append(feats.atom_fields.astype(np.int8))
        pos.append(feats.pos.astype(np.float32))
        bond_pairs.append(feats.bond_pairs.astype(np.int16))
        bond_fields.append(feats.bond_fields.astype(np.int8))
        bond_class.append(feats.bond_class.astype(np.int8))
        spd.append(feats.spd.astype(np.int8))
        shared, spd_l = build_line_graph_arrays(feats)
        line_shared.append(shared)
        spd_line.append(spd_l)
        targets.append(rec["targets"])
        index.append(rec["index"])
        atom_off.append(atom_off[-1] + n)
        bond_off.append(bond_off[-1] + m)
        spd_off.append(spd_off[-1] + n * n)
        spd_line_off.append(spd_line_off[-1] + m * m)
        line_off.append(line_off[-1] + m * m)

    arrays = {
        "z": np.concatenate(z) if z else np.zeros(0, np.int16),
        "atom_class": np.concatenate(atom_class) if atom_class else np.zeros(0, np.int8),
        "atom_fields": np.concatenate(atom_fields) if atom_fields else np.zeros((0, 0), np.int8),
        "pos": np.concatenate(pos) if pos else np.zeros((0, 3), np.float32),
        "atom_offsets": np.asarray(atom_off, dtype=np.int64),
        "bond_pairs": np.concatenate(bond_pairs) if bond_pairs else np.zeros((0, 2), np.int16),
        "bond_fields": np.concatenate(bond_fields) if bond_fields else np.zeros((0, 0), np.int8),
        "bond_class": np.concatenate(bond_class) if bond_class else np.zeros(0, np.int8),
        "bond_offsets": np.asarray(bond_off, dtype=np.int64),
        "spd": np.concatenate([a.ravel() for a in spd]) if spd else np.zeros(0, np.int8),
        "spd_offsets": np.asarray(spd_off, dtype=np.int64),
        "spd_line": np.concatenate([a.ravel() for a in spd_line]) if spd_line else np.zeros(0, np.int8),
        "spd_line_offsets": np.asarray(spd_line_off, dtype=np.int64),
        "line_shared": np.concatenate([a.ravel() for a in line_shared]) if line_shared else np.zeros(0, np.int16),
        "line_shared_offsets": np.asarray(line_off, dtype=np.int64),
        "targets": np.stack(targets) if targets else np.zeros((0, N_TARGETS), np.float32),
        "index": np.asarray(index, dtype=np.int64),
    }
    for name, arr in arrays.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)
    stats = {
        "n_molecules": len(index),
        "n_atoms": int(arrays["atom_offsets"][-1]),
        "n_bonds": int(arrays["bond_offsets"][-1]),
        "max_atoms": int(np.diff(arrays["atom_offsets"]).max()) if index else 0,
        "max_bonds": int(np.diff(arrays["bond_offsets"]).max()) if index else 0,
    }
    import json
    with open(os.path.join(out_dir, "cache_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    return stats


def make_split(n: int, seed: int = 42, n_val: int = N_VAL, n_test: int = N_TEST) -> np.ndarray:
    """0 = train, 1 = validation, 2 = test (random, reproducible)."""
    split = np.zeros(n, dtype=np.int8)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split[perm[:n_val]] = 1
    split[perm[n_val : n_val + n_test]] = 2
    return split


# --------------------------------------------------------------------------
# subset slicing -- cut a self-contained cache out of an existing one
# --------------------------------------------------------------------------
# Which offsets array governs each ragged data array.
CACHE_OFFSETS = {
    "z": "atom_offsets",
    "atom_class": "atom_offsets",
    "atom_fields": "atom_offsets",
    "pos": "atom_offsets",
    "bond_pairs": "bond_offsets",
    "bond_fields": "bond_offsets",
    "bond_class": "bond_offsets",
    "spd": "spd_offsets",
    "spd_line": "spd_line_offsets",
    "line_shared": "line_shared_offsets",
}
# One row per molecule, no offsets involved.
CACHE_PER_MOLECULE = ("targets", "index")
CACHE_OFFSET_ARRAYS = (
    "atom_offsets", "bond_offsets", "spd_offsets",
    "spd_line_offsets", "line_shared_offsets",
)


def load_cache_arrays(cache_dir: str) -> dict:
    """Memory-map every array of a cache directory (fails loudly if incomplete)."""
    missing = [
        name for name in CACHE_ARRAYS
        if not os.path.exists(os.path.join(cache_dir, f"{name}.npy"))
    ]
    if missing:
        raise FileNotFoundError(
            f"{cache_dir} 不是完整的 DeMol 缓存，缺少 {len(missing)} 个数组: {missing[:4]}"
        )
    return {
        name: np.load(os.path.join(cache_dir, f"{name}.npy"), mmap_mode="r")
        for name in CACHE_ARRAYS
    }


def _rebuild_offsets(offsets: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Offsets of the sliced cache: prefix sums of the selected slice lengths."""
    off = np.asarray(offsets)
    sizes = off[rows + 1] - off[rows]
    new_offsets = np.zeros(rows.shape[0] + 1, dtype=np.int64)
    np.cumsum(sizes, out=new_offsets[1:])
    return new_offsets


def _gather_ragged(arr: np.ndarray, offsets: np.ndarray, rows: np.ndarray):
    """Concatenate ``arr`` slices selected by ``rows``; returns (data, new_offsets)."""
    starts = np.asarray(offsets)[rows]
    sizes = np.asarray(offsets)[rows + 1] - starts
    new_offsets = _rebuild_offsets(offsets, rows)
    total = int(new_offsets[-1])
    if total == 0:
        return np.zeros((0,) + tuple(arr.shape[1:]), dtype=arr.dtype), new_offsets
    width = int(sizes.max())
    slots = np.arange(width, dtype=np.int64)
    positions = starts[:, None] + slots[None, :]
    keep = slots[None, :] < sizes[:, None]
    return np.asarray(arr[positions[keep]]), new_offsets


def slice_cache(
    cache_dir: str,
    out_dir: str,
    rows,
    split: np.ndarray | None = None,
) -> dict:
    """Write a standalone cache holding only the molecules listed in ``rows``.

    Offsets are rebuilt from scratch, so the result is a valid cache on its own --
    it can be memory-mapped by :class:`QM9Dataset` exactly like a full one.  This
    is what ``scripts/pack_subset.py`` uses to build the subset that gets pushed
    to ModelScope and pulled down on the training server.

    ``rows`` may be in any order; it is deduplicated and sorted, because the
    offset arrays require increasing molecule indices.  ``split`` (when given)
    must already be aligned with the *sliced* rows and is written to ``split.npy``.
    """
    arrays = load_cache_arrays(cache_dir)
    rows = np.unique(np.asarray(rows, dtype=np.int64).reshape(-1))
    n_total = int(len(arrays["index"]))
    if rows.size == 0:
        raise ValueError("rows 不能为空")
    if rows.min() < 0 or rows.max() >= n_total:
        raise IndexError(f"rows 超出范围: [{int(rows.min())}, {int(rows.max())}] / {n_total}")
    if split is not None:
        split = np.asarray(split, dtype=np.int8).reshape(-1)
        if split.shape[0] != rows.shape[0]:
            raise ValueError(f"split 长度 {split.shape[0]} != rows 长度 {rows.shape[0]}")

    out: dict[str, np.ndarray] = {}
    for name, off_name in CACHE_OFFSETS.items():
        data, new_offsets = _gather_ragged(arrays[name], arrays[off_name], rows)
        out[name] = data
        previous = out.get(off_name)
        if previous is None:
            out[off_name] = new_offsets
        elif not np.array_equal(previous, new_offsets):
            raise ValueError(f"{name} 与 {off_name} 不自洽，源缓存可能已损坏")
    for name in CACHE_PER_MOLECULE:
        out[name] = np.asarray(arrays[name][rows])
    for name in CACHE_OFFSET_ARRAYS:
        if name not in out:  # 没有关联数据数组时也要把 offsets 切对
            out[name] = _rebuild_offsets(arrays[name], rows)

    os.makedirs(out_dir, exist_ok=True)
    for name in CACHE_ARRAYS:
        np.save(os.path.join(out_dir, f"{name}.npy"), out[name])
    if split is not None:
        np.save(os.path.join(out_dir, "split.npy"), split)

    stats = {
        "n_molecules": int(rows.shape[0]),
        "n_atoms": int(out["atom_offsets"][-1]),
        "n_bonds": int(out["bond_offsets"][-1]),
        "max_atoms": int(np.diff(out["atom_offsets"]).max()),
        "max_bonds": int(np.diff(out["bond_offsets"]).max()),
    }
    src_stats_path = os.path.join(cache_dir, "cache_stats.json")
    merged: dict = {}
    if os.path.exists(src_stats_path):
        import json
        with open(src_stats_path, "r", encoding="utf-8") as fh:
            src = json.load(fh)
        for key in ("backend", "bond_threshold_alpha", "n_targets", "targets"):
            if key in src:
                merged[key] = src[key]
    merged.update(stats)
    merged["sliced_from"] = os.path.abspath(cache_dir)
    if split is not None:
        counts = np.bincount(split, minlength=3)
        merged["split_counts"] = {
            "train": int(counts[0]), "val": int(counts[1]), "test": int(counts[2])
        }
    import json
    with open(os.path.join(out_dir, "cache_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    return merged


class QM9Dataset(torch.utils.data.Dataset):
    """Memory-mapped QM9 cache; yields ragged numpy dicts for `qm9_collate`."""

    def __init__(self, cache_dir: str, split_ids: tuple[int, ...] = (0,)):
        self.dir = cache_dir
        self.arrays = {
            name: np.load(os.path.join(cache_dir, f"{name}.npy"), mmap_mode="r")
            for name in CACHE_ARRAYS
        }
        split_path = os.path.join(cache_dir, "split.npy")
        if os.path.exists(split_path):
            split = np.load(split_path)
        else:
            split = make_split(len(self.arrays["index"]))
            np.save(split_path, split)
        self.split_ids = tuple(split_ids)
        self.rows = np.flatnonzero(np.isin(split, self.split_ids)).astype(np.int64)

    def close(self) -> None:
        """Release the memory-map handles.

        On Windows an open ``mmap_mode="r"`` mapping keeps the ``.npy`` file locked,
        so ``shutil.rmtree`` on the cache directory fails (silently, when
        ``ignore_errors=True``).  Call this before deleting a cache you created --
        the smoke test and the random probe both do.
        """
        for arr in self.arrays.values():
            mapping = getattr(arr, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self.arrays = {}
        self.rows = np.zeros(0, dtype=np.int64)

    def __enter__(self) -> "QM9Dataset":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def __getitem__(self, k: int) -> dict:
        row = int(self.rows[k])
        a0, a1 = self.arrays["atom_offsets"][row], self.arrays["atom_offsets"][row + 1]
        b0, b1 = self.arrays["bond_offsets"][row], self.arrays["bond_offsets"][row + 1]
        n, m = int(a1 - a0), int(b1 - b0)
        s0, s1 = self.arrays["spd_offsets"][row], self.arrays["spd_offsets"][row + 1]
        l0, l1 = self.arrays["spd_line_offsets"][row], self.arrays["spd_line_offsets"][row + 1]
        h0, h1 = self.arrays["line_shared_offsets"][row], self.arrays["line_shared_offsets"][row + 1]
        return {
            "n": n,
            "m": m,
            "z": np.asarray(self.arrays["z"][a0:a1], dtype=np.int64),
            "atom_class": np.asarray(self.arrays["atom_class"][a0:a1], dtype=np.int64),
            "atom_fields": np.asarray(self.arrays["atom_fields"][a0:a1], dtype=np.int64),
            "pos": np.asarray(self.arrays["pos"][a0:a1], dtype=np.float32),
            "bond_pairs": np.asarray(self.arrays["bond_pairs"][b0:b1], dtype=np.int64),
            "bond_fields": np.asarray(self.arrays["bond_fields"][b0:b1], dtype=np.int64),
            "bond_class": np.asarray(self.arrays["bond_class"][b0:b1], dtype=np.int64),
            "spd": np.asarray(self.arrays["spd"][s0:s1], dtype=np.int64).reshape(n, n),
            "spd_line": np.asarray(self.arrays["spd_line"][l0:l1], dtype=np.int64).reshape(m, m),
            "line_shared": np.asarray(self.arrays["line_shared"][h0:h1], dtype=np.int64).reshape(m, m),
            "targets": np.asarray(self.arrays["targets"][row], dtype=np.float32),
        }


# --------------------------------------------------------------------------
# dense batching
# --------------------------------------------------------------------------
def _pad_stack(items: list[np.ndarray], shape_tail: tuple[int, ...], dtype) -> np.ndarray:
    """Stack ragged arrays with zero padding to the batch maximum."""
    sizes = [max(1, it.shape[0]) for it in items]
    length = max(sizes) if sizes else 1
    out = np.zeros((len(items), length) + shape_tail, dtype=dtype)
    for b, it in enumerate(items):
        if it.shape[0]:
            out[b, : it.shape[0]] = it
    return out


def qm9_collate(batch: list[dict], atom_cutoff: float = chem.ATOM_MASK_CUTOFF_ANGSTROM) -> dict:
    """Build the dense padded batch consumed by :class:`demol.model.DeMol`."""
    b = len(batch)
    n_max = max(it["n"] for it in batch)
    m_max = max(it["m"] for it in batch)

    atom_fields = np.zeros((b, n_max, batch[0]["atom_fields"].shape[1]), dtype=np.int64)
    atom_class = np.zeros((b, n_max), dtype=np.int64)
    atom_mask = np.zeros((b, n_max), dtype=bool)
    pos = np.zeros((b, n_max, 3), dtype=np.float32)
    spd = np.zeros((b, n_max, n_max), dtype=np.int64)
    aa_mask = np.zeros((b, n_max, n_max), dtype=bool)
    bond_gt = np.zeros((b, n_max, n_max), dtype=bool)

    bond_fields = np.zeros((b, m_max, batch[0]["bond_fields"].shape[1]), dtype=np.int64)
    bond_class = np.zeros((b, m_max), dtype=np.int64)
    bond_mask = np.zeros((b, m_max), dtype=bool)
    bond_idx = np.zeros((b, m_max, 2), dtype=np.int64)
    spd_line = np.zeros((b, m_max, m_max), dtype=np.int64)
    bb_mask = np.zeros((b, m_max, m_max), dtype=bool)
    ab_mask = np.zeros((b, n_max, m_max), dtype=bool)
    # 目标宽度跟数据走（QM9 是 12，PCQ 微调缓存是 1），不再写死 N_TARGETS
    targets = np.zeros((b, batch[0]["targets"].shape[0]), dtype=np.float32)

    for k, it in enumerate(batch):
        n, m = it["n"], it["m"]
        atom_fields[k, :n] = it["atom_fields"]
        atom_class[k, :n] = it["atom_class"]
        atom_mask[k, :n] = True
        pos[k, :n] = it["pos"]
        spd[k, :n, :n] = it["spd"]
        targets[k] = it["targets"]

        adj = np.zeros((n, n), dtype=bool)
        if m:
            adj[it["bond_pairs"][:, 0], it["bond_pairs"][:, 1]] = True
            adj[it["bond_pairs"][:, 1], it["bond_pairs"][:, 0]] = True
        dist = chem.pairwise_distances(it["pos"])
        near = dist < atom_cutoff
        eye_n = np.eye(n, dtype=bool)
        aa = (near | adj | eye_n)
        aa_mask[k, :n, :n] = aa
        bond_gt[k, :n, :n] = adj

        bond_fields[k, :m] = it["bond_fields"]
        bond_class[k, :m] = it["bond_class"]
        bond_mask[k, :m] = True
        if m:
            bond_idx[k, :m] = it["bond_pairs"]
            spd_line[k, :m, :m] = it["spd_line"]
            line_adj = it["line_shared"] >= 0
            conj = it["bond_fields"][:, 1] == 2
            two_hop = line_adj @ line_adj
            conj_pair = two_hop & conj[:, None] & conj[None, :]
            bb = line_adj | conj_pair | np.eye(m, dtype=bool)
            bb_mask[k, :m, :m] = bb
            # atom-bond incidence: bond (u, v) is incident to atoms u and v
            inc = np.zeros((n, m), dtype=bool)
            inc[it["bond_pairs"][:, 0], np.arange(m)] = True
            inc[it["bond_pairs"][:, 1], np.arange(m)] = True
            ab_mask[k, :n, :m] = inc

    to_t = lambda arr: torch.from_numpy(np.ascontiguousarray(arr))  # noqa: E731
    return {
        "atom_fields": to_t(atom_fields),
        "atom_class": to_t(atom_class),
        "atom_mask": to_t(atom_mask),
        "pos": to_t(pos),
        "spd": to_t(spd),
        "aa_mask": to_t(aa_mask),
        "bond_gt": to_t(bond_gt),
        "bond_fields": to_t(bond_fields),
        "bond_class": to_t(bond_class),
        "bond_mask": to_t(bond_mask),
        "bond_idx": to_t(bond_idx),
        "spd_line": to_t(spd_line),
        "bb_mask": to_t(bb_mask),
        "ab_mask": to_t(ab_mask),
        "targets": to_t(targets),
    }
