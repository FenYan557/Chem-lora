"""Atom / bond featurisation for DeMol.

The paper only says that "raw atom feature x_i are embedded into h_i^(a,0)" and
that bond features e_ij are embedded the same way, so we use the standard
categorical field set from Graphormer / Transformer-M (element, degree, charge,
num-H, hybridisation, aromaticity, ring membership) and (bond type, conjugation,
ring membership, endpoint elements).  Every field is an index into its own
embedding table; index 0 is reserved for padding.

Two perception back-ends are provided:

* ``rdkit``   - builds a mol from the DFT geometry with
  ``rdDetermineBonds.DetermineBonds`` (xyz2mol), giving exact bond orders,
  aromaticity, conjugation and hybridisation.  Used automatically when rdkit is
  importable (the GPU server has it).
* ``geometry`` - pure numpy fallback for machines without rdkit (the local
  `deeplearn` env).  Bond orders come from length ratios relative to the sum of
  covalent radii, rings from a cycle basis, hybridisation from the pi-bond count.

Both back-ends produce identical output shapes and the same vocabulary, so a
cache built locally can be re-built on the server without touching the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import chem

# --------------------------------------------------------------------------
# Vocabularies.  Field value 0 == PAD, so real values start at 1.
# --------------------------------------------------------------------------
# 元素词表，**只允许追加，不允许重排**：QM9 检查点/缓存里的类别索引按旧
# 顺序解释，append-only 保证旧索引语义不变（OTHER 固定在索引 5）。
# 后 8 个是 PCQM4Mv2 里出现而 QM9 没有的元素（LoRA 微调用）。
ATOM_VOCAB = (
    "H", "C", "N", "O", "F", "OTHER",
    "B", "Si", "P", "S", "Cl", "Br", "I", "Se",
)

ATOM_FIELDS: tuple[tuple[str, int], ...] = (
    ("type", len(ATOM_VOCAB) + 1),        # 7
    ("degree", 8),                        # 0..6 + pad
    ("charge", 4),                        # -1, 0, +1 + pad
    ("num_h", 6),                         # 0..4 + pad
    ("hybrid", 6),                        # unk, s, sp, sp2, sp3 + pad
    ("aromatic", 3),                      # no, yes + pad
    ("ring", 3),                          # no, yes + pad
)

BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

BOND_FIELDS: tuple[tuple[str, int], ...] = (
    ("type", len(BOND_TYPES) + 1),        # 5
    ("conjugated", 3),                    # no, yes + pad
    ("ring", 3),                          # no, yes + pad
    ("atom_i", len(ATOM_VOCAB) + 1),      # 7  (endpoint with the smaller index)
    ("atom_j", len(ATOM_VOCAB) + 1),      # 7
)

ATOM_FIELD_NAMES = tuple(name for name, _ in ATOM_FIELDS)
BOND_FIELD_NAMES = tuple(name for name, _ in BOND_FIELDS)
ATOM_FIELD_SIZES = tuple(size for _, size in ATOM_FIELDS)
BOND_FIELD_SIZES = tuple(size for _, size in BOND_FIELDS)

# Element -> class index used for the Gaussian-basis alpha/beta lookup.
_Z_TO_CLASS = {
    1: 0, 6: 1, 7: 2, 8: 3, 9: 4,          # H C N O F
    5: 6, 14: 7, 15: 8, 16: 9, 17: 10, 35: 11, 53: 12, 34: 13,  # B Si P S Cl Br I Se
}
N_ATOM_CLASSES = len(ATOM_VOCAB)
N_BOND_CLASSES = len(BOND_TYPES) + 1     # + pad/unknown

# 词表扩展前的旧布局（QM9 预训练检查点）：零样本跨数据集评估时把
# 新元素类别降级回 OTHER。
LEGACY_ATOM_VOCAB_SIZE = 6               # ("H","C","N","O","F","OTHER")
OTHER_CLASS = ATOM_VOCAB.index("OTHER")  # 5

HYBRID_UNKNOWN, HYBRID_S, HYBRID_SP, HYBRID_SP2, HYBRID_SP3 = 1, 2, 3, 4, 5


@dataclass
class MolFeatures:
    """Featurised molecule (ragged numpy arrays, no padding)."""

    atomic_numbers: np.ndarray      # (N,) int64
    atom_class: np.ndarray          # (N,) int64 in [0, N_ATOM_CLASSES)
    atom_fields: np.ndarray         # (N, n_atom_fields) int16
    pos: np.ndarray                 # (N, 3) float32, Angstrom
    bond_pairs: np.ndarray          # (M, 2) int32, i < j
    bond_fields: np.ndarray         # (M, n_bond_fields) int16
    bond_class: np.ndarray          # (M,) int64 in [0, N_BOND_CLASSES)
    spd: np.ndarray                 # (N, N) int16 shortest path distance
    backend: str = "geometry"       # "rdkit" | "geometry"


def atom_class_of(atomic_numbers: np.ndarray) -> np.ndarray:
    z = np.asarray(atomic_numbers)
    out = np.full(z.shape, OTHER_CLASS, dtype=np.int64)           # OTHER
    for zi, cls in _Z_TO_CLASS.items():
        out[z == zi] = cls
    return out


def rdkit_available() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import rdDetermineBonds  # noqa: F401
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------
# rdkit back-end
# --------------------------------------------------------------------------
def featurize_with_rdkit(pos: np.ndarray, atomic_numbers: np.ndarray) -> MolFeatures | None:
    """Perceive bonds/orders from the DFT geometry using rdkit (xyz2mol)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except Exception:
        return None

    n = len(atomic_numbers)
    xyz_block = [f"{n}", "qm9"]
    symbols = [chem.ELEMENT_SYMBOL.get(int(z), "X") for z in atomic_numbers]
    for sym, (x, y, zc) in zip(symbols, np.asarray(pos, dtype=np.float64)):
        xyz_block.append(f"{sym} {x:.8f} {y:.8f} {zc:.8f}")
    mol = Chem.MolFromXYZBlock("\n".join(xyz_block))
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        try:
            rdDetermineBonds.DetermineConnectivity(mol)
        except Exception:
            return None

    atom_fields = np.zeros((n, len(ATOM_FIELDS)), dtype=np.int16)
    atom_class = atom_class_of(atomic_numbers)
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() != int(atomic_numbers[idx]):
            return None                     # ordering mismatch -> bail out
        atom_fields[idx, 0] = atom_class[idx] + 1
        atom_fields[idx, 1] = min(atom.GetDegree(), 6) + 1
        charge = atom.GetFormalCharge()
        atom_fields[idx, 2] = 2 if charge == 0 else (1 if charge < 0 else 3)
        atom_fields[idx, 3] = min(atom.GetTotalNumHs(), 4) + 1
        hyb = str(atom.GetHybridization())
        atom_fields[idx, 4] = {
            "S": HYBRID_S, "SP": HYBRID_SP, "SP2": HYBRID_SP2, "SP3": HYBRID_SP3
        }.get(hyb, HYBRID_UNKNOWN)
        atom_fields[idx, 5] = 2 if atom.GetIsAromatic() else 1
        atom_fields[idx, 6] = 2 if atom.IsInRing() else 1

    bonds = list(mol.GetBonds())
    pairs = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in bonds], dtype=np.int32)
    if pairs.size:
        pairs = np.sort(pairs, axis=1)
        order_key = pairs[:, 0] * n + pairs[:, 1]
        pairs = pairs[np.argsort(order_key)]
        bonds = [bonds[i] for i in np.argsort(order_key)]
    bond_fields = np.zeros((pairs.shape[0], len(BOND_FIELDS)), dtype=np.int16)
    bond_class = np.zeros((pairs.shape[0],), dtype=np.int64)
    for k, b in enumerate(bonds):
        bt = str(b.GetBondType())
        cls = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}.get(bt, 0)
        bond_class[k] = cls
        bond_fields[k, 0] = cls
        bond_fields[k, 1] = 2 if b.GetIsConjugated() else 1
        bond_fields[k, 2] = 2 if b.IsInRing() else 1
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        bond_fields[k, 3] = atom_class[i] + 1
        bond_fields[k, 4] = atom_class[j] + 1

    adj = chem.adjacency_from_pairs(pairs, n)
    return MolFeatures(
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int64),
        atom_class=atom_class,
        atom_fields=atom_fields,
        pos=np.asarray(pos, dtype=np.float32),
        bond_pairs=pairs,
        bond_fields=bond_fields,
        bond_class=bond_class,
        spd=chem.shortest_path_distances(adj),
        backend="rdkit",
    )


# --------------------------------------------------------------------------
# pure-numpy back-end
# --------------------------------------------------------------------------
def _fallback_hybridization(
    atom_class: np.ndarray, pi_count: np.ndarray, aromatic: np.ndarray
) -> np.ndarray:
    hyb = np.full(pi_count.shape, HYBRID_UNKNOWN, dtype=np.int16)
    heavy = atom_class != 0                     # not hydrogen
    hyb[heavy & (pi_count >= 2)] = HYBRID_SP
    sp2 = heavy & (pi_count == 1) & ~aromatic
    hyb[sp2] = HYBRID_SP2
    hyb[heavy & aromatic] = HYBRID_SP2
    hyb[heavy & (pi_count == 0) & ~aromatic] = HYBRID_SP3
    return hyb


def featurize_from_geometry(pos: np.ndarray, atomic_numbers: np.ndarray) -> MolFeatures:
    """Algorithm 1 bond perception + length-ratio bond orders (no rdkit)."""
    pos = np.asarray(pos, dtype=np.float32)
    z = np.asarray(atomic_numbers, dtype=np.int64)
    n = z.shape[0]
    pairs = chem.perceive_bonds(pos, z)
    adj = chem.adjacency_from_pairs(pairs, n)
    atom_class = atom_class_of(z)

    degree = adj.sum(1).astype(np.int64)
    num_h = np.zeros(n, dtype=np.int64)
    for h_idx in np.flatnonzero(z == 1):
        for nb in np.flatnonzero(adj[h_idx]):
            num_h[nb] += 1

    ratio = chem.bond_length_ratio(pos, pairs, z)
    order = chem.bond_order_from_ratio(ratio)          # 1,2,3 or 4=aromatic
    atom_ring, bond_ring = chem.ring_membership(adj)

    pi_count = np.zeros(n, dtype=np.int64)
    for (i, j), o in zip(pairs, order):
        if o == 4:
            pi_count[i] += 1
            pi_count[j] += 1
        elif o > 1:
            pi_count[i] += o - 1
            pi_count[j] += o - 1
    aromatic_bond = order == 4
    atom_aromatic = np.zeros(n, dtype=bool)
    for (i, j), aro in zip(pairs, aromatic_bond):
        if aro:
            atom_aromatic[i] = True
            atom_aromatic[j] = True

    # Conjugation: aromatic bonds, plus multiple bonds separated by one single
    # bond from another multiple/aromatic bond (the classic alternating pattern).
    m = pairs.shape[0]
    multiple = (order >= 2) | aromatic_bond
    conjugated = aromatic_bond.copy()
    if m:
        line_adj = chem.line_graph_adjacency(pairs)
        for b in range(m):
            if not multiple[b]:
                continue
            for s in np.flatnonzero(line_adj[b]):
                s = int(s)
                if multiple[s] or order[s] != 1:
                    continue
                for c in np.flatnonzero(line_adj[s]):
                    c = int(c)
                    if c != b and multiple[c]:
                        conjugated[b] = True
                        conjugated[s] = True
                        conjugated[c] = True

    atom_fields = np.zeros((n, len(ATOM_FIELDS)), dtype=np.int16)
    atom_fields[:, 0] = atom_class + 1
    atom_fields[:, 1] = np.clip(degree, 0, 6) + 1
    atom_fields[:, 2] = 2                                   # neutral
    atom_fields[:, 3] = np.clip(num_h, 0, 4) + 1
    atom_fields[:, 4] = _fallback_hybridization(atom_class, pi_count, atom_aromatic)
    atom_fields[:, 5] = np.where(atom_aromatic, 2, 1)
    atom_fields[:, 6] = np.where(atom_ring, 2, 1)

    bond_fields = np.zeros((m, len(BOND_FIELDS)), dtype=np.int16)
    if m:
        bond_fields[:, 0] = order
        bond_fields[:, 1] = np.where(conjugated, 2, 1)
        in_ring = np.array([bond_ring[i, j] for i, j in pairs], dtype=bool)
        bond_fields[:, 2] = np.where(in_ring, 2, 1)
        bond_fields[:, 3] = atom_class[pairs[:, 0]] + 1
        bond_fields[:, 4] = atom_class[pairs[:, 1]] + 1

    return MolFeatures(
        atomic_numbers=z,
        atom_class=atom_class,
        atom_fields=atom_fields,
        pos=pos,
        bond_pairs=pairs.astype(np.int32),
        bond_fields=bond_fields,
        bond_class=order.astype(np.int64) if m else np.zeros((0,), dtype=np.int64),
        spd=chem.shortest_path_distances(adj),
        backend="geometry",
    )


def featurize_molecule(
    pos: np.ndarray,
    atomic_numbers: np.ndarray,
    prefer_rdkit: bool = True,
) -> MolFeatures:
    """Featurise one molecule, falling back to the geometry back-end on error."""
    if prefer_rdkit and rdkit_available():
        feats = featurize_with_rdkit(pos, atomic_numbers)
        if feats is not None:
            return feats
    return featurize_from_geometry(pos, atomic_numbers)
