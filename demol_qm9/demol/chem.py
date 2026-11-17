"""Chemistry primitives: covalent radii, bond perception, geometry.

Implements Appendix B "Algorithm 1: Bond Prediction based on Covalent Radii"
of the DeMol paper, plus the geometric quantities used by the bond-centric
channel (bond midpoints, bond angles cos(theta_ijk), inter-bond cos(phi_ijkl)).

Everything here is pure numpy so it runs identically on the laptop (no rdkit in
the local `deeplearn` conda env) and on the GPU server.
"""

from __future__ import annotations

import numpy as np

# --- Bond perception threshold (paper: alpha = 1.15, after Lu & Chen 2012) ---
BOND_THRESHOLD_ALPHA = 1.15

# Cordero et al. (2008) covalent radii in Angstrom, keyed by atomic number.
COVALENT_RADII = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66,
    9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60,
    23: 1.53, 24: 1.39, 25: 1.50, 26: 1.42, 27: 1.38, 28: 1.24, 29: 1.32,
    30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16,
    44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 78: 1.36, 79: 1.36,
}
DEFAULT_COVALENT_RADIUS = 1.5

ELEMENT_SYMBOL = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 35: "Br", 53: "I",
}
SYMBOL_TO_Z = {v: k for k, v in ELEMENT_SYMBOL.items()}

# Structure-aware mask cutoffs (paper Section 3.2, "Structure-aware Mask").
ATOM_MASK_CUTOFF_ANGSTROM = 5.0   # keep pairs closer than 5 A (weak interactions)
BOND_CANDIDATE_CUTOFF_ANGSTROM = 5.0


def covalent_radius(z: np.ndarray) -> np.ndarray:
    """Vectorised lookup of covalent radii for an array of atomic numbers."""
    z = np.asarray(z)
    out = np.full(z.shape, DEFAULT_COVALENT_RADIUS, dtype=np.float32)
    for zi, r in COVALENT_RADII.items():
        out[z == zi] = r
    return out


def pairwise_distances(pos: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix of shape (N, N) for pos of shape (N, 3)."""
    pos = np.asarray(pos, dtype=np.float64)
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff * diff).sum(-1)).astype(np.float32)


def perceive_bonds(
    pos: np.ndarray,
    atomic_numbers: np.ndarray,
    alpha: float = BOND_THRESHOLD_ALPHA,
) -> np.ndarray:
    """Algorithm 1 of the paper: bond if d_ij <= alpha * (r_i + r_j).

    Returns an (M, 2) int32 array of atom index pairs with i < j.
    """
    pos = np.asarray(pos, dtype=np.float32)
    z = np.asarray(atomic_numbers)
    n = z.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int32)
    radii = covalent_radius(z)
    thresh = alpha * (radii[:, None] + radii[None, :])
    dist = pairwise_distances(pos)
    adj = dist <= thresh
    np.fill_diagonal(adj, False)
    iu, ju = np.triu_indices(n, k=1)
    keep = adj[iu, ju]
    return np.stack([iu[keep], ju[keep]], axis=1).astype(np.int32)


def adjacency_from_pairs(pairs: np.ndarray, n_atoms: int) -> np.ndarray:
    """Boolean (N, N) adjacency matrix from an (M, 2) pair array."""
    adj = np.zeros((n_atoms, n_atoms), dtype=bool)
    if pairs.shape[0] > 0:
        i, j = pairs[:, 0], pairs[:, 1]
        adj[i, j] = True
        adj[j, i] = True
    return adj


def bond_length_ratio(
    pos: np.ndarray, pairs: np.ndarray, atomic_numbers: np.ndarray
) -> np.ndarray:
    """d_ij / (r_i + r_j) for every bond; ~1.0 single, ~0.91 aromatic,
    ~0.87 double, ~0.79 triple for carbon-carbon bonds."""
    if pairs.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float32)
    radii = covalent_radius(np.asarray(atomic_numbers))
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(pos[i] - pos[j], axis=1)
    return (d / np.maximum(radii[i] + radii[j], 1e-6)).astype(np.float32)


# Ratio bands used when rdkit is unavailable (see features.py for the mapping
# to categorical bond types).  Calibrated on QM9 DFT geometries:
#   C-C 1.53 -> 1.007, aromatic C-C 1.39 -> 0.914, C=C 1.33 -> 0.875,
#   C#C 1.20 -> 0.789, C=O 1.21 -> 0.852, C-O 1.43 -> 1.007.
BOND_ORDER_BANDS = (
    (0.825, 3),    # triple
    (0.895, 2),    # double
    (0.965, 4),    # aromatic / 1.5
)


def bond_order_from_ratio(ratio: np.ndarray) -> np.ndarray:
    """Map length ratios to {1, 2, 3, 4=aromatic} without a chemistry toolkit."""
    order = np.ones_like(np.asarray(ratio, dtype=np.float32), dtype=np.int8)
    for cutoff, value in sorted(BOND_ORDER_BANDS, key=lambda kv: kv[0]):
        order[ratio < cutoff] = value
    return order


def fundamental_cycles(adj: np.ndarray) -> list[list[int]]:
    """Dependency-free cycle basis: BFS spanning tree + one back edge each."""
    n = adj.shape[0]
    if n == 0:
        return []
    seen = np.zeros(n, dtype=bool)
    parent = -np.ones(n, dtype=np.int64)
    tree_edges: set[tuple[int, int]] = set()
    used_back_edges: set[tuple[int, int]] = set()
    cycles: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in np.flatnonzero(adj[u]):
                v = int(v)
                if not seen[v]:
                    seen[v] = True
                    parent[v] = u
                    tree_edges.add((min(u, v), max(u, v)))
                    queue.append(v)
                elif parent[u] != v:
                    key = (min(u, v), max(u, v))
                    if key in tree_edges or key in used_back_edges:
                        continue
                    used_back_edges.add(key)
                    cycles.append(_cycle_path(parent, u, v))
    return [c for c in cycles if len(c) >= 3]


def _cycle_path(parent: np.ndarray, u: int, v: int) -> list[int]:
    def path_to_root(x: int) -> list[int]:
        out = [x]
        while parent[out[-1]] != -1:
            out.append(int(parent[out[-1]]))
        return out

    pu, pv = path_to_root(u), path_to_root(v)
    su, sv = set(pu), set(pv)
    lca = next(x for x in pu if x in sv)
    cycle = pu[: pu.index(lca) + 1]
    cycle += pv[: pv.index(lca)][::-1]
    return cycle


def ring_membership(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (atom_in_ring (N,), bond_in_ring (N, N)) from a cycle basis."""
    n = adj.shape[0]
    atom_ring = np.zeros(n, dtype=bool)
    bond_ring = np.zeros((n, n), dtype=bool)
    for cycle in fundamental_cycles(adj):
        atom_ring[np.asarray(cycle, dtype=np.int64)] = True
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            bond_ring[a, b] = True
            bond_ring[b, a] = True
    return atom_ring, bond_ring


def ring_sizes(adj: np.ndarray) -> np.ndarray:
    """Smallest ring size each atom belongs to (0 when acyclic)."""
    n = adj.shape[0]
    sizes = np.zeros(n, dtype=np.int64)
    for cycle in fundamental_cycles(adj):
        for a in cycle:
            if sizes[a] == 0 or len(cycle) < sizes[a]:
                sizes[a] = len(cycle)
    return sizes


def shortest_path_distances(adj: np.ndarray, cap: int = 15) -> np.ndarray:
    """All-pairs shortest path length on an unweighted graph, clipped to `cap`.

    Disconnected pairs get `cap + 1` (they are masked out of attention anyway
    by the structure-aware mask).  Implemented with a per-source BFS so that no
    scipy/networkx dependency is required.
    """
    n = adj.shape[0]
    unreachable = cap + 1
    out = np.full((n, n), unreachable, dtype=np.int16)
    if n == 0:
        return out
    nbrs = [np.flatnonzero(adj[i]).astype(np.int64) for i in range(n)]
    for src in range(n):
        out[src, src] = 0
        frontier = [src]
        dist = 0
        visited = np.zeros(n, dtype=bool)
        visited[src] = True
        while frontier and dist < cap:
            dist += 1
            nxt: list[int] = []
            for u in frontier:
                for v in nbrs[u]:
                    v = int(v)
                    if not visited[v]:
                        visited[v] = True
                        out[src, v] = dist
                        nxt.append(v)
            frontier = nxt
    return out


def bond_vectors(pos: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Directed bond vectors r_j - r_i with the canonical orientation i < j."""
    if pairs.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    i, j = pairs[:, 0], pairs[:, 1]
    return (np.asarray(pos, dtype=np.float32)[j] - np.asarray(pos, dtype=np.float32)[i]).astype(
        np.float32
    )


def bond_midpoints(pos: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Bond coordinates r_ij = (r_i + r_j) / 2 (paper Section 3.2)."""
    if pairs.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float32)
    return (0.5 * (pos[pairs[:, 0]] + pos[pairs[:, 1]])).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-8)


def torsion_features(
    pos: np.ndarray, pairs: np.ndarray, shared_atom: np.ndarray
) -> np.ndarray:
    """Per-bond-pair torsion scalar exactly as defined in the paper.

    For two bonds sharing an atom (line-graph neighbours) the paper uses the
    bond angle:  cos(theta_ijk) = (r_ij . r_jk) / (|r_ij| |r_jk|).
    Otherwise it uses the angle between the two bond direction vectors:
    cos(phi_ijkl) = (r_ij . r_kl) / (|r_ij| |r_kl|).
    (The paper's cos(phi) is *not* the classical IUPAC dihedral; we follow the
    written definition and flag it in the README.)

    `shared_atom[m1, m2]` holds the index of the shared atom, or -1 when the two
    bonds are not line-graph neighbours.  Returns an (M, M) float32 matrix of
    cosines in [-1, 1].  Fully vectorised (no per-pair Python loop).
    """
    m = pairs.shape[0]
    if m == 0:
        return np.zeros((0, 0), dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float32)
    vecs = bond_vectors(pos, pairs)                       # (M, 3)
    cos_far = _cosine(vecs[:, None, :], vecs[None, :, :])  # inter-bond angle
    shared = np.asarray(shared_atom, dtype=np.int64)
    has_shared = shared >= 0
    if not has_shared.any():
        return cos_far.astype(np.float32)
    safe = np.where(has_shared, shared, 0)                 # avoid index -1
    i_idx = pairs[:, 0].astype(np.int64)
    j_idx = pairs[:, 1].astype(np.int64)
    # Other endpoint of bond m1 with respect to the shared atom, per pair.
    other1 = np.where(j_idx[:, None] == safe, i_idx[:, None], j_idx[:, None])
    other2 = np.where(j_idx[None, :] == safe, i_idx[None, :], j_idx[None, :])
    v1 = pos[other1] - pos[safe]                           # (M, M, 3)
    v2 = pos[other2] - pos[safe]
    cos_angle = _cosine(v1, v2)
    out = np.where(has_shared, cos_angle, cos_far)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def shared_atom_matrix(pairs: np.ndarray) -> np.ndarray:
    """(M, M) int32 matrix with the shared atom index of two bonds, else -1."""
    m = pairs.shape[0]
    if m == 0:
        return np.zeros((0, 0), dtype=np.int32)
    i_idx = pairs[:, 0].astype(np.int64)
    j_idx = pairs[:, 1].astype(np.int64)
    shared = np.full((m, m), -1, dtype=np.int64)
    for left, right in (
        (i_idx[:, None], i_idx[None, :]),
        (i_idx[:, None], j_idx[None, :]),
        (j_idx[:, None], i_idx[None, :]),
        (j_idx[:, None], j_idx[None, :]),
    ):
        hit = left == right
        shared = np.where(hit & (shared < 0), np.broadcast_to(left, (m, m)), shared)
    np.fill_diagonal(shared, -1)
    return shared.astype(np.int32)


def line_graph_adjacency(pairs: np.ndarray) -> np.ndarray:
    """(M, M) boolean line-graph adjacency: two bonds share an atom."""
    return shared_atom_matrix(pairs) >= 0
