"""Structure encodings for DeMol (paper Section 3.2, eqs. 5-10).

Three ingredients, all of which end up as *additive attention biases*:

``Phi_dist``  Gaussian-basis kernel expansion of a scalar pair quantity
              (eq. 5) followed by ``GELU(phi W1) W2``.  The learnable scalars
              ``alpha``/``beta`` are indexed by the pair of atom types (atom
              channel) or the pair of bond types (bond channel), exactly as the
              paper prescribes.
``Phi_SPD``   a learnable embedding of the shortest-path distance.
``Phi_tors``  the bond-channel-only torsion encoding, i.e. the same Gaussian
              basis applied to ``cos(theta_ijk)`` for line-graph neighbours and
              ``cos(phi_ijkl)`` otherwise.

The Gaussian basis itself (mu_k, sigma_k, alpha, beta) is computed **once** and
shared by all layers; every layer owns its own ``W1/W2`` projection and SPD
embedding so that ``Phi^(l)`` in eqs. (6)-(10) is layer specific, matching
Transformer-M, which the paper cites for this design.

Interpretation notes (also in README.md):

* The paper states ``Phi_dist`` has shape ``N x N`` (a single scalar per pair)
  while the attention in eq. (6) is multi-head.  ``n_bias_heads`` defaults to
  the number of attention heads (per-head bias, as in Transformer-M); set it to
  1 to broadcast one shared bias over all heads, which is the literal reading.
* For the cross-graph (Double-Helix) attentions the paper writes ``+Phi^(l)``
  and ``+Phi_b^(l)`` although those matrices are indexed by atom-atom and
  bond-bond pairs while the attentions are indexed by atom-bond pairs.  For
  atom<-bond we gather ``Phi^(l)[i, j]`` where ``j`` is the *other* endpoint of
  the incident bond.  For both directions we additionally build a dedicated
  atom-bond encoding from the bond-midpoint/atom distance and the graph
  distance ``min(SPD[i,u], SPD[i,v])``, which is the natural atom-bond analogue.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "gaussian_centers",
    "GaussianBasis",
    "ScalarBasisEncoder",
    "SPDEncoder",
    "shared_atom_index",
    "torsion_cosine",
    "pairwise_distance",
    "masked_pool",
]


# --------------------------------------------------------------------------
# Gaussian basis kernel (eq. 5)
# --------------------------------------------------------------------------
def gaussian_centers(n_kernels: int, width: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``mu_k = w * (k - 1) / K`` and ``sigma_k = w / K`` (1-based k)."""
    k = torch.arange(n_kernels, dtype=torch.float32)
    mu = width * k / n_kernels
    sigma = torch.full((n_kernels,), width / n_kernels, dtype=torch.float32)
    return mu, sigma


class GaussianBasis(nn.Module):
    """``phi_k(x) = -1/(sqrt(2 pi)|sigma_k|) exp(-0.5 ((alpha x + beta - mu_k)/sigma_k)^2)``.

    ``alpha`` and ``beta`` are learnable scalars indexed by the pair of node
    types (atom types for the atom channel, bond types for the bond channel).
    For the atom-bond cross encoding the table is rectangular
    (``n_types`` rows x ``n_types_col`` columns).
    """

    def __init__(
        self,
        n_kernels: int = 128,
        width: float = 10.0,
        n_types: int = 6,
        n_types_col: int | None = None,
    ):
        super().__init__()
        mu, sigma = gaussian_centers(n_kernels, width)
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)
        self.register_buffer("norm", -1.0 / (math.sqrt(2.0 * math.pi) * sigma))
        self.n_kernels = n_kernels
        self.width = width
        cols = n_types if n_types_col is None else n_types_col
        self.alpha = nn.Parameter(torch.ones(n_types, cols))
        self.beta = nn.Parameter(torch.zeros(n_types, cols))

    def forward(self, x: torch.Tensor, ti: torch.Tensor, tj: torch.Tensor) -> torch.Tensor:
        """``x`` is (B, X, Y); ``ti`` (B, X) and ``tj`` (B, Y) are type indices.

        Returns (B, X, Y, K).
        """
        a = self.alpha[ti[:, :, None], tj[:, None, :]].unsqueeze(-1)
        b = self.beta[ti[:, :, None], tj[:, None, :]].unsqueeze(-1)
        z = (a * x.unsqueeze(-1) + b - self.mu) / self.sigma
        return self.norm * torch.exp(-0.5 * z * z)


class ScalarBasisEncoder(nn.Module):
    """``Phi = GELU(phi W1) W2`` with ``W1 in R^{K x K}``, ``W2 in R^{K x H}``."""

    def __init__(self, n_kernels: int, n_out: int):
        super().__init__()
        self.fc1 = nn.Linear(n_kernels, n_kernels)
        self.fc2 = nn.Linear(n_kernels, n_out)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """(B, X, Y, K) -> (B, H, X, Y)."""
        out = self.fc2(F.gelu(self.fc1(phi)))
        return out.permute(0, 3, 1, 2)


class SPDEncoder(nn.Module):
    """Learnable embedding of the (capped) shortest-path distance."""

    def __init__(self, n_buckets: int, n_out: int):
        super().__init__()
        self.emb = nn.Embedding(n_buckets, n_out)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, spd: torch.Tensor) -> torch.Tensor:
        """(B, X, Y) long -> (B, H, X, Y)."""
        return self.emb(spd).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------
# torch-side geometry helpers (mirror demol.chem, but differentiable inputs)
# --------------------------------------------------------------------------
def pairwise_distance(pos: torch.Tensor) -> torch.Tensor:
    """(B, N, 3) -> (B, N, N) Euclidean distances."""
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    return torch.sqrt((diff * diff).sum(-1).clamp_min(1e-12))


def gather_atoms(pos: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``pos`` (B, N, 3), ``idx`` (B, ...) long -> (B, ..., 3).

    ``idx`` may carry any number of trailing dimensions.  ``torch.gather`` insists
    that the index has the same rank as the input, so the trailing dimensions are
    flattened away and restored afterwards -- that is what lets
    :func:`torsion_cosine` gather with an (B, M, M) shared-atom matrix.
    """
    flat = idx.reshape(idx.shape[0], -1).unsqueeze(-1).expand(-1, -1, pos.shape[-1])
    out = torch.gather(pos, 1, flat)
    return out.reshape(*idx.shape, pos.shape[-1])


def shared_atom_index(bond_idx: torch.Tensor) -> torch.Tensor:
    """(B, M, 2) atom-index pairs -> (B, M, M) shared atom index, -1 if none.

    Torch equivalent of :func:`demol.chem.shared_atom_matrix`; the diagonal is
    forced to -1 because a bond trivially shares both endpoints with itself.
    """
    i0, i1 = bond_idx[..., 0], bond_idx[..., 1]
    b, m, _ = bond_idx.shape
    shared = torch.full((b, m, m), -1, dtype=torch.long, device=bond_idx.device)
    for left, right in ((i0, i0), (i0, i1), (i1, i0), (i1, i1)):
        hit = left[:, :, None] == right[:, None, :]
        cand = left[:, :, None].expand(b, m, m)
        shared = torch.where(hit & (shared < 0), cand, shared)
    eye = torch.eye(m, dtype=torch.bool, device=bond_idx.device)[None]
    return shared.masked_fill(eye, -1)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    return (a * b).sum(-1) / (na * nb).clamp_min(1e-8)


def torsion_cosine(
    pos: torch.Tensor, bond_idx: torch.Tensor, shared: torch.Tensor | None = None
) -> torch.Tensor:
    """The paper's torsion scalar for every bond pair, as a (B, M, M) matrix.

    * line-graph neighbours (bonds ``ijk`` sharing atom ``j``):
      ``cos(theta_ijk) = (r_ij . r_jk) / (|r_ij| |r_jk|)``, i.e. the bond angle
      measured at the shared atom;
    * all other pairs: ``cos(phi_ijkl) = (r_ij . r_kl) / (|r_ij| |r_kl|)``, the
      angle between the two bond direction vectors.  (This is the definition
      written in the paper; it is *not* the classical IUPAC dihedral, which
      would need four distinct atoms.  See README.)
    """
    b, m, _ = bond_idx.shape
    if shared is None:
        shared = shared_atom_index(bond_idx)
    vecs = gather_atoms(pos, bond_idx[..., 1]) - gather_atoms(pos, bond_idx[..., 0])
    cos_far = _cosine(vecs[:, :, None, :], vecs[:, None, :, :])          # (B, M, M)

    has_shared = shared >= 0
    safe = torch.where(has_shared, shared, torch.zeros_like(shared))
    i0, i1 = bond_idx[..., 0], bond_idx[..., 1]
    # endpoint of bond m1 that is *not* the shared atom (per pair)
    other1 = torch.where(i1[:, :, None] == safe, i0[:, :, None], i1[:, :, None]).expand(b, m, m)
    other2 = torch.where(i1[:, None, :] == safe, i0[:, None, :], i1[:, None, :]).expand(b, m, m)
    p_s = gather_atoms(pos, safe)                                        # (B, M, M, 3)
    v1 = gather_atoms(pos, other1) - p_s
    v2 = gather_atoms(pos, other2) - p_s
    cos_angle = _cosine(v1, v2)
    return torch.where(has_shared, cos_angle, cos_far).clamp(-1.0, 1.0)


def masked_pool(
    x: torch.Tensor, mask: torch.Tensor, modes: tuple[str, ...] = ("mean", "sum")
) -> torch.Tensor:
    """Pool (B, N, D) over valid entries of ``mask`` (B, N) -> (B, D*len(modes)).

    Rows without any valid entry yield zeros instead of NaNs.
    """
    m = mask.unsqueeze(-1).to(x.dtype)
    denom = m.sum(1).clamp_min(1.0)
    out = []
    for mode in modes:
        total = (x * m).sum(1)
        out.append(total / denom if mode == "mean" else total)
    return torch.cat(out, dim=-1)


# --------------------------------------------------------------------------
# per-layer structure encodings
# --------------------------------------------------------------------------
class AtomStructureEncoding(nn.Module):
    """``Phi = Phi_dist + Phi_SPD`` for one layer of the atom channel."""

    def __init__(self, n_kernels: int, n_bias_heads: int, spd_buckets: int):
        super().__init__()
        self.dist = ScalarBasisEncoder(n_kernels, n_bias_heads)
        self.spd = SPDEncoder(spd_buckets, n_bias_heads)

    def forward(self, phi_dist: torch.Tensor, spd: torch.Tensor) -> torch.Tensor:
        return self.dist(phi_dist) + self.spd(spd)


class BondStructureEncoding(nn.Module):
    """``Phi_b = Phi_dist_b + Phi_SPD_b + Phi_tors_b`` for one layer."""

    def __init__(self, n_kernels: int, n_bias_heads: int, spd_buckets: int):
        super().__init__()
        self.dist = ScalarBasisEncoder(n_kernels, n_bias_heads)
        self.spd = SPDEncoder(spd_buckets, n_bias_heads)
        self.tors = ScalarBasisEncoder(n_kernels, n_bias_heads)

    def forward(
        self, phi_dist: torch.Tensor, spd: torch.Tensor, phi_tors: torch.Tensor
    ) -> torch.Tensor:
        return self.dist(phi_dist) + self.spd(spd) + self.tors(phi_tors)


class CrossStructureEncoding(nn.Module):
    """Atom-bond structure encoding used as the Double-Helix attention bias."""

    def __init__(self, n_kernels: int, n_bias_heads: int, spd_buckets: int):
        super().__init__()
        self.dist = ScalarBasisEncoder(n_kernels, n_bias_heads)
        self.spd = SPDEncoder(spd_buckets, n_bias_heads)

    def forward(self, phi: torch.Tensor, spd: torch.Tensor) -> torch.Tensor:
        return self.dist(phi) + self.spd(spd)
