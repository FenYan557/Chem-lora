"""DeMol: dual-graph (atom + bond / line-graph) transformer.

Faithful reimplementation of Liu et al., ICLR 2026, "Enhancing Molecular
Property Predictions by Learning from Bond Modelling and Interactions"
(Section 3.2, eqs. 5-10, Appendix B Algorithm 1).  No official code exists, so
every place where the paper is ambiguous carries an explicit ``NOTE`` here and
a matching entry in README.md.

Layout of one Double-Helix block (paper Figure 3)::

    h_a --[atom self-attn + FFN, bias Phi^(l)]----> h_a'
    h_b --[bond self-attn + FFN, bias Phi_b^(l)]--> h_b'
    h_a' --[atom<-bond cross-attn + FFN]---------> h_a''   (eq. 9)
    h_b' --[bond<-atom cross-attn + FFN]---------> h_b''   (eq. 10)

Both cross-attentions read the post-self-attention representations, so one
block is a symmetric rung of the helix rather than a sequential chain.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .encoding import (
    AtomStructureEncoding,
    BondStructureEncoding,
    CrossStructureEncoding,
    GaussianBasis,
    gather_atoms,
    masked_pool,
    pairwise_distance,
    shared_atom_index,
    torsion_cosine,
)
from .features import (
    ATOM_FIELD_SIZES,
    BOND_FIELD_SIZES,
    N_ATOM_CLASSES,
    N_BOND_CLASSES,
)

__all__ = ["DeMolConfig", "DeMol", "count_parameters"]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass
class DeMolConfig:
    """Hyper-parameters; defaults are the paper's atom/bond channel settings.

    The full paper setting (12 layers, d_a = d_b = 768, 128 Gaussian kernels,
    8 heads) is what these defaults describe.  Note that with four attention
    sub-layers per block and ``ffn_mult=4`` this yields ~350M parameters,
    about twice the 186M quoted in Table 1.  The quoted number matches the
    Graphormer-style FFN whose hidden dim equals the model dim -- the paper's
    own Table 1 lists Graphormer at 47.1M, which is exactly L=12, d=768,
    ffn_mult=1; with ``ffn_mult=1`` this implementation measures 180.7M.
    Pass ``--set-model ffn_mult=1`` for the paper-scale model.  See README.md.
    """

    n_layers: int = 12
    d_atom: int = 768
    d_bond: int = 768
    n_heads: int = 8
    n_kernels: int = 128
    kernel_width: float = 10.0        # w in mu_k = w(k-1)/K, sigma_k = w/K
    ffn_mult: int = 4

    dropout_input: float = 0.0
    dropout_attn: float = 0.1
    dropout_hidden: float = 0.0
    pre_ln: bool = True

    spd_buckets: int = 17             # 0..15 reachable + 16 = unreachable
    n_atom_types: int = N_ATOM_CLASSES
    n_bond_types: int = N_BOND_CLASSES
    atom_field_sizes: tuple[int, ...] = ATOM_FIELD_SIZES
    bond_field_sizes: tuple[int, ...] = BOND_FIELD_SIZES
    n_targets: int = 12

    # --- structure-encoding / memory trade-offs ---------------------------
    per_head_bias: bool = True        # False -> one bias shared by all heads
    share_structure_encoding: bool = False   # True -> one Phi for all layers
    share_cross_encoding: bool = True        # atom-bond bias shared by layers
    use_endpoint_bias: bool = True    # add Phi^(l)[i, j] to atom<-bond attn
    gradient_checkpointing: bool = False

    # --- heads ------------------------------------------------------------
    pool_modes: tuple[str, ...] = ("mean", "sum")
    cross_ffn: bool = True
    bond_head_rank: int = 0           # 0 -> d_atom // 4
    coord_target: str = "delta"       # "delta" (recover p) | "noise" (predict eps)
    bond_threshold_alpha: float = 1.15

    def __post_init__(self) -> None:
        if self.d_atom % self.n_heads or self.d_bond % self.n_heads:
            raise ValueError("d_atom and d_bond must be divisible by n_heads")
        if self.coord_target not in ("delta", "noise"):
            raise ValueError("coord_target must be 'delta' or 'noise'")
        for mode in self.pool_modes:
            if mode not in ("mean", "sum"):
                raise ValueError("pool_modes entries must be 'mean' or 'sum'")

    @property
    def n_bias_heads(self) -> int:
        return self.n_heads if self.per_head_bias else 1

    @property
    def d_readout(self) -> int:
        return (self.d_atom + self.d_bond) * len(self.pool_modes)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["atom_field_sizes"] = list(self.atom_field_sizes)
        out["bond_field_sizes"] = list(self.bond_field_sizes)
        out["pool_modes"] = list(self.pool_modes)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeMolConfig":
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in data.items() if k in known}
        for key in ("atom_field_sizes", "bond_field_sizes", "pool_modes"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
class MaskedMultiHeadAttention(nn.Module):
    """Dense multi-head attention with an additive bias and a key mask.

    ``bias`` is (B, H_bias, X, Y) where ``H_bias`` is either ``n_heads`` or 1.
    Query rows whose keys are all masked produce zeros instead of NaNs, which
    keeps padded slots from poisoning the masked pooling and the losses.
    """

    def __init__(
        self,
        d_query: int,
        d_key: int,
        n_heads: int,
        d_head: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else d_query // n_heads
        inner = n_heads * self.d_head
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.q_proj = nn.Linear(d_query, inner, bias=False)
        self.k_proj = nn.Linear(d_key, inner, bias=False)
        self.v_proj = nn.Linear(d_key, inner, bias=False)
        self.o_proj = nn.Linear(inner, d_query)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        bias: torch.Tensor | None,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, x, _ = query.shape
        y = key_value.shape[1]
        h, dk = self.n_heads, self.d_head
        qh = self.q_proj(query).view(b, x, h, dk).transpose(1, 2)
        kh = self.k_proj(key_value).view(b, y, h, dk).transpose(1, 2)
        vh = self.v_proj(key_value).view(b, y, h, dk).transpose(1, 2)

        logits = torch.matmul(qh, kh.transpose(-1, -2)) * self.scale
        if bias is not None:
            if bias.shape[1] != h:
                bias = bias.expand(b, h, x, y)
            logits = logits + bias

        km = key_mask[:, None, None, :]                      # (B, 1, 1, Y)
        logits = logits.masked_fill(~km, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        # a row with no valid key would otherwise become uniform noise
        attn = torch.where(km.any(dim=-1, keepdim=True), attn, torch.zeros_like(attn))
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, vh).transpose(1, 2).reshape(b, x, h * dk)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mult)
        self.fc2 = nn.Linear(d_model * mult, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class SubLayer(nn.Module):
    """One attention (+ optional FFN) update with residual connections.

    ``ctx is None`` means self-attention.  With ``pre_ln`` the context stream
    of a cross-attention gets its own LayerNorm (``d_ctx``), because ``ln_attn``
    belongs to the query stream.
    """

    def __init__(
        self,
        attn: MaskedMultiHeadAttention,
        d_model: int,
        ffn: FeedForward | None,
        dropout_hidden: float,
        pre_ln: bool,
        d_ctx: int | None = None,
    ):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.pre_ln = pre_ln
        self.ln_attn = nn.LayerNorm(d_model)
        self.drop_attn = nn.Dropout(dropout_hidden)
        self.ctx_norm = (
            nn.LayerNorm(d_ctx) if (pre_ln and d_ctx is not None) else None
        )
        if ffn is not None:
            self.ln_ffn = nn.LayerNorm(d_model)
            self.drop_ffn = nn.Dropout(dropout_hidden)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor | None,
        bias: torch.Tensor | None,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pre_ln:
            hq = self.ln_attn(x)
            hc = hq if ctx is None else (self.ctx_norm(ctx) if self.ctx_norm else ctx)
            x = x + self.drop_attn(self.attn(hq, hc, bias, key_mask))
            if self.ffn is not None:
                x = x + self.drop_ffn(self.ffn(self.ln_ffn(x)))
        else:
            x = self.ln_attn(x + self.drop_attn(self.attn(x, x if ctx is None else ctx, bias, key_mask)))
            if self.ffn is not None:
                x = self.ln_ffn(x + self.drop_ffn(self.ffn(x)))
        return x


class DoubleHelixBlock(nn.Module):
    """Atom self-attn, bond self-attn, then both cross-attentions (eqs. 6-10)."""

    def __init__(self, cfg: DeMolConfig):
        super().__init__()
        h = cfg.n_heads
        d_head_a = cfg.d_atom // h
        d_head_b = cfg.d_bond // h
        ffn_a = FeedForward(cfg.d_atom, cfg.ffn_mult, cfg.dropout_hidden)
        ffn_b = FeedForward(cfg.d_bond, cfg.ffn_mult, cfg.dropout_hidden)
        ffn_ax = FeedForward(cfg.d_atom, cfg.ffn_mult, cfg.dropout_hidden) if cfg.cross_ffn else None
        ffn_bx = FeedForward(cfg.d_bond, cfg.ffn_mult, cfg.dropout_hidden) if cfg.cross_ffn else None

        self.atom_self = SubLayer(
            MaskedMultiHeadAttention(cfg.d_atom, cfg.d_atom, h, d_head_a, cfg.dropout_attn),
            cfg.d_atom, ffn_a, cfg.dropout_hidden, cfg.pre_ln,
        )
        self.bond_self = SubLayer(
            MaskedMultiHeadAttention(cfg.d_bond, cfg.d_bond, h, d_head_b, cfg.dropout_attn),
            cfg.d_bond, ffn_b, cfg.dropout_hidden, cfg.pre_ln,
        )
        self.atom_cross = SubLayer(
            MaskedMultiHeadAttention(cfg.d_atom, cfg.d_bond, h, d_head_a, cfg.dropout_attn),
            cfg.d_atom, ffn_ax, cfg.dropout_hidden, cfg.pre_ln, d_ctx=cfg.d_bond,
        )
        self.bond_cross = SubLayer(
            MaskedMultiHeadAttention(cfg.d_bond, cfg.d_atom, h, d_head_b, cfg.dropout_attn),
            cfg.d_bond, ffn_bx, cfg.dropout_hidden, cfg.pre_ln, d_ctx=cfg.d_atom,
        )

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        phi_a: torch.Tensor,
        phi_b: torch.Tensor,
        bias_ab: torch.Tensor,
        bias_ba: torch.Tensor,
        atom_mask: torch.Tensor,
        bond_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_a1 = self.atom_self(h_a, None, phi_a, atom_mask)
        h_b1 = self.bond_self(h_b, None, phi_b, bond_mask)
        h_a2 = self.atom_cross(h_a1, h_b1, bias_ab, bond_mask)
        h_b2 = self.bond_cross(h_b1, h_a1, bias_ba, atom_mask)
        return h_a2, h_b2


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class DeMol(nn.Module):
    """Dual-graph transformer with the four pre-training heads of the paper."""

    def __init__(self, config: DeMolConfig | None = None, **overrides: Any):
        super().__init__()
        cfg = config if config is not None else DeMolConfig()
        if overrides:
            cfg = DeMolConfig.from_dict({**cfg.to_dict(), **overrides})
        self.cfg = cfg
        nb = cfg.n_bias_heads
        k = cfg.n_kernels

        # --- token embeddings (sum over categorical fields) ---------------
        self.atom_emb = nn.ModuleList([nn.Embedding(s, cfg.d_atom) for s in cfg.atom_field_sizes])
        self.bond_emb = nn.ModuleList([nn.Embedding(s, cfg.d_bond) for s in cfg.bond_field_sizes])
        for table in list(self.atom_emb) + list(self.bond_emb):
            nn.init.normal_(table.weight, std=0.02)
        # learnable [MASK] token used by the masked-atom-prediction task
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_atom))
        nn.init.normal_(self.mask_token, std=0.02)
        self.input_drop = nn.Dropout(cfg.dropout_input)

        # --- shared Gaussian bases (eq. 5) --------------------------------
        self.basis_aa = GaussianBasis(k, cfg.kernel_width, cfg.n_atom_types)
        self.basis_bb = GaussianBasis(k, cfg.kernel_width, cfg.n_bond_types)
        # the torsion input is (cos + 1) / 2 in [0, 1], hence width 1.0
        self.basis_tors = GaussianBasis(k, 1.0, cfg.n_bond_types)
        self.basis_ab = GaussianBasis(k, cfg.kernel_width, cfg.n_atom_types, cfg.n_bond_types)

        # --- structure encodings (per layer, as Phi^(l) in eqs. 6-10) -----
        n_enc = 1 if cfg.share_structure_encoding else cfg.n_layers
        self.atom_enc = nn.ModuleList(
            [AtomStructureEncoding(k, nb, cfg.spd_buckets) for _ in range(n_enc)]
        )
        self.bond_enc = nn.ModuleList(
            [BondStructureEncoding(k, nb, cfg.spd_buckets) for _ in range(n_enc)]
        )
        n_cross = 1 if cfg.share_cross_encoding else cfg.n_layers
        self.cross_ab_enc = nn.ModuleList(
            [CrossStructureEncoding(k, nb, cfg.spd_buckets) for _ in range(n_cross)]
        )
        self.cross_ba_enc = nn.ModuleList(
            [CrossStructureEncoding(k, nb, cfg.spd_buckets) for _ in range(n_cross)]
        )

        self.blocks = nn.ModuleList([DoubleHelixBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm_a = nn.LayerNorm(cfg.d_atom)
        self.final_norm_b = nn.LayerNorm(cfg.d_bond)

        # --- heads --------------------------------------------------------
        d_read = cfg.d_readout
        self.prop_head = nn.Sequential(
            nn.Linear(d_read, d_read),
            nn.GELU(),
            nn.Dropout(cfg.dropout_hidden),
            nn.Linear(d_read, cfg.n_targets),
        )
        self.mask_head = nn.Linear(cfg.d_atom, cfg.n_atom_types)
        self.coord_head = nn.Linear(cfg.d_atom, 3)
        self.bond_head_rank = cfg.bond_head_rank or max(1, cfg.d_atom // 4)
        self.bond_left = nn.Linear(cfg.d_atom, self.bond_head_rank)
        self.bond_right = nn.Linear(cfg.d_atom, self.bond_head_rank)
        self.bond_bias = nn.Parameter(torch.zeros(1))

    # -- helpers ----------------------------------------------------------
    def _embed(self, fields: torch.Tensor, tables: nn.ModuleList) -> torch.Tensor:
        out = tables[0](fields[..., 0])
        for f in range(1, len(tables)):
            out = out + tables[f](fields[..., f])
        return out

    def _atom_bond_graph_distance(self, spd: torch.Tensor, bond_idx: torch.Tensor) -> torch.Tensor:
        """``min(SPD[i, u], SPD[i, v])`` for bond ``(u, v)`` -> (B, N, M)."""
        b, n, _ = spd.shape
        m = bond_idx.shape[1]
        iu = bond_idx[..., 0][:, None, :].expand(b, n, m)
        iv = bond_idx[..., 1][:, None, :].expand(b, n, m)
        return torch.minimum(torch.gather(spd, 2, iu), torch.gather(spd, 2, iv))

    def _other_endpoint(self, bond_idx: torch.Tensor, n_atoms: int) -> torch.Tensor:
        """For atom ``i`` and incident bond ``(u, v)``, the endpoint != ``i``.

        Returns (B, N, M).  Entries where the bond is not incident to ``i`` are
        arbitrary and are excluded by the atom-bond attention mask.
        """
        b, m, _ = bond_idx.shape
        ar = torch.arange(n_atoms, device=bond_idx.device)[None, :, None].expand(b, n_atoms, m)
        iu = bond_idx[..., 0][:, None, :].expand(b, n_atoms, m)
        iv = bond_idx[..., 1][:, None, :].expand(b, n_atoms, m)
        return torch.where(iu == ar, iv, iu)

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        mask_ratio: float = 0.0,
        coord_noise: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Run the dual-graph transformer.

        ``mask_ratio`` enables the masked-atom task (Lmask); ``coord_noise`` is
        the Gaussian std in Angstrom added to the input coordinates for the
        coordinate-recovery task (Lcoord).  Pass 0.0 for both at inference.
        """
        cfg = self.cfg
        atom_fields = batch["atom_fields"].long()
        atom_class = batch["atom_class"].long()
        atom_mask = batch["atom_mask"].bool()
        pos = batch["pos"].float()
        spd = batch["spd"].long()
        aa_mask = batch["aa_mask"].bool()
        bond_fields = batch["bond_fields"].long()
        bond_class = batch["bond_class"].long()
        bond_mask = batch["bond_mask"].bool()
        bond_idx = batch["bond_idx"].long()
        spd_line = batch["spd_line"].long()
        bb_mask = batch["bb_mask"].bool()
        ab_mask = batch["ab_mask"].bool()

        b, n = atom_mask.shape
        m = bond_mask.shape[1]
        if m == 0:
            raise ValueError(
                "DeMol needs at least one bond in the batch (the bond channel "
                "is empty); every QM9 molecule has bonds."
            )
        device = pos.device

        # --- Lcoord: perturb the input geometry, keep the clean target -----
        if coord_noise > 0.0:
            # mask the noise itself so that coord_input == pos + noise holds
            # everywhere and padding atoms stay exactly at their (zero) position;
            # losses.py masks coord_loss anyway, so this is purely for consistency.
            noise = torch.randn_like(pos) * coord_noise * atom_mask.unsqueeze(-1)
            pos_in = pos + noise
        else:
            noise = torch.zeros_like(pos)
            pos_in = pos

        # --- token embeddings ---------------------------------------------
        h_a = self._embed(atom_fields, self.atom_emb)
        h_b = self._embed(bond_fields, self.bond_emb)

        # --- Lmask: hide the atom-type field of a random subset ------------
        masked_atoms = torch.zeros((b, n), dtype=torch.bool, device=device)
        if mask_ratio > 0.0:
            masked_atoms = (torch.rand(b, n, device=device) < mask_ratio) & atom_mask
            m3 = masked_atoms.unsqueeze(-1).to(h_a.dtype)
            type_emb = self.atom_emb[0](atom_fields[..., 0])
            h_a = h_a - type_emb * m3 + self.mask_token * m3

        h_a = self.input_drop(h_a)
        h_b = self.input_drop(h_b)

        # --- geometry -> Gaussian bases (computed once, shared by layers) --
        dist_aa = pairwise_distance(pos_in)
        mid = 0.5 * (
            gather_atoms(pos_in, bond_idx[..., 0]) + gather_atoms(pos_in, bond_idx[..., 1])
        )
        dist_bb = pairwise_distance(mid)
        shared = shared_atom_index(bond_idx)
        tors = torsion_cosine(pos_in, bond_idx, shared)
        dist_ab = (mid[:, None, :, :] - pos_in[:, :, None, :]).norm(dim=-1)   # (B, N, M)
        spd_ab = self._atom_bond_graph_distance(spd, bond_idx)

        phi_aa = self.basis_aa(dist_aa, atom_class, atom_class)               # (B,N,N,K)
        phi_bb = self.basis_bb(dist_bb, bond_class, bond_class)               # (B,M,M,K)
        phi_tors = self.basis_tors((tors + 1.0) * 0.5, bond_class, bond_class)
        phi_ab = self.basis_ab(dist_ab, atom_class, bond_class)               # (B,N,M,K)

        # --- atom-bond cross bias ----------------------------------------
        bias_ab = self.cross_ab_enc[0](phi_ab, spd_ab.clamp(0, cfg.spd_buckets - 1))
        bias_ba = self.cross_ba_enc[0](
            phi_ab.transpose(1, 2), spd_ab.transpose(1, 2).clamp(0, cfg.spd_buckets - 1)
        )
        other = self._other_endpoint(bond_idx, n) if cfg.use_endpoint_bias else None

        for layer, block in enumerate(self.blocks):
            enc = 0 if cfg.share_structure_encoding else layer
            phi_a = self.atom_enc[enc](phi_aa, spd.clamp(0, cfg.spd_buckets - 1))
            phi_b = self.bond_enc[enc](
                phi_bb, spd_line.clamp(0, cfg.spd_buckets - 1), phi_tors
            )
            b_ab, b_ba = bias_ab, bias_ba
            if not cfg.share_cross_encoding:
                enc_x = layer
                b_ab = self.cross_ab_enc[enc_x](phi_ab, spd_ab.clamp(0, cfg.spd_buckets - 1))
                b_ba = self.cross_ba_enc[enc_x](
                    phi_ab.transpose(1, 2),
                    spd_ab.transpose(1, 2).clamp(0, cfg.spd_buckets - 1),
                )
            if other is not None:
                # NOTE: eq. (9) adds Phi^(l) to an atom x bond attention whose
                # indices are (i, ij); we use Phi^(l)[i, j] with j the bond's
                # other endpoint.
                idx = other[:, None, :, :].expand(b, phi_a.shape[1], n, m)
                b_ab = b_ab + torch.gather(phi_a, 3, idx)

            if cfg.gradient_checkpointing and self.training:
                h_a, h_b = torch_checkpoint(
                    block,
                    h_a, h_b, phi_a, phi_b, b_ab, b_ba, atom_mask, bond_mask,
                    use_reentrant=False,
                )
            else:
                h_a, h_b = block(h_a, h_b, phi_a, phi_b, b_ab, b_ba, atom_mask, bond_mask)

        h_a = self.final_norm_a(h_a)
        h_b = self.final_norm_b(h_b)

        # --- readout ------------------------------------------------------
        graph_repr = torch.cat(
            [
                masked_pool(h_a, atom_mask, cfg.pool_modes),
                masked_pool(h_b, bond_mask, cfg.pool_modes),
            ],
            dim=-1,
        )
        prop = self.prop_head(graph_repr)

        # --- auxiliary heads ----------------------------------------------
        atom_logits = self.mask_head(h_a)                                   # (B,N,C)
        coord_delta = self.coord_head(h_a)                                  # (B,N,3)
        coord_pred = pos_in + coord_delta if cfg.coord_target == "delta" else coord_delta

        left = self.bond_left(h_a)
        right = self.bond_right(h_a)
        bond_logits = torch.einsum("bir,bjr->bij", left, right)
        bond_logits = (bond_logits + bond_logits.transpose(1, 2)) / math.sqrt(self.bond_head_rank)
        bond_logits = bond_logits + self.bond_bias

        return {
            "prop": prop,
            "graph_repr": graph_repr,
            "atom_repr": h_a,
            "bond_repr": h_b,
            "atom_logits": atom_logits,
            "masked_atoms": masked_atoms,
            "coord_pred": coord_pred,
            "coord_target": pos,
            "coord_noise": noise,
            "coord_input": pos_in,
            "bond_logits": bond_logits,
            "atom_mask": atom_mask,
            "bond_mask": bond_mask,
            "aa_mask": aa_mask,
            "ab_mask": ab_mask,
            "bb_mask": bb_mask,
            "bond_idx": bond_idx,
        }

    # -- convenience ------------------------------------------------------
    def load_pretrained(self, path: str, strict: bool = False) -> tuple[Any, Any]:
        """Load encoder weights from a pre-training checkpoint."""
        state = torch.load(path, map_location="cpu")
        weights = state.get("model", state) if isinstance(state, dict) else state
        return self.load_state_dict(weights, strict=strict)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
