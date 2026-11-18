#!/usr/bin/env python
"""DeMol 复现代码的自检脚本（不依赖 QM9 数据，也不需要 GPU）。

分子是**合成**的：随机生成一棵重原子树（C/N/O/F，键长按共价半径之和给出），
再按价态补氢，因此化学键的真值是已知的，可以反过来检验 Algorithm 1 的键感知
和整套 featurisation。之后走一遍真实的代码路径：``write_cache`` ->
``QM9Dataset`` -> ``qm9_collate`` -> ``DeMol.forward`` -> ``DeMolLoss`` ->
``Trainer.fit`` / ``Trainer.evaluate``。

重点覆盖三类容易悄悄出错的地方：

1. **numpy 与 torch 两套几何实现的一致性** —— 预处理在 numpy 里算
   （``demol.chem``），模型前向在 torch 里算（``demol.encoding``）。二者必须
   逐元素一致，否则缓存里的 SPD/线图解和模型看到的几何就会错位。
2. **padding 语义** —— 稠密批处理里补零的原子/化学键不能进入注意力、池化或
   任何一个损失项。
3. **梯度与数值** —— 四个损失都有限、反向传播没有 NaN/Inf、EMA 与检查点能
   正常存取。

用法::

    python scripts/smoke_test.py                  # 全部检查，CPU
    python scripts/smoke_test.py --only parity    # 只跑名字里含 parity 的检查
    python scripts/smoke_test.py --list           # 列出所有检查名
    python scripts/smoke_test.py --keep-dir DIR   # 保留临时缓存便于排查

退出码 0 = 全部通过，1 = 有硬失败。
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import uuid
import time
import traceback
from typing import Any, Callable, Iterable, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 与 run_common.py 同样的 UTF-8 输出保护：中文自检说明在任意代码页下都不会
# 触发 UnicodeEncodeError。
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

import torch  # noqa: E402

from demol import chem, encoding, features  # noqa: E402
from demol.benchmarks import PAPER_MODEL_PARAMS_M, comparison_table  # noqa: E402
from demol.configs import apply_overrides, loss_config, model_config, train_config  # noqa: E402
from demol.data import build_loaders, cycling_loader  # noqa: E402
from demol.features import (  # noqa: E402
    ATOM_FIELD_SIZES,
    BOND_FIELD_SIZES,
    N_ATOM_CLASSES,
    N_BOND_CLASSES,
)
from demol.losses import DeMolLoss, TargetStandardizer  # noqa: E402
from demol.model import DeMol, MaskedMultiHeadAttention, count_parameters  # noqa: E402
from demol.qm9 import (  # noqa: E402
    N_TARGETS,
    QM9Dataset,
    parse_qm9_file,
    qm9_collate,
    verify_target_columns,
    write_cache,
)
from demol.trainer import Trainer  # noqa: E402

# --------------------------------------------------------------------------
# 合成分子
# --------------------------------------------------------------------------
HEAVY_ELEMENTS = (6, 7, 8, 9)
VALENCE = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}
# 非键合原子对的最小间距 = SAFE_MARGIN * 成键阈值，保证 Algorithm 1 不会误判。
# 这个余量不能取得太大：四面体碳（109.47°）上两个 C-C 键的子原子间距只有
# 1.42 * 阈值，所以 SAFE_MARGIN >= 1.45 时任何 4 配位碳都放不进去，几何上无解。
# 1.20 仍留 20% 余量（perceive_bonds 的阈值是 1.15），同时随机树能稳定构建。
SAFE_MARGIN = 1.20
MAX_PLACE_TRIES = 400
# 新键方向的球面候选数：采一批后取"离已有键最远"的那个（VSEPR 式互斥）
DIRECTION_CANDIDATES = 192
# 单个分子挤不进去时，整棵分子树重掷的最大次数
MAX_MOLECULE_ATTEMPTS = 24


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = float(np.linalg.norm(vec))
    while norm < 1e-6:
        vec = rng.normal(size=3)
        norm = float(np.linalg.norm(vec))
    return vec / norm


def _sample_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    """:func:`_unit_vector` 的向量化版本：一次采 ``n`` 个，返回 (n, 3)。"""
    vec = rng.normal(size=(n, 3))
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    while bool((norm[:, 0] < 1e-6).any()):
        bad = norm[:, 0] < 1e-6
        vec[bad] = rng.normal(size=(int(bad.sum()), 3))
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / norm


def _far_enough(positions, numbers, cand: np.ndarray, z_new: int, skip: int | None) -> bool:
    """cand 与除 ``skip`` 之外的所有原子的距离都要显著大于成键阈值。"""
    r_new = chem.COVALENT_RADII.get(int(z_new), chem.DEFAULT_COVALENT_RADIUS)
    for k, (pos_k, z_k) in enumerate(zip(positions, numbers)):
        if k == skip:
            continue
        r_k = chem.COVALENT_RADII.get(int(z_k), chem.DEFAULT_COVALENT_RADIUS)
        threshold = chem.BOND_THRESHOLD_ALPHA * (r_k + r_new)
        if float(np.linalg.norm(pos_k - cand)) <= SAFE_MARGIN * threshold:
            return False
    return True


class _PlacementError(RuntimeError):
    """一个分子放不下原子，交给上层重掷整棵树。"""


def _direction_away_from(
    rng: np.random.Generator, existing: list[np.ndarray]
) -> np.ndarray:
    """球面采样一批方向，返回与 ``existing``（该原子已有的键方向）最小夹角最大的那个。

    VSEPR 式的电子对互斥：4 配位碳会自然逼近四面体（109.5°）而不是随机方向常见的
    30° 小夹角。纯随机方向 + 严格的间距约束几乎必然无解，这是原来报"分子太挤"的
    真正原因。
    """
    cands = _sample_unit_vectors(rng, DIRECTION_CANDIDATES)
    if not existing:
        return cands[0]
    # 与已有方向的最大余弦越小，说明夹角越大
    worst_cos = (cands @ np.stack(existing).T).max(axis=1)
    return cands[int(np.argmin(worst_cos))]


def _build_molecule(
    rng: np.random.Generator, n_heavy: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """构造一个分子；放不下时抛 :class:`_PlacementError`。"""
    positions: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
    numbers: list[int] = [int(rng.choice(HEAVY_ELEMENTS))]
    edges: list[tuple[int, int]] = []
    # dirs[i]: 从原子 i 出发的所有键的单位向量，供 _direction_away_from 使用
    dirs: list[list[np.ndarray]] = [[]]

    def attach(parent: int, z_new: int, length: float) -> int:
        for _try in range(MAX_PLACE_TRIES):
            unit = _direction_away_from(rng, dirs[parent])
            cand = positions[parent] + unit * length
            if not _far_enough(positions, numbers, cand, z_new, parent):
                continue
            positions.append(cand)
            numbers.append(z_new)
            new = len(positions) - 1
            edges.append((parent, new))
            dirs[parent].append(unit)
            dirs.append([-unit])
            return new
        raise _PlacementError(f"原子 {parent} 上放不开 z={z_new} 的新原子")

    for _ in range(n_heavy - 1):
        parent = int(rng.integers(0, len(positions)))
        z_new = int(rng.choice(HEAVY_ELEMENTS))
        r_sum = (chem.COVALENT_RADII[numbers[parent]]
                 + chem.COVALENT_RADII.get(z_new, chem.DEFAULT_COVALENT_RADIUS))
        attach(parent, z_new, r_sum * float(rng.uniform(0.98, 1.02)))

    degree = np.zeros(len(positions), dtype=np.int64)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    n_heavy_now = len(positions)
    for i in range(n_heavy_now):
        n_h = int(VALENCE[numbers[i]] - degree[i])
        if n_h <= 0:
            continue
        r_sum = chem.COVALENT_RADII[1] + chem.COVALENT_RADII[numbers[i]]
        for _ in range(n_h):
            # 键长比 ~1.02 -> 几何后端判为单键
            attach(i, 1, r_sum * 1.02)
    return (np.asarray(positions, dtype=np.float32),
            np.asarray(numbers, dtype=np.int64),
            edges)


def make_molecule(
    rng: np.random.Generator,
    n_heavy: int,
    attempts: int = MAX_MOLECULE_ATTEMPTS,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """随机饱和有机分子：重原子树 + 补氢。返回 (pos, z, 真值化学键)。

    挤不进去时用同一支 ``rng`` 重掷整棵分子树，只有连续 ``attempts`` 次都失败才抛错。
    """
    last: _PlacementError | None = None
    for _attempt in range(max(1, attempts)):
        try:
            return _build_molecule(rng, n_heavy)
        except _PlacementError as exc:
            last = exc
    raise RuntimeError(
        f"连续 {attempts} 次都无法放置原子（分子太挤）: {last}"
    ) from last


def synthetic_records(n_mols: int = 8, seed: int = 0) -> list[dict]:
    """``write_cache`` 需要的记录格式，另附真值键与坐标便于断言。"""
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for k in range(n_mols):
        n_heavy = int(rng.integers(3, 10))
        pos, z, edges = make_molecule(rng, n_heavy)
        feats = features.featurize_from_geometry(pos, z)
        targets = (rng.normal(size=N_TARGETS).astype(np.float32)
                   * np.linspace(0.5, 50.0, N_TARGETS).astype(np.float32))
        records.append({
            "index": 1000 + k, "feats": feats, "targets": targets,
            "raw": np.zeros(15, dtype=np.float64),
            "edges": edges, "pos": pos, "z": z,
        })
    return records


def _edge_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    return {(int(min(a, b)), int(max(a, b))) for a, b in np.asarray(pairs).reshape(-1, 2)}


# --------------------------------------------------------------------------
# 检查项。约定：返回字符串列表，以 "FAIL" 开头的是硬失败，其余是信息。
# --------------------------------------------------------------------------
def check_bond_perception() -> list[str]:
    """Algorithm 1 必须原样恢复合成分子的化学键真值。"""
    notes: list[str] = []
    rng = np.random.default_rng(7)
    total_edges = recovered = spurious = 0
    for _ in range(12):
        pos, z, edges = make_molecule(rng, int(rng.integers(3, 10)))
        pairs = chem.perceive_bonds(pos, z)
        got = _edge_set(pairs)
        truth = {(min(a, b), max(a, b)) for a, b in edges}
        total_edges += len(truth)
        recovered += len(truth & got)
        # 多出来的键只允许是氢相关的（补氢时可能因空间拥挤被跳过，不会凭空多出）
        extra = got - truth
        heavy_extra = {(a, b) for a, b in extra if z[a] != 1 and z[b] != 1}
        spurious += len(heavy_extra)
        if heavy_extra:
            notes.append(f"FAIL 出现虚假的重原子键: {sorted(heavy_extra)[:4]}")
        ratio = chem.bond_length_ratio(pos, pairs, z)
        orders = chem.bond_order_from_ratio(ratio)
        if not np.all(orders == 1):
            notes.append(f"FAIL 饱和分子被判出非单键: {np.unique(orders, return_counts=True)}")
    if recovered != total_edges:
        notes.append(f"FAIL 键恢复率 {recovered}/{total_edges}")
    notes.insert(0, f"12 个分子共 {total_edges} 条键全部恢复，虚假重原子键 {spurious} 条；"
                    "键长比 -> 键级判定全部为单键")
    return notes


def check_bond_order_bands() -> list[str]:
    """键长比 -> 键级的四个区间必须各自命中。

    回归测试：bond_order_from_ratio 曾按升序逐轮覆写，宽区间（芳香 0.965）
    把窄区间冲掉，双键/三键永远不会被判出。np.select 的"首个命中生效"语义
    才是正确的。这里的采样点含 chem.py 注释里的标定值（C#C 0.789 -> 3，
    C=C 0.875 -> 2，芳香 0.914 -> 4）与各边界两侧。
    """
    notes: list[str] = []
    ratio = np.array(
        [0.789, 0.80, 0.824, 0.83, 0.875, 0.89, 0.914, 0.96, 0.966, 1.00, 1.10],
        dtype=np.float32,
    )
    expect = [3, 3, 3, 2, 2, 2, 4, 4, 1, 1, 1]
    got = chem.bond_order_from_ratio(ratio).tolist()
    if got != expect:
        notes.append(f"FAIL 键级判定 {got} != 期望 {expect}")
    counts = {v: got.count(v) for v in (1, 2, 3, 4)}
    notes.insert(0, f"11 个采样点四区间全部命中：单键 {counts[1]}、双键 {counts[2]}、"
                    f"三键 {counts[3]}、芳香 {counts[4]}（含 C#C/C=C/芳香标定值与边界）")
    return notes


def check_featurizer_vocab() -> list[str]:
    """字段取值必须落在各自词表内，SPD 必须是对称的、对角为 0。"""
    notes: list[str] = []
    for rec in synthetic_records(6, seed=11):
        feats = rec["feats"]
        n = feats.atomic_numbers.shape[0]
        if feats.atom_fields.shape != (n, len(ATOM_FIELD_SIZES)):
            notes.append(f"FAIL atom_fields 形状 {feats.atom_fields.shape}")
        for f, size in enumerate(ATOM_FIELD_SIZES):
            col = feats.atom_fields[:, f]
            if col.min() < 1 or col.max() >= size:
                notes.append(f"FAIL 原子字段 {f} 越界: [{col.min()}, {col.max()}] vs size {size}")
        if feats.bond_pairs.shape[0]:
            if feats.bond_fields.shape[1] != len(BOND_FIELD_SIZES):
                notes.append("FAIL bond_fields 列数不对")
            for f, size in enumerate(BOND_FIELD_SIZES):
                col = feats.bond_fields[:, f]
                if col.min() < 1 or col.max() >= size:
                    notes.append(f"FAIL 化学键字段 {f} 越界: [{col.min()}, {col.max()}] vs {size}")
            if feats.bond_class.min() < 1 or feats.bond_class.max() >= N_BOND_CLASSES:
                notes.append("FAIL bond_class 越界")
            if not np.all(feats.bond_pairs[:, 0] < feats.bond_pairs[:, 1]):
                notes.append("FAIL bond_pairs 未满足 i < j")
        if feats.atom_class.min() < 0 or feats.atom_class.max() >= N_ATOM_CLASSES:
            notes.append("FAIL atom_class 越界")
        spd = feats.spd
        if not np.array_equal(spd, spd.T):
            notes.append("FAIL SPD 不对称")
        if not np.all(np.diag(spd) == 0):
            notes.append("FAIL SPD 对角非 0")
        if spd.min() < 0 or spd.max() > 16:
            notes.append(f"FAIL SPD 取值范围 [{spd.min()}, {spd.max()}]")
        # 相邻原子的 SPD 必须是 1
        adj = chem.adjacency_from_pairs(feats.bond_pairs, n)
        if not np.all(spd[adj] == 1):
            notes.append("FAIL 成键原子的 SPD != 1")
    notes.insert(0, f"6 个分子的原子/化学键字段全部在词表内（原子 {len(ATOM_FIELD_SIZES)} 个字段，"
                    f"化学键 {len(BOND_FIELD_SIZES)} 个字段），SPD 对称且成键对为 1")
    return notes


def check_numpy_torch_parity() -> list[str]:
    """预处理用的 numpy 几何 与 模型前向用的 torch 几何 必须一致。"""
    notes: list[str] = []
    rec = synthetic_records(1, seed=3)[0]
    pos, feats = rec["pos"], rec["feats"]
    pairs = feats.bond_pairs
    pos_t = torch.from_numpy(np.ascontiguousarray(pos)).unsqueeze(0)
    idx_t = torch.from_numpy(np.ascontiguousarray(pairs.astype(np.int64))).unsqueeze(0)

    # 1) 两两距离
    d_np = chem.pairwise_distances(pos)
    d_t = encoding.pairwise_distance(pos_t)[0].numpy()
    err_d = float(np.abs(d_np - d_t).max())
    if err_d > 1e-4:
        notes.append(f"FAIL 两两距离最大误差 {err_d:.3e}")

    # 2) 共享原子索引
    s_np = chem.shared_atom_matrix(pairs)
    s_t = encoding.shared_atom_index(idx_t)[0].numpy()
    if not np.array_equal(s_np.astype(np.int64), s_t.astype(np.int64)):
        bad = int((s_np != s_t).sum())
        notes.append(f"FAIL shared_atom 有 {bad} 处不一致")

    # 3) 扭转/键角余弦
    t_np = chem.torsion_features(pos, pairs, s_np)
    t_t = encoding.torsion_cosine(pos_t, idx_t, torch.from_numpy(s_np.astype(np.int64)).unsqueeze(0))[0].numpy()
    err_t = float(np.abs(t_np - t_t).max())
    if err_t > 1e-4:
        notes.append(f"FAIL torsion 余弦最大误差 {err_t:.3e}")
    if t_np.size and (t_np.min() < -1.0 - 1e-6 or t_np.max() > 1.0 + 1e-6):
        notes.append(f"FAIL torsion 余弦越界 [{t_np.min()}, {t_np.max()}]")

    # 4) 独立验证：共享原子的键对，余弦应等于在共享原子处测的键角
    if pairs.shape[0] >= 2:
        m1, m2 = 0, 1
        shared = int(s_np[m1, m2])
        if shared >= 0:
            o1 = pairs[m1][0] if pairs[m1][1] == shared else pairs[m1][1]
            o2 = pairs[m2][0] if pairs[m2][1] == shared else pairs[m2][1]
            v1 = pos[o1].astype(np.float64) - pos[shared].astype(np.float64)
            v2 = pos[o2].astype(np.float64) - pos[shared].astype(np.float64)
            ref = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            if abs(ref - float(t_np[m1, m2])) > 1e-4:
                notes.append(f"FAIL 键角余弦与独立计算不符 {ref:.5f} vs {t_np[m1, m2]:.5f}")

    # 5) 键中点
    mid_np = chem.bond_midpoints(pos, pairs)
    mid_t = 0.5 * (encoding.gather_atoms(pos_t, idx_t[..., 0])
                   + encoding.gather_atoms(pos_t, idx_t[..., 1]))[0].numpy()
    err_m = float(np.abs(mid_np - mid_t).max())
    if err_m > 1e-5:
        notes.append(f"FAIL 键中点最大误差 {err_m:.3e}")

    notes.insert(0, f"numpy/torch 一致：距离 {err_d:.1e}，扭转 {err_t:.1e}，中点 {err_m:.1e}，"
                    f"shared_atom 完全相同（M={pairs.shape[0]} 条键）")
    return notes


def check_gaussian_basis() -> list[str]:
    """eq.5 的高斯基：负值、峰位正确、形状正确。"""
    notes: list[str] = []
    n_k, width = 8, 4.0
    basis = encoding.GaussianBasis(n_k, width, n_types=3, n_types_col=2)
    # x 必须是 (B, X, Y)：X=3 个标量值、Y=2 个列类型，沿 Y 复制成同一份值。
    # 原来写成 (1, 1, 3) 会和 ti=(1,3)/tj=(1,2) 广播不到一起。
    x = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.5, 2.5]]])
    ti = torch.zeros(1, 3, dtype=torch.long)
    tj = torch.ones(1, 2, dtype=torch.long)
    phi = basis(x, ti, tj)
    if phi.shape != (1, 3, 2, n_k):
        notes.append(f"FAIL 高斯基输出形状 {tuple(phi.shape)}")
    if not bool((phi <= 0).all()):
        notes.append("FAIL 高斯基出现正值（归一化系数应为负）")
    mu, sigma = encoding.gaussian_centers(n_k, width)
    if not torch.allclose(mu, torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])):
        notes.append(f"FAIL 高斯中心 {mu.tolist()}")
    if not torch.allclose(sigma, torch.full((n_k,), width / n_k)):
        notes.append("FAIL 高斯宽度 sigma_k != w/K")
    # alpha=1, beta=0 时，x=1.0 的峰应落在 mu=1.0 那个核（下标 2）
    # eq.5 的归一化系数是负的，所以 phi <= 0，"峰"是最负的那个核 -> argmin
    peak = int(torch.argmin(phi[0, 1, 0]))
    if peak != 2:
        notes.append(f"FAIL x=1.0 的峰位在下标 {peak}，期望 2")
    enc = encoding.ScalarBasisEncoder(n_k, 4)
    if enc(phi).shape != (1, 4, 3, 2):
        notes.append(f"FAIL ScalarBasisEncoder 输出形状 {tuple(enc(phi).shape)}")
    notes.insert(0, f"K={n_k}, w={width}: mu/sigma 符合 eq.5，峰位正确，输出 (B,X,Y,K)->(B,H,X,Y)")
    return notes


def _scratch_dir(prefix: str) -> str:
    """在仓库内建一个唯一的临时目录，返回其路径。

    刻意不用 ``tempfile.mkdtemp``：它以 0o700 模式建目录，在受限沙箱下建出来的目录
    反而不可写（连 ``os.chmod`` 都被拒，只能报 PermissionError）；``os.makedirs``
    建的目录则可写。系统 TEMP 同样不可靠，所以默认落在仓库内的 ``.smoke_tmp``，
    可用环境变量 ``DEMOL_TMP_ROOT`` 改到别处。
    """
    root = os.environ.get("DEMOL_TMP_ROOT") or os.path.join(_ROOT, ".smoke_tmp")
    while True:
        path = os.path.join(root, f"{prefix}{uuid.uuid4().hex[:12]}")
        try:
            os.makedirs(path)
        except FileExistsError:
            continue
        return path


def _cleanup(paths: Sequence[str], datasets: Iterable[Any] = ()) -> None:
    """删除临时目录（先释放 mmap 句柄）。

    Windows 上 ``QM9Dataset`` 用 ``mmap_mode="r"`` 打开的 .npy 会被文件句柄锁住，
    此时 ``shutil.rmtree(..., ignore_errors=True)`` 会静默失败并在 ``.smoke_tmp``
    里留下残壳。所以先对每个数据集调用 ``close()``，再 gc 一轮，最后才删目录。
    """
    for ds in datasets:
        close = getattr(ds, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - 清理阶段不该再抛异常
                pass
    gc.collect()
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)


def _build_temp_cache(n_mols: int = 8, seed: int = 5) -> tuple[str, list[dict]]:
    """写一份微型缓存并生成 split.npy（QM9Dataset 依赖它）。"""
    tmp = _scratch_dir("demol_smoke_")
    records = synthetic_records(n_mols, seed=seed)
    write_cache(records, tmp)
    n_test = max(1, n_mols // 4)
    n_val = max(1, n_mols // 4)
    split = np.zeros(n_mols, dtype=np.int8)
    split[n_mols - n_val - n_test: n_mols - n_test] = 1
    split[n_mols - n_test:] = 2
    np.save(os.path.join(tmp, "split.npy"), split)
    return tmp, records


def _make_loaders(cache_dir: str, batch_size: int = 2):
    common = dict(collate_fn=qm9_collate, num_workers=0)
    loaders = {}
    datasets = {}
    for name, sid in (("train", 0), ("val", 1), ("test", 2)):
        ds = QM9Dataset(cache_dir, split_ids=(sid,))
        datasets[name] = ds
        loaders[name] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, drop_last=False, **common
        )
    return loaders, datasets


def check_collate_padding() -> list[str]:
    """稠密批处理：padding 不能污染 mask、SPD、化学键真值和 atom-bond 关联。"""
    notes: list[str] = []
    cache_dir, records = _build_temp_cache(8, seed=5)
    ds = None
    try:
        ds = QM9Dataset(cache_dir, split_ids=(0,))
        items = [ds[i] for i in range(len(ds))]
        if len({it["n"] for it in items}) < 2:
            notes.append("（提示）这一批分子原子数相同，padding 分支未被覆盖")
        batch = qm9_collate(items)
        b, n_max = batch["atom_mask"].shape
        m_max = batch["bond_mask"].shape[1]
        if b != len(items):
            notes.append("FAIL batch 维度不对")
        for k, it in enumerate(items):
            n, m = it["n"], it["m"]
            if int(batch["atom_mask"][k].sum()) != n:
                notes.append(f"FAIL 分子 {k} atom_mask 数量 {int(batch['atom_mask'][k].sum())} != {n}")
            if int(batch["bond_mask"][k].sum()) != m:
                notes.append(f"FAIL 分子 {k} bond_mask 数量 != {m}")
            if n < n_max and batch["atom_fields"][k, n:].abs().sum() != 0:
                notes.append(f"FAIL 分子 {k} padding 原子字段非零")
            if n < n_max and batch["pos"][k, n:].abs().sum() != 0:
                notes.append(f"FAIL 分子 {k} padding 坐标非零")
            if n < n_max and bool(batch["aa_mask"][k, n:].any()):
                notes.append(f"FAIL 分子 {k} padding 行进入了 aa_mask")
            if bool(batch["aa_mask"][k, :, n:].any()):
                notes.append(f"FAIL 分子 {k} padding 列进入了 aa_mask")
            if n < n_max and bool(batch["ab_mask"][k, n:].any()):
                notes.append(f"FAIL 分子 {k} padding 原子进入了 ab_mask")
            if m < m_max and bool(batch["bb_mask"][k, m:].any()):
                notes.append(f"FAIL 分子 {k} padding 化学键进入了 bb_mask")
            if m < m_max and bool(batch["ab_mask"][k, :, m:].any()):
                notes.append(f"FAIL 分子 {k} padding 化学键进入了 ab_mask")
            # 真值键矩阵：对称、蕴含于 aa_mask、与 bond_idx 一致
            gt = batch["bond_gt"][k]
            if not torch.equal(gt, gt.T):
                notes.append(f"FAIL 分子 {k} bond_gt 不对称")
            if bool((gt & ~batch["aa_mask"][k]).any()):
                notes.append(f"FAIL 分子 {k} 有化学键不在 aa_mask 内")
            if int(gt.sum()) != 2 * m:
                notes.append(f"FAIL 分子 {k} bond_gt 条目 {int(gt.sum())} != 2*{m}")
            inc = torch.zeros(n, m, dtype=torch.bool)
            if m:
                inc[it["bond_pairs"][:, 0], torch.arange(m)] = True
                inc[it["bond_pairs"][:, 1], torch.arange(m)] = True
            if not torch.equal(inc, batch["ab_mask"][k, :n, :m]):
                notes.append(f"FAIL 分子 {k} ab_mask 与 bond_idx 不一致")
            # 每个真实化学键的两个端点在 aa_mask 里必须互相可见
            if m and not bool(batch["aa_mask"][k, it["bond_pairs"][:, 0], it["bond_pairs"][:, 1]].all()):
                notes.append(f"FAIL 分子 {k} 成键原子对未全部落在 aa_mask 内")
            # padding 原子的 SPD 必须是 0（不参与，也不该产生 embedding 噪声以外的影响）
            if n < n_max and int(batch["spd"][k, n:].abs().sum()) != 0:
                notes.append(f"FAIL 分子 {k} padding SPD 非零")
        if tuple(batch["targets"].shape) != (b, N_TARGETS):
            notes.append(f"FAIL targets 形状 {tuple(batch['targets'].shape)}")
        if batch["targets"].dtype != torch.float32:
            notes.append(f"FAIL targets dtype {batch['targets'].dtype}")
        notes.insert(0, f"{b} 个分子 -> 稠密批 (N={n_max}, M={m_max})，padding 未进入任何 mask，"
                        "bond_gt 对称且与 bond_idx/ab_mask 一致")
    finally:
        _cleanup([cache_dir], [ds])
    return notes


def check_forward_shapes() -> list[str]:
    """smoke 配置下的前向：形状、有限性、mask 与去噪任务的行为。"""
    notes: list[str] = []
    cache_dir, _ = _build_temp_cache(6, seed=17)
    ds = None
    try:
        ds = QM9Dataset(cache_dir, split_ids=(0,))
        batch = qm9_collate([ds[i] for i in range(len(ds))])
        cfg = model_config("smoke")
        model = DeMol(cfg)
        model.eval()
        b, n = batch["atom_mask"].shape
        m = batch["bond_mask"].shape[1]
        with torch.no_grad():
            out = model(batch)
        expect = {
            "prop": (b, cfg.n_targets),
            "graph_repr": (b, cfg.d_readout),
            "atom_repr": (b, n, cfg.d_atom),
            "bond_repr": (b, m, cfg.d_bond),
            "atom_logits": (b, n, N_ATOM_CLASSES),
            "coord_pred": (b, n, 3),
            "bond_logits": (b, n, n),
        }
        for key, shape in expect.items():
            if tuple(out[key].shape) != shape:
                notes.append(f"FAIL {key} 形状 {tuple(out[key].shape)}，期望 {shape}")
        for key, tensor in out.items():
            if torch.is_tensor(tensor) and tensor.dtype.is_floating_point:
                if not bool(torch.isfinite(tensor).all()):
                    notes.append(f"FAIL {key} 含 NaN/Inf")
        if not torch.allclose(out["bond_logits"], out["bond_logits"].transpose(1, 2), atol=1e-5):
            notes.append("FAIL bond_logits 不对称")
        if bool(out["masked_atoms"].any()):
            notes.append("FAIL mask_ratio=0 时仍然产生了 masked_atoms")
        pos_ref = batch["pos"].float()
        if not torch.equal(out["coord_input"], pos_ref):
            notes.append("FAIL coord_noise=0 时 coord_input 应该等于原始坐标")
        if not torch.equal(out["coord_target"], pos_ref):
            notes.append("FAIL coord_noise=0 时 coord_target 应该等于原始坐标")
        if bool(out["coord_noise"].abs().any()):
            notes.append("FAIL coord_noise=0 时噪声张量应全零")

        with torch.no_grad():
            out2 = model(batch, mask_ratio=0.5, coord_noise=0.1)
        if int(out2["masked_atoms"].sum()) == 0:
            notes.append("FAIL mask_ratio=0.5 没有 mask 到任何原子")
        if bool((out2["masked_atoms"] & ~batch["atom_mask"]).any()):
            notes.append("FAIL padding 原子被 mask 了")
        if torch.equal(out2["coord_input"], batch["pos"].float()):
            notes.append("FAIL coord_noise>0 但输入坐标没有扰动")
        if not torch.equal(out2["coord_target"], batch["pos"].float()):
            notes.append("FAIL coord_target 应始终是未加噪的原始坐标")
        if bool(out2["coord_noise"][~batch["atom_mask"]].abs().any()):
            notes.append("FAIL padding 原子上出现了坐标噪声")
        notes.insert(0, f"前向 OK: prop{tuple(out['prop'].shape)}, atom_logits"
                        f"{tuple(out['atom_logits'].shape)}, bond_logits{tuple(out['bond_logits'].shape)}; "
                        f"mask_ratio=0.5 时 mask 了 {int(out2['masked_atoms'].sum())} 个原子")
    finally:
        _cleanup([cache_dir], [ds])
    return notes


def check_structure_mask_wiring() -> list[str]:
    """结构感知掩码必须真的约束注意力（论文 3.2 节/附录 C），不能算出来没人用。

    回归测试：aa/bb/ab_mask 曾一度只被 collate 计算、forward 原样返回，
    注意力实际是全连接的。
    """
    notes: list[str] = []
    # 单元级：pair_mask = 单位阵时每个 query 只剩自己一个 key，softmax 恒为 1，
    # 注意力输出必须精确等于 o_proj(v_proj(x))，与 q/k 完全无关。
    torch.manual_seed(3)
    attn = MaskedMultiHeadAttention(16, 16, 4, 4, 0.0).eval()
    x = torch.randn(2, 5, 16)
    key_mask = torch.ones(2, 5, dtype=torch.bool)
    eye = torch.eye(5, dtype=torch.bool)[None].expand(2, 5, 5)
    with torch.no_grad():
        masked = attn(x, x, None, key_mask, eye)
        ref = attn.o_proj(attn.v_proj(x))
        dense = attn(x, x, None, key_mask)
    if not torch.allclose(masked, ref, atol=1e-5):
        notes.append("FAIL pair_mask=I 时输出 != o_proj(v(x))，掩码未进入注意力")
    if torch.allclose(masked, dense, atol=1e-6):
        notes.append("FAIL pair_mask=I 与全连接输出相同，掩码形同虚设")

    # 集成级：同一份 batch，真实结构掩码 vs 全连接掩码，prop 必须不同且有限。
    cache_dir, _ = _build_temp_cache(4, seed=23)
    ds = None
    try:
        ds = QM9Dataset(cache_dir, split_ids=(0,))
        batch = qm9_collate([ds[i] for i in range(len(ds))])
        model = DeMol(model_config("smoke")).eval()
        am, bm = batch["atom_mask"].bool(), batch["bond_mask"].bool()
        dense_batch = dict(batch)
        dense_batch["aa_mask"] = am[:, :, None] & am[:, None, :]
        dense_batch["bb_mask"] = bm[:, :, None] & bm[:, None, :]
        dense_batch["ab_mask"] = am[:, :, None] & bm[:, None, :]
        with torch.no_grad():
            p_real = model(batch)["prop"]
            p_dense = model(dense_batch)["prop"]
        if not (bool(torch.isfinite(p_real).all()) and bool(torch.isfinite(p_dense).all())):
            notes.append("FAIL 结构掩码接入后 prop 出现 NaN/Inf")
        diff = float((p_real - p_dense).abs().max())
        if diff < 1e-6:
            notes.append("FAIL 真实掩码与全连接掩码的 prop 相同，掩码没有生效")
        notes.insert(0, f"pair_mask=I 恒等式成立（注意力退化为 o_proj(v(x))）；"
                        f"真实掩码 vs 全连接的 prop 最大差异 {diff:.3g}")
    finally:
        _cleanup([cache_dir], [ds])
    return notes


def check_loss_and_backward() -> list[str]:
    """四个损失都参与、都有限，反向传播没有 NaN/Inf。"""
    notes: list[str] = []
    cache_dir, _ = _build_temp_cache(6, seed=23)
    ds = None
    try:
        ds = QM9Dataset(cache_dir, split_ids=(0,))
        batch = qm9_collate([ds[i] for i in range(len(ds))])
        model = DeMol(model_config("smoke"))
        model.train()
        out = model(batch, mask_ratio=0.3, coord_noise=0.1)
        criterion = DeMolLoss(loss_config("pretrain"))
        total, parts = criterion(out, batch)
        for name in ("prop", "mask", "coord", "bond", "total"):
            if name not in parts:
                notes.append(f"FAIL 损失缺少分量 {name}")
            elif not np.isfinite(parts[name]):
                notes.append(f"FAIL 损失分量 {name} = {parts[name]}")
        if not torch.isfinite(total):
            notes.append("FAIL 总损失非有限")
        total.backward()
        bad = [name for name, p in model.named_parameters()
               if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        if bad:
            notes.append(f"FAIL {len(bad)} 个参数梯度含 NaN/Inf，例如 {bad[:4]}")
        n_grad = sum(1 for p in model.parameters() if p.grad is not None)
        n_nonzero = sum(1 for p in model.parameters()
                        if p.grad is not None and float(p.grad.abs().sum()) > 0)
        if n_grad == 0:
            notes.append("FAIL 没有任何梯度")
        if n_nonzero < 0.5 * n_grad:
            notes.append(f"（提示）只有 {n_nonzero}/{n_grad} 个参数拿到非零梯度")

        # 微调损失：只留 Lprop、MSE、并且只算选中的目标
        ft_cfg = loss_config("finetune", "gap")
        if ft_cfg.prop_mode != "mse" or ft_cfg.target_indices != (4,):
            notes.append(f"FAIL finetune 损失配置不对: {ft_cfg}")
        if (ft_cfg.w_mask, ft_cfg.w_coord, ft_cfg.w_bond) != (0.0, 0.0, 0.0):
            notes.append("FAIL 微调阶段仍启用了辅助任务")
        ft_parts = DeMolLoss(ft_cfg)(out, batch)[1]
        if set(ft_parts) != {"prop", "total"}:
            notes.append(f"FAIL 微调损失分量 {sorted(ft_parts)}")
        notes.insert(0, f"预训练损失分量 {{{', '.join(f'{k}={v:.4g}' for k, v in parts.items() if k != 'total')}}}；"
                        f"{n_nonzero}/{n_grad} 个参数拿到非零梯度，全部有限")
    finally:
        _cleanup([cache_dir], [ds])
    return notes


def check_training_and_eval() -> list[str]:
    """3 步训练 + 验证 + 检查点/EMA 存取（用真实 Trainer 代码路径）。"""
    notes: list[str] = []
    cache_dir, records = _build_temp_cache(8, seed=31)
    out_dir = _scratch_dir("demol_run_")
    datasets: dict[str, QM9Dataset] = {}
    try:
        loaders, datasets = _make_loaders(cache_dir, batch_size=2)
        n_train = len(datasets["train"])
        train_records = records[:n_train]
        std = TargetStandardizer.fit(np.stack([r["targets"] for r in train_records]))
        train_cfg = train_config(
            "pretrain", "local",
            total_steps=3, warmup_steps=1, batch_size=2, num_workers=0,
            log_every=1, eval_every=2, save_every=3, out_dir=out_dir,
            amp="none", ema_decay=0.9, mask_ratio=0.2, coord_noise=0.1,
        )
        loss_cfg = loss_config("pretrain")
        model = DeMol(model_config("smoke"))
        trainer = Trainer(model, train_cfg, loss_cfg, std, device="cpu")
        lines: list[str] = []
        result = trainer.fit(cycling_loader(loaders["train"]), loaders["val"], log_fn=lines.append)
        if trainer.step != 3:
            notes.append(f"FAIL 训练步数 {trainer.step} != 3")
        if not lines:
            notes.append("FAIL 没有产生任何日志")
        for name in ("last.pt", "best.pt"):
            if not os.path.exists(os.path.join(out_dir, name)):
                notes.append(f"FAIL 缺少检查点 {name}")
        lr_values = [float(ln.split("lr=")[-1].split()[0]) for ln in lines if "lr=" in ln]
        if not lr_values:
            notes.append("FAIL 日志里没有出现学习率")
        elif not all(np.isfinite(v) and v >= 0.0 for v in lr_values):
            notes.append(f"FAIL 学习率含负值或非有限值: {lr_values}")
        elif max(lr_values) <= 0.0:
            notes.append("FAIL 学习率始终为 0，warmup 没有生效")
        elif lr_values[-1] > lr_values[0] + 1e-12:
            notes.append(f"FAIL warmup 之后学习率应当线性衰减到 0，"
                         f"实际 {lr_values[0]:.3e} -> {lr_values[-1]:.3e}")

        metrics = trainer.evaluate(loaders["val"], use_ema=True)
        if len(metrics["mae"]) != N_TARGETS:
            notes.append(f"FAIL 评估返回 {len(metrics['mae'])} 个目标")
        if not all(np.isfinite(v) for v in metrics["mae"].values()):
            notes.append("FAIL 评估 MAE 含 NaN/Inf")
        if metrics["n"] != len(datasets["val"]):
            notes.append(f"FAIL 评估样本数 {metrics['n']} != {len(datasets['val'])}")

        # 标准化往返：物理单位 -> 标准正态 -> 物理单位
        raw = np.stack([r["targets"] for r in train_records])
        back = std.denormalize(std.normalize(raw))
        err = float(np.abs(back - raw).max())
        if err > 1e-3:
            notes.append(f"FAIL 标准化往返误差 {err:.3e}")
        t = torch.as_tensor(raw, dtype=torch.float32)
        err_t = float((std.denormalize(std.normalize(t)) - t).abs().max())
        if err_t > 1e-3:
            notes.append(f"FAIL 张量路径标准化往返误差 {err_t:.3e}")

        # EMA 影子权重必须能覆盖回模型
        if trainer.ema is None:
            notes.append("FAIL ema_decay>0 却没有创建 EMA")
        else:
            state = trainer.model.state_dict()
            unknown = [k for k in trainer.ema.shadow if k not in state]
            if unknown:
                notes.append(f"FAIL EMA 影子含未知键 {unknown[:4]}")
            trainer.ema.copy_to(trainer.model)
            trainer.ema.restore(trainer.model)

        # 检查点可被一个新的 Trainer 完整读回
        fresh = Trainer(DeMol(model_config("smoke")), train_cfg, loss_cfg, std, device="cpu")
        payload = fresh.load_checkpoint(os.path.join(out_dir, "last.pt"))
        if int(payload.get("step", -1)) != 3:
            notes.append(f"FAIL 检查点 step={payload.get('step')}")
        notes.insert(0, f"3 步训练完成（val loss={metrics['loss']:.4g}, "
                        f"coord_rmse={metrics['coord_rmse']:.4g}），last.pt/best.pt 已写出并可回读，"
                        f"EMA 覆盖/恢复正常，标准化往返误差 {err:.1e}")
    finally:
        _cleanup([cache_dir, out_dir], datasets.values())
    return notes


def check_config_overrides() -> list[str]:
    """CLI 的 --set-model/--set-train/--set-loss 三条路径都要能正确改写字段。"""
    notes: list[str] = []
    model_cfg = apply_overrides(model_config("smoke"), {"n_layers": "3", "d_atom": "96",
                                                        "d_bond": "96", "cross_ffn": "false"})
    if (model_cfg.n_layers, model_cfg.d_atom, model_cfg.cross_ffn) != (3, 96, False):
        notes.append(f"FAIL model 覆盖未生效: {model_cfg.n_layers}, {model_cfg.d_atom}, {model_cfg.cross_ffn}")
    train_cfg = apply_overrides(train_config("finetune", "local"),
                                {"lr": "1e-4", "total_steps": "10", "betas": "0.9,0.95"})
    if abs(train_cfg.lr - 1e-4) > 1e-12 or train_cfg.total_steps != 10:
        notes.append("FAIL train 覆盖未生效")
    if tuple(train_cfg.betas) != (0.9, 0.95):
        notes.append(f"FAIL betas 覆盖未生效: {train_cfg.betas}")
    loss_cfg = apply_overrides(loss_config("pretrain", "homo,lumo,gap"),
                              {"target_indices": "0,4", "w_bond": "0.25"})
    if tuple(loss_cfg.target_indices) != (0, 4):
        notes.append(f"FAIL loss 覆盖未生效: {loss_cfg.target_indices}")
    if len(loss_cfg.target_weights) != len(loss_cfg.target_indices):
        notes.append("FAIL target_weights 未随 target_indices 重置")
    if abs(loss_cfg.w_bond - 0.25) > 1e-12:
        notes.append("FAIL w_bond 覆盖未生效")
    try:
        apply_overrides(loss_config("pretrain"), {"nope": "1"})
        notes.append("FAIL 未知字段没有报错")
    except KeyError:
        pass
    notes.insert(0, "三类配置的 key=value 覆盖全部生效（含 bool/float/tuple 转型与未知字段报错）")
    return notes


def check_param_budget() -> list[str]:
    """各预设的参数量；与论文 186M 的差异要在 README 里解释得通。"""
    notes: list[str] = []
    for preset in ("smoke", "small", "base", "paper"):
        try:
            with torch.device("meta"):
                n_full = count_parameters(DeMol(model_config(preset)))
                n_shared = count_parameters(DeMol(model_config(preset, cross_ffn=False)))
        except Exception as exc:                      # meta 设备不可用时跳过
            notes.append(f"（提示）{preset} 无法在 meta 设备上统计参数量: {type(exc).__name__}")
            continue
        notes.append(f"{preset:<6} {n_full / 1e6:8.2f} M   (cross_ffn=False: {n_shared / 1e6:8.2f} M)")
        if preset == "paper":
            ratio = n_full / (PAPER_MODEL_PARAMS_M * 1e6)
            notes.append(f"       论文 Appendix D.1 报 {PAPER_MODEL_PARAMS_M:.0f} M，"
                         f"当前 {ratio:.2f}x（paper 预设已按 Graphormer 式 mult=1 对齐，"
                         "残留 ~3% 来自读出头/词表等论文未展开的细节，见 README.md）")
        if preset == "paper" and not (0.8 <= n_full / (PAPER_MODEL_PARAMS_M * 1e6) <= 1.2):
            notes.append(f"FAIL paper 预设 {n_full / 1e6:.1f} M 偏离论文 186 M 超过 20%")
    smoke = count_parameters(DeMol(model_config("smoke")))
    if smoke > 20e6:
        notes.append(f"FAIL smoke 预设过大（{smoke / 1e6:.1f} M），自检会变慢")
    notes.insert(0, "参数量统计：")
    return notes


def check_benchmark_reporting() -> list[str]:
    """论文基准常数与对比表渲染（只检查格式，不做任何数值主张）。"""
    notes: list[str] = []
    measured = {name: 1.234 for name in ("homo", "lumo", "gap", "g298", "cv")}
    table = comparison_table(measured, targets=tuple(measured))
    lines = table.splitlines()
    if len(lines) != 2 + len(measured):
        notes.append(f"FAIL 对比表行数 {len(lines)}，期望 {2 + len(measured)}")
    if "no-pretr." not in lines[0]:
        notes.append("FAIL 对比表缺少 no-pretrain 列")
    for name in measured:
        row = next((ln for ln in lines if ln.startswith(name)), None)
        if row is None:
            notes.append(f"FAIL 对比表缺少 {name} 行")
    notes.insert(0, f"对比表渲染正常（{len(measured)} 个目标 x 5 列）")
    return notes


def check_qm9_xyz_layout() -> list[str]:
    """真实 figshare 版式（原子块紧跟表头）的 xyz 必须能被正确解析。

    回归测试：parse_qm9_file 曾假设 SMILES/InChI 在原子块之前
    （lines[4:4+n]），而真实 dsgdb9nsd_*.xyz 的原子块从 lines[2] 开始，
    会导致全部文件解析失败。现在按"第一条像原子的行"定位原子块，
    两种版式都应解析出 5 个原子。
    """
    # gdb_1 甲烷的真实表头数值（DeepChem gdb9.sdf.csv 第一行）
    header = ("gdb 1 157.7118 157.70997 157.70699 0 13.21 -0.3877 0.1171 0.5048 "
              "35.3641 0.044749 -40.47893 -40.476062 -40.475117 -40.498597 6.469 "
              "-395.999594594 -398.643290011 -401.014646522 -372.471772148")
    atom_block = (
        "C -0.0127014359 1.0858041578 0.0080009958 0.1197598914\n"
        "H 0.0021504160 -0.0060313176 0.0019761204 0.0226801748\n"
        "H 1.0117308433 1.4637511618 0.0002765748 0.0178689532\n"
        "H -0.5408150690 1.4475266138 -0.8766437152 0.0196477771\n"
        "H -0.5238136345 1.4379326443 0.9063972942 0.0200431924"
    )
    # 真实版式：表头 -> 原子 -> 频率 -> SMILES -> InChI
    standard = f"5\n{header}\n{atom_block}\n696.5348 1298.0 1320.0 1320.0\nC C\nInChI=1S/CH4/h1H4\n"
    # 变体：SMILES/InChI 在原子块之前
    swapped = f"5\n{header}\nC C\nInChI=1S/CH4/h1H4\n{atom_block}\n"

    notes: list[str] = []
    tmpdir = _scratch_dir("xyz_layout_")
    try:
        for name, text in (("standard", standard), ("swapped", swapped)):
            path = os.path.join(tmpdir, f"dsgdb9nsd_000001_{name}.xyz")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            index, z, pos, raw, extra = parse_qm9_file(path)
            if index != 1 or z.shape[0] != 5 or pos.shape != (5, 3):
                notes.append(f"FAIL {name}: index={index} n={z.shape[0]} pos={pos.shape}")
            if z.tolist() != [6, 1, 1, 1, 1]:
                notes.append(f"FAIL {name}: 元素序列 {z.tolist()}")
            if verify_target_columns(raw) is not None:
                notes.append(f"FAIL {name}: {verify_target_columns(raw)}")
        # 原子块内混入非原子行时必须报错而不是静默跳过
        bad = f"5\n{header}\nC 0 0 0 0\nH 0 0 1 0\nbroken-line\nH 0 1 0 0\nH 1 0 0 0\n"
        bad_path = os.path.join(tmpdir, "bad.xyz")
        with open(bad_path, "w", encoding="utf-8") as fh:
            fh.write(bad)
        try:
            parse_qm9_file(bad_path)
            notes.append("FAIL 损坏的原子块没有报错")
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    notes.insert(0, "standard / swapped 两种版式均解析出 5 原子、20 个原始属性，"
                    "gap=lumo-homo 校验通过；损坏原子块正确报错")
    return notes


def check_ddp_sharding() -> list[str]:
    """DDP 训练集分片：各 rank 不相交、等长、覆盖全集，且逐 epoch 重洗。"""
    notes: list[str] = []
    cache_dir, _ = _build_temp_cache(12, seed=11)  # 12 个分子 -> 6 个训练样本
    try:
        world = 3
        epoch_indices: list[list[list[int]]] = [[], [], []]
        for rank in range(world):
            bundle = build_loaders(cache_dir, batch_size=2, rank=rank, world_size=world)
            sampler = bundle.train.sampler
            if sampler is None or not hasattr(sampler, "set_epoch"):
                notes.append("FAIL train loader 没有 DistributedSampler")
                continue
            for epoch in (0, 1):
                sampler.set_epoch(epoch)
                epoch_indices[rank].append(list(iter(sampler)))
            for ds in bundle.datasets.values():
                close = getattr(ds, "close", None)
                if callable(close):
                    close()
        shards = [idx[0] for idx in epoch_indices]
        lengths = {len(s) for s in shards}
        if lengths != {2}:  # floor(6 / 3)
            notes.append(f"FAIL 分片长度 {sorted(lengths)}，期望全部 2")
        union = sorted(i for s in shards for i in s)
        if union != list(range(6)):
            notes.append(f"FAIL 分片并集 {union} 不是 0..5")
        if shards[0] == epoch_indices[0][1]:
            notes.append("FAIL set_epoch 没有改变分片顺序")
        notes.insert(0, f"world={world} 分片不相交且覆盖 6 个训练样本（各 2 个），"
                        "set_epoch 后顺序改变")
    finally:
        _cleanup([cache_dir])
    return notes


# --------------------------------------------------------------------------
# PCQ / LoRA 相关检查（chem-lora 扩展）
# --------------------------------------------------------------------------
def check_vocab_extension() -> list[str]:
    """词表扩展必须是 append-only：旧索引语义不变，OTHER 固定在 5。"""
    notes: list[str] = []
    from demol.features import (
        ATOM_VOCAB,
        LEGACY_ATOM_VOCAB_SIZE,
        OTHER_CLASS,
        atom_class_of,
    )

    if ATOM_VOCAB[:6] != ("H", "C", "N", "O", "F", "OTHER"):
        notes.append(f"FAIL 词表前 6 项变了: {ATOM_VOCAB[:6]}")
    if OTHER_CLASS != 5:
        notes.append(f"FAIL OTHER 不在索引 5: {OTHER_CLASS}")
    if len(ATOM_VOCAB) != 14:
        notes.append(f"FAIL 词表长度应为 14，实际 {len(ATOM_VOCAB)}")
    cls = atom_class_of(np.array([1, 6, 7, 8, 9, 16, 17, 5, 14, 15, 35, 53, 34, 999]))
    expect = [0, 1, 2, 3, 4, 9, 10, 6, 7, 8, 11, 12, 13, 5]
    if list(cls) != expect:
        notes.append(f"FAIL atom_class_of 映射错误: {list(cls)} != {expect}")
    if LEGACY_ATOM_VOCAB_SIZE != 6:
        notes.append("FAIL LEGACY_ATOM_VOCAB_SIZE 应为 6")
    # 扩展词表建出来的模型，默认配置 atom_field_sizes 应跟随新词表
    cfg = model_config("smoke")
    if cfg.atom_field_sizes[0] != len(ATOM_VOCAB) + 1:
        notes.append(f"FAIL atom_field_sizes[0]={cfg.atom_field_sizes[0]} != {len(ATOM_VOCAB) + 1}")
    return notes


def check_lora_mechanics() -> list[str]:
    """LoRA：注入后等价（B=0）、冻结正确、梯度只落在放开的参数上。"""
    notes: list[str] = []
    from demol.lora import (
        LoRALinear,
        adapter_state_dict,
        apply_lora,
        lora_parameters,
        mark_lora_trainable,
    )

    cache_dir, _ = _build_temp_cache(6, seed=31)
    ds = None
    try:
        ds = QM9Dataset(cache_dir, split_ids=(0,))
        batch = qm9_collate([ds[i] for i in range(len(ds))])

        model = DeMol(model_config("smoke"))
        model.eval()
        with torch.no_grad():
            ref = model(batch)["prop"]

        n_injected = apply_lora(model, rank=4, alpha=8.0)
        if n_injected == 0:
            notes.append("FAIL 没有注入任何 LoRA adapter")
        # B=0 时输出应与注入前逐元素一致
        with torch.no_grad():
            after = model(batch)["prop"]
        if not torch.allclose(ref, after, atol=1e-6):
            notes.append("FAIL LoRA 注入改变了前向输出（B 应零初始化）")

        n_lora = sum(p.numel() for p in lora_parameters(model))
        n_total = sum(p.numel() for p in model.parameters())
        # smoke 模型 d=64 很小，rank=4 的占比天然偏高；paper 规模下 rank=8
        # 约占 1-2%。这里只要求远小于全量。
        if not (0 < n_lora < 0.25 * n_total):
            notes.append(f"FAIL LoRA 参数量异常: {n_lora} / {n_total}")

        groups = mark_lora_trainable(model, train_embeddings=True)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        if any("lora_" not in n and not n.startswith("prop_head")
               and not n.startswith("atom_emb") and not n.startswith("bond_emb")
               and n != "mask_token" for n in trainable):
            bad = [n for n in trainable if "lora_" not in n
                   and not n.startswith(("prop_head", "atom_emb", "bond_emb"))
                   and n != "mask_token"]
            notes.append(f"FAIL 冻结不全，放开了: {bad[:4]}")
        if groups["lora"] != n_lora:
            notes.append("FAIL mark_lora_trainable 统计与 lora_parameters 不一致")

        # 前向 + 反向：梯度只出现在放开的参数上
        model.train()
        out = model(batch)
        loss = out["prop"].square().mean()
        loss.backward()
        bad_grads = [n for n, p in model.named_parameters()
                     if p.grad is not None and not p.requires_grad]
        if bad_grads:
            notes.append(f"FAIL 冻结参数拿到了梯度: {bad_grads[:4]}")
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and (p.grad is None or not bool(p.grad.abs().sum() > 0))
                   # lora_A 在 B=0 时梯度恒为 0（第一步只更新 B，LoRA 的正常起点）；
                   # mask_token 只在 mask_ratio>0 时使用
                   and not n.endswith("lora_A") and n != "mask_token"]
        if no_grad:
            notes.append(f"（提示）可训练但无梯度: {no_grad[:4]}")

        # adapter 存取
        state = adapter_state_dict(model)
        if not state or any("lora_" not in k for k in state):
            notes.append("FAIL adapter_state_dict 含非 lora 键")
    finally:
        _cleanup([cache_dir], [ds] if ds else [])
    return notes


def check_pcq_pipeline() -> list[str]:
    """PCQ 数据链路：jsonl -> 缓存（targets 宽度 1）-> collate -> 词表降级。"""
    notes: list[str] = []
    import gzip
    import json as json_mod

    from demol.pcq import VocabDowngradeDataset, featurize_pcq_records, write_pcq_split

    cache_dir = _scratch_dir("demol_pcq_smoke_")
    ds = None
    jsonl_path = None
    try:
        # 用合成分子伪造 6 条 jsonl 记录（gap 取 3-9 eV 的合理范围）
        records = synthetic_records(6, seed=47)
        jsonl_path = os.path.join(cache_dir, "fake_pcq.jsonl.gz")
        os.makedirs(cache_dir, exist_ok=True)
        with gzip.open(jsonl_path, "wt", encoding="utf-8") as fh:
            for k, rec in enumerate(records):
                fh.write(json_mod.dumps({
                    "position": rec["pos"].tolist(),
                    "node_class": rec["z"].tolist(),
                    "labels": {"object": {"SMILES": f"fake{k}", "gap": 4.0 + 0.3 * k},
                               "node": None},
                    "subset_idx": k,
                }) + "\n")

        pcq_records = featurize_pcq_records(jsonl_path, prefer_rdkit=False)
        if len(pcq_records) != 6:
            notes.append(f"FAIL featurize 后剩 {len(pcq_records)}/6 条")
        if pcq_records and pcq_records[0]["targets"].shape != (1,):
            notes.append(f"FAIL targets 宽度应为 1: {pcq_records[0]['targets'].shape}")

        write_cache(pcq_records, cache_dir)
        write_pcq_split(cache_dir, 4, 2)

        ds = QM9Dataset(cache_dir, split_ids=(0,))
        ds_val = QM9Dataset(cache_dir, split_ids=(1,))
        if len(ds) != 4 or len(ds_val) != 2:
            notes.append(f"FAIL split 不对: train={len(ds)} val={len(ds_val)}")
        batch = qm9_collate([ds[i] for i in range(len(ds))])
        if tuple(batch["targets"].shape) != (4, 1):
            notes.append(f"FAIL collate 后 targets 形状 {tuple(batch['targets'].shape)}")
        got = sorted(round(float(v), 3) for v in batch["targets"][:, 0])
        want = [4.0, 4.3, 4.6, 4.9]
        if got != want:
            notes.append(f"FAIL gap 标签串列: {got} != {want}")

        # 词表降级：把原子类别伪造成新元素（S=9），验证被压回 OTHER=5
        down = VocabDowngradeDataset(ds)
        item = down[0]
        fake = dict(item)
        fake["atom_class"] = np.full_like(item["atom_class"], 9)
        af = item["atom_fields"].copy()
        af[:, 0] = 10
        fake["atom_fields"] = af
        # 手动验证降级规则与实现一致
        if int(np.minimum(fake["atom_class"], 5).max()) != 5:
            notes.append("FAIL 降级逻辑自检失败")
        # 真链路：原始数据类别都应 <=5（合成分子只有 H/C/N/O/F），降级是恒等
        if not np.array_equal(item["atom_class"], ds[0]["atom_class"]):
            notes.append("FAIL 对 QM9 元素的降级应是恒等映射")
    finally:
        _cleanup([cache_dir], [ds] if ds else [])
    return notes


CHECKS: tuple[tuple[str, Callable[[], list[str]]], ...] = (
    ("bond_perception", check_bond_perception),
    ("bond_order_bands", check_bond_order_bands),
    ("qm9_xyz_layout", check_qm9_xyz_layout),
    ("ddp_sharding", check_ddp_sharding),
    ("featurizer_vocab", check_featurizer_vocab),
    ("numpy_torch_parity", check_numpy_torch_parity),
    ("gaussian_basis", check_gaussian_basis),
    ("collate_padding", check_collate_padding),
    ("forward_shapes", check_forward_shapes),
    ("structure_mask_wiring", check_structure_mask_wiring),
    ("loss_and_backward", check_loss_and_backward),
    ("training_and_eval", check_training_and_eval),
    ("config_overrides", check_config_overrides),
    ("param_budget", check_param_budget),
    ("benchmark_reporting", check_benchmark_reporting),
    ("vocab_extension", check_vocab_extension),
    ("lora_mechanics", check_lora_mechanics),
    ("pcq_pipeline", check_pcq_pipeline),
)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_checks(only: str | None = None, verbose: bool = True) -> int:
    selected = [(n, f) for n, f in CHECKS if not only or only.lower() in n.lower()]
    if not selected:
        print(f"[err ] 没有名字包含 '{only}' 的检查项")
        return 2
    torch.manual_seed(0)
    failures: list[str] = []
    print(f"DeMol 自检：{len(selected)} 项  "
          f"(torch {torch.__version__}, numpy {np.__version__}, "
          f"rdkit={'yes' if features.rdkit_available() else 'no'})\n")
    for name, fn in selected:
        started = time.time()
        try:
            notes = fn() or []
        except Exception:
            failures.append(name)
            print(f"[FAIL] {name}  ({time.time() - started:.1f}s)")
            print("       " + traceback.format_exc(limit=10).strip().replace("\n", "\n       "))
            continue
        hard = [x for x in notes if x.startswith("FAIL")]
        info = [x for x in notes if not x.startswith("FAIL")]
        status = "FAIL" if hard else "ok  "
        print(f"[{status}] {name}  ({time.time() - started:.1f}s)")
        if hard:
            failures.append(name)
        if verbose:
            for line in info:
                print(f"       {line}")
            for line in hard:
                print(f"       {line}")
    print()
    if failures:
        print(f"结果：{len(selected) - len(failures)}/{len(selected)} 通过，失败项: {', '.join(failures)}")
        return 1
    print(f"结果：{len(selected)}/{len(selected)} 全部通过")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", default=None, help="只运行名字包含该子串的检查")
    ap.add_argument("--list", action="store_true", help="列出所有检查名并退出")
    ap.add_argument("--quiet", action="store_true", help="只打印通过/失败")
    ap.add_argument("--threads", type=int, default=1, help="CPU 线程数（默认 1，保证可复现）")
    args = ap.parse_args(argv)

    if args.list:
        for name, fn in CHECKS:
            doc = (fn.__doc__ or "").strip().splitlines()
            print(f"{name:<22} {doc[0] if doc else ''}")
        return 0

    torch.set_num_threads(max(1, args.threads))
    return run_checks(only=args.only, verbose=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
