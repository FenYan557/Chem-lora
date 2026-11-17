# -*- coding: utf-8 -*-
"""select_subset / cluster_members / cluster_quotas / typicality_scores 回归测试。

三条锚点:
  1. cluster_members (排序+searchsorted) 与朴素 np.where 完全等价;
  2. **旧行为可复现**: select_subset(power=0.5, within='fps', noise_frac=0.05,
     noise_order='outlier') 必须与重构前那版实现产出同一批下标 —— 旧版慢且有
     `i < 50*簇数` 的硬上限, 所以断言写成"新版是旧版的延长";
  3. **默认(代表性)模式**行为正确: 正比配额让每簇采样率趋同、命中 target、无重复。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_subset import (cluster_members, cluster_quotas, pick_medoids,
                            select_subset, typicality_scores)

N_CHECK = 0


def check(cond, msg):
    global N_CHECK
    N_CHECK += 1
    assert cond, f"FAIL: {msg}"


# ------------------------------------------------------- 重构前那版实现(基准)
def legacy_select_subset(V_all, labels, strengths, target, noise_frac=0.05, top_m=100):
    """重构前的 select_subset 原样照抄, 只把 role 字面量换成新命名"""
    N = len(labels)
    target = int(min(target, N))
    uniq = np.unique(labels[labels >= 0])
    counts = np.bincount(labels[labels >= 0], minlength=int(uniq.max()) + 1)[uniq] \
        if len(uniq) else np.empty(0, dtype=np.int64)
    budget_c = int(round(target * (1 - noise_frac)))
    quotas = cluster_quotas(counts, budget_c, 0.5) if len(uniq) \
        else np.empty(0, dtype=np.int64)

    picked, meta = [], []
    for c, q in zip(uniq, quotas):
        idx = np.where(labels == c)[0]
        sel = pick_medoids(V_all, idx, strengths, int(q), top_m)
        picked.extend(sel.tolist())
        meta.extend([(int(c), "cluster")] * len(sel))

    n_noise = target - len(picked)
    noise_idx = np.where(labels == -1)[0]
    if n_noise > 0 and len(noise_idx) > 0:
        take = min(n_noise, len(noise_idx))
        sel_noise = noise_idx[np.argsort(strengths[noise_idx])[:take]]
        picked.extend(sel_noise.tolist())
        meta.extend([(-1, "noise")] * take)

    short = target - len(picked)
    if short > 0 and len(uniq):
        used = {}
        for c, _ in meta:
            used[c] = used.get(c, 0) + 1
        order = np.argsort(-counts)
        i = 0
        while short > 0 and i < 50 * len(order):
            c = uniq[order[i % len(order)]]
            idx = np.where(labels == c)[0]
            k = used.get(int(c), 0) + 1
            if k <= len(idx):
                sel = pick_medoids(V_all, idx, strengths, k, top_m)
                new = [s for s in sel.tolist() if s not in set(picked)]
                if new:
                    picked.append(new[0])
                    meta.append((int(c), "cluster"))
                    used[int(c)] = k
                    short -= 1
            i += 1
    return np.array(picked, dtype=np.int64), meta


LEGACY_KW = dict(power=0.5, within="fps", noise_order="outlier")


# ---------------------------------------------------------------- 用例
def test_cluster_members():
    rng = np.random.default_rng(7)
    for trial in range(12):
        n = int(rng.integers(50, 4000))
        k = int(rng.integers(1, 25))
        labels = rng.integers(-1, k, size=n).astype(np.int32)
        uniq = np.unique(labels[labels >= 0])
        got = cluster_members(labels, uniq)
        check(set(got) == set(int(c) for c in uniq), f"t{trial} 簇号集合不符")
        for c in uniq:
            check(np.array_equal(np.sort(got[int(c)]), np.where(labels == c)[0]),
                  f"t{trial} 簇 {c} 成员不符")
        check(sum(len(v) for v in got.values()) == int((labels >= 0).sum()),
              f"t{trial} 成员总数不符")
    lab = np.full(100, -1, dtype=np.int32)
    check(cluster_members(lab, np.unique(lab[lab >= 0])) == {}, "全噪声应返回空 dict")


def test_quotas_power():
    counts = np.array([10000, 2500, 625, 100], dtype=np.int64)
    budget = 1000
    qp = cluster_quotas(counts, budget, 1.0)
    check(int(qp.sum()) == budget, f"正比配额总和 {qp.sum()} != {budget}")
    # 正比: 采样率应几乎相同
    rates = qp / counts
    check(float(rates.max() / rates.min()) < 1.15,
          f"正比配额各簇采样率不齐: {rates}")
    qs = cluster_quotas(counts, budget, 0.5)
    check(int(qs.sum()) == budget, f"sqrt 配额总和 {qs.sum()} != {budget}")
    # sqrt: 小簇采样率应显著高于大簇
    check(qs[3] / counts[3] > qs[0] / counts[0], "sqrt 配额没有抬高小簇占比")
    check(float(qs[3] / counts[3]) > 1.5 * float(qp[3] / counts[3]),
          "sqrt 相对正比对小簇的抬升不明显")
    # 保底 1 且不超过容量
    tiny = np.array([3, 2, 1], dtype=np.int64)
    for p in (0.5, 1.0, 2.0):
        q = cluster_quotas(tiny, 100, p)
        check(bool((q >= 1).all()) and bool((q <= tiny).all()), f"power={p} 越界: {q}")
        check(int(q.sum()) == int(tiny.sum()), f"power={p} 总和 {q.sum()} != {tiny.sum()}")


def test_typicality():
    rng = np.random.default_rng(11)
    # 两个明显分开的簇, 各 400 点; 每簇再放一个远离质心的点
    A = rng.normal(0, 0.1, (400, 6)).astype(np.float32)
    B = rng.normal(8, 0.1, (400, 6)).astype(np.float32)
    far = np.array([[3.0] * 6, [5.0] * 6], dtype=np.float32)
    V = np.vstack([A, far[:1], B, far[1:]])
    labels = np.concatenate([np.zeros(401, np.int32), np.ones(400, np.int32),
                             np.full(1, -1, np.int32)])
    uniq = np.unique(labels[labels >= 0])
    memb = cluster_members(labels, uniq)
    s = typicality_scores(V, labels, uniq, memb, need_noise_dist=True)
    check(s.shape == (len(V),), "typicality 形状不对")
    check(float(s.min()) >= 0 and float(s.max()) <= 1, "typicality 未落在 [0,1]")
    check(len(np.unique(np.round(s, 4))) > 8,
          f"typicality 取值太少({len(np.unique(np.round(s,4)))}), 疑似退化")
    # 簇心附近的点应比簇边缘点更典型
    core = np.linalg.norm(A - A.mean(0), axis=1).argmin()
    edge = np.linalg.norm(A - A.mean(0), axis=1).argmax()
    check(s[core] > s[edge], f"簇心({s[core]:.3f}) 未比簇边缘({s[edge]:.3f}) 更典型")
    # 噪声点: 离所有簇都远 -> 分数应低
    nz = int(np.nonzero(labels == -1)[0][0])       # = 801, vstack 后的最后一行
    check(s[nz] < float(np.median(s[:401])),
          f"远离簇的噪声点典型性分数未偏低: s[{nz}]={s[nz]:.3f} "
          f"vs 簇内中位 {float(np.median(s[:401])):.3f}")


def test_representative_default():
    """默认模式: 正比配额 -> 各簇采样率趋同 (这才是"分布匹配")"""
    rng = np.random.default_rng(99)
    sizes = [6000, 2500, 900, 400, 200]
    n = sum(sizes)
    labels = np.concatenate([np.full(s, i, np.int32) for i, s in enumerate(sizes)])
    # 掺 10% 噪声
    nz = rng.choice(n, size=n // 10, replace=False)
    labels[nz] = -1
    V = rng.normal(size=(n, 15)).astype(np.float32)
    s = rng.random(n).astype(np.float32)
    target = 1000

    picked, meta = select_subset(V, labels, s, target, rng=np.random.default_rng(42))
    check(len(picked) == target, f"默认模式未命中 target: {len(picked)}")
    check(len(set(picked.tolist())) == target, "默认模式有重复")
    check(all(r in ("cluster", "noise") for _, r in meta), "role 取值非法")
    check(all((c == -1) == (r == "noise") for c, r in meta), "role 与 cluster 不一致")

    n_noise_lib = float((labels == -1).mean())
    n_noise_sub = sum(1 for _, r in meta if r == "noise") / target
    check(abs(n_noise_sub - n_noise_lib) < 0.03,
          f"噪声层占比 {n_noise_sub:.3f} 未跟随实测噪声率 {n_noise_lib:.3f}")

    rates = []
    for c in range(len(sizes)):
        m = labels == c
        got = sum(1 for cc, _ in meta if cc == c)
        rates.append(got / int(m.sum()))
    spread = max(rates) / min(rates)
    check(spread < 1.35, f"正比配额下各簇采样率不齐 (max/min={spread:.2f}): {rates}")
    print(f"  per-cluster sampling rates: {[round(r,4) for r in rates]} "
          f"(max/min={spread:.2f})")

    # 同一 seed 必须可复现
    p2, _ = select_subset(V, labels, s, target, rng=np.random.default_rng(42))
    check(np.array_equal(picked, p2), "同 seed 不可复现")


def test_legacy_equivalence():
    rng = np.random.default_rng(2026)
    cases = extended = 0
    for trial in range(30):
        n = int(rng.integers(200, 3000))
        V = rng.normal(size=(n, 15)).astype(np.float32)
        strength = rng.random(n).astype(np.float32)
        ncl = int(rng.integers(1, 12))
        labels = rng.integers(0, ncl, size=n).astype(np.int32)
        noise_p = float(rng.choice([0.0, 0.02, 0.2, 0.6]))
        labels[rng.random(n) < noise_p] = -1
        if not (labels >= 0).any():
            continue
        target = int(rng.integers(10, min(n, 800)))
        noise_frac = float(rng.choice([0.0, 0.05, 0.3]))
        top_m = int(rng.choice([5, 100]))

        got, gmeta = select_subset(V, labels, strength, target,
                                   noise_frac=noise_frac, top_m=top_m,
                                   rng=np.random.default_rng(0), **LEGACY_KW)
        exp, emeta = legacy_select_subset(V, labels, strength, target,
                                          noise_frac=noise_frac, top_m=top_m)

        check(len(got) == len(set(got.tolist())), f"t{trial} 有重复")
        check(len(got) <= target, f"t{trial} 超出 target")
        check(int(got.min()) >= 0 and int(got.max()) < n, f"t{trial} 下标越界")
        check(len(got) == len(gmeta), f"t{trial} picked/meta 长度不符")
        # legacy 路径不消耗 rng(fps + outlier 都是确定性排序), 所以必须逐项相同
        check(len(got) >= len(exp), f"t{trial} 新版比旧版少: {len(got)} < {len(exp)}")
        check(np.array_equal(got[:len(exp)], exp), f"t{trial} 前缀与旧版不一致")
        check(gmeta[:len(emeta)] == emeta, f"t{trial} meta 前缀与旧版不一致")
        if len(got) != len(exp):
            print(f"  t{trial}: n={n} ncl={ncl} target={target} "
                  f"noise_frac={noise_frac} top_m={top_m} "
                  f"-> legacy={len(exp)}(触到 50*簇数 上限) new={len(got)}")
            extended += 1
        cases += 1
    check(cases >= 20, f"有效用例太少: {cases}")
    print(f"  legacy-equivalence cases: {cases}, 新版补足更完整: {extended}")


def test_extreme_shapes():
    rng = np.random.default_rng(5)
    n = 500
    V = rng.normal(size=(n, 15)).astype(np.float32)
    s = rng.random(n).astype(np.float32)

    lab = rng.integers(0, 3, size=n).astype(np.int32)
    got, _ = select_subset(V, lab, s, target=10 * n, rng=np.random.default_rng(1))
    check(len(got) == n and len(set(got.tolist())) == n, "target>N 时应返回全部 N 且不重复")

    lab = np.full(n, -1, dtype=np.int32)
    got, meta = select_subset(V, lab, s, target=50, rng=np.random.default_rng(1))
    check(len(got) == 50 and all(r == "noise" for _, r in meta), f"全噪声异常: {len(got)}")

    lab = np.zeros(n, dtype=np.int32)
    got, _ = select_subset(V, lab, s, target=80, rng=np.random.default_rng(1))
    check(len(got) == 80 and len(set(got.tolist())) == 80, "单簇场景异常")

    lab = np.repeat(np.arange(250), 2).astype(np.int32)
    got, _ = select_subset(V, lab, s, target=300, rng=np.random.default_rng(1))
    check(len(set(got.tolist())) == len(got) and len(got) <= 300, "微簇场景异常")
    print(f"  micro-cluster picked={len(got)}/300")

    # 三种 within 模式都要能跑通且不重复
    lab = rng.integers(0, 5, size=n).astype(np.int32)
    for mode in ("random", "medoid", "fps"):
        got, _ = select_subset(V, lab, s, target=120, within=mode,
                               rng=np.random.default_rng(3))
        check(len(set(got.tolist())) == len(got), f"within={mode} 有重复")
        check(len(got) == 120, f"within={mode} 未命中 target: {len(got)}")


if __name__ == "__main__":
    test_cluster_members()
    print("[ok] cluster_members")
    test_quotas_power()
    print("[ok] cluster_quotas (power 可调)")
    test_typicality()
    print("[ok] typicality_scores (替代退化的 probabilities_)")
    test_extreme_shapes()
    print("[ok] extreme shapes / within 三种模式")
    test_representative_default()
    print("[ok] 默认代表性模式")
    test_legacy_equivalence()
    print("[ok] 旧行为可复现")
    print(f"\nALL PASS ({N_CHECK} assertions)")
