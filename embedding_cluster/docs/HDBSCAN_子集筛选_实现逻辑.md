# PCQM4Mv2 子集蒸馏：Embedding + HDBSCAN 实现逻辑与优化

目标：从 PCQM4Mv2（train 约 337.8 万分子）中选出一个**有代表性的子集**——用尽量少的分子覆盖尽量全的化学空间。

---

## 1. 整体流水线

```
SMILES --> 分子指纹 embedding --> 降维 --> HDBSCAN 聚类
                                              |
                                              v
                        每个簇选代表点（聚类中心/核心样本）
                                              |
                                              v
                                   输出子集 CSV（ID + SMILES + 簇号）
```

---

## 2. Embedding：Morgan 指纹（ECFP）

对每个分子计算半径 2、2048 位的 Morgan 指纹（等价 ECFP4）：

- 每个原子以其局部环境（半径 0,1,2 的子结构）哈希成位向量；
- 得到稀疏二值向量 x 属于 {0,1}^2048。

**分子间距离用 Tanimoto 距离**（化学领域标准，不是欧氏距离）：

    d_T(a,b) = 1 - c / (n_a + n_b - c)

其中 c = 两指纹共同置 1 的位数，n_a、n_b 为各自置 1 的位数。

> 优化点：指纹用 RDKit 的 BulkTanimotoSimilarity，底层 C++ 位运算，比 Python 双层循环快 3 个数量级以上；337 万分子绝不计算全对距离矩阵（见第 5 节）。

---

## 3. HDBSCAN 核心公式

HDBSCAN = 层次化 DBSCAN，不需要预设簇数，自动识别不同密度的簇 + 噪声点。

**(1) 核心距离（core distance）**

    core_k(x) = d(x, N_k(x))

即点 x 到第 k 近邻的距离。k = min_samples（建议 5~15）。

**(2) 互可达距离（mutual reachability distance）**——HDBSCAN 的关键改造：

    d_mreach(a,b) = max{ core_k(a), core_k(b), d(a,b) }

作用：稀疏区域的点之间距离被"撑大"，稀疏噪声不会把稠密簇连起来。

**(3) 最小生成树（MST）**

在互可达距离构成的图上求 MST（实现用 Boruvka 算法 + k-d 树/BallTree 加速），MST 按边权从小到大删边即得到层次聚类树（dendrogram）。

**(4) 簇压缩（condense tree）**

引入 lambda = 1/d（密度尺度）。从 lambda=0 开始增加，点逐步"掉出"簇。当某簇的规模 < min_cluster_size 时判定消亡。

**(5) 稳定性选择（决定最终保留哪些簇）**

    S(C_i) = sum_{p in C_i} ( lambda_p - lambda_birth(C_i) )

lambda_p 是点 p 离开簇时的密度水平。HDBSCAN 选择一组互不嵌套、总稳定性最大的簇——这是它比固定 eps 的 DBSCAN 强的地方：eps 等价于在层次树上砍一刀（lambda 固定），只能找到单一密度的簇。

**(6) 成员强度（membership strength）**

每个点带一个 0~1 的强度值：在该簇内存活得越久（密度越高、越靠核心）强度越大；边缘点强度趋近 0；噪声点标签为 -1。

---

## 4. 挑代表点（聚类中心）

对每个簇 C_i，两种选法，推荐 B：

**方法 A：簇内强度最高者**

    x* = argmax_{x in C_i} strength(x)

即该簇最典型的分子。零额外计算，直接取 HDBSCAN 输出。

**方法 B：簇内中心性最高者（medoid）**

    x* = argmin_{x in C_i} sum_{y in C_i} d(x, y)

选簇内到其他点总距离最小的点（medoid，比质心鲁棒且保证是真实分子）。

> 优化：簇大时全对距离太贵，只对簇内强度 top-M（如 100）的点之间算距离再取 medoid。

**子集配额分配**：按簇大小比例分配名额（sqrt 缩放更均衡）：

    n_i = N_target * sqrt(|C_i|) / sum_j sqrt(|C_j|)

大簇多给、小簇保底 1 个，避免巨型簇吃掉全部名额，同时保证稀有化学型不丢。

**噪声点（label = -1）处理**：它们是最离群的分子。建议按离群程度保留少量（约总预算 5%），这些往往是有价值的边缘化学型。

---

## 5. 337 万规模的工程优化（关键）

直接对 337 万点跑 HDBSCAN 不现实：

| 步骤 | 复杂度 | 337 万点的问题 |
|---|---|---|
| 全对距离 | O(n^2) | 约 5.7 万亿对，不可行 |
| MST 存图 | O(n^2) 内存 | 数十 TB，不可行 |

**优化策略：分层采样 + 近似分配（推荐）**

```
337 万分子
   |
   +-- 随机抽 30~50 万（种子固定，可复现）
   |      |
   |      +-- Morgan 指纹 2048 位
   |      +-- 可选：UMAP 降到 15 维（metric='jaccard'），加速后续 kNN
   |      +-- HDBSCAN（50 万点，内存 <10GB，可跑）
   |
   +-- 其余 290 万用 hdbscan.approximate_predict / membership_vector
          批量归入最近簇
               |
               v
       全库每个分子都有簇标签 + 强度
               |
               v
       按第 4 节配额 + medoid 规则选代表点
```

要点：

1. **聚类只发生在采样子集上**。50 万点 HDBSCAN（hdbscan 库，core_dist_n_jobs=-1，algorithm 用 boruvka_kdtree 对低维数据最快）在服务器上几小时内可完成。
2. **近似分配**：approximate_predict 对每个新点只找最近的已有核心点，O(n log n)，数百万点分钟级。
3. **降维可选**：直接在 2048 位指纹上用 metric='jaccard' 也可行，但树索引对高维稀疏向量效率差；UMAP 降到 15 维后用欧氏 HDBSCAN 通常快 5~10 倍，且 UMAP 保局部邻域结构，对聚类友好。注意 UMAP 要固定 random_state。
4. **内存**：指纹用 scipy.sparse CSR 存（2048 位 x 337 万，平均每分子约 80 个非零位，总共约 1GB），不要转稠密矩阵（约 800GB 会爆）。
5. **可复现性**：采样种子、UMAP random_state、min_cluster_size/min_samples 全部写入输出 CSV 元信息头或单独 JSON。

**超参数起步建议**：

| 参数 | 建议值 | 说明 |
|---|---|---|
| 指纹 | Morgan r=2, 2048 bit | ECFP4，化学空间标准 |
| min_cluster_size | 200~500（对 50 万采样） | 太小会碎成几千个簇 |
| min_samples | 10 | 控制噪声判定严格度 |
| 目标子集大小 | 5 万（约 1.5%） | 可按训练预算调 |

---

## 6. 伪代码

```python
# 1. 指纹
fps = [morgan_fp(mol, r=2, nbits=2048) for mol in train_smiles]  # sparse

# 2. 采样聚类
idx = rng.choice(N, size=500_000, replace=False)
emb = UMAP(n_components=15, metric='jaccard', random_state=42).fit_transform(fps[idx])
clusterer = hdbscan.HDBSCAN(min_cluster_size=300, min_samples=10, core_dist_n_jobs=-1)
labels_sub = clusterer.fit_predict(emb)

# 3. 全库分配
labels_all = np.full(N, -1)
labels_all[idx] = labels_sub
rest = np.setdiff1d(np.arange(N), idx)
for chunk in batches(rest, 100_000):
    lab, s = hdbscan.approximate_predict(clusterer, transform(umap, fps[chunk]))
    labels_all[chunk] = lab

# 4. 选代表点
for c in unique(labels_all >= 0):
    quota_i = quota(c)                       # sqrt 配额
    cand = top_strength_points(c, M=100)     # 簇内强度 top100
    picked += greedy_medoids(cand, quota_i)  # 迭代取 medoid
subset = picked + top_noise_points(0.05 * budget)
subset.to_csv('pcqm4mv2_subset.csv')         # id, smiles, cluster, strength, role
```

---

## 7. 质量自检（选完后必做）

1. **覆盖率**：子集指纹 vs 全库指纹，每个子集点的 Tanimoto 0.4 邻域覆盖全库的比例（目标 >80%）；
2. **性质分布对齐**：子集与全库的 HOMO-LUMO gap、重原子数、元素组成直方图对比（KS 检验）；
3. **scaffold 多样性**：子集的 Bemis-Murcko 骨架数 / 子集大小，应显著高于随机抽样。
