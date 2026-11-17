# DeMol × QM9 复现

对 Liu et al., *"Enhancing Molecular Property Predictions by Learning from Bond
Modelling and Interactions"*, ICLR 2026（下称"论文"）的架构复现，目标是 QM9 上的
12 个量子化学性质预测（论文 Table 3）。论文没有官方代码，所以这里的每一处实现
选择都尽量对应到论文的具体公式/附录编号，凡论文没写清楚的地方都在下面
[与论文的偏差](#与论文的偏差) 一节明确列出。

> **当前状态：11 项自检 + 随机数据探针全部通过（CPU 实跑）。**
> 前向、反向、AdamW 优化、EMA、检查点存取都有运行期证据；四个损失分量在随机
> 数据上都能被压下去（预训练 200 步降 79%，微调单目标 120 步降 99%）。
> 还**没有**跑过的是真实 QM9 数据上的完整训练——数据集尚未下载。

---

## 目录结构

```
demol_qm9/
├── demol/                     # 模型本体（可 import 的包）
│   ├── chem.py                # Algorithm 1 成键判定、共价半径、几何量（纯 numpy）
│   ├── features.py            # 原子/化学键特征化，rdkit 与 geometry 两套后端
│   ├── encoding.py            # 高斯基、SPD、扭转编码 → 注意力偏置（eq. 5-8）
│   ├── model.py               # DeMolConfig + DeMol（双图 Transformer，eq. 6-10）
│   ├── losses.py              # 四个预训练损失 + 标签标准化
│   ├── qm9.py                 # QM9 解析/缓存/划分/稠密批处理
│   ├── hubio.py               # manifest/sha256/归档工具（ModelScope 打包与下载共用）
│   ├── data.py                # DataLoader 装配
│   ├── trainer.py             # 训练循环、AdamW+warmup/衰减、EMA、检查点
│   ├── configs.py             # preset（模型尺寸）× scale（优化预算）预设
│   └── benchmarks.py          # 论文 Table 3 / Table 8 的对照常数（仅用于报告）
└── scripts/
    ├── download_qm9.py        # 下载并解包 QM9
    ├── preprocess_qm9.py      # 特征化 → .npy 内存映射缓存
    ├── pack_subset.py         # 从缓存切子集 + manifest + 可选归档（本地打包端）
    ├── upload_modelscope.py   # 把打包目录推到 ModelScope 数据集仓库
    ├── download_modelscope.py # 服务器端：拉取 + sha256 校验 + 归位成缓存
    ├── run_common.py          # 两个 run 脚本共享的 CLI/构建/报告管线
    ├── run_pretrain.py        # 预训练入口
    ├── run_finetune.py        # 微调入口（含与论文 Table 3 的对比表）
    ├── smoke_test.py          # 11 项自检，不需要 QM9 数据也不需要 GPU
    └── random_probe.py        # 随机数据上的前向/反向/优化器探针（看损失是否真的降）
```

---

## 环境

本机 `deeplearn` conda 环境（`E:\anconda\envs\deeplearn\python.exe`）：

| 组件 | 版本 | 影响 |
| --- | --- | --- |
| Python | 3.11.16 | — |
| torch | 2.14.0+cu132 | RTX 4060 Laptop 8 GB |
| numpy / scipy / scikit-learn | 已装 | — |
| **rdkit** | **未安装** | 化学键感知走纯 numpy 几何后端 |
| **torch_geometric** | **未安装** | 全程用稠密 padding 批处理，不依赖 PyG |

没有 rdkit/PyG 不是缺陷而是设计约束：`chem.py` + `features.py` 的几何后端与
`qm9.py` 的稠密 collate 让整条链路只依赖 numpy + torch。服务器上如果装了
rdkit，`preprocess_qm9.py` 会自动切到 rdkit 后端（`--backend` 可强制指定），
两套后端输出**相同的形状和相同的词表**，所以缓存可以互换、模型代码不用改。

---

## 五步跑通

```powershell
cd E:\deeplearn\创新创业\Chem-dbscan\demol_qm9

# 0) 先自检（合成分子，秒级，不需要数据/GPU）
python scripts/smoke_test.py
python scripts/random_probe.py --steps 60    # 可选：确认优化器真的在降损失

# 1) 下载 QM9（133,885 个分子，压缩包约 1.6 GB）
python scripts/download_qm9.py --out data/qm9

#    国内/服务器网络如果 figshare 不通，走 hf-mirror 的 DeepChem 镜像
#    （内容是 qm9.sdf + qm9.sdf.csv，需要一步转换）：
python scripts/download_qm9.py --out data/qm9 --source moldata-mirror
python scripts/sdf_to_xyz.py --sdf data/qm9/qm9.sdf --csv data/qm9/qm9.sdf.csv `
                             --out-dir data/qm9/dsgdb9nsd.xyz

# 2) 特征化 → 缓存（全量峰值内存约 1.5 GB；先用 --limit 试跑）
python scripts/preprocess_qm9.py --raw-dir data/qm9/dsgdb9nsd.xyz `
                                 --cache-dir data/qm9_cache

# 3) 预训练（本地预算）
python scripts/run_pretrain.py --preset small --scale local `
                               --cache-dir data/qm9_cache `
                               --out-dir runs/pretrain_small

# 4) 微调单个目标并打印与论文 Table 3 的对比
python scripts/run_finetune.py --pretrained runs/pretrain_small/ema_model.pt `
                               --preset small --scale local --targets gap `
                               --out-dir runs/ft_gap
```

任何一步都可以加 `--dry-run`：只构建配置与模型、打印参数预算、写 `config.json`，
**不读数据、不执行任何前向/反向**。微调的 `--dry-run` 还会额外做一次
"预训练权重能否对上"的形状对齐检查。

---

## 多卡训练（DDP）

两个入口脚本都支持 torchrun 多卡：检测到 ``RANK/WORLD_SIZE/LOCAL_RANK``
环境变量即进入 DDP（NCCL；无 NCCL 的平台退回 gloo + CPU，仅用于联调）。
训练集按 rank 分片、逐 epoch 重洗；日志与检查点只由 rank 0 写出；
检查点保存的是**裸模型**（无 ``module.`` 前缀），单卡/微调可直接加载。

``batch_size`` 是**每卡**的：全局 batch = batch_size × 卡数。论文全局
batch 1024，4 卡时每卡 256：

```bash
# 先验证多卡路径（合成数据，秒级，不需要 QM9）
python scripts/ddp_smoke.py --nproc 4 --set-train amp=bf16

# 正式预训练：paper 预设 + 4 卡 DDP + bf16，全局 batch 1024，5 万步
torchrun --nproc_per_node=4 scripts/run_pretrain.py \
    --preset paper --cache-dir data/qm9_cache --out-dir runs/pretrain_paper \
    --set-train batch_size=256 total_steps=50000 amp=bf16
```

EMA（decay 0.999）与 warmup 10% + 线性衰减到 0 的调度都由配置自带，
**不需要早停**（decay-to-zero + best.pt 兜底）。

---

## 把缓存传到服务器（ModelScope）

DeMol 的 paper 预设（~181 M 参数）需要租大显卡训练。服务器上重新下载 1.6 GB
原始 QM9 + 跑一遍特征化又慢又容易失败（figshare 在国内常连不上），所以缓存
在**本地**做好，通过 ModelScope 数据集仓库中转：

```powershell
# 本地（deeplearn 环境已装 modelscope 1.40）
python scripts/pack_subset.py --cache-dir data/qm9_cache `
                              --out-dir data/qm9_subset_20k --size 20000
$env:MODELSCOPE_API_TOKEN="xxxx"   # 魔搭网页 → 账号设置 → 访问令牌
python scripts/upload_modelscope.py --local-dir data/qm9_subset_20k `
                                    --repo-id <你的用户名>/demol-qm9-20k
```

```bash
# 服务器
pip install modelscope
export MODELSCOPE_API_TOKEN=xxxx    # 私有仓库需要；公开仓库可省
python scripts/download_modelscope.py --repo-id <你的用户名>/demol-qm9-20k \
                                      --cache-dir data/qm9_cache
python scripts/run_pretrain.py --cache-dir data/qm9_cache --preset base --scale cluster
```

要点：

* **不传 `--size` 就是整份缓存**。QM9 全量特征化后只有几十 MB（17 个 `.npy`），
  通常直接全传；`--size 20000` 切 2 万子集用于先跑通流程。
* `--select stratified`（默认）按源缓存的 train/val/test 比例分层抽样，子集里
  三块都在，服务器上可以直接跑"训练+验证+测试"全流程；另有 `random` / `head`。
* 打包时会对子集与源缓存做**逐字段抽查自检**，并写 `manifest.json`
  （每个文件的 sha256 + 每个数组的 shape/dtype）；服务器下载后按它逐文件校验，
  校验不过会拒绝归位。
* 仓库默认建为**私有**（`--visibility private`），下载要同一个 token；
  `--visibility public` 可公开。token 优先级：`--token` > 环境变量
  `MODELSCOPE_API_TOKEN`。
* 想走单文件传输：`pack_subset.py --archive tar.gz` 会把 manifest 一起打进归档，
  下载端发现归档会自动解包再校验。
* 上传/下载默认走 modelscope SDK；SDK 异常时两个脚本都支持 `--use-cli`
  兜底（`python -m modelscope.cli.cli`）。
* 上传前可先 `upload_modelscope.py --dry-run` 做纯本地校验（不打网络）。

本地全链路验证过：打包 6 分子子集 → manifest 自检 → dry-run 上传计划 →
模拟服务器下载 + 校验 + 归位 → `QM9Dataset` 打开做前向，全部通过。

---

## 论文 → 代码映射

| 论文位置 | 内容 | 代码 |
| --- | --- | --- |
| Sec. 3.2, Fig. 3 | 原子通道 + 化学键通道（线图）双图表示 | `model.py: DeMol`（`h_a` / `h_b` 两条 token 序列） |
| Appendix B, Alg. 1 | 基于共价半径的成键判定，`d_ij < α(r_i+r_j)`，α=1.15 | `chem.py: perceive_bonds`，`BOND_THRESHOLD_ALPHA = 1.15` |
| Sec. 3.2 "Structure-aware Mask" | 稀疏注意力掩码：原子对只保留 5 Å 内+成键，键对只保留邻接+共轭两跳，交叉只保留原子-键关联 | `chem.py: ATOM_MASK_CUTOFF_ANGSTROM = 5.0` → `qm9.py: qm9_collate` 的 `aa/bb/ab_mask` → `model.py` 注意力的 `pair_mask` 参数 |
| eq. (5) | 高斯基核展开 `exp(-α(x-μ_k)²/σ_k²)`，α/β 按原子类型对索引 | `encoding.py: GaussianBasis, ScalarBasisEncoder` |
| eq. (6) | 原子自注意力 + `Φ^(l) = Φ_dist + Φ_SPD` | `encoding.py: AtomStructureEncoding` → `model.py` 原子自注意力子层 |
| eq. (7)(8) | 化学键自注意力 + `Φ_b = Φ_dist + Φ_SPD + Φ_tors` | `encoding.py: BondStructureEncoding, torsion_cosine` |
| eq. (9) | atom ← bond 交叉注意力（Double-Helix） | `model.py` 交叉注意力子层 + `encoding.py: CrossStructureEncoding` |
| eq. (10) | bond ← atom 交叉注意力 | 同上，反方向 |
| Appendix D.1 | 四个预训练损失 `Lprop/Lmask/Lcoord/Lbond` | `losses.py: DeMolLoss` |
| Appendix D.1 | 12 层 / d_a=d_b=768 / K=128 / 8 头 | `configs.py: MODEL_PRESETS["paper"]` |
| Appendix D.1 | batch 1024、1.5M step、150k warmup、lr 2e-4、clip 5.0、EMA 0.999 | `configs.py: _TRAIN[("pretrain","paper")]` |
| Appendix D.4 | QM9 微调：lr 7e-5、batch 128、600k step、60k warmup 线性衰减到 0、训练 MSE / 评估 MAE | `_TRAIN[("finetune","paper")]` + `losses.py` 的 `prop_mode` |
| Appendix D.4 | 划分 10,000 val / 10,831 test / 其余训练 | `qm9.py: N_VAL, N_TEST, make_split` |
| Table 3 / Table 8 | 各目标 MAE 与无预训练消融 | `benchmarks.py`（**只读常数**，模型/损失/训练器都不 import 它） |

### 一个 Double-Helix block 的数据流

```
h_a ──[原子自注意力 + FFN，偏置 Φ^(l)]──────► h_a'
h_b ──[化学键自注意力 + FFN，偏置 Φ_b^(l)]──► h_b'
h_a' ──[atom ← bond 交叉注意力 + FFN]───────► h_a''    (eq. 9)
h_b' ──[bond ← atom 交叉注意力 + FFN]───────► h_b''    (eq. 10)
```

两个交叉注意力都读**自注意力之后**的表示，所以一个 block 是双螺旋对称的一"级"，
而不是串行的链。每个 block 因此有 4 个注意力子层。

---

## 与论文的偏差

按影响从大到小排列。前两条会直接改变数值结果，请务必知悉。

### 1. 预训练语料：PCQM4Mv2 → QM9 自身

论文在 PCQM4Mv2（3.37M 分子，8×A6000，1.5M step）上预训练，再迁移到 QM9。
本地不具备这个条件，按需求改为**用 QM9 自身做多任务预训练**：四个目标同时训练，
`Lprop` 覆盖全部 12 个 QM9 目标（论文预训练时只监督 HOMO-LUMO gap 一个标量）。

**后果**：微调数值不能与论文 Table 3 直接等同——辅助任务的信号来自一个小得多、
多样性低得多的语料，而且 `Lprop` 已经在同一批标签上训练过，微调阶段的"提升"
会显著偏乐观。`run_finetune.py` 打印的对比表里带了这句提醒。

### 2. 交叉注意力偏置的索引歧义

论文 eq. (9)(10) 写的是 `+Φ^(l)` 和 `+Φ_b^(l)`，但这两个矩阵分别按
**原子-原子**（N×N）和**化学键-化学键**（M×M）索引，而交叉注意力是按
**原子-化学键**（N×M）索引的——字面写法维度对不上。本实现的处理：

* atom ← bond：取 `Φ^(l)[i, j]`，其中 `j` 是该化学键的**另一个端点**；
* 两个方向都额外构建一个专门的 atom-bond 编码，标量取"原子到键中点的距离"，
  图距离取 `min(SPD[i,u], SPD[i,v])`，即 atom-bond 的自然类比。

对应 `encoding.py: CrossStructureEncoding` 与 `model.py` 第 496 行附近的 NOTE。

### 3. 扭转角的定义按论文原文，不是 IUPAC 二面角

论文 Appendix A.3 定义：两条化学键**共享一个原子**时取键角 `cos θ_ijk`，
**不共享**时取两键向量的夹角 `cos φ_ijkl = (r_ij·r_kl)/(|r_ij||r_kl|)`。
`encoding.py: torsion_cosine` 严格按这个定义实现，**不是**标准的四原子二面角。
这是有意为之（跟随论文），但和化学直觉不同，读代码时容易误会。

### 4. 参数量对不上论文标称的 186 M

见下面[参数量](#参数量为什么和论文的-186-m-对不上)一节。

### 5. 微调是单目标还是联合，论文没说

Appendix D.4 没有明确 12 个目标是一个联合模型还是每目标一个模型；它跟随的
Transformer-M 是**每目标一个模型**。`run_finetune.py` 两种都支持：
`--targets all` 是联合，`--targets gap` 是单目标。属性头始终保持论文的 12 路输出，
只在损失和评估里挑选指定列，所以联合与单目标之间可以直接复用检查点。
想完整复现 Table 3 就跑 12 次单目标（脚本 docstring 里有 PowerShell 循环命令）。

### 6. 特征化细节论文没给

论文只说"raw atom feature `x_i` 被嵌入为 `h_i^(a,0)`"。本实现采用
Graphormer / Transformer-M 的标准类别字段集：

* 原子 7 个字段：元素类型(6+pad)、度数(0-6)、形式电荷(-1/0/+1)、氢原子数(0-4)、
  杂化(unk/s/sp/sp2/sp3)、芳香性、是否在环内；
* 化学键 5 个字段：键级(单/双/三/芳香)、共轭、是否在环内、两个端点的元素类型。

每个字段有独立的 embedding 表，索引 0 保留给 padding。

### 7. 无 rdkit 时键级由几何推断

几何后端用"键长 / 共价半径之和"的比值分档得到键级，环由圈基判定，杂化由
π 键数推断。对 QM9 这种 DFT 优化几何通常够用，但和 rdkit 的 `xyz2mol`
在少数共振/strained 体系上会有分歧。实际用了哪套后端记录在缓存目录的
`cache_stats.json` 里，**不要混用两套后端产生的缓存**。

### 8. `Lbond` 的负样本集合

论文的 `E_all` 未明确定义。本实现只在 structure-aware mask（5 Å 内）覆盖的
原子对上算 BCE：所有真实化学键都在 5 Å 内，因此正样本完整保留，
被丢掉的是 O(N²) 的远场负样本。见 `losses.py` 的模块 docstring。

---

## 参数量为什么和论文的 186 M 对不上

论文 Table 1（PCQM4Mv2 榜单）标称 DeMol = **186 M**，对应 Appendix D.1 的
12 层 / da=db=768 / K=128 / 8 头模型。我们最初的读法是 eq. (6)-(10) 字面实现：
每 block 4 个注意力子层、每个各自带一个 **mult=4** 的 FFN → **350.7 M**，
接近标称值两倍。

**用论文自己的 Table 1 可以反推出答案**：同一张表里 Graphormer = 47.1 M，
而 Graphormer 是公开的 L=12、d=768、**FFN 隐藏维 = d（mult=1）** 模型
（每层注意力 2.36M + FFN 1.18M ≈ 3.54M × 12 = 42.5M + 开销 ≈ 47M，完全吻合）。
D.1 也说 "Similar to Graphormer, Transformer-M and Uni-Mol"——这三个模型的
FFN 隐藏维都等于 d。把这条用到 DeMol 上：

| 配置（12 层、da=db=768、8 头，用本仓库代码实测） | 参数量 |
| --- | --- |
| mult=4，4 个独立 FFN（最初的默认） | 350.65 M |
| mult=4，交叉注意力共享 FFN（`cross_ffn=false`） | 237.27 M |
| mult=2，4 个独立 FFN | 237.33 M |
| **mult=1，4 个独立 FFN（Graphormer 式）** | **180.67 M** |
| mult=2，共享交叉 FFN | 180.61 M |
| 6 层 block，mult=4，4 个独立 FFN | 180.26 M |

三个不同读法都收敛到 **~180.6 M**，与标称 186 M 只差 ~3%
（差值在读出头规模、词表嵌入等论文未展开的细节里）。结论：

* 186 M **排除了 mult=4**；最吻合的是 **Graphormer 式 mult=1、四个更新各自带 FFN**
  （`cross_ffn=True`）。**`paper` 预设已默认 `ffn_mult=1`**，实测 180.67 M。
* 论文写 `W2_D ∈ R^{K×1}`、`Φdist` 形状 N×N——按字面，结构偏置在所有注意力头间
  **共享**（`per_head_bias=false`）；我们默认的每头一份是 Transformer-M 式推广。
* 若要进一步连偏置粒度也按论文字面，加 `--set-model per_head_bias=false`
  （对参数量影响可忽略，主要是实现读法问题）。

还可以进一步压缩（都会偏离论文字面实现，需要自己权衡）：

| 开关 | 作用 |
| --- | --- |
| `--set-model cross_ffn=false` | 交叉注意力不再单独带 FFN |
| `--set-model share_structure_encoding=true` | 所有层共用一份 `Φ`，而不是每层一份 |
| `--set-model per_head_bias=false` | 偏置在所有头上广播，而不是每头一份（论文的字面读法） |
| `--set-model bond_head_rank=64` | 低秩分解 `N×N` 的键预测头 |
| `--set-model gradient_checkpointing=true` | 不换参数量，换显存（8 GB 卡跑 base/paper 时基本必需） |

**准确数字请用 `--dry-run` 看**：它会按组件打印参数分布，并直接把当前配置与
186 M 并排列出。`smoke_test.py` 的 `param_budget` 一项做的也是这件事。

---

## 配置预设

两个正交的轴：`--preset` 决定模型尺寸，`--scale` 决定优化预算。

### `--preset`（模型尺寸）

| preset | 层数 | d_atom / d_bond | 头数 | 高斯基 K | 用途 |
| --- | --- | --- | --- | --- | --- |
| `smoke` | 2 | 64 | 4 | 16 | 自检，CPU 秒级 |
| `small` | 4 | 128 | 8 | 32 | 笔记本 GPU，端到端跑通 |
| `base` | 8 | 384 | 8 | 64 | 单张 24 GB 卡，真实 QM9 实验 |
| `paper` | 12 | 768 | 8 | 128 | Appendix D.1 原始配置 |

FFN 宽度：`paper` 用 mult=1（Graphormer 式，与论文 186 M 口径对齐，实测
180.67 M）；`small`/`base` 用 mult=4，`smoke` 用 mult=2（这两个是本地自定义
尺寸，不对应论文）。

### `--scale`（优化预算）

| 阶段 | scale | lr | batch | steps | warmup | mask_ratio | coord_noise |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 预训练 | `paper` | 2e-4 | 1024 | 1,500,000 | 150,000 | 0.15 | 0.1 Å |
| 预训练 | `local` | 3e-4 | 32 | 100,000 | 5,000 | 0.15 | 0.1 Å |
| 微调 | `paper` | 7e-5 | 128 | 600,000 | 60,000 | 0 | 0 |
| 微调 | `local` | 2e-4 | 32 | 60,000 | 3,000 | 0 | 0 |

两个 scale 共用同一套 schedule 形状（线性 warmup → 线性衰减到 0）、
grad clip 5.0、weight decay 0、AdamW(β=0.9/0.999, eps=1e-8)、EMA 0.999。
`local` 是为一张 8 GB 笔记本卡缩短的，目的是把整条流水线跑通，
**不是**为了复现论文精度。

任意字段都能用 `--set-model / --set-train / --set-loss KEY=VALUE` 覆盖，
可重复传；`--targets` 接受 `all` 或逗号分隔的目标名。

---

## QM9 的 12 个目标与论文基准

顺序即 `demol/qm9.py: TARGET_NAMES`，也是标签向量的列顺序。能量类目标
（u0/u298/h298/g298）已按论文惯例做**原子化**（减去每元素参考能量，
取自 xyz 第 2 行的第 15-18 列）。

| # | 目标 | 单位 | 论文 Table 3 (DeMol) | 论文 Table 8 (无预训练) |
| --- | --- | --- | --- | --- |
| 0 | `mu` | D | 0.024 | — |
| 1 | `alpha` | Bohr³ | 0.042 | — |
| 2 | `homo` | meV | 16.4 | 21.2 |
| 3 | `lumo` | meV | 15.7 | 19.3 |
| 4 | `gap` | meV | 26.8 | 30.4 |
| 5 | `r2` | Bohr² | 0.101 | — |
| 6 | `zpve` | meV | 1.16 | — |
| 7 | `u0` | kcal/mol | 6.24 | — |
| 8 | `u298` | kcal/mol | 6.10 | — |
| 9 | `h298` | kcal/mol | 6.35 | — |
| 10 | `g298` | kcal/mol | 6.29 | 10.97 |
| 11 | `cv` | cal/(mol·K) | 0.021 | 0.029 |

论文 Table 3 最后一列：DeMol 在 12 个目标上的平均排名 **2.67**。

这些数字全部集中在 `benchmarks.py`，是**纯报告用常数**——模型、损失、训练器
都不 import 它，所以对不上时不可能悄悄改变一次运行。`run_finetune.py` 结束时
会打印一张对照表，列为 `target / unit / this run / paper / ratio / no-pretr.`，
其中 `ratio` 是本次结果除以论文值（>1 表示更差）；`--no-compare` 可关闭。

---

## 产物文件

预训练和微调都往 `--out-dir` 写：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 解析后的 model/train/loss 配置 + 环境信息 + 权重迁移报告 |
| `standardizer.json` | 训练集标签均值/方差，**微调必须复用同一份** |
| `last.pt` / `best.pt` | Trainer 检查点（含 optimizer / scheduler / EMA / step） |
| `ema_model.pt` | 纯 EMA 权重，直接喂给 `run_finetune.py --pretrained` |
| `final_metrics.json` | 各 split 的物理单位 MAE + 最近 20 条验证历史 |
| `train_log.txt` | 与屏幕同步的完整日志 |

`ema_model.pt` 之所以要单独导出：EMA 影子只保存浮点张量，而高斯基中心这类
整数/缓冲区不在其中，所以导出时把影子权重**合并回完整 state_dict** 再存成
`{"model": ...}`，这样 `load_pretrained_weights` 一次就能装回去。

微调载入预训练权重时按**名字 + 形状**双重过滤，属性头对不上就跳过并如实汇报，
不会因为头不匹配而崩（`--reinit-head` 可以主动随机重置属性头）。

---

## 自检脚本

`scripts/smoke_test.py` 用**合成分子**（只含 C/N/O/F，键长按共价半径之和生成，
新键方向按 VSEPR 式互斥挑选，非键原子对强制拉开到 `1.20 ×` 成键阈值以上）跑
11 项检查，不需要 QM9 数据也不需要 GPU：

```powershell
python scripts/smoke_test.py              # 全部
python scripts/smoke_test.py --list       # 只列名字和说明
python scripts/smoke_test.py --only bond  # 名字模糊匹配
```

| 检查项 | 验证什么 |
| --- | --- |
| `bond_perception` | Algorithm 1 必须原样恢复合成分子的化学键真值 |
| `featurizer_vocab` | 字段取值落在各自词表内；SPD 对称、对角为 0 |
| `numpy_torch_parity` | 预处理用的 numpy 几何 与 前向用的 torch 几何 必须一致 |
| `gaussian_basis` | eq. (5) 的高斯基：负值、峰位、形状 |
| `collate_padding` | 稠密批处理：padding 不污染 mask / SPD / 键真值 / atom-bond 关联 |
| `forward_shapes` | 前向输出形状、有限性、mask 与去噪任务的行为 |
| `loss_and_backward` | 四个损失都参与、都有限，反向无 NaN/Inf |
| `training_and_eval` | 3 步训练 + 验证 + 检查点/EMA 存取（走真实 Trainer 路径） |
| `config_overrides` | `--set-model/--set-train/--set-loss` 三条覆盖路径 |
| `param_budget` | 各预设参数量，与论文 186 M 的差异 |
| `benchmark_reporting` | 基准常数与对比表渲染 |

`numpy_torch_parity` 是最值得留意的一项：特征化在 numpy 里算一遍几何量，
前向在 torch 里再算一遍，两者必须逐元素一致，否则预处理的缓存和模型看到的
东西就不是同一个分子。

### 随机数据探针

`smoke_test.py` 逐条断言"每个部件对不对"，`scripts/random_probe.py` 则回答一个
更朴素的问题：**接上优化器之后，损失到底降不降。**

```powershell
python scripts/random_probe.py                                  # 预训练：四个损失一起
python scripts/random_probe.py --steps 200 --mols 16 --log-every 25
python scripts/random_probe.py --stage finetune --targets gap `
                               --mask-ratio 0 --coord-noise 0    # 微调：只留 Lprop
```

它拿一批随机合成分子（目标值也是随机数）反复优化同一个 batch，打印损失曲线、
全局梯度范数和参数位移；退出码非 0 表示有参数没动或损失没降。最近一次实测：

| 阶段 | 步数 | 首步 loss | 末步 loss | 分量变化 |
| --- | --- | --- | --- | --- |
| `pretrain`（12 目标 + 三个辅助任务） | 200 | 28.53 | 5.98 | prop 23.7→4.6，mask 1.83→0.57，coord 1.88→0.034，bond 1.14→0.81 |
| `finetune`（只 Lprop，`gap`，MSE） | 120 | 285.6 | 2.08 | 164/164 个参与梯度的参数被更新 |

微调时那 10 个"没动"的参数（`mask_token`、`mask_head.*`、`bond_bias` 等）是
**应该**没动的：`finetune` 把 `w_mask/w_coord/w_bond` 置零，这些头不参与损失，
自然收不到梯度。探针按阶段区分这一点，不会误报。

### 运行期验证修掉的三个 bug

* `encoding.gather_atoms` 原本只支持 2 维索引，而 `torsion_cosine` 要用 (B, M, M)
  的共享原子矩阵去 gather 坐标，直接抛 `Index tensor must have the same number of
  dimensions as input tensor`。改成先摊平再还原，任意尾随维度都能用。
  **这是模型本体的真 bug：键通道的扭转偏置此前根本跑不通。**
* `DeMol.forward` 给 padding 原子也加了坐标噪声（只在 `pos_in` 上事后乘 mask，
  返回的 `coord_noise` 没乘）。现在噪声本身就带 mask，`coord_input == pos + noise`
  处处成立。损失那边本来就乘了 `atom_mask`，所以不影响数值，只是消除歧义。
* `smoke_test.py`/`random_probe.py` 的临时目录改用仓库内的 `.smoke_tmp`（可用
  `DEMOL_TMP_ROOT` 改位置），并且**不用 `tempfile.mkdtemp`**：它以 0o700 模式建
  目录，在受限沙箱下建出来的目录反而不可写（连 `chmod` 都被拒），只能用
  `os.makedirs`。

---

## 已知限制与下一步

* **只验证到随机数据。** 11 项自检和随机探针都过了，但还没在真实 QM9 上跑过完整
  流程。第一次跑请先 `smoke_test.py`，再用 `--preset smoke` 走一遍
  `preprocess → pretrain → finetune` 的极小流程，确认产物文件齐全后才放大到
  `small`/`base`。
* **8 GB 显存。** `base`/`paper` 预设在这张卡上基本需要
  `gradient_checkpointing=true` + 小 batch + 梯度累积，`local` scale 的
  batch 32 就是按这个约束定的。
* **预训练语料偏差**（见偏差 1）意味着这套代码适合验证**架构是否有效**
  （有/无 bond 通道、有/无扭转编码的消融），不适合声称复现了论文精度。
  想做架构消融，用 `run_finetune.py` 不带 `--pretrained` 跑对照组，
  对应论文 Table 8 的 "No Pre-training"。
* **rdkit 后端未在本机验证过**（环境里没装）。服务器上第一次用
  `--backend rdkit` 建缓存时，建议同时用 `--limit 200` 跑一遍几何后端，
  对比两者在 `bond_class` 上的分歧率。

---

## 参考

* 论文原文：`E:\deeplearn\创新创业\文献\补充文献\B5_分子性质预测\Enhancing Molecular Property Predictions by Learning from Bond Modelling and Interactions (Liu 2026).pdf`
* 提取出的纯文本（本次复现的依据）：`E:\deeplearn\创新创业\Chem-dbscan\_tmp_demol_paper.txt`
* QM9 划分协议跟随 Thölke & De Fabritiis (2022)；损失与微调设置跟随
  Transformer-M；结构编码（每层独立的 `W1/W2` 投影 + SPD embedding）跟随
  Graphormer / Transformer-M。
