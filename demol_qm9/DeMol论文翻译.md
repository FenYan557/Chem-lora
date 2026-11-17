# 通过键建模与键交互增强分子性质预测

> **原文**：*Enhancing Molecular Property Predictions by Learning from Bond Modelling
> and Interactions*，Yunqing Liu、Yi Zhou、Wenqi Fan（香港理工大学），
> ICLR 2026 会议论文，arXiv:2603.00568v1 [cs.LG]，2026-02-28。
>
> **翻译说明**：公式按论文编号重排为 LaTeX；表格按原数据重排为 Markdown
> （个别在 PDF 提取中丢失的数值以原文为准）；图片无法嵌入，仅保留图题与描述；
> 参考文献保留英文原文。译文中〔译注〕标记的是译者为方便理解补充的说明。

---

## 摘要

分子表示学习对于理解和预测分子性质至关重要。然而，传统的以原子为中心的模型
（atom-centric models）仅将化学键视为成对的相互作用，往往忽略了共振
（resonance）和立体选择性（stereoselectivity）等复杂的键级现象。这一疏忽限制
了它们对精细化学行为的预测精度。为解决这个问题，我们提出 DeMol——一个双图
（dual-graph）框架，其架构设计源于严格的信息论分析，该分析证明了以键为中心
（bond-centric）的视角能带来信息增益。DeMol 通过并行的以原子为中心和以键为
中心的两条通道对分子进行显式建模，二者由多尺度的双螺旋块（Double-Helix
Blocks）协同融合，用以学习复杂的原子-原子、原子-键和键-键相互作用。框架的
几何一致性还通过一个基于共价半径的正则化项得到增强，以强制生成化学上合理的
结构。在 PCQM4Mv2、OC20 IS2RE、QM9 和 MoleculeNet 等多个基准上的全面评估
表明，DeMol 建立了新的最优水平（state-of-the-art），超过现有方法。这些结果
证实了显式建模键信息及键间相互作用的优越性，为更鲁棒、更准确的分子机器学习
铺平了道路。

---

## 1 引言

> **图 1**：苯（C₆H₆）中不同类型化学键的示例。由 p 轨道重叠形成的离域 π 体系
> 在环平面上/下方产生电子云密度，赋予分子稳定性和独特的反应活性。
> 〔译注：图中对比了"孤立的键"与"共轭的键"。〕

分子是物质的基本构建单元，从简单化合物到复杂生物体系，万物的性质与行为都由
分子支配（Fan et al., 2025; Liu et al., 2025b; Zhou et al., 2025）。分子精密的
三维结构与化学相互作用决定了药物、材料、催化剂和生物分子的功能，使得分子理解
在医学（Li et al., 2024b; Zhao et al., 2024）、能源（Wang et al., 2025a;
Hu et al., 2023）和环境科学（Liu et al., 2025a; 2023）等领域不可或缺。破译
这些关系不仅需要实验技术，还需要能够大规模建模和预测分子行为的计算方法——
正是这一挑战推动了分子表示学习的兴起。

传统的分子分析方法依赖手工设计的描述符（如分子指纹）或基于物理的模拟，既耗费
人力，又往往难以捕捉复杂的结构依赖（Duvenaud et al., 2015; Collins & Bettens,
2015; Li et al., 2024a;c）。随着 AI 技术中图神经网络（GNN）的出现（Fan et al.,
2022; Zhang et al., 2024; Ma et al., 2026），分子如今被广泛描述为图：原子
（节点）和化学键（边）构成相互连接的网络，从而可以端到端地学习特定任务的嵌入
（Luo et al., 2022; Lu et al., 2023）。最近的方法进一步转向将分子建模为 3D
几何图，利用原子的空间坐标来捕捉对精确性质预测至关重要的定向成键模式、位阻
效应和非共价相互作用（Liao & Smidt, 2022; Lu et al., 2023; Hussain et al.,
2024）。

然而，大多数现有方法把原子当作主要实体，忽略了化学键本身蕴含的丰富信息
（Ying et al., 2021; Xia et al., 2023; Luo et al., 2022）。化学键不仅仅是成对
的相互作用，还携带键级、键长、杂化状态等属性，直接影响分子的反应活性和稳定性
（Evans, 2001）。例如，如图 1 所示，苯环上交替的单双键并非彼此孤立，而是通过
键间共振形成离域 π 电子体系。这种集体行为无法用成对的原子相互作用来描述。
分子建模中另一个关键但尚未被充分探索的方面，是对**化学键之间相互作用**的显式
捕捉。尽管在图模型中化学键通常被当作相互独立的边，真实分子却表现出非加性的键
间相互作用，主导着立体选择性、空间协同性等现象（Ding et al., 2019; Weng et
al., 2021）。例如，如图 2 所示，顺铂（cisplatin）是一种抗癌药，属于细胞周期
非特异性药物，对肉瘤、恶性上皮肿瘤、淋巴瘤和生殖细胞肿瘤有治疗作用。具体而言，
顺铂是一种含铂化合物，两个氨配体和两个氯离子以顺式构型结合在中心铂原子上。
这种空间排布使顺铂能够交联 DNA 链，破坏复制并诱导癌细胞凋亡（Dong et al.,
2019）。相反，顺铂的立体异构体反铂（transplatin）具有完全相同的原子组成，但
配体为反式构型，因而不具药理活性。因此，这些惊人差异并非来自单个键属性（如键
长或杂化）的变化，而是来自键与键之间相对取向的集体效应。

> **图 2**：影响分子性质的键-键相互作用示例。左为抗癌药顺铂（cisplatin）：两个
> 氨配体和两个氯离子以顺式构型结合在中心铂原子上。右为反铂（transplatin）：
> 原子组成相同，但配体为反式构型，因而没有药理活性。

为消除这些局限，我们提出将分子键相互作用纳入分子表示学习，从而把键与相互作用
信息整合进分子性质预测。我们首先在 3.1 节从理论上确立以键为中心的图在分子表
示学习中捕捉键中心属性与相互作用的必要性。我们提出新颖的**双图增强多尺度交互
分子表示学习框架 DeMol**（Dual-graph enhanced Multi-scale interaction framework
for Molecule representation learning）——一个层级化的多尺度框架，通过双图表示
和几何约束显式地同时建模原子与化学键，包含以原子为中心的通道、以键为中心的通
道以及双螺旋块。如图 3 所示，两条通道分别把分子编码为以原子为中心的图和以键
为中心的图；双螺旋块随后在两个通道之间进行多尺度的信息交换与融合，使模型能够
捕捉复杂的原子-原子、原子-键、键-键相互作用，从而全面理解分子。为保证几何一致
性，我们还引入基于共价半径的键预测作为正则化项（见算法 1），通过惩罚偏离预期
成键距离的结构来强制化学合理性。最后，我们在多种分子性质预测任务上评估模型。
DeMol 在 PCQM4Mv2、OC20 IS2RE、QM9 和 MoleculeNet 数据集上均超过了现有最优
方法，验证了 DeMol 的性能优势。

> **图 3**：DeMol 通过双图表示整合以原子为中心的通道与以键为中心的通道。跨层次
> （原子-键）的相互作用由双螺旋块强制实现，从而保证几何一致性。

---

## 2 相关工作

**基于图的分子表示学习。** 分子表示学习旨在把分子结构编码到连续向量空间中，
捕捉其化学知识以服务下游任务（Guo et al., 2022）。作为最具代表性的 AI 技术
之一（Xu et al., 2025; Ning et al., 2025; Qu et al., 2025; 2026; Wang et al.,
2025b; Fan et al., 2024; Ning et al., 2024），GNN 与图 Transformer 通过消息传
递把原子拓扑信息编码进分子表示（Gasteiger et al., 2020; 2021; Ying et al.,
2021; Rampášek et al., 2022; Wang et al., 2020; Fan et al., 2019; 2020）。各种
几何感知 GNN 与等变神经网络还额外学习分子的空间几何知识（Liu et al., 2022b;
Liao & Smidt, 2022; Zhou et al., 2023; Wang et al., 2023b）。此外，也有一些统一
的 2D/3D 分子表示学习框架被提出，以整体地利用拓扑与几何特征（Liu et al.,
2021; Luo et al., 2022; Lu et al., 2023; Stärk et al., 2022）。

**分子键建模。** 近年来，3D 分子表示学习的进展已隐式地纳入键相关信息以增强原
子表示。DimeNet（Gasteiger et al., 2020）把原子间距离和角度作为结构约束建模；
GemNet（Gasteiger et al., 2021）在几何表示与消息传递中引入二面角；
Transformer-M（Luo et al., 2022）把 3D 距离编码整合进注意力机制。与此同时，
也有个别开创性研究尝试更多地聚焦化学键本身：LEMON（Chen et al., 2025）以线图
（line graph）作为分子图的对照；GEM（Fang et al., 2022）通过在键角图上建模
来编码分子几何；ALIGNN（Choudhary & DeCost, 2021）使用线图，将其作为消息传
递的路径；ESA（Buterez et al., 2025）提出一种端到端注意力架构，把图当作边的
集合以显式学习边表示。这些方法证明了键特征的价值——无论作为辅助信息还是独立
建模元素。然而，尽管已取得这些成功，现有分子建模方法仍存在原子/键建模不平衡
与几何简化的问题。要理解更完整的分子语义，仍需要一个能显式学习动态原子-原子、
原子-键、键-键相互作用的协同框架。

---

## 3 方法

我们首先从理论上确立以键为中心的图在分子表示学习中捕捉键中心属性与相互作用的
必要性。这一理论基础催生了我们的 DeMol 框架——它显式建模原子-原子、原子-键、
键-键相互作用，以更全面地理解分子结构。

**问题表述与记号。** 如图 3 所示，分子 $\mathcal{M}$ 由两张图表示。以原子为中心
的图 $G=(V,E)$ 中，节点 $v_i\in V$ 表示原子，特征为 $x_i\in\mathbb{R}^{d_a}$；边
$e_{ij}\in E$ 编码键属性 $e_{ij}\in\mathbb{R}^{d_b}$。以键为中心的图
$L(G)=(V',E')$ 中，节点 $v'_{ij}\in V'$ 表示来自 $G$ 的键，特征由 $e_{ij}$ 派生；
边 $e'_{(ij)(jk)}\in E'$ 连接共享公共原子 $j$ 的键，编码角度关系 $\theta_{ijk}$ 与
扭转角 $\varphi_{ijkl}$。我们把双图表示学习的目标定义为映射
$(G, L(G))\rightarrow(H^{(a)}, H^{(b)})$，其中 $H^{(a)}\in\mathbb{R}^{N\times d_a}$
是从 $G$ 得到的原子级嵌入，$H^{(b)}\in\mathbb{R}^{M\times d_b}$ 是从 $L(G)$ 得到
的键级嵌入。我们的目标是在保持几何一致性的同时，最大化表示与分子性质之间的互
信息。

### 3.1 理论动机与合理性

**命题 1（边邻接模式带来的信息增益）.** 对于非平凡的图 $G$，$L(G)$ 的熵满足：

$$
H(L(G)) = H(G) + \underbrace{H(E' \mid E)}_{\Delta I_{\text{edge-structure}}},
\quad \text{且 } \Delta I_{\text{edge-structure}} > 0.
\tag{1}
$$

这表明，尽管以键为中心的图是从原图派生的，它却编码了原图中不存在的独特结构信
息。论证见附录 A.1。为了利用这一独特的信息源，我们为 DeMol 设计了两条并行的
处理流：不再把化学键仅仅当作以原子为中心的图中的边，而是创建一条专门的以键为
中心的通道，把 $L(G)$ 当作一等实体来处理。这种双通道设计确保原子与键各自固有
的独特结构信息从一开始就得到独立捕捉与提炼。

**命题 2（互信息分解）.** 对于任意分子图 $G$ 及其双图表示 $(H^{(a)}, H^{(b)})$，
互信息满足：

$$
I(G, L(G);\, H^{(a)}, H^{(b)}) =
\underbrace{I(G;\, H^{(a)})}_{\text{原子中心信息}}
+ \underbrace{I(L(G);\, H^{(b)} \mid H^{(a)})}_{\text{键中心信息}}
+ \underbrace{I(G;\, H^{(b)} \mid L(G),\, H^{(a)})}_{\text{残余原子依赖}}.
\tag{2}
$$

该分解表明，当 $I(L(G); H^{(b)} \mid H^{(a)}) > 0$ 且
$I(G; H^{(b)} \mid L(G), H^{(a)}) > 0$ 时，双图学习保留的信息严格多于单图方法。
论证见附录 A.2。命题 2 对总互信息做了分解，揭示出双图表示通过结合原子中心信息、
键中心信息及其残余依赖，能比任一单图捕捉严格更多的信息。这也说明最终的预测能
力不在于单独的表示，而在于它们的有效融合。简单的特征拼接无法捕捉原子与键之间
复杂的交叉依赖。为实现真正的协同融合，我们引入双螺旋块（Double-Helix Blocks）：
一种双向交叉注意力模块，在多尺度上促进原子表示与键表示之间的动态交互。这使模
型能够显式学习并整合复杂的原子-原子、键-键、原子-键相互作用，直接优化我们的
分析所强调的关键信息融合。

**命题 3（几何信息增益）.** 记键角为 $\theta_{ijk}$、二面角为 $\varphi_{ijkl}$。
若 $L(G)$ 通过 $E'$ 编码角度关系，则键建模带来的增益满足：

$$
\Delta I = I(L(G);\, H^{(b)} \mid G;\, H^{(a)}) \;\propto\;
\mathbb{E}_{i,j,k,l}\left[
\log \frac{p(\theta_{ijk}, \varphi_{ijkl} \mid h_i^{(a)}, h_j^{(a)}, h_k^{(a)}, h_l^{(a)})}
{p(\theta_{ijk}, \varphi_{ijkl})}
\right].
\tag{3}
$$

命题 3 揭示：以键为中心的图是表示键角（$\theta_{ijk}$）与二面角
（$\varphi_{ijkl}$）等复杂几何关系的天然载体——这些关系在以原子为中心的视角中
只能被隐式捕捉。$L(G)$ 中"键即节点"的结构使相邻键之间的关系（键角）与键对之间
的关系（二面角）变得显式。论证见附录 A.3。为直接利用这一几何优势，我们在以键
为中心的通道内专门引入扭转编码矩阵（$\Phi^{tors}_b$）。通过把键角和键间二面角
作为该通道的注意力偏置，我们把关键的 3D 信息注入到它最自然的表示位置。这一由
理论洞见引导的针对性设计选择，确保模型能更有效地从分子完整的几何构象中学习。

**命题 4（双图信息瓶颈）.** 对于分子性质预测 $Y$，最优编码器 $\Phi$ 最小化：

$$
\min_{\Phi}\Big[\, I(G, L(G);\, H^{(a)}, H^{(b)}) \;-\; \beta\, I(H^{(a)}, H^{(b)};\, Y) \,\Big].
\tag{4}
$$

命题 4 把信息瓶颈（Information Bottleneck）原理应用于我们的双图设定。它表明最
优模型必须学会把来自两张图的丰富信息压缩成对目标性质最具预测性的最小表示。双
图系统更丰富的输入为该优化提供了更紧的界，但也增加了学到虚假或非物理关联的风
险。这促使我们采用端到端学习，最终由一个利用融合后原子-键表示的预测头输出。
此外，为确保模型达成有效的信息瓶颈权衡，我们引入两个关键的正则化与效率机制：
**基于共价半径的键预测**与**结构感知掩码**。

综上，我们的理论分析确立了：双图表示（以原子为中心的图 + 以键为中心的图）保留
的信息严格多于单图方法（命题 1、2），并通过显式编码角度关系改善几何一致性
（命题 3）；最优编码器在信息压缩与任务相关预测之间取得平衡（命题 4）。

### 3.2 提出的框架

上述理论基础催生了我们的双图增强多尺度交互框架（DeMol），如图 3 所示。为同时
建模 $G$ 与 $L(G)$，DeMol 采用双编码通道，并行地激活以原子为中心和以键为中心
的表示。

**$G$ 上的原子中心通道。** 该通道处理以原子为中心的图 $G$，学习原子嵌入
$\{h_i^{(a)}\}$。首先，原始原子特征 $x_i$ 被嵌入为
$h_i^{(a,0)} \in \mathbb{R}^{d_a}$。与先前工作类似，我们用结构编码来编码 3D 空间
信息与 2D 图位置信息（Zhou et al., 2023; Luo et al., 2022）。对于 3D 结构编码，
我们编码欧氏距离以反映 3D 空间中任意原子对之间的空间关系。对每个原子对
$(i,j)$，先用高斯基核函数（Scholkopf et al., 1997）处理其欧氏距离：

$$
\varphi_k^{(i,j)} = -\frac{1}{\sqrt{2\pi}\,|\sigma_k|}
\exp\!\left(-\frac{1}{2}\left(
\frac{\alpha^{(i,j)} \|r_i - r_j\| + \beta^{(i,j)} - \mu_k}{|\sigma_k|}
\right)^2\right), \quad k = \{1, 2, \ldots, K\},
\tag{5}
$$

其中 $K$ 是高斯基核的数量，第 $i$ 个原子的 3D 坐标记为 $r_i \in \mathbb{R}^3$。
$\alpha^{(i,j)}$ 与 $\beta^{(i,j)}$ 是按**原子类型对**索引的可学习标量，
$\mu_k$、$\sigma_k$ 是预定义常数：$\mu_k = w\times(k-1)/K$，$\sigma_k = w/K$，
宽度 $w$ 为超参数。于是 3D 距离编码为
$\Phi^{dist}_{(i,j)} = \mathrm{GELU}(\varphi^{(i,j)} W^1_D)\, W^2_D$，其中
$\varphi^{(i,j)} = [\varphi_1^{(i,j)}; \ldots; \varphi_K^{(i,j)}]^\top$；
$W^1_D \in \mathbb{R}^{K\times K}$、$W^2_D \in \mathbb{R}^{K\times 1}$ 为可学习参
数。记 $\Phi^{dist}$ 为 3D 距离编码的矩阵形式，形状为 $N\times N$。对于 2D 图
结构编码，我们编码两原子间的最短路径距离（SPD）以反映其空间关系，记原子 $i$
与 $j$ 之间的 SPD 编码为 $\Phi^{SPD}_{ij}$。对大多数分子，任意两原子间只存在一
条不同的最短路径。记从 $i$ 到 $j$ 的最短路径上的边为
$SP_{ij} = (s_1, s_2, \ldots, e_N)$，$\Phi^{SPD} \in \mathbb{R}^{N\times N}$ 为 SPD
编码矩阵。综合以上，结构编码为 $\Phi = \Phi^{dist} + \Phi^{SPD}$。对每个原子 $i$，
更新过程包含一个自注意力和一个前馈网络（FFN）层。注意力权重与原子表示按下式
更新：

$$
\alpha^{(l)}_{ij} = \mathrm{Softmax}_j\!\left(
\frac{(W_q^{(l)} h_i^{(a,l)})^\top (W_k^{(l)} h_j^{(a,l)})}{\sqrt{d_k}}
+ \Phi^{(l)} \right), \qquad
h_i^{(a,l+1)} = \mathrm{FFN}\!\left( h_i^{(a,l)} +
\sum_{j \in V(i)} \alpha^{(l)}_{ij} W_v^{(l)} h_j^{(a,l)} \right),
\tag{6}
$$

其中 $W_q, W_k, W_v \in \mathbb{R}^{d_a \times d_h}$ 为可学习参数。

**$L(G)$ 上的键中心通道。** 与 $G$ 类似，我们处理以键为中心的图 $L(G)$ 来学习
键嵌入 $\{h_{ij}^{(b)}\}$。首先把键特征 $e_{ij}$ 编码为
$h_{i,j}^{(b,0)} \in \mathbb{R}^{d_b}$。类似地，我们用 3D 空间与 2D 图位置信息
得到结构编码。记 $\Phi^{SPD}_b$ 与 $\Phi^{distance}_b$ 为键 $e_{ij}$ 与键 $e_{jk}$
之间的 SPD 编码与 3D 距离编码，形状均为 $M\times M$。具体地，我们用两个原子
坐标的中点作为键的坐标 $r_{ij} = \frac{1}{2}(r_i + r_j)$。除此之外，我们还为键中
心通道提出一种新的 3D 结构位置编码——**扭转编码** $\Phi^{tors}_b$，即形状为
$M\times M$ 的键角 $\theta_{ijk}$ 与二面角 $\varphi_{ijkl}$ 矩阵。若两条键共享一
个原子，键角按 $\cos(\theta_{ijk}) = \frac{r_{ij}\cdot r_{jk}}{\|r_{ij}\|\,\|r_{jk}\|}$
计算；否则二面角按
$\cos(\varphi_{ijkl}) = \frac{r_{ij}\cdot r_{kl}}{\|r_{ij}\|\,\|r_{kl}\|}$ 计算。
与原子中心通道类似，我们用高斯基核函数处理扭转编码。综合以上，键通道的结构编
码为 $\Phi_b = \Phi^{dist}_b + \Phi^{SPD}_b + \Phi^{tors}_b$。对每条键 $ij$，注意力
权重与键表示按下式更新：

$$
\beta^{(l)}_{(ij)(jk)} = \mathrm{Softmax}_{(jk)}\!\left(
\frac{(W_q^{(l)} h_{ij}^{(b,l)})^\top (W_k^{(l)} h_{jk}^{(b,l)})}{\sqrt{d_k}}
+ \Phi_b^{(l)} \right),
\tag{7}
$$

$$
h_{ij}^{(b,l+1)} = \mathrm{FFN}\!\left( h_{ij}^{(b,l)} +
\sum_{k \in V'(ij)} \beta^{(l)}_{(ij)(jk)} W_v^{(l)} h_{jk}^{(b,l)} \right),
\tag{8}
$$

其中 $W_q, W_k, W_v \in \mathbb{R}^{d_b \times d_h}$ 为可学习参数。

**通过双螺旋块的跨图交互。** 为对齐原子与键表示，我们引入双螺旋块，通过交叉注
意力强制多尺度一致性。对原子 $i$，在其关联键集合 $\{ij\}$ 上做交叉注意力：

$$
\gamma^{(l)}_{i,ij} = \mathrm{Softmax}_{ij}\!\left(
\frac{(W_q^{(l)} h_i^{(a,l)})^\top (W_k^{(l)} h_{ij}^{(b,l)})}{\sqrt{d_k}}
+ \Phi^{(l)} \right), \qquad
h_i^{(a,l+1)} = \mathrm{FFN}\!\left( h_i^{(a,l)} +
\sum_{k \in V(i)} \gamma^{(l)}_{i,ij} W_v^{(l)} h_{ij}^{(b,l)} \right).
\tag{9}
$$

对键 $ij$，在其原子 $i, j$ 上做交叉注意力：

$$
\delta^{(l)}_{ij,i} = \mathrm{Softmax}_{i}\!\left(
\frac{(W_q^{(l)} h_{ij}^{(b,l)})^\top (W_k^{(l)} h_i^{(a,l)})}{\sqrt{d_k}}
+ \Phi_b^{(l)} \right), \qquad
h_{ij}^{(b,l+1)} = \mathrm{FFN}\!\left( h_{ij}^{(b,l)} +
\sum_{k \in \{i,j\}} \delta^{(l)}_{ij,k} W_v^{(l)} h_k^{(a,l)} \right).
\tag{10}
$$

**基于共价半径的键预测。** 为确保分子表示学习中的几何一致性，DeMol 通过一个
建立在共价半径约束上的键预测模块来显式建模原子与键的相互作用。给定分子
$\mathcal{M}$ 的原子 $\{a_i\}_{i=1}^N$、其 3D 坐标
$\{\vec{p}_i = (x_i, y_i, z_i)\}_{i=1}^N$ 与共价半径字典 $R_{cov}$，我们的算法
（附录 B 算法 1）通过比较原子间距离与共价半径导出的阈值来预测键。依据
(Lu & Chen, 2012)，键阈值因子 $\alpha$ 取 1.15。"基于共价半径的键预测"模块充当
正则化项：它惩罚生成几何上不一致结构的模型，确保学到的表示符合基本化学原理。

**结构感知掩码。** 由于引入了 $L(G)$ 上的键中心通道与跨图交互块，模型的时间复
杂度可能上升，详见附录 C 的复杂度分析。为此，我们引入结构感知掩码来缓解：它是
由化学价规则导出的稀疏注意力掩码——共价分子中原子间键长一般小于 3 Å
（Nikolaienko et al., 2019）；考虑到原子之间仍存在一些弱相互作用力，我们以原子
间距离小于 5 Å 作为是否掩码的依据。与使用邻接矩阵作掩码不同，结构感知掩码既保
留原子间的近邻连接性，又捕捉潜在的长程作用力。对于键的结构感知掩码，我们只考
虑邻接与共轭关系，以增强捕捉边之间共轭关系的能力（Nikolaienko et al., 2019）。

---

## 4 实验

本节通过实证研究 DeMol 的性能。首先在 PCQM4Mv2（来自 OGB 大规模挑战赛，
Hu et al., 2021）的训练集上预训练模型，然后在 PCQM4Mv2、Open Catalyst 2020
IS2RE、QM9 和 MoleculeNet 等下游任务上微调评估，最后通过一系列消融实验考察模
型的关键设计。

**预训练数据集。** 预训练数据集 PCQM4Mv2（Hu et al., 2021）旨在促进预测量子化
学（QC）性质——即 HOMO-LUMO 能隙——的机器学习模型的发展。该性质是最高占据
分子轨道（HOMO）与最低未占分子轨道（LUMO）之间的能量差。数据集包含 337 万个
以 SMILES 表示的分子，训练集与验证集提供 HOMO-LUMO 能隙标签，训练集还包含
DFT 平衡构象（验证集不包含）。在该基准上，模型在推理时只能使用 SMILES，不能
使用 DFT 平衡构象来预测 HOMO-LUMO 能隙。

**PCQM4Mv2.** 预训练后，我们在 PCQM4Mv2 验证集上评估 DeMol。任务是预测
HOMO-LUMO 能隙，评估指标为平均绝对误差（MAE）。由于预训练目标已经包含
HOMO-LUMO 能隙预测，我们无需额外微调即可评估。我们选取了多种基线，包括图神经
网络（GNN）和 Transformer 变体。设置与基线的详细描述见附录 D.2。

**表 1：PCQM4Mv2 验证集性能（Hu et al., 2021）。加粗为最佳。**

| 模型 | 参数量 | MAE (↓) |
| --- | --- | --- |
| MLP-Fingerprint | 16.1M | 0.1735 |
| GCN | 2.0M | 0.1379 |
| GIN | 3.8M | 0.1195 |
| GINE-VN | 13.2M | 0.1167 |
| GCN-VN | 4.9M | 0.1153 |
| GIN-VN | 6.7M | 0.1083 |
| DeeperGCN-VN | 25.5M | 0.1021 |
| GraphGPS_SMALL | 6.2M | 0.0938 |
| TokenGT | 48.5M | 0.0910 |
| GRPE_BASE | 46.2M | 0.0890 |
| EGT | 89.3M | 0.0869 |
| GRPE_LARGE | 46.2M | 0.0867 |
| Graphormer | 47.1M | 0.0864 |
| GraphGPS_BASE | 19.4M | 0.0858 |
| GraphGPS_DEEP | 13.8M | 0.0852 |
| GEM-2 | 32.1M | 0.0793 |
| GPS++ | 44.3M | 0.0778 |
| Transformer-M | 69M | 0.0772 |
| Unimol+ | 77M | 0.0693 |
| TGT-At | 203M | 0.0671 |
| **DeMol** | **186M** | **0.0603** |

表 1 的对比结果表明，DeMol 以 0.0603 eV 的 MAE 建立了新的最优水平，在更高精
度捕捉分子图信息上实现了显著飞跃。具体而言，DeMol 相对此前的最佳模型 TGT-At
（0.0671）提升了 0.0068 eV（10.1%），并显著优于 Transformer-M（0.0772）与
GraphGPS_DEEP（0.0852）等方法。值得注意的是，此前表现最好的模型依赖大规模集成
（如 GPS++ 的 112 模型集成），而 DeMol 仅用单一模型就取得了更优性能，展示了其
固有的鲁棒性与泛化能力。

**Open Catalyst 2020 IS2RE.** Open Catalyst 2020 挑战赛（Chanussot et al., 2021）
的目标是用机器学习预测分子在催化剂表面的吸附能。具体地，我们聚焦 IS2RE
（Initial Structure to Relaxed Energy，初始结构到弛豫能量）任务，约含 46 万样本。
该任务中，模型获得晶体与吸附质的初始 DFT 结构，二者在弛豫过程中相互作用，模型
的目标是预测体系最终的弛豫能量。此外，训练时提供 DFT 平衡构象，但推理时不可使
用。设置与基线的详细描述见附录 D.3。

**表 2：OC20 IS2RE 验证集性能。NN 指 "Noisy Nodes"（Godwin et al., 2021）。
加粗为最佳。**

| 模型 | ID MAE↓ | OOD Ads.↓ | OOD Cat.↓ | OOD Both↓ | 平均 MAE↓ | ID EwT↑ | OOD Ads.↑ | OOD Cat.↑ | OOD Both↑ | 平均 EwT↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SchNet | 0.6465 | 0.7074 | 0.6475 | 0.6626 | 0.6660 | 2.96 | 2.22 | 3.03 | 2.38 | 2.65 |
| DimeNet++ | 0.5636 | 0.7127 | 0.5612 | 0.6492 | 0.6217 | 4.25 | 2.48 | 4.40 | 2.56 | 3.42 |
| GemNet-T | 0.5561 | 0.7342 | 0.5659 | 0.6964 | 0.6382 | 4.51 | 2.24 | 4.37 | 2.38 | 3.38 |
| SphereNet | 0.5632 | 0.6682 | 0.5590 | 0.6190 | 0.6024 | 4.56 | 2.70 | 4.59 | 2.70 | 3.64 |
| Graphormer-3D | 0.4329 | 0.5850 | 0.4441 | 0.5299 | 0.4980 | - | - | - | - | - |
| GNS | 0.54 | 0.65 | 0.55 | 0.59 | 0.5825 | - | - | - | - | - |
| GNS+NN | 0.47 | 0.51 | 0.48 | 0.46 | 0.4800 | - | - | - | - | - |
| DRFormer | 0.4222 | 0.5420 | 0.4231 | 0.4754 | 0.4657 | 7.23 | 3.77 | 7.13 | 4.10 | 5.56 |
| EquiFormer+NN | 0.4156 | 0.4976 | 0.4165 | 0.4344 | 0.4410 | 7.47 | 4.64 | 7.19 | 4.84 | 6.04 |
| DRFormer | 0.4187 | 0.4863 | 0.4321 | 0.4332 | 0.4425 | 8.39 | 5.42 | 8.12 | 5.44 | 6.84 |
| Unimol+ | 0.3795 | 0.4526 | 0.4011 | 0.4021 | 0.4088 | 11.15 | 6.71 | 9.90 | 6.68 | 8.61 |
| TGT-At | 0.3813 | 0.4454 | 0.3917 | 0.3936 | 0.4030 | 11.15 | 6.87 | 10.47 | 6.80 | 8.82 |
| **DeMol** | **0.3663** | **0.4302** | **0.3746** | **0.3804** | **0.3879** | **12.04** | **6.98** | **10.86** | **7.01** | **9.23** |

〔译注：能量 MAE 单位为 eV；EwT 为"落在阈值内的能量百分比"，两个 EwT 列分别为
0.02 eV 与 0.05 eV 阈值口径的均值。原文表中 DRFormer 出现两行，系原文如此。〕

表 2 用两个关键指标比较了各模型在 OC20 IS2RE 验证集上的性能：能量 MAE（eV）
与阈值内能量百分比（EwT%），MAE 越低越好、EwT 越高越好。DeMol 取得最优结果，
在两项指标上均超过所有基线。具体地，DeMol 取得最低的平均能量 MAE（0.3879 eV），
相对 Unimol+（0.4088 eV）与 TGT-At（0.4030 eV）分别提升 5.1% 与 3.7%。在 EwT
上，DeMol 以平均 9.23% 显著领先（Unimol+ 为 8.61%，TGT-At 为 8.82%）。值得注意
的是，DeMol 在域内（ID）、域外吸附（OOD Ads.）、域外催化（OOD Cat.）、域外双重
（OOD Both）所有类别上都一致取得最高 EwT。这种一致的优势凸显了 DeMol 在处理域
内与多样域外场景时的鲁棒性。

**QM9.** 我们使用 QM9 数据集（Ramakrishnan et al., 2014）在 3D 格式的分子任务
上评估模型。QM9 是一个量子化学基准，包含 13.4 万个稳定的小有机分子，对应 GDB-17
化学宇宙（1660 亿有机分子）中全部 133,885 个物种的一个子集。每个分子带有 12 个
预测目标，涵盖能量、电子与热力学性质。模型输入为分子的 3D 几何结构。遵循
(Thölke & De Fabritiis, 2022)，我们随机选取 10,000 个分子用于验证、10,831 个用于
测试，其余用于微调 DeMol。基线与实验设置见附录 D.4。

**表 3：QM9 结果。评估指标为 MAE。基线为附录 D.4 报告的官方结果。加粗为最佳。**

| 模型 | μ | α | ε_HOMO | ε_LUMO | Δε | R² | ZPVE | U₀ | U | H | G | C_v | 平均排名# |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EdgePred | 0.039 | 0.086 | 37.4 | 31.9 | 58.2 | 0.112 | 1.81 | 14.7 | 14.2 | 14.8 | 14.5 | 0.038 | 18.92 |
| AttrMask | 0.020 | 0.072 | 31.3 | 37.8 | 50.0 | 0.423 | 1.90 | 10.7 | 10.8 | 11.4 | 11.2 | 0.062 | 15.25 |
| InfoGraph | 0.041 | 0.099 | 48.1 | 38.1 | 72.2 | 0.114 | 1.69 | 16.4 | 14.9 | 14.5 | 16.5 | 0.030 | 19.08 |
| GraphCL | 0.027 | 0.066 | 26.8 | 22.9 | 45.5 | 0.095 | 1.42 | 9.6 | 9.7 | 9.6 | 10.2 | 0.028 | 10.0 |
| GPT-GNN | 0.039 | 0.103 | 35.7 | 28.8 | 54.1 | 0.158 | 1.75 | 12.0 | 24.8 | 14.8 | 12.2 | 0.032 | 18.17 |
| GraphMVP | 0.031 | 0.070 | 28.5 | 26.3 | 46.9 | 0.082 | 1.63 | 10.2 | 10.3 | 10.4 | 11.2 | 0.033 | 12.00 |
| GEM | 0.034 | 0.081 | 33.8 | 27.7 | 52.1 | 0.089 | 1.73 | 13.4 | 12.6 | 13.3 | 13.2 | 0.035 | 16.00 |
| 3D Infomax | 0.034 | 0.075 | 29.8 | 25.7 | 48.8 | 0.122 | 1.67 | 12.7 | 12.5 | 12.4 | 13.0 | 0.033 | 14.92 |
| PosPred | 0.024 | 0.067 | 25.1 | 20.9 | 40.6 | 0.115 | 1.46 | 10.2 | 10.3 | 10.2 | 10.9 | 0.035 | 11.08 |
| 3D-MGP | 0.020 | 0.057 | 21.3 | 18.2 | 37.1 | 0.092 | 1.38 | 8.6 | 8.6 | 8.7 | 9.3 | 0.026 | 6.83 |
| SchNet | 0.033 | 0.235 | 41 | 34 | 63 | 0.073 | 1.7 | 14 | 19 | 14 | 14 | 0.033 | 17.33 |
| PhysNet | 0.0529 | 0.0615 | 32.9 | 24.7 | 42.5 | 0.765 | 1.39 | 8.15 | 8.34 | 8.42 | 9.4 | 0.028 | 11.67 |
| Cormorant | 0.038 | 0.085 | 34 | 38 | 61 | 0.961 | 2.027 | 22 | 21 | 21 | 20 | 0.026 | 20.08 |
| DimeNet++ | 0.0297 | 0.0435 | 24.6 | 19.5 | 32.6 | 0.331 | 1.21 | 6.32 | 6.28 | 6.53 | 7.56 | 0.023 | 6.33 |
| PaiNN | 0.012 | 0.045 | 27.6 | 20.4 | 45.7 | 0.066 | 1.28 | 5.85 | 5.83 | 5.98 | 7.35 | 0.024 | 4.67 |
| ALIGNN | 0.0146 | 0.0561 | 21.4 | 19.5 | 38.1 | 0.543 | 3.1 | 15.3 | 14.4 | 14.7 | 14.4 | 0.033 | 13.83 |
| LieTF | 0.041 | 0.082 | 33 | 27 | 51 | 0.448 | 2.10 | 17 | 16 | 17 | 19 | 0.035 | 19.58 |
| TorchMD-Net | 0.011 | 0.059 | 20.3 | 17.5 | 36.1 | 0.033 | 1.84 | 6.15 | 6.38 | 6.16 | 7.62 | 0.026 | 5.25 |
| EGNN | 0.029 | 0.071 | 29 | 25 | 48 | 0.106 | 1.55 | 11 | 12 | 12 | 12 | 0.031 | 13.00 |
| NoisyNode | 0.025 | 0.052 | 20.4 | 18.6 | 28.6 | 0.70 | 1.16 | 7.30 | 7.57 | 7.43 | 8.30 | 0.025 | 6.75 |
| Transformer-M | 0.037 | 0.041 | 17.5 | 16.2 | 27.4 | 0.075 | 1.18 | 9.37 | 9.41 | 9.39 | 9.63 | 0.022 | 6.00 |
| ESA | 0.024 | 0.041 | 17.4 | 16.0 | 27.1 | 0.101 | 1.20 | 6.42 | 6.62 | 6.45 | 8.39 | 0.024 | 4.17 |
| **DeMol** | 0.024 | 0.042 | **16.4** | **15.7** | **26.8** | 0.101 | **1.16** | **6.24** | **6.1** | 6.35 | **6.29** | **0.021** | **2.67** |

〔译注：各列单位——μ 为 D（德拜），α 为 Bohr³，ε_HOMO/ε_LUMO/Δε/ZPVE 为 meV，
R² 为 Bohr²，U₀/U/H/G 为 kcal/mol（原子化能口径），C_v 为 cal/(mol·K)。〕

评估结果见表 3，表明 DeMol 相对强基线取得了有竞争力的性能。特别地，DeMol 在
HOMO、LUMO、HOMO-LUMO 能隙、ZPVE、G 和 C_v 预测上达到最优，在其余指标上也
表现相当。我们的见解是：这一结果源于 QM9 数据集本身的性质与 DeMol 特定优势之间
的关系。QM9 由 13.4 万个非常小而简单的有机分子组成，重原子不超过 9 个
（C、O、N、F）。DeMol 专为捕捉复杂的非局部键级现象（如共振）与精细的 3D 键-键
相互作用（决定立体选择性等性质）而设计；相比 PCQM4Mv2 中更大、更多样的分子或
OC20 中复杂的多组分体系，这些复杂现象在 QM9 的小而简单的分子中远不那么普遍或
重要。这些发现表明，DeMol 通过对原子与键相互作用的建模成功捕捉了关键的量子力学
信息，从而在多样的分子性质上取得准确预测，凸显了其架构在学习可迁移表示方面的有
效性。

**MoleculeNet.** 预训练后，我们在 MoleculeNet 基准（Wu et al., 2018）上评估
DeMol，它包含 8 个广泛使用的二分类数据集，涵盖量子化学、物理化学、生物物理和
生理等多个领域，为化学信息学建模提供了全面的评测资源。遵循既有惯例，我们采用
骨架划分（scaffold splitting，Ramsundar et al., 2019），按分子结构划分数据，使评
估更好地反映真实使用场景。基线与设置详见附录 D.5。

> **图 4**：分子性质分类任务结果。（表格版本见附录表 6。）

图 4 与附录表 6 比较了各分子表示学习方法在 8 个 MoleculeNet 基准数据集上的性能，
评估采用骨架划分下 10 个随机种子的 ROC-AUC 分数。我们的 DeMol 以平均 ROC-AUC
79.96 达到最优，超过所有现有基线。模型在 8 个基准数据集中的 7 个（BBBP、Tox21、
SIDER、ClinTox、MUV、HIV、BACE）上表现最佳，同时在 ToxCast 上相比 GEM（Fang
et al., 2022）和 Uni-Mol（Zhou et al., 2023）保持有竞争力的性能。具体地，与同样
采用键中心图的 Galformer（Bai et al., 2023）、LEMON（Chen et al., 2025）和 GEM
（Fang et al., 2022）相比，DeMol 在所有分子性质预测基准上都取得更优性能。这些结
果展示了 DeMol 在多样化学与生物性质上的优越性与鲁棒性。此外，其优势不仅来自键
中心图表示，还来自对键-键与原子-键相互作用的显式建模，二者共同提升了分子性质预
测精度。

### 4.1 消融研究

**表 4：各组件影响的消融研究。实验在 PCQM4Mv2 数据集上进行。**

| 原子中心图 | 键中心图 | 共价半径键预测 | 扭转角位置编码 | 结构感知掩码 | 键-键注意力 | 键-原子注意力 | 验证集 MAE(meV)↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ✗ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | 89.9 |
| ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 77.2 |
| ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | 76.4 |
| ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | 75.5 |
| ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | 65.4 |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 64.8 |
| ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | 61.1 |
| ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | 61.7 |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **60.3** |

我们在 PCQM4Mv2 数据集上以 MAE（meV）为指标对 DeMol 的关键组件做了系统消融
（表 4）。为公平比较，所有超参数与附录 D.2 的设置保持一致。结果表明：(1) 仅使用
键中心图的 MAE 为 89.9 meV，仅使用原子中心图则降至 77.2 meV。(2) 原子中心图结
合扭转角位置编码进一步改善到 75.5 meV，而初步结合原子中心图与键中心图则把 MAE
显著降到 65.4 meV，凸显两种图表示的互补性。(3) 加入共价半径预测、扭转角编码与
结构感知掩码后，模型能显式建模角度关系并剪除非物理相互作用，几何一致性增强。
(4) 键-键与键-原子注意力机制通过捕捉显式的化学键相互作用与跨层次原子-键关系带
来额外增益。这些发现共同验证了每个组件对模型性能都有独特贡献：双图架构提供基础
的原子/键表示，而几何约束与注意力机制通过物理合理性约束与交互建模对其加以提炼。

---

## 5 结论

本文提出 DeMol——一个用于分子表示学习的双图增强多尺度交互框架，以弥补显式建模
化学键及其相互作用这一关键空白。大量实验表明，DeMol 在 PCQM4Mv2、OC20 IS2RE、
QM9 和 MoleculeNet 等数据集上的多种分子性质预测任务中均达到最优性能。

---

## 致谢

本研究部分受到香港研究资助局 General Research Funds（项目号 PolyU 15200023、
15206024、15224524）、香港理工大学校内研究基金（项目号 P0042693、P0048625、
P0051361、P0059586）以及 Sheertek International (HK) Limited 的资助。计算资源由
香港理工大学大型 AI 模型中心（CLAIM）提供。

## 伦理声明

本工作遵守 ICLR 伦理准则。本研究不涉及人类受试者或动物实验。所有使用的数据集
（PCQM4Mv2、OC2020 IS2RE、QM9 与 MoleculeNet）均按相关使用准则获取，确保不
侵犯隐私。我们在研究过程中注意避免任何偏见或歧视性结果，未使用任何个人可识别信
息，未开展可能引发隐私或安全顾虑的实验。我们承诺在整个研究过程中保持透明与诚信。

---

## 参考文献

〔译注：按学术惯例，参考文献保留英文原文。〕

Brandon Anderson, Truong Son Hy, and Risi Kondor. Cormorant: Covariant molecular neural networks. *Advances in Neural Information Processing Systems*, 32, 2019.

Peizhen Bai, Xianyuan Liu, and Haiping Lu. Geometry-aware line graph transformer pre-training for molecular property prediction. *arXiv preprint arXiv:2309.00483*, 2023.

Rémy Brossard, Oriel Frigo, and David Dehaene. Graph convolutions that can finally model local structure. *arXiv preprint arXiv:2011.15069*, 2020.

Kieron Burke. Perspective on density functional theory. *The Journal of Chemical Physics*, 136(15), 2012.

David Buterez, Jon Paul Janet, Dino Oglic, and Pietro Liò. An end-to-end attention-based approach for learning on graphs. *Nature Communications*, 16(1):5244, 2025.

Lowik Chanussot, Abhishek Das, Siddharth Goyal, Thibaut Lavril, Muhammed Shuaibi, Morgane Riviere, Kevin Tran, Javier Heras-Domingo, Caleb Ho, Weihua Hu, et al. Open catalyst 2020 (OC20) dataset and community challenges. *ACS Catalysis*, 11(10):6059–6072, 2021.

Xueyuan Chen, Shangzhe Li, Ruomei Liu, Bowen Shi, Jiaheng Liu, Junran Wu, and Ke Xu. Molecular graph contrastive learning with line graph. *Pattern Recognition*, pp. 111380, 2025.

Kamal Choudhary and Brian DeCost. Atomistic line graph neural network for improved materials property predictions. *npj Computational Materials*, 7(1):185, 2021.

Michael A Collins and Ryan PA Bettens. Energy-based molecular fragmentation methods. *Chemical Reviews*, 115(12):5607–5642, 2015.

Thomas M Cover, Joy A Thomas, et al. Entropy, relative entropy and mutual information. *Elements of Information Theory*, 2(1):12–13, 1991.

Kingma Diederik. Adam: A method for stochastic optimization. 2014.

Bo Ding, Yue Weng, Yunqing Liu, Chunlan Song, Le Yin, Jiafan Yuan, Yanrui Ren, Aiwen Lei, and Chien-Wei Chiang. Selective photoredox trifluoromethylation of tryptophan-containing peptides. *European Journal of Organic Chemistry*, 2019(46):7596–7605, 2019.

Yu Dong, Tao Xu, Shouping Zhong, Bo Wang, Huimin Zhang, Xin Wang, Peng Wang, Guangshun Li, and Shuanying Yang. Circ_0076305 regulates cisplatin resistance of non-small cell lung cancer via positively modulating stat3 by sponging mir-296-5p. *Life Sciences*, 239:116984, 2019.

David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. *Advances in Neural Information Processing Systems*, 28, 2015.

Evan Evans. Probing the relation between force—lifetime—and chemistry in single molecular bonds. *Annual Review of Biophysics and Biomolecular Structure*, 30(1):105–128, 2001.

Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. Graph neural networks for social recommendation. In *The World Wide Web Conference*, pp. 417–426, 2019.

Wenqi Fan, Yao Ma, Qing Li, Jianping Wang, Guoyong Cai, Jiliang Tang, and Dawei Yin. A graph neural network framework for social recommendations. *IEEE Transactions on Knowledge and Data Engineering*, 2020.

Wenqi Fan, Xiaorui Liu, Wei Jin, Xiangyu Zhao, Jiliang Tang, and Qing Li. Graph trend filtering networks for recommendation. In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 112–121, 2022.

Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey on RAG meeting LLMs: Towards retrieval-augmented large language models. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 6491–6501, 2024.

Wenqi Fan, Yi Zhou, Shijie Wang, Yuyao Yan, Hui Liu, Qian Zhao, Le Song, and Qing Li. Computational protein science in the era of large language models (LLMs). *arXiv preprint arXiv:2501.10282*, 2025.

Xiaomin Fang, Lihang Liu, Jieqiong Lei, Donglong He, Shanzhuo Zhang, Jingbo Zhou, Fan Wang, Hua Wu, and Haifeng Wang. Geometry-enhanced molecular representation learning for property prediction. *Nature Machine Intelligence*, 4(2):127–134, 2022.

Johannes Gasteiger, Janek Groß, and Stephan Günnemann. Directional message passing for molecular graphs. *arXiv preprint arXiv:2003.03123*, 2020.

Johannes Gasteiger, Florian Becker, and Stephan Günnemann. GemNet: Universal directional graph neural networks for molecules. *Advances in Neural Information Processing Systems*, 34:6790–6802, 2021.

Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In *International Conference on Machine Learning*, pp. 1263–1272. PMLR, 2017.

Jonathan Godwin, Michael Schaarschmidt, Alexander Gaunt, Alvaro Sanchez-Gonzalez, Yulia Rubanova, Petar Veličković, James Kirkpatrick, and Peter Battaglia. Simple GNN regularisation for 3D molecular property prediction & beyond. *arXiv preprint arXiv:2106.07971*, 2021.

Zhichun Guo, Kehan Guo, Bozhao Nan, Yijun Tian, Roshni G Iyer, Yihong Ma, Olaf Wiest, Xiangliang Zhang, Wei Wang, Chuxu Zhang, et al. Graph-based molecular representation learning. *arXiv preprint arXiv:2207.04869*, 2022.

Hanxu Hu, Yunqing Liu, Zhongyi Yu, and Laura Perez-Beltrachini. Improving user controlled table-to-text generation robustness. In *Findings of the Association for Computational Linguistics: EACL 2023*, pp. 2317–2324, 2023.

Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. Strategies for pre-training graph neural networks. *arXiv preprint arXiv:1905.12265*, 2019.

Weihua Hu, Matthias Fey, Hongyu Ren, Maho Nakata, Yuxiao Dong, and Jure Leskovec. OGB-LSC: A large-scale challenge for machine learning on graphs. *arXiv preprint arXiv:2103.09430*, 2021.

Md Shamim Hussain, Mohammed J Zaki, and Dharmashankar Subramanian. Global self-attention as a replacement for graph convolution. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 655–665, 2022.

Md Shamim Hussain, Mohammed J Zaki, and Dharmashankar Subramanian. Triplet interaction improves graph transformers: Accurate molecular graph learning with triplet graph transformers. *arXiv preprint arXiv:2402.04538*, 2024.

Michael J Hutchinson, Charline Le Lan, Sheheryar Zaidi, Emilien Dupont, Yee Whye Teh, and Hyunjik Kim. LieTransformer: Equivariant self-attention for Lie groups. In *International Conference on Machine Learning*, pp. 4533–4543. PMLR, 2021.

Bo Jiang, Ziyan Zhang, Doudou Lin, Jin Tang, and Bin Luo. Semi-supervised learning with graph learning-convolutional networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11313–11320, 2019.

Rui Jiao, Jiaqi Han, Wenbing Huang, Yu Rong, and Yang Liu. Energy-motivated equivariant pre-training for 3D molecular graphs. In *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 37, pp. 8096–8104, 2023.

Jinwoo Kim, Dat Nguyen, Seonwoo Min, Sungjun Cho, Moontae Lee, Honglak Lee, and Seunghoon Hong. Pure transformers are powerful graph learners. *Advances in Neural Information Processing Systems*, 35:14582–14595, 2022.

Guohao Li, Chenxin Xiong, Ali Thabet, and Bernard Ghanem. DeeperGCN: All you need to train deeper GCNs. *arXiv preprint arXiv:2006.07739*, 2020.

Jiatong Li, Junxian Li, Weida Wang, Yunqing Liu, Changmeng Zheng, Dongzhan Zhou, Xiao-yong Wei, and Qing Li. Speak-to-structure: Evaluating LLMs in open-domain natural language-driven molecule generation. *arXiv preprint arXiv:2412.14642*, 2024a.

Jiatong Li, Yunqing Liu, Wenqi Fan, Xiao-Yong Wei, Hui Liu, Jiliang Tang, and Qing Li. Empowering molecule discovery for molecule-caption translation with large language models: A ChatGPT perspective. *IEEE Transactions on Knowledge and Data Engineering*, 2024b.

Jiatong Li, Yunqing Liu, Wei Liu, Jingdi Le, Di Zhang, Wenqi Fan, Dongzhan Zhou, Yuqiang Li, and Qing Li. MolReflect: Towards in-context fine-grained alignments between molecules and texts. *arXiv preprint arXiv:2411.14721*, 2024c.

Yi-Lun Liao and Tess Smidt. Equiformer: Equivariant graph attention transformer for 3D atomistic graphs. *arXiv preprint arXiv:2206.11990*, 2022.

Chengyi Liu, Wenqi Fan, Yunqing Liu, Jiatong Li, Hang Li, Hui Liu, Jiliang Tang, and Qing Li. Generative diffusion models on graphs: methods and applications. In *Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI '23)*, 2023.

Lihang Liu, Donglong He, Xiaomin Fang, Shanzhuo Zhang, Fan Wang, Jingzhou He, and Hua Wu. GEM-2: next generation molecular property prediction network by modeling full-range many-body interactions. *arXiv preprint arXiv:2208.05863*, 2022a.

Shengchao Liu, Hanchen Wang, Weiyang Liu, Joan Lasenby, Hongyu Guo, and Jian Tang. Pre-training molecular graph representation with 3D geometry. *arXiv preprint arXiv:2110.07728*, 2021.

Xin Liu, Arindam Sau, Alexander R Green, Mihai V Popescu, Nicholas F Pompetti, Yingzi Li, Yucheng Zhao, Robert S Paton, Niels H Damrauer, and Garret M Miyake. Photocatalytic C–F bond activation in small molecules and polyfluoroalkyl substances. *Nature*, 637(8046):601–607, 2025a.

Yi Liu, Limei Wang, Meng Liu, Yuchao Lin, Xuan Zhang, Bora Oztekin, and Shuiwang Ji. Spherical message passing for 3D molecular graphs. In *International Conference on Learning Representations (ICLR)*, 2022b.

Yunqing Liu, Wenqi Fan, Xiaoyong Wei, and Li Qing. GLProtein: Global-and-local structure aware protein representation learning. In *Findings of the Association for Computational Linguistics: EMNLP 2025*, pp. 4355–4372, 2025b.

Shuqi Lu, Zhifeng Gao, Di He, Linfeng Zhang, and Guolin Ke. Highly accurate quantum chemical property prediction with Uni-Mol+. *arXiv preprint arXiv:2303.16982*, 2023.

Tian Lu and Feiwu Chen. Multiwfn: A multifunctional wavefunction analyzer. *Journal of Computational Chemistry*, 33(5):580–592, 2012.

Shengjie Luo, Tianlang Chen, Yixian Xu, Shuxin Zheng, Tie-Yan Liu, Liwei Wang, and Di He. One transformer can understand both 2D & 3D molecular data. *arXiv preprint arXiv:2210.01765*, 2022.

Hongxu Ma, Han Zhou, Kai Tian, Xuefeng Zhang, Chunjie Chen, Han Li, Jihong Guan, and Shuigeng Zhou. Gor: A unified and extensible generative framework for ordinal regression. In *The Fourteenth International Conference on Learning Representations*, 2026.

Nakata Maho. The PubChemQC project: A large chemical database from the first principle calculations. In *AIP Conference Proceedings*, vol. 1702. AIP Publishing, 2015.

Dominic Masters, Josef Dean, Kerstin Klaser, Zhiyi Li, Sam Maddrell-Mander, Adam Sanders, Hatem Helal, Deniz Beker, Ladislav Rampášek, and Dominique Beaini. GPS++: An optimised hybrid MPNN/Transformer for molecular property prediction. *arXiv preprint arXiv:2212.02229*, 2022.

Maho Nakata and Tomomi Shimazaki. PubChemQC project: a large-scale first-principles electronic structure database for data-driven chemistry. *Journal of Chemical Information and Modeling*, 57(6):1300–1308, 2017.

Tymofii Yu Nikolaienko, Valerii S Chuiko, and Leonid A Bulavin. The dataset of covalent bond lengths resulting from the first-principle calculations. *Computational and Theoretical Chemistry*, 1163:112508, 2019.

Liang-bo Ning, Shijie Wang, Wenqi Fan, Qing Li, Xin Xu, Hao Chen, and Feiran Huang. CheatAgent: Attacking LLM-empowered recommender systems via LLM agent. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 2284–2295, 2024.

Liangbo Ning, Ziran Liang, Zhuohang Jiang, Haohao Qu, Yujuan Ding, Wenqi Fan, Xiao-yong Wei, Shanru Lin, Hui Liu, Philip S Yu, et al. A survey of WebAgents: Towards next-generation AI agents for web automation with large foundation models. In *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2*, pp. 6140–6150, 2025.

Wonpyo Park, Woonggi Chang, Donggeon Lee, Juntae Kim, and Seung-won Hwang. GRPE: Relative positional encoding for graph transformer. *arXiv preprint arXiv:2201.12787*, 2022.

Haohao Qu, Wenqi Fan, Zihuai Zhao, and Qing Li. TokenRec: Learning to tokenize ID for LLM-based generative recommendations. *IEEE Transactions on Knowledge and Data Engineering*, 2025.

Haohao Qu, Shanru Lin, Yujuan Ding, Yiqi Wang, and Wenqi Fan. Diffusion generative recommendation with continuous tokens. 2026.

Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific Data*, 1(1):1–7, 2014.

Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. Recipe for a general, powerful, scalable graph transformer. *Advances in Neural Information Processing Systems*, 35:14501–14515, 2022.

Bharath Ramsundar, Peter Eastman, Pat Walters, and Vijay Pande. Deep learning for the life sciences: applying deep learning to genomics, microscopy, drug discovery, and more. *O'Reilly Media*, 2019.

Víctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) equivariant graph neural networks. In *International Conference on Machine Learning*, pp. 9323–9332. PMLR, 2021.

Bernhard Scholkopf, Kah-Kay Sung, Christopher JC Burges, Federico Girosi, Partha Niyogi, Tomaso Poggio, and Vladimir Vapnik. Comparing support vector machines with Gaussian kernels to radial basis function classifiers. *IEEE Transactions on Signal Processing*, 45(11):2758–2765, 1997.

Kristof Schütt, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Alexandre Tkatchenko, and Klaus-Robert Müller. SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *Advances in Neural Information Processing Systems*, 30, 2017.

Kristof Schütt, Oliver Unke, and Michael Gastegger. Equivariant message passing for the prediction of tensorial properties and molecular spectra. In *International Conference on Machine Learning*, pp. 9377–9388. PMLR, 2021.

Yu Shi, Shuxin Zheng, Guolin Ke, Yifei Shen, Jiacheng You, Jiyan He, Shengjie Luo, Chang Liu, Di He, and Tie-Yan Liu. Benchmarking Graphormer on large-scale molecular modeling datasets. *arXiv preprint arXiv:2203.04810*, 2022.

Anuroop Sriram, Abhishek Das, Brandon M Wood, Siddharth Goyal, and C Lawrence Zitnick. Towards training billion parameter graph neural networks for atomic simulations. *arXiv preprint arXiv:2203.09697*, 2022.

Hannes Stärk, Dominique Beaini, Gabriele Corso, Prudencio Tossou, Christian Dallago, Stephan Günnemann, and Pietro Liò. 3D Infomax improves GNNs for molecular property prediction. In *International Conference on Machine Learning*, pp. 20479–20502. PMLR, 2022.

Fan-Yun Sun, Jordan Hoffmann, Vikas Verma, and Jian Tang. InfoGraph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization. *arXiv preprint arXiv:1908.01000*, 2019.

Philipp Thölke and Gianni De Fabritiis. TorchMD-Net: equivariant transformers for neural network based molecular potentials. *arXiv preprint arXiv:2202.02541*, 2022.

Bowen Wang, Chen Liang, Jiaze Wang, Furui Liu, Shaogang Hao, Dong Li, Jianye Hao, Guangyong Chen, Xiaolong Zou, and Pheng-Ann Heng. DR-label: Improving GNN models for catalysis systems by label deconstruction and reconstruction. *arXiv preprint arXiv:2303.02875*, 2023a.

Peng-Lai Wang, Stefan Borsley, Martin J Power, Alessandro Cavasso, Nicolas Giuseppone, and David A Leigh. Transducing chemical energy through catalysis by an artificial molecular motor. *Nature*, 637(8046):594–600, 2025a.

Shijie Wang, Wenqi Fan, Yue Feng, Lin Shanru, Xinyu Ma, Shuaiqiang Wang, and Dawei Yin. Knowledge graph retrieval-augmented generation for LLM-based recommendation. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 27152–27168, 2025b.

Xiaoyang Wang, Yao Ma, Yiqi Wang, Wei Jin, Xin Wang, Jiliang Tang, Caiyan Jia, and Jian Yu. Traffic flow prediction via spatial temporal graph neural network. In *Proceedings of the Web Conference 2020*, pp. 1082–1092, 2020.

Xu Wang, Huan Zhao, Wei-wei Tu, and Quanming Yao. Automated 3D pre-training for molecular property prediction. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 2419–2430, 2023b.

Yue Weng, Bo Ding, Yunqing Liu, Chunlan Song, Lo-Ying Chan, and Chien-Wei Chiang. Late-stage photoredox C–H amidation of N-unprotected indole derivatives: Access to N-(indol-2-yl) amides. *Organic Letters*, 23(7):2710–2714, 2021.

Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9(2):513–530, 2018.

Jun Xia, Chengshuai Zhao, Bozhen Hu, Zhangyang Gao, Cheng Tan, Yue Liu, Siyuan Li, and Stan Z Li. Mole-BERT: Rethinking pre-training graph neural networks for molecules. 2023.

Gelei Xu, Xueyang Li, Yixiong Chen, Yuying Duan, Shuqing Wu, Alexander Yu, Ching-Hao Chiu, Juntong Ni, Ningzhi Tang, Toby Jia-Jun Li, et al. A comprehensive survey of agentic AI in healthcare. *Authorea Preprints*, 2025.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? *arXiv preprint arXiv:1810.00826*, 2018.

Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. Do transformers really perform badly for graph representation? *Advances in Neural Information Processing Systems*, 34:28877–28888, 2021.

Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen. Graph contrastive learning with augmentations. *Advances in Neural Information Processing Systems*, 33:5812–5823, 2020.

Jiahao Zhang, Rui Xue, Wenqi Fan, Xin Xu, Qing Li, Jian Pei, and Xiaorui Liu. Linear-time graph neural networks for scalable recommendations. In *Proceedings of the ACM on Web Conference 2024*, pp. 3533–3544, 2024.

Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen Wen, Fei Wang, Xiangyu Zhao, Jiliang Tang, and Qing Li. Recommender systems in the era of large language models (LLMs). *IEEE Transactions on Knowledge and Data Engineering*, 36(11):6889–6907, 2024.

Gengmo Zhou, Zhifeng Gao, Qiankun Ding, Hang Zheng, Hongteng Xu, Zhewei Wei, Linfeng Zhang, and Guolin Ke. Uni-Mol: A universal 3D molecular representation learning framework. 2023.

Yi Zhou, Haohao Qu, Yunqing Liu, Shanru Lin, Le Song, and Wenqi Fan. HD-Prot: A protein language model for joint sequence-structure modeling with continuous structure tokens. *arXiv preprint arXiv:2512.15133*, 2025.

---

# 附录

## 附录 A：命题论证

### A.1 命题 1 的论证

**命题 1（边邻接模式带来的信息增益）.** 对于非平凡的图 $G$，$L(G)$ 的熵满足：
$H(L(G)) = H(G) + \underbrace{H(E'|E)}_{\Delta I_{\text{edge-structure}}}$，且
$\Delta I_{\text{edge-structure}} > 0$。

**论证：** 在信息论中，图 $G$ 的信息含量可以通过其结构熵来量化。对于以原子为中
心的图 $G = (V, E)$，其熵分解为：

$$H(G) = H(V) + H(E \mid V), \tag{11}$$

其中 $H(V)$ 是节点分布的熵（例如度分布的熵），$H(E|V)$ 是在给定节点分布下边连
接的条件熵，反映边放置的随机性。以键为中心的图 $L(G)$ 把 $G$ 的边 $E$ 提升为节
点，并编码边之间的邻接关系。其熵为：

$$H(L(G)) = H(E) + H(E' \mid E), \tag{12}$$

其中 $H(E)$ 是原边分布的熵（例如边权重的多样性），$H(E'|E)$ 是 $L(G)$ 中边邻接
模式的条件熵，反映边-边连接的复杂性。

衡量 $G$ 与 $L(G)$ 之间共享信息的互信息 $I(G; L(G))$ 定义为：

$$I(G; L(G)) = H(G) + H(L(G)) - H(G; L(G)). \tag{13}$$

由于 $L(G)$ 完全由 $G$ 构造，$H(G; L(G)) = H(L(G))$，从而：

$$I(G; L(G)) = H(L(G)). \tag{14}$$

这表明 $L(G)$ 的熵完全源自 $G$。然而，$L(G)$ 的独特信息在于其条件熵 $H(E'|E)$，
它捕捉了 $G$ 中未显式编码的边邻接模式。因此，对于非平凡的图 $G$，$L(G)$ 的熵满
足 $H(L(G)) = H(G) + \Delta I_{\text{edge-structure}}$ 且
$\Delta I_{\text{edge-structure}} > 0$。$H(E'|E)$ 量化了 $G$ 中边邻接模式的随机
性，引入了 $G$ 中不存在的高阶结构信息，这保证了
$\Delta I_{\text{edge-structure}} > 0$。

### A.2 命题 2 的论证

**命题 2（互信息分解）.** 对于任意分子图 $G$ 及其双图表示 $(H^{(a)}, H^{(b)})$，
互信息满足：$I(G, L(G); H^{(a)}, H^{(b)}) = \underbrace{I(G; H^{(a)})}_{\text{原子信息}} + \underbrace{I(L(G); H^{(b)}|H^{(a)})}_{\text{键中心图信息}} + \underbrace{I(G; H^{(b)}|L(G), H^{(a)})}_{\text{残余原子依赖}}$。

**论证：**

**引理 1（互信息链式法则）**（Cover et al., 1991）：

$$
\begin{aligned}
I(X, Z; Y) &= I(X; Y) + I(Z; Y \mid X) \\
&= \mathbb{E}_{p(x,z,y)}\left[\log \frac{p(z, y, x)}{p(y)p(z, x)}\right] \\
&= \mathbb{E}_{p(x,z,y)}\left[\log \frac{p(x, y)\,p(z, y, x)}{p(y)\,p(z, x)\,p(y, x)}\right] \\
&= \mathbb{E}_{p(x,z,y)}\left[\log \frac{p(x, y)}{p(x)p(y)} \cdot \frac{p(z, y|x)}{p(z|x)p(y|x)}\right] \\
&= \mathbb{E}_{p(x,y)}\left[\log \frac{p(x, y)}{p(x)p(y)}\right] + \mathbb{E}_{p(x,z,y)}\left[\log \frac{p(z, y|x)}{p(z|x)p(y|x)}\right] \\
&= I(X; Y) + I(Z; Y \mid X)
\end{aligned}
\tag{15}
$$

从总互信息 $I(G, L(G); H^{(a)}, H^{(b)})$ 出发，应用互信息链式法则：

$$I(G, L(G); H^{(a)}, H^{(b)}) = I(G, L(G); H^{(a)}) + I(G, L(G); H^{(b)} \mid H^{(a)}).$$

由于 $L(G)$ 是 $G$ 的确定性函数，已知 $G$ 不会提供关于 $L(G)$ 的额外信息，因此
$I(L(G); H^{(a)}|G) = 0$。由此化简第一项：

$$I(G, L(G); H^{(a)}) = I(G; H^{(a)}) + I(L(G); H^{(a)}|G) = I(G; H^{(a)}).$$

第二项 $I(G, L(G); H^{(b)}|H^{(a)})$ 可展开为：

$$I(G, L(G); H^{(b)}|H^{(a)}) = I(L(G); H^{(b)}|H^{(a)}) + I(G; H^{(b)}|L(G), H^{(a)}).$$

合并即得完整分解：

$$I(G, L(G); H^{(a)}, H^{(b)}) = I(G; H^{(a)}) + I(L(G); H^{(b)}|H^{(a)}) + I(G; H^{(b)}|L(G), H^{(a)}).$$

该分解表明，当 $I(L(G); H^{(b)}|H^{(a)}) > 0$ 且 $I(G; H^{(b)}|L(G), H^{(a)}) > 0$
时，双图学习保留的信息严格多于单图方法。

### A.3 命题 3 的论证

**命题 3（几何信息增益）.** 记键角为 $\theta_{ijk}$、二面角为 $\varphi_{ijkl}$。
若 $L(G)$ 通过 $E'$ 编码角度关系，则键建模增益满足：

$$
\Delta I = I(L(G); H^{(b)} \mid G; H^{(a)}) \propto
\mathbb{E}_{i,j,k,l}\left[\log \frac{p(\theta_{ijk}, \varphi_{ijkl} \mid h_i^{(a)}, h_j^{(a)}, h_k^{(a)}, h_l^{(a)})}{p(\theta_{ijk}, \varphi_{ijkl})}\right].
\tag{16}
$$

**论证：** 该命题为"引入以键为中心的图 $L(G)$ 能增强所学分子表示的几何丰富性"
提供了理论依据。论证核心在于把 $L(G)$ 的结构性质与其编码高阶几何信息的能力联系
起来，并用信息论框架加以量化。

**1. 定义几何信息增益（ΔI）：** 所关注的项 $\Delta I = I(L(G); H^{(b)}|G; H^{(a)})$
是一个条件互信息。它量化了在已知以原子为中心的图 $G$ 及其对应原子表示 $H^{(a)}$
的条件下，以键为中心的图 $L(G)$ 为键表示 $H^{(b)}$ 提供的额外信息量。在我们的框
架中，该项代表可归因于键中心视角的边际信息增益，我们认为其本质主要是几何性的。

**2. $L(G)$ 在几何编码上的结构适配性：** 以原子为中心的图 $G$ 显式建模原子连接
性（0 跳与 1 跳关系）。相比之下，以键为中心的图 $L(G)$ 把键提升为节点，从而使键
之间的关系变得显式：

* 键角（$\theta_{ijk}$）本质上是由共享一个公共原子的两条相邻键定义的性质，在
  $L(G)$ 中对应两个节点之间的直接边；
* 二面角（$\varphi_{ijkl}$）描述三条连续键之间的关系，在 $L(G)$ 中对应一条 2 跳
  路径。

因此，$L(G)$ 的拓扑结构天然比 $G$ 更适合显式表示这些高阶几何特征——在 $G$ 中
这些信息只能是隐式的，必须经由推断获得。

**3. 通过预测能力量化信息增益：** 表示学习的一个核心原则是：高质量嵌入应当捕捉
输入数据的显著性质。对分子而言，3D 几何是根本性质。因此，学到的表示
$(H^{(a)}, H^{(b)})$ 的质量可以通过其重构或预测这些几何性质的能力来衡量。命题右
端的表达式是在数据分布上取平均的 Kullback-Leibler（KL）散度
$D_{KL}(p(\theta, \varphi|H) \| p(\theta, \varphi))$。该 KL 散度衡量在观测到学到的
表示 $H$ 之后，关于几何变量 $(\theta, \varphi)$ 不确定性的减少程度。

$L(G)$ 的结构特性使模型能够学到几何信息更丰富的键表示 $H^{(b)}$。这些表示捕捉几
何信息的程度，直接由它们预测真实几何参数 $(\theta_{ijk}, \varphi_{ijkl})$ 的能力来
衡量。信息量更大的表示会使后验分布 $p(\theta, \varphi|H)$ 在真实值附近更尖锐，从
而与无信息先验 $p(\theta, \varphi)$ 之间产生更大的 KL 散度。因此，来自键中心通道
的几何信息增益 $\Delta I$ 与模型预测这些几何量的增强能力成正比。这证明了我们在
框架中把扭转角等几何特征专门整合进键中心通道这一架构选择的合理性。

### A.4 命题 4 的论证

**命题 4（双图信息瓶颈）.** 对于分子性质预测 $Y$，最优编码器 $\Phi$ 最小化：

$$\min_{\Phi}\left[ I(G, L(G); H^{(a)}, H^{(b)}) - \beta\, I(H^{(a)}, H^{(b)}; Y) \right]. \tag{17}$$

**论证：** 该命题把分子表示学习任务纳入信息瓶颈（IB）原理的框架，目标是证明：
相比单一的原子中心图，双图输入为求解 IB 优化问题提供了更有利的起点。

**1. 信息瓶颈原理：** IB 原理指出，为预测目标变量 $Y$，输入变量 $X$ 的最优表示
（编码）应满足两个相互竞争的目标：

* **压缩**：尽可能压缩输入 $X$，丢弃非本质信息——通过最小化输入 $X$ 与表示 $Z$
  之间的互信息 $I(X; Z)$ 实现；
* **预测**：尽可能保留关于目标 $Y$ 的信息——通过最大化互信息 $I(Z; Y)$ 实现。

拉格朗日乘子 $\beta$ 控制这两个目标之间的权衡。

**2. 把 IB 原理映射到 DeMol 框架：** 在我们的语境中，各变量映射如下：

* 输入 $X$ 对应分子的完整双图描述：$(G, L(G))$；
* 学到的表示 $Z$ 对应原子与键嵌入的集合：$(H^{(a)}, H^{(b)})$；
* 目标变量 $Y$ 是待预测的分子性质。

因此，命题 4 的目标函数是 IB 原理在我们框架上的直接应用，其中编码器 $\Phi$ 学习
从双图输入到潜在嵌入的映射。

**3. 双图输入的优势：** 在 IB 框架下，双图方法优越性的核心论点在于它为编码器提
供了更丰富、更结构化的输入：

* **更丰富的信息含量**：如命题 1 所确立，双图输入 $(G, L(G))$ 包含的信息严格多
  于单独的原子中心图 $G$，特别是在显式的几何与键关系信息方面；
* **改善压缩-预测权衡**：从信息量更大、更结构化的输入出发，编码器 $\Phi$ 更有条
  件找到 IB 权衡的更优解。模型能更有效地分辨分子结构的哪些方面对目标性质 $Y$ 有
  预测力（例如对生物活性至关重要的特定扭转角），哪些只是结构上的伪影。只在 $G$
  上操作的编码器可能被迫保留更多含糊的结构信息，因为关键几何线索只能被隐式表
  示；双图输入让编码器更有选择性，有望得到既更具压缩性（丢弃冗余原子信息）又更
  具预测性（聚焦真正重要的显式键级特征）的表示，从而在优化过程中达成更好的权衡
  （更紧的界）。

**4. 与架构选择的联系：** 这一理论动机直接指导了模型设计。IB 原理不仅为端到端
学习目标提供了依据，也说明了为实现有效压缩而引入正则化的必要性。结构感知掩码
（Structure-aware Mask）与基于共价半径的键预测模块可以被解释为辅助压缩目标的归
纳偏置：通过强制化学与物理合理性，这些组件引导编码器忽略潜在表示空间中对应不真
实分子构象的大片区域，从而帮助它找到更紧凑、更有意义的表示。

综上，命题 4 通过把双图框架置于成熟的信息瓶颈原理之中论证了其合理性：双图表示
提供的更丰富输入使压缩-预测权衡的优化更高效、更有效，最终得到既简洁又对分子性
质高度可预测的嵌入。

---

## 附录 B：基于共价半径的键预测算法

> **算法 1：基于共价半径的键预测**
>
> **输入**：分子 $\mathcal{M}$ 的 $N$ 个原子列表 $A = \{a_1, a_2, \ldots, a_N\}$；
> 每个原子的类型（如 'H'、'C'、'O'）$T$；3D 坐标 $\vec{p} = (x, y, z)$；
> 共价半径字典 $R_{cov}$；成键阈值因子 $\alpha = 1.15$。
>
> **输出**：预测得到的化学键集合 $B$。
>
> 1: $B \leftarrow \emptyset$  ▷ 初始化空键集
> 2: **for** $i \leftarrow 1$ **to** $N-1$ **do**
> 3:   $T_i \leftarrow \text{type}(a_i)$  ▷ 取原子类型
> 4:   $r_i \leftarrow R_{cov}[T_i]$  ▷ 查共价半径
> 5:   $\vec{p}_i \leftarrow (x_i, y_i, z_i)$  ▷ 取坐标
> 6:   **for** $j \leftarrow i+1$ **to** $N$ **do**
> 7:    $T_j \leftarrow \text{type}(a_j)$  ▷ 取原子类型
> 8:    $r_j \leftarrow R_{cov}[T_j]$  ▷ 查共价半径
> 9:    $\vec{p}_j \leftarrow (x_j, y_j, z_j)$  ▷ 取坐标
> 10:    $R_{sum} \leftarrow r_i + r_j$  ▷ 共价半径之和
> 11:    $D_{threshold} \leftarrow \alpha \times R_{sum}$  ▷ 成键距离阈值
> 12:    $D_{ij} \leftarrow \sqrt{(x_i-x_j)^2 + (y_i-y_j)^2 + (z_i-z_j)^2}$  ▷ 欧氏距离
> 13:    **if** $D_{ij} \le D_{threshold}$ **then**
> 14:     $B \leftarrow B \cup \{(i, j)\}$  ▷ 将该原子对加入键集
> 15:    **end if**
> 16:   **end for**
> 17: **end for**
> 18: **return** $B$  ▷ 返回预测的键集

---

## 附录 C：时间复杂度分析

本节分析 DeMol 框架的时间复杂度，帮助理解双图架构的计算效率，并说明设计选择如
何影响可扩展性。

DeMol 框架运行在双图表示上，包括：

1. 以原子为中心的图 $G = (V, E)$，节点表示原子，边表示化学键；
2. 由 $G$ 派生的以键为中心的图 $L(G)$，节点表示化学键，边编码键之间的空间或拓
   扑关系。

记分子中原子数为 $|V| = N$，键数为 $|E| = M$。原子与键的特征维度分别记为 $d$ 与
$d_b$。假设原子中心通道与键中心通道都使用 $h$ 头、$L$ 层的多头注意力。

**原子中心图。** 原子中心图的每层处理涉及图注意力机制。对于稀疏分子图，边数
$M \sim O(N)$，因此每层时间复杂度为 $O(h \cdot M \cdot d)$，$L$ 层总计
$O(L \cdot h \cdot N \cdot d)$。

**键中心图。** 键中心图有 $M$ 个节点（每条键一个），其边集 $E'$ 编码键之间的邻
接。最坏情况下 $|E'| \sim O(|E|^2)$，特别是在考虑键之间所有可能的空间相互作用时。
因此键中心通道每层复杂度为
$O(h \cdot |E'| \cdot d_b) = O(h \cdot |E|^2 \cdot d_b) = O(h \cdot M^2 \cdot d_b)$，
$L$ 层总计 $O(L \cdot h \cdot M^2 \cdot d_b)$。

**跨层次注意力。** 为对齐两图表示，DeMol 在原子与键之间采用跨层次注意力，引入
额外的成对注意力计算。假设原子与键之间完全交互，复杂度为 $O(h \cdot N \cdot M
\cdot d)$。由于 $M$ 的二次增长与乘积 $N \cdot M$，该项在大分子中主导整体复杂度。

综合以上各组件，DeMol 的总时间复杂度为
$O\big(Lh(Nd + M^2 d_b + NMd)\big)$。

**使用结构感知掩码后。** 为缓解高计算开销，我们引入结构感知掩码，基于化学价规
则与共价半径约束，把注意力计算限制在物理上合理的原子与键相互作用上。更新后的复
杂度组成：

1. **原子中心图**：施加掩码后，边数仍为 $N$ 的线性量级，复杂度保持
   $O(LhNd)$；
2. **键中心图**：由于掩码，键中心图中的边数从 $O(|E|^2)$ 降为 $O(|E|)$，复杂度
   为 $O(Lh|E|d_b)$；
3. **跨层次注意力**：有效的原子-键相互作用数正比于 $|E|$ 而非 $N \cdot |E|$，复
   杂度为 $O(LhKd)$，其中 $K \ll N \cdot |E|$ 为掩码后保留的相互作用数。

**总复杂度。** 在结构感知掩码策略下，DeMol 的总时间复杂度变为
$O\big(Lh(Nd + |E|d_b)\big) = O\big(Lh(Nd + Md_b)\big)$。事实上，大多数分子图满足
边数约为原子数 1.1~1.2 倍的条件，即 $M \approx 1.2N$，于是总复杂度可写为
$O\big(Lh(Nd + (d + d_b)N^2)\big)$。
〔译注：原文此处的公式推导排版有若干笔误（括号不配对、$M^2$ 与 $M \approx 1.2N$
并存），我们按语义整理如上，供参考。〕

### C.1 推理时间分析

我们进一步给出在 NVIDIA A6000 GPU 上、PCQM4Mv2 数据集上的推理时间对比。

**表 5：PCQM4Mv2 数据集上的推理时间对比。**

| 方法 | 推理时间 |
| --- | --- |
| Transformer-M | ~28 ms/分子 |
| GPS++（集成） | ~100 ms/分子 |
| DeMol | ~34 ms/分子 |

如表 5 所示，由于额外的键通道计算，DeMol 比 Transformer-M 约慢 20%，但带来了
显著的性能增益；同时它远快于基于集成的方法（如 GPS++），且以单模型取得了更优
性能。

---

## 附录 D：实验细节

### D.1 大规模预训练

**数据集。** DeMol 在 OGB 大规模挑战赛（Hu et al., 2021）的 PCQM4Mv2 训练集上
预训练。PCQM4Mv2 是最初由 PubChemQC 项目（Maho, 2015; Nakata & Shimazaki, 2017）
整理的量子化学数据集，训练样本总数为 337 万。训练集中每个分子都同时具有 2D 图结
构与 3D 几何结构，并提供由基于 DFT 的几何优化得到的 HOMO-LUMO 能隙（Burke,
2012）。根据 OGB-LSC（Hu et al., 2021），HOMO-LUMO 能隙是分子最具实际意义的量
子化学性质之一，因为它与反应活性、光激发和电荷传输相关。作为公开可用的最大分子
性质预测数据集，PCQM4Mv2 被认为是对分子模型极具挑战性的基准。

**设置。** 与 Graphormer（Ying et al., 2021）、Transformer-M（Luo et al., 2022）
和 Uni-mol（Zhou et al., 2023）类似，DeMol 包含 12 层，原子表示维度 $d_a = 768$，
键表示维度 $d_b = 768$，采用 128 个高斯核。优化器为 AdamW（Diederik, 2014），超参
数 $\epsilon = 10^{-8}$，$(\beta_1, \beta_2) = (0.9, 0.999)$。梯度裁剪范数为 5.0。
峰值学习率为 $2\times10^{-4}$，批大小为 1024。模型训练 150 万步，其中 15 万步
warmup。还使用了衰减率 0.999 的指数移动平均（EMA）。训练过程在 8 张 NVIDIA
A6000 GPU 上进行了约 7 天。

**预训练策略。** 如第 3 节图 3 所示，DeMol 采用多任务预训练策略。我们遵循
Transformer-M（Luo et al., 2022）与 Unimol（Zhou et al., 2023）的设定：主目标是
对量子化学性质（HOMO-LUMO 能隙）的有监督回归，同时利用若干辅助任务来强化几何
与结构理解。

**性质预测损失（L_prop）。** 这是主要的有监督任务。对 PCQM4Mv2 数据集，我们最小
化预测的 HOMO-LUMO 能隙 $\hat{y}$ 与真值 $y$ 之间的平均绝对误差（MAE）：

$$L_{prop} = \frac{1}{|B|} \sum_{i \in B} |y_i - \hat{y}_i|,$$

其中 $B$ 为一个分子批次。

**掩码原子预测损失（L_mask）。** 为捕捉拓扑上下文，我们随机掩码一部分原子节点特
征，掩码原子预测头以交叉熵损失预测原子类型 $t_i$：

$$L_{mask} = -\sum_{j \in \mathcal{M}} \log P(t_j \mid G_{masked}),$$

其中 $\mathcal{M}$ 为被掩码的索引集合。

**坐标恢复损失（L_coord）。** 为强化 3D 空间感知，我们向输入坐标添加噪声，坐标
恢复头学习去噪、恢复真值坐标 $\vec{p}_i$，通常以均方误差（MSE）最小化：

$$L_{coord} = \sum_{i=1}^{N} \|\vec{p}_i - \hat{\vec{p}}_i\|^2.$$

**基于共价半径的键预测损失（L_bond）。** 如算法 1 所述，我们基于共价半径约束
（$D_{ij} < \alpha(r_i + r_j)$）确定真值化学键 $B_{gt}$。键预测头学习对有效相互作
用进行分类，作为惩罚几何不一致结构的正则化项：

$$L_{bond} = -\sum_{(i,j) \in E_{all}} \Big[ \mathbb{I}_{(i,j) \in B_{gt}} \log(p_{ij}) + (1 - \mathbb{I}_{(i,j) \in B_{gt}}) \log(1 - p_{ij}) \Big],$$

其中 $p_{ij}$ 为原子 $i$ 与 $j$ 之间存在键的预测概率。

### D.2 PCQM4Mv2

**基线。** 我们将 DeMol 与若干有竞争力的基线比较，分为两类：消息传递神经网络
（MPNN）变体与图 Transformer。

MPNN 变体包括两种广泛使用的模型 GCN（Jiang et al., 2019）和 GIN（Xu et al.,
2018）及它们的虚拟节点（VN）版本（Gilmer et al., 2017; Jiang et al., 2019），此
外还比较了 GINE-VN（Brossard et al., 2020; Gilmer et al., 2017; Luo et al., 2022）
与 DeeperGCN-VN（Li et al., 2020; Luo et al., 2022）。GINE 是 GIN 的多跳版本；
DeeperGCN 是精心设计了聚合器的 12 层 GNN。MLP-Fingerprint（Hu et al., 2021）的
结果也一并报告。

图 Transformer 方面：Graphormer（Ying et al., 2021）把图结构编码整合进标准
Transformer；TokenGT（Kim et al., 2022）采用标准 Transformer 架构而不做图特定改
造；GraphGPS（Rampášek et al., 2022）提出将位置/结构编码、局部消息传递与全局注意
力整合进 Transformer 的框架；GPS++（Masters et al., 2022）优化了混合
MPNN/Transformer；GRPE（Park et al., 2022）提出图特定的相对位置编码，同时考虑
节点-空间与节点-边关系；EGT（Hussain et al., 2022）完全用全局自注意力替代静态局
部卷积聚合，并利用边通道捕捉结构信息；GEM-2（Liu et al., 2022a）建模分子中的全
程多体相互作用，借助轴向注意力机制高效捕捉复杂的量子相互作用；Transformer-M
（Luo et al., 2022）把 2D 与 3D 空间编码整合为注意力偏置以增强分子表示；
Unimol+（Lu et al., 2023）提出分子构象更新机制以精确预测量子化学性质；TGT
（Hussain et al., 2024）整合三元组相互作用以改进图 Transformer。

### D.3 Open Catalyst 2020 IS2RE

Open Catalyst 2020 挑战赛（Chanussot et al., 2021）旨在用机器学习预测分子在催化剂
表面的吸附能。我们聚焦 IS2RE（Initial Structure to Relaxed Energy）任务：模型获得
晶体与吸附质的初始 DFT 结构，二者相互作用达到弛豫结构，此时测量体系的弛豫能量。
该任务约含 46 万训练数据点。训练时提供 DFT 平衡构象，但推理时不允许使用。

**设置。** OC20 实验使用默认的 12 层设置。首先，由于 OC20 缺少图信息，模型中排除
了图相关特征。我们采用（Shi et al., 2022）提出的方案处理周期性边界条件：预先扩
展邻近晶胞，然后施加半径截断以减少原子数。训练使用 AdamW 优化器，共 150 万步，
含 15 万步 warmup；学习率 $2\times10^{-4}$，批大小 64，$(\beta_1, \beta_2) =
(0.9, 0.999)$，梯度裁剪 5.0。训练在 8 张 NVIDIA A6000 GPU 上进行了约 14 天。
〔译注：原文此处为 "8 NVIDIA A600 GPUs"，应为 A6000 的笔误。〕

**基线。** 基线分为两类：基于 3D 的模型与基于图的模型。SchNet（Schütt et al.,
2017）提出开创性的 3D 卷积神经网络，用连续滤波器卷积基于原子间距离建模原子相互
作用；DimeNet++（Sriram et al., 2022）引入定向消息传递以更好捕捉原子间的角度依
赖；GemNet-T（Gasteiger et al., 2021）是能同时处理分子几何与电子性质的图神经网
络，关注平移等变性；SphereNet（Liu et al., 2022b）用球谐函数表示分子，可高效计算
旋转等变特征。基于图的模型方面：Graphormer-3D（Shi et al., 2022）把 Transformer
架构与 3D 图表示结合以捕捉分子结构中的长程依赖；GNS（Godwin et al., 2021）使用
生成式神经模拟器，通过建模原子间相互作用来学习预测分子动力学；DRFormer（Wang
et al., 2023a）捕捉分子图中的关系信息用于药物发现；EquiFormer（Liao & Smidt,
2022）提出在分子数据中保持对称性、并结合神经网络层增强性能的等变 Transformer 架
构；UniMol+（Lu et al., 2023）结合多种模态（如 3D 坐标与图表示）以提升在多样分
子数据集上的泛化能力；TGT（Hussain et al., 2024）整合三元组相互作用改进图
Transformer。

### D.4 QM9

我们使用 QM9 数据集（Ramakrishnan et al., 2014）在 3D 数据格式的分子任务上评估模
型。QM9 是量子化学基准，包含 13.4 万个稳定的小有机分子，对应 GDB-17 化学宇宙
（1660 亿有机分子）中全部 133,885 个物种的一个子集。每个分子带有 12 个目标，涵
盖能量、电子与热力学性质。输入为分子的 3D 几何结构。遵循（Thölke & De Fabritiis,
2022），我们随机选取 10,000 个分子用于验证、10,831 个用于测试，其余分子用于微调
DeMol。我们注意到此前一些工作使用了不同的数据划分比例或未描述评估细节；为公平
比较，我们选取在原始论文中使用相似划分比例的基线。

**设置。** 我们在 QM9 数据集上微调预训练好的 DeMol。遵循 Transformer-M（Luo et
al., 2022），训练时采用均方误差（MSE）损失、评估时使用平均绝对误差（MAE），并
采用标签标准化以稳定训练。优化器为 AdamW，超参数 $\epsilon = 10^{-8}$，
$(\beta_1, \beta_2) = (0.9, 0.999)$。梯度裁剪范数 5.0。峰值学习率 $7\times10^{-5}$，
批大小 128。输入嵌入、注意力矩阵、隐藏表示的 dropout 比率分别为 0.0、0.1、0.0。
权重衰减为 0.0。模型微调 60 万步，含 6 万步 warmup；warmup 结束后学习率线性衰减
到零。所有模型均在 8 张 NVIDIA A6000 GPU 上训练。

**基线。** 我们同时与预训练方法和 3D 分子模型做全面比较。首先遵循（Luo et al.,
2022）比较若干预训练方法：AttrMask（Hu et al., 2019）提出通过节点级与图级任务预
训练 GNN 的策略；InfoGraph（Sun et al., 2019）以最大化图级表示与子结构表示之间
的互信息作为预训练任务；GraphCL（You et al., 2020）用对比学习预训练 GNN。也有工
作在预训练中利用 3D 几何结构：GraphMVP（Jiang et al., 2019）最大化 2D 与 3D 表示
之间的互信息；GEM（Fang et al., 2022）同时利用局部与全局 3D 结构学习空间信息；
3D Infomax（Stärk et al., 2022）用两个编码器分别捕捉 2D 与 3D 结构信息并最大化
互信息；PosPred（Jiao et al., 2023）采用等变能量模型并设计了用于力预测的节点级
预训练损失。我们还遵循（Thölke & De Fabritiis, 2022）比较 3D 分子模型：SchNet
（Schütt et al., 2017）用连续滤波器卷积层建模分子中的量子相互作用；Cormorant
（Anderson et al., 2019）提出激活函数对旋转协变的 GNN；DimeNet++（Gasteiger et
al., 2020）提出利用原子对嵌入与原子间方向信息的方向性消息传递；PaiNN（Schütt et
al., 2021）提出使用等变消息传递机制的可极化原子相互作用神经网络；LieTF
（Hutchinson et al., 2021）基于对任意李群及其他离散子群等变的注意力层构建
Transformer；TorchMD-Net（Thölke & De Fabritiis, 2022）同样开发了由物理化学先验
知识设计层的 Transformer 变体；EGNN（Satorras et al., 2021）提出的模型无需在中间
层使用计算昂贵的高阶表示即可保持等变性，且易于扩展到更高维空间；NoisyNode
（Godwin et al., 2021）提出 3D 位置去噪任务并在 Graph Network-based Simulator
（GPS）模型（Masters et al., 2022）上验证；Transformer-M（Luo et al., 2022）把 2D
与 3D 空间编码整合为注意力偏置；ALIGNN（Choudhary & DeCost, 2021）使用线图作为
消息传递的路径；ESA（Buterez et al., 2025）提出把图当作边集合以显式学习边表示的
端到端注意力架构。

### D.5 MoleculeNet

MoleculeNet（Wu et al., 2018）是流行的分子性质预测基准，包含聚焦不同分子性质的
数据集，涵盖量子力学、物理化学到生物物理与生理。我们使用的数据集细节如下：

**BBBP.** BBBP 数据集用于预测分子能否穿过血脑屏障（BBB）——药物递送与疗效的关
键因素。数据集包含 2,039 个分子，每个分子根据实验数据被标记为可透过或不可透过。
任务为二分类，要求模型识别影响 BBB 通透性的结构特征。该数据集被广泛用于评估机器
学习模型捕捉影响 BBB 穿透的细微分子性质的能力，是药物发现与药代动力学相关任务
的基准。

**Tox21.** Tox21 数据集聚焦化合物的毒性预测，包含 8,000 多个分子，每个分子关联
12 个不同的毒性终点（如核受体激活、应激反应通路）。数据来自美国环保署（EPA）与
美国国立卫生研究院（NIH）的高通量筛选实验。Tox21 挑战模型准确分类分子潜在毒性
的能力，强调在生化层面理解分子相互作用的重要性；其多样的终点使其成为评估毒性预
测模型的综合资源。

**ToxCast.** ToxCast 是 Tox21 的扩展，提供更广泛的毒性预测：超过 10,000 个分子、
覆盖 600 多种不同的毒性测定，跨越多种生物学通路与机制。数据集旨在评估化学物质对
人类健康与环境的潜在不利影响。鉴于其对生化相互作用的广泛覆盖与多样的分子结构，
ToxCast 特别适用于评估模型在多个毒性终点上的泛化能力。

**SIDER.** 〔译注：原文此处段落标题为 SIDER，但正文实际描述的是 ClinTox 数据集，
系原文笔误，按原文如实翻译。〕ClinTox 数据集用于预测药物的临床试验结果，聚焦毒
性与疗效。它包含 1,478 种药物，每种药物有两个二分类标签：临床试验中"有毒/无毒"，
以及基于临床结果的"有效/无效"。ClinTox 来自公开来源，包括 DrugBank 数据库与临床
试验记录。该数据集对评估模型同时预测安全性与有效性的能力很有价值——这是药物开
发中的关键因素。

**MUV.** MUV 是虚拟筛选任务的基准，包含 150,000 个分子、17 个不同的生物靶点。
每个分子针对特定靶点被标记为活性或非活性，构成二分类问题。MUV 旨在评估模型通过
区分活性与非活性化合物来识别潜在候选药物的性能。由于规模庞大且需要平衡假阳性与
假阴性，该数据集尤其具有挑战性，是分子性质预测模型的标准基准。

**HIV.** HIV 数据集聚焦预测化合物抗 HIV（人类免疫缺陷病毒）的活性，包含 41,127
个分子，按抑制 HIV 复制的能力标记为活性或非活性。数据来自 Developmental
Therapeutics Program（DTP）AIDS Antiviral Screen 的高通量筛选实验。HIV 是二分类任
务，要求模型识别与抗病毒活性相关的结构特征，对抗 HIV 药物发现研究至关重要。

**BACE.** BACE（β-分泌酶切割位点）数据集用于预测对 β-分泌酶的抑制作用——该酶在
阿尔茨海默病中起关键作用。数据集包含 1,513 个分子，按抑制酶的能力标记为活性或非
活性。BACE 是二分类任务，评估模型识别具有治疗神经退行性疾病潜力的分子的能力。数
据来自实验筛选，被广泛用于酶抑制与药物发现场景的模型基准。

**设置。** 参考先前工作 Uni-Mol（Zhou et al., 2023），我们使用骨架划分把数据集按
8:1:1 分为训练、验证与测试集。骨架划分比随机划分更具挑战性，因为不同子集中分子的
骨架集合不相交，这种划分检验模型的泛化能力并反映真实场景（Wu et al., 2018）。所
有实验中，我们选择验证损失最好的检查点，并报告该检查点在测试集上的结果。参考先
前工作，我们用网格搜索为分子性质任务寻找最佳超参数组合；为降低时间开销，对大数
据集设置了更小的搜索空间。具体搜索空间见表 7。小数据集在单张 NVIDIA A6000 GPU
上运行；大数据集与 HIV 在 4 张 NVIDIA A6000 GPU 上运行。

**表 7：超参数搜索空间。小数据集：BBBP、BACE、ClinTox、Tox21、ToxCast、SIDER；
大数据集：MUV 和 HIV。**

| 超参数 | 小数据集 | 大数据集（MUV） | HIV |
| --- | --- | --- | --- |
| 学习率 | [5e-5, 8e-5, 1e-4, 4e-4, 5e-4] | [2e-5, 1e-4] | [2e-5, 5e-5] |
| 批大小 | [32, 64, 128, 256] | [128, 256] | [128, 256] |
| 训练轮数 | [40, 60, 80, 100] | [20, 40] | [2, 5, 10] |
| Pooler dropout | [0.0, 0.1, 0.2, 0.5] | [0.0, 0.1] | [0.0, 0.2] |
| Warmup 比例 | [0.0, 0.06, 0.1] | [0.0, 0.06] | [0.0, 0.1] |

---

## 附录 E：自注意力图的可视化

为提高可解释性，我们分别对原子中心图的原子-原子注意力与键中心图的键-键注意力的
自注意力图做了可视化，如图 5 所示。图 5 上半部分展示原子中心图表示的自注意力权
重，每个子图对应一个注意力头，显示分子中原子两两之间的注意力权重；下半部分展示
键中心图表示的自注意力权重，每个子图同样对应一个注意力头，显示分子中键两两之间
的注意力权重。

> **图 5**：多头自注意力图的逐头可视化。（上：原子中心图的原子-原子注意力权重，
> 8 个头；下：键中心图的键-键注意力权重，8 个头。）

对比两部分可以观察到：原子中心注意力倾向于呈现更多样、更复杂的模式，反映原子相
互作用与官能团的丰富多样性；键中心注意力则呈现更结构化、更局部化的模式，凸显键
特异性特征在捕捉分子性质中的重要性。对角线与非对角线注意力模式的组合表明，模型
能够同时有效利用局部与全局结构信息。这两种表示的互补性说明：结合原子级与键级信
息可以对分子结构形成更全面的理解。

---

## 附录 F：键级相互作用的定性分析

我们进一步设计了一个包含 100 个分子的精选数据集，用以可视化 DeMol、GEM 和
LEMON 的表示空间，检验特定的键级能力。数据集包含四类分子，用于测试两种键级现象：

* **立体化学（3D 几何）**：25 个顺式异构体（如顺铂）与 25 个反式异构体（如反
  铂），二者仅在 3D 键取向（二面角/键角）上不同；
* **共振（2D 电子结构）**：25 个芳香环（如苯）与 25 个非芳香环（如环己二烯），
  二者在键级与离域程度上不同。

我们用 t-SNE 对这些分子的最终图级嵌入做了投影。如图 6 所示，DeMol、GEM 与
LEMON 都能成功分离芳香（绿色）与非芳香（红色）簇，证明它们能较好地捕捉 2D 拓
扑特征。然而，顺式（蓝色）与反式（橙色）异构体在 LEMON 中完全混成一个簇；
GEM 虽然显式建模了键角图，但其基于 GNN 的消息传递似乎把这些特征"平滑"掉了，
导致表示纠缠——3D 几何差异在最终嵌入空间中未被严格保持。DeMol 则产生了四个清
晰分离的簇：既区分顺式与反式异构体（捕捉 3D 几何），又区分芳香与非芳香环（捕捉
键共振）。这直接定性地证明了 DeMol 的独特架构成功学会"解耦"其他键感知模型或缺
失或混淆的复杂键级现象。

> **图 6**：键级相互作用定性分析的可视化（t-SNE 投影）。

---

## 附录 G：局限性与未来工作

**局限性。** 尽管 DeMol 通过双图表示显式建模了原子-键与键-键相互作用，键中心通
道与跨层次注意力块的引入也增加了计算开销。我们通过由化学价规则导出的结构感知掩
码缓解了这一问题，但对于实时应用仍需进一步优化。此外，当前框架主要在小有机分子
与简单无机配合物上得到验证；其对具有非常规成键的体系（如金属-有机框架、有机金
属化合物）或动态共价相互作用（如超分子组装体中的可逆键）的适用性，仍有待进一步
研究。

**未来工作。** 未来可以探索轻量级交叉注意力设计（如通过可学习图稀疏化的稀疏注意
力，或核化近似），在保持几何一致性的同时降低计算成本。此外，把显式的电子描述符
（如部分电荷、轨道杂化）与量子力学约束（如 HOMO-LUMO 能隙）纳入键中心通道，以
及探索把 DeMol 与物理信息神经网络结合的混合模型以弥合经典与量子表示，都是有前
景的方向。

---

## 附录 H：预训练对 PCQM4Mv2 的影响

我们在 QM9 数据集上做了补充实验：不经过 PCQM4Mv2 预训练直接训练 DeMol。选取五
个 QM9 目标做比较，预训练与微调的所有超参数保持一致。

**表 8：预训练（PCQM4Mv2）对 QM9 的影响。**

| 方法 | ε_HOMO | ε_LUMO | Δε | G | C_v |
| --- | --- | --- | --- | --- | --- |
| 无预训练 | 21.2 | 19.3 | 30.4 | 10.97 | 0.029 |
| 在 PCQM4Mv2 上预训练 | **16.4** | **15.7** | **26.8** | **6.29** | **0.021** |

如表 8 所示，在 PCQM4Mv2 数据集上预训练确实能改善下游任务性能，这与
Graphormer（Ying et al., 2021）、Transformer-M（Luo et al., 2022）和 Uni-mol
（Zhou et al., 2023）中观察到的性能增益一致。

---

## 附录 I：两种图表示的示例

图 7 展示了甲烷分子（CH₄）的两种不同图表示，凸显原子中心与键中心图表示的互补
性。该图旨在全面展示两种表示如何编码结构信息，包括原子间距离与键间角度。最左面
板给出化学式（CH₄）及其 2D、3D 结构，作为甲烷分子的视觉参照。

在原子中心图中，原子被表示为节点（v₁, v₂, v₃, v₄, v₅），连接它们的边
（e₁, e₂, e₃, e₄）反映分子连接性。该表示显式编码原子间距离等结构信息。结构编
码包括邻接矩阵、最短路径距离编码、3D 距离编码等。邻接矩阵提供图连接性的数值表
示——例如 "1" 表示两个原子直接相连，"0" 表示无直接连接。2D 图可视化描绘了原子
间的最短路径距离，强调分子内的空间关系（节点之间由长度反映最短路径的边连接）。
最右侧是甲烷的 3D 分子模型，化学键以红色高亮。该可视化整合了几何信息，说明原子
中心图如何捕捉三维空间关系。

键中心图把关注点从原子转向化学键：键被视为节点（e₁, e₂, e₃, e₄），节点之间的
边表示相邻键之间的关系，从而能显式编码键间角度等结构特征。与原子中心图类似，图
中给出了彩色编码矩阵（结构编码）来可视化键之间的邻接关系——该矩阵捕捉键之间的
连接模式，反映其空间排布。键中心图的邻接矩阵以数值编码键之间的连接性，"1" 表示
直接关系，"0" 表示无直接关系。2D 图可视化展示键之间的最短路径距离，强调键中心图
内的拓扑关系。最右侧再次给出甲烷的 3D 分子模型，但这次聚焦键之间的角度：键以红
色高亮，可视化强调键中心图如何捕捉三维空间中的角度关系。

> **图 7**：甲烷分子的两种图表示示例。原子中心图：以原子为节点的图表示，可显式
> 编码原子间距离等结构信息。键中心图：以化学键为节点的图表示，可显式编码键间角
> 度等结构信息。

---

## 附录 J：LLM 使用声明

作者仅将大语言模型（LLM）作为辅助工具用于本文写作的润色与精炼，仅用于改善语法
流畅性、句子结构与稿件整体清晰度。所有构思、理论发展、实证研究与技术结论完全出
自作者。作者对 LLM 生成并呈现在本文中的所有内容承担全部责任。

---

> **MoleculeNet 主结果表（表 6，正文图 4 的表格版）**：骨架划分下 10 个随机种子
> 的 ROC-AUC（均值 ± 标准差，越高越好），加粗为最佳。

| 方法 | BBBP↑ | Tox21↑ | ToxCast↑ | SIDER↑ | ClinTox↑ | MUV↑ | HIV↑ | BACE↑ | 平均↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AttrMask | 65.0 ± 2.3 | 74.8 ± 0.2 | 62.9 ± 0.1 | 61.2 ± 0.1 | 87.7 ± 1.1 | 73.4 ± 2.0 | 76.8 ± 0.5 | 79.7 ± 0.3 | 72.68 |
| ContextPred | 65.7 ± 0.6 | 74.2 ± 0.0 | 62.5 ± 0.3 | 62.2 ± 0.5 | 77.2 ± 0.8 | 75.3 ± 1.5 | 77.1 ± 0.8 | 76.0 ± 2.0 | 71.28 |
| GraphCL | 69.7 ± 0.6 | 73.9 ± 0.6 | 62.4 ± 0.5 | 60.5 ± 0.8 | 76.0 ± 2.6 | 69.8 ± 2.6 | 78.5 ± 1.2 | 75.4 ± 1.4 | 70.78 |
| InfoGraph | 67.5 ± 0.1 | 73.2 ± 0.4 | 63.7 ± 0.5 | 59.9 ± 0.3 | 76.5 ± 1.0 | 74.1 ± 0.7 | 75.1 ± 0.9 | 77.8 ± 0.8 | 70.98 |
| GROVER | 70.0 ± 0.1 | 74.3 ± 0.1 | 65.4 ± 0.4 | 64.8 ± 0.6 | 81.2 ± 3.0 | 67.3 ± 1.8 | 62.5 ± 0.9 | 82.6 ± 0.7 | 71.01 |
| MolCLR | 66.6 ± 1.8 | 73.0 ± 0.1 | 62.9 ± 0.3 | 57.5 ± 1.7 | 86.1 ± 0.9 | 72.5 ± 2.3 | 76.2 ± 1.5 | 71.5 ± 3.1 | 70.79 |
| GraphMAE | 72.0 ± 0.6 | 75.5 ± 0.6 | 64.1 ± 0.3 | 60.3 ± 1.1 | 82.3 ± 1.2 | 76.3 ± 2.4 | 77.2 ± 1.0 | 83.1 ± 0.9 | 73.85 |
| Mole-BERT | 71.9 ± 1.6 | 76.8 ± 0.5 | 64.3 ± 0.2 | 62.8 ± 1.1 | 78.9 ± 3.0 | 78.6 ± 1.8 | 78.2 ± 0.8 | 80.8 ± 1.4 | 74.04 |
| MoleculeSDE | 71.8 ± 0.7 | 76.8 ± 0.3 | 65.0 ± 0.2 | 60.8 ± 0.3 | 87.0 ± 0.5 | 80.9 ± 0.3 | 78.8 ± 0.9 | 79.5 ± 2.1 | 75.07 |
| 3D InfoMax | 70.4 ± 1.0 | 75.5 ± 0.5 | 63.1 ± 0.7 | 64.1 ± 0.1 | 89.8 ± 1.2 | 72.8 ± 1.0 | 74.9 ± 0.3 | 80.7 ± 0.6 | 73.91 |
| Galformer | 71.6 ± 0.9 | 75.7 ± 0.8 | 64.0 ± 0.4 | 64.3 ± 0.5 | 84.7 ± 0.9 | 73.4 ± 1.1 | 75.6 ± 0.4 | 80.9 ± 1.0 | 74.02 |
| GraphMVP | 71.5 ± 1.3 | 76.1 ± 0.9 | 64.3 ± 0.6 | 64.7 ± 0.7 | 85.4 ± 0.8 | 74.9 ± 1.2 | 76.0 ± 0.6 | 81.5 ± 1.2 | 74.86 |
| MoleBlend | 73.0 ± 0.8 | 77.8 ± 0.8 | 66.1 ± 0.0 | 64.9 ± 0.3 | 87.6 ± 0.7 | 77.2 ± 2.3 | 79.0 ± 0.8 | 83.7 ± 1.4 | 76.16 |
| LEMON | 73.7 ± 1.1 | 77.5 ± 0.6 | 65.1 ± 0.5 | 64.3 ± 0.9 | 85.9 ± 3.2 | 79.4 ± 4.3 | 79.3 ± 1.1 | 87.8 ± 1.4 | 76.62 |
| GEM | 72.4 ± 0.4 | 78.1 ± 0.1 | 69.2 ± 0.4 | 67.2 ± 0.4 | 90.1 ± 1.3 | 81.7 ± 0.5 | 80.6 ± 0.9 | 85.6 ± 1.1 | 78.18 |
| Uni-Mol | 72.9 ± 0.6 | 79.6 ± 0.5 | 69.6 ± 0.1 | 65.9 ± 1.3 | 91.9 ± 1.8 | 82.1 ± 1.3 | 80.8 ± 0.3 | 85.7 ± 0.2 | 78.63 |
| **DeMol** | **75.1 ± 0.6** | **80.9 ± 0.4** | 69.3 ± 0.5 | **68.4 ± 0.3** | **92.6 ± 0.7** | **82.5 ± 1.4** | **81.2 ± 0.9** | **89.0 ± 1.1** | **79.96** |
