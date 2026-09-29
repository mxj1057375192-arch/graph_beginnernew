# 任务三 · 图分类

基于 PyTorch Geometric 实现 GCN / GAT / GraphSAGE / GIN 在 TUDataset(MUTAG、
PROTEINS、ENZYMES)与 ZINC 上的图级任务,并分析**不同池化方式**
(AvgPooling / MaxPooling / MinPooling)对性能的影响。

> **与任务一、二的区别**:图级任务**没有"全图 vs 采样"的对比**。数据本身是一堆
> 独立的小图,天然按 mini-batch 训练,"全图"在这里没有对应物。所以任务书对
> 任务三也没有提出这个要求(任务要求第 3.2 条明确写了"任务四不需要",
> 而任务三的考点是池化)。

---

## 一、环境

同任务一,见 [task1 的 README](../task1_node_classification/README.md#一环境)。
本任务**不需要** pyg-lib(不做子图采样)。

---

## 二、数据集

| 数据集 | 图数 | 类别 | 任务类型 | 指标 | 训练/验证/测试 |
|---|---|---|---|---|---|
| MUTAG | 188 | 2 | 分类 | 准确率 ↑ | 150 / 18 / 20 |
| PROTEINS | 1,113 | 2 | 分类 | 准确率 ↑ | 890 / 111 / 112 |
| ENZYMES | 600 | 6 | 分类 | 准确率 ↑ | 480 / 60 / 60 |
| ZINC | 10,000 | — | **回归** | **MAE ↓** | 官方划分 |

> ⚠️ **ZINC 的代码已实现但实验未运行** —— 本仓库的实测结果只覆盖三个 TU 数据集。
> 原因(计算预算)与由此带来的缺口写在[第九节](#91-zinc-未运行重要)。

**TUDataset 没有官方划分**(MUTAG 等只有整份数据),必须自己切。本项目用
80/10/10 随机划分,**用带种子的生成器**而非全局随机状态 —— 否则同一份代码在不同
调用顺序下会切出不同的集合,实验不可复现。测试脚本会用 checkpoint 里存的种子
重新切出同一份划分。

**ZINC 用官方划分**,并且用的是 `subset=True` 的 1 万张图子集(完整版 25 万张,
纯 CPU 上不可行)。

---

## 三、目录结构

```
task3_graph_classification/
├── data/
│   ├── TU/                  # MUTAG / PROTEINS / ENZYMES
│   └── ZINC/
├── models/
├── README.md
└── code/
    ├── train.py
    ├── test.py
    └── report.py
```

---

## 四、训练脚本

[`code/train.py`](code/train.py)

```bash
# 单次运行
python task3_graph_classification/code/train.py --dataset MUTAG --model GCN --pooling mean

# 分类数据集主实验:MUTAG + PROTEINS + ENZYMES × 4 模型 × 3 池化 × 3 种子
python task3_graph_classification/code/train.py \
       --all --datasets MUTAG PROTEINS ENZYMES --seeds 3

# ZINC 是回归任务,慢两个数量级,单独跑
python task3_graph_classification/code/train.py --all --datasets ZINC --seeds 1

# 保存权重
python task3_graph_classification/code/train.py --all --save-model task3_graph_classification/models
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | MUTAG | MUTAG / PROTEINS / ENZYMES / ZINC |
| `--model` | GCN | GCN / GAT / GraphSAGE / GIN |
| `--pooling` | mean | `mean` / `max` / `min` |
| `--layers` | 3 | 图级任务常用 3 层(比节点分类深一层) |
| `--epochs` | 按数据集 | ZINC 取 60,TU 数据集取 120 |
| `--batch-size` | 64 | 每批图数(ZINC 自动提到 128) |

`--dataset ZINC` 会自动切到**回归**模式:单标量输出、MAE 评价、官方划分。

---

## 五、测试脚本

```bash
python task3_graph_classification/code/test.py \
       --checkpoint task3_graph_classification/models/MUTAG_GCN_mean_seed0.pt
python task3_graph_classification/code/test.py --checkpoints task3_graph_classification/models
```

**TUDataset 的划分是随机切的**,所以测试时会用 checkpoint 里存的种子重新切出
同一份 —— 否则评测的图集与训练时不是同一批。ZINC 用官方划分,不受影响。

---

## 六、池化的实现

三种池化在 [`common/models.py`](../common/models.py) 的 `pool_nodes` 里统一实现:

| 池化 | 实现 | 说明 |
|---|---|---|
| 平均 | `scatter_mean(h, batch)` | 对邻域规模不敏感 |
| 最大 | `scatter_max(h, batch)` | 取最显著的特征 |
| 最小 | `-scatter_max(-h, batch)` | **PyG 没有 `global_min_pool`**,用取负再取最大实现 |

> **关于 min pooling 的一个预期修正**:动手前曾预期 min pooling 会因为
> "ReLU 之后负值全被截断、最小值恒为 0"而表现最差。**实测结果不支持这个预期** ——
> 它在三个数据集上的平均准确率反而是三者中最高的(0.5485),且在 PROTEINS 上
> **四个模型一致地**取得最优。详见第八节。
>
> 之所以没有塌向 0:池化发生在最后一层编码器的 LayerNorm + ReLU 之后,
> 但各维度的激活分布不同,最小值并非恒为 0;而且"哪个特征最弱"本身携带信息。
> 这类"想当然的预期被数据推翻"正是要跑实验的原因。

---

## 七、ZINC 的两个坑(都踩过)

### 7.1 节点特征是原子类型编号,必须走 embedding

ZINC 的 `x` 是 `[N, 1]` 的 **int64** 原子类型编号,**不是**可直接使用的连续特征
(实测 `num_features == 1`,不是文档常说的 28 维 one-hot)。直接喂进卷积会出两种
后果,而且**两者不一样**:

| 模型 | 表现 |
|---|---|
| GCN | 报错 —— `gcn_norm(..., dtype=x.dtype)` 取到 int64,`deg.pow_(-0.5)` 抛 `RuntimeError` |
| GAT / GraphSAGE | 报错 —— 线性层 matmul 的 dtype 不匹配 |
| **GIN** | **不报错,但结果是错的** |

GIN 之所以不报错,是因为 `GINConv.forward` 里有:

```python
out = out + (1 + self.eps) * x_r     # eps 是 float
```

`eps` 是浮点数,PyTorch 的类型提升规则会把整数**静默转成浮点**。于是模型把
"碳=0、氮=1、氧=2"当成数值大小 0.0 / 1.0 / 2.0 在使用 —— 而元素编号的顺序本身
是任意的、没有度量含义,所以算出来的 MAE 没有意义。

> 修好之后 GIN 的 MAE 从 0.8992 变成 1.0516 —— **错误版本的数字反而更好看**。
> 这正是静默错误最危险的地方。

修法:ZINC 走 `num_embeddings` embedding 层。同时
[`common/models.py`](../common/models.py) 里加了一道守卫:**整数特征没配
embedding 就直接抛异常**,不给 GIN 这类"静默通过"的机会。

### 7.2 标签是浮点回归目标,不能转 long

TUDataset 的 `y` 有时是 `[1]`、有时是 `[1,1]`,需要压平成 `[N]` 的类别索引;
但 ZINC 的 `y` 是**浮点**回归目标,对它做 `.long()` 会直接毁掉任务。
两者的处理必须分开,见 `run_graph_classification`。

---

## 八、结论摘要

> 详细表格见 [`results/report_task3.md`](../results/report_task3.md)。

### 8.1 池化方式的差异整体很小 —— 但 PROTEINS 上例外

三个数据集、四个模型、共 36 个组合的平均准确率:

| 池化方式 | 平均准确率 | 与最优的差 |
|---|---|---|
| **最小池化** | **0.5485** | — |
| 平均池化 | 0.5400 | −0.0085 |
| 最大池化 | 0.5393 | −0.0092 |

**整体差 0.009,小于各组合的标准差**(普遍在 ±0.03~0.08),
所以**在总体层面不应下"某种池化更好"的结论**。

但拆到单个数据集看,有一个**跨模型一致**的模式值得注意:

| 数据集 | 最优池化(4 个模型中) |
|---|---|
| PROTEINS | **最小池化 4/4** |
| MUTAG | 平均池化 2/4,最大 1/4,最小 1/4 |
| ENZYMES | 平均 2/4,最小 1/4,最大 1/4 |

**PROTEINS 上四个模型一致选择最小池化**,且优势不小(如 GAT:0.7411 vs 平均 0.6637,
差 0.077)。单个模型的差异可能是噪声,但四个模型方向一致、且幅度都超过各自标准差的
情况,更可能是真实效应。合理解释:PROTEINS 是"是否酶"的二分类,而**"哪个特征最弱"
本身携带判别信息**,最大池化只保留最显著的激活、会丢掉这部分信号。

### 8.2 模型对比:模型间差距大于池化间差距

| 数据集 | 表现 |
|---|---|
| MUTAG | GraphSAGE 0.7500 / GCN 0.7333 领先,GIN 0.6333 垫底 |
| PROTEINS | GAT(最小池化)0.7411 / GCN 0.7292 领先 |
| ENZYMES | 各模型 0.18~0.27,**差距很小且都接近随机** |

**池化方式带来的差异(≈0.009)比模型选择带来的差异小得多**,说明在这三个数据集上,
"选哪个池化"远不如"选哪个模型"重要。

### 8.3 ⚠️ ENZYMES 上的结果不理想,如实说明

ENZYMES 是 6 分类,随机猜的准确率是 **0.167**。而所有 12 个组合(4 模型 × 3 池化)
的结果落在 **0.178 ~ 0.272**,只比随机高一点。

**这说明模型在 ENZYMES 上基本没学到东西。** 可能的原因:

- **数据太少**:ENZYMES 只有 600 张图,训练集 480 张,而输入特征有 18 维(可区分性低)
- **模型容量与训练预算**:`layers=3, hidden=64` 是为 MUTAG/PROTEINS 定的,
  在小样本多分类上容易过拟合
- **划分的随机性**:验证集只有 60 张图,早停依据很不稳定

文献上 ENZYMES 用合适的配置能到 0.5~0.7,所以**本任务的数字不代表该数据集的上限**,
只反映"在当前这套统一配置下"的表现。报告中没有对 ENZYMES 做过任何逐数据集的调参,
这是为了保持四个数据集之间配置一致 —— **代价就是 ENZYMES 的结果偏弱**。
这一点在解读时不应忽略。

### 8.4 梯度消失的观察:GIN 在小图上依然最弱

GIN 在 MUTAG(0.6333)和 PROTEINS(0.6250~0.7232)上普遍是四个模型里最低的。
与任务一在 Cora 上"GIN 加深度会崩"的发现一致,呼应了 GIN 的 sum 聚合在
小图上容易让表示趋同的问题。

---

## 九、本任务未覆盖的部分

### 9.1 ZINC 未运行(重要)

**任务书要求使用 TUDataset 与 ZINC,本仓库实际只跑了 TUDataset 的三个数据集。**
ZINC 的代码路径已经实现并验证可跑通(见第七节,踩过的两个坑都在 ZINC 上调通的),
但**实验未执行**:

| 数据集 | 状态 |
|---|---|
| MUTAG / PROTEINS / ENZYMES | ✅ 已跑 |
| **ZINC** | ❌ **未跑** |

原因是计算预算:ZINC 有 10,000 张图、每 epoch 79 个批次、单次 epoch 约 2 秒,
是 MUTAG 的 **60 倍**。4 模型 × 3 池化 × 1 种子的最小网格就要约 1 小时,
在纯 CPU 上超出本次作业的时间预算。

**这带来两个具体缺口:**

1. **缺少图回归任务的实验结果**。三个 TU 数据集都是小图的**分类**任务,
   而 ZINC 是**回归**任务(指标 MAE)—— 本任务的报告里没有任何回归结果。
2. **池化方式的结论只在小图上成立**。ZINC 的图平均约 23 个节点,
   与 TU 数据集规模相近,但结论能否推广到更大的分子图没有实验支持。

ZINC 的完整实现(`run_graph_classification` 的 `regression` 分支、
`--dataset ZINC` 的自动切换、`num_embeddings` embedding 层、
`eval_graph_classification` 的回归评测)都已就绪,补跑只需:

```bash
python task3_graph_classification/code/train.py --all --datasets ZINC --seeds 1
```

### 9.2 其他

- **未使用完整的 ZINC**(25 万张图),代码中用的是 `subset=True` 的 1 万张。
- **三个 TU 数据集都用了 3 个种子**,但划分是 80/10/10 随机切分 ——
  验证集只有 18~111 张图,指标本身噪声不小。
