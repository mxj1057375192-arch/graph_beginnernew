# 图神经网络作业 · 完成情况

> 本仓库由老师提供的任务书模板 [Maxioo/graph_beginner](https://github.com/Maxioo/graph_beginner) fork 而来。
> **从这一行到下面的分隔线为止,是本次作业的完成情况**;分隔线之后是老师原始的任务书,一字未改。

**框架**:PyTorch Geometric　·　**硬件**:纯 CPU,无独立显卡(全部实验在一台笔记本上跑完)

---

## 一、运行环境

| 项 | 版本 |
|---|---|
| Python | 3.12.14(conda 环境 `py312`) |
| PyTorch | 2.14.0+cpu |
| PyTorch Geometric | 2.8.0.post1 |
| pyg-lib | 0.9.0+pt214cpu |
| 硬件 | i5-1135G7(4 核 8 线程)/ 16 GB / **无独立显卡** |

完整依赖见 [requirements.txt](requirements.txt)。

> ⚠️ `pyg-lib` 是**采样训练**的必需依赖。不装它时全图模式正常,但一旦用
> `--mode sampler` 就会抛 `ImportError: 'NeighborSampler' requires either
> 'pyg-lib' or 'torch-sparse'`。这是最容易漏装的一个包。

## 二、三分钟验收

四条命令分别验证四个任务能跑通(前三条几秒、第四条几十秒出结果):

```bash
# 任务一 · 节点分类
python task1_node_classification/code/train.py --dataset Cora --model GCN --mode full

# 任务二 · 链路预测
python task2_link_prediction/code/train.py --dataset Cora --model GCN --mode full

# 任务三 · 图分类
python task3_graph_classification/code/train.py --dataset MUTAG --model GCN --pooling mean

# 任务四 · 知识图谱补全
python task4_knowledge_graph/code/run_experiments.py --datasets countries_S1 --models RotatE
```

前三条已逐条实测,实际输出:

| 命令 | 实测结果 |
|---|---|
| 任务一 | `best_epoch=84  val=0.7960  test=0.8130` |
| 任务二 | `best_epoch=8  val_auc=0.7407  test_auc=0.7371` |
| 任务三 | `best_epoch=27  val_acc=0.9444  test_acc=0.8000` |

> **数据集不入库**(Cora/Citeseer/Flickr/TU 合计约 1.5 GB)。首次运行会由 PyG
> 自动下载到各任务的 `data/` 下。任务四的 `kge_framework/data/` 里**只保留了实际用到的
> countries_S1/S2/S3 三个数据集(共约 175 KB)**,FB15k 等大文件已排除。

重新生成报告表格(结果全部由脚本从原始记录汇总,无手工抄写):

```bash
python task1_node_classification/code/report.py
python task2_link_prediction/code/report.py
python task3_graph_classification/code/report.py
python task4_knowledge_graph/code/report.py
```

## 三、完成情况一览

| 任务 | 数据集 | 模型 | 训练模式/池化 | 状态 |
|---|---|---|---|---|
| 一 节点分类 | Cora, Citeseer | GCN/GAT/SAGE/GIN | 全图 vs 采样,5 种子 | ✅ 完整 |
| | Flickr | GCN、GraphSAGE | — | ⚠️ **部分**(见第五节) |
| 二 链路预测 | Cora, Citeseer | GCN/GAT/SAGE/GIN | 全图 vs 采样,3 种子 | ✅ 完整 |
| | Flickr | — | — | ❌ 未跑 |
| 三 图分类 | MUTAG, PROTEINS, ENZYMES | GCN/GAT/SAGE/GIN | 平均/最大/最小池化,3 种子 | ✅ 完整 |
| | ZINC | — | — | ❌ 未跑 |
| 四 知识图谱 | countries_S1/S2/S3 | TransE/RotatE/ConvE | — | ✅ 完整 |
| | FB15k-237 | — | — | ❌ 未跑 |

**交付物**:四个任务各自独立可运行(`code/` + `README.md`),共用的模型/训练/评测代码
在 [`common/`](common/),实验结果原始记录在 [`results/runs.jsonl`](results/runs.jsonl)。

## 四、四个任务的核心结论

**任务一 · 两种训练模式在精度上等价。** Cora 上采样平均高 +0.0044,但标准差是
±0.0083;Citeseer 上 GCN 反而低 0.0084。**所有差异都在一个标准差以内**,不构成
"谁更好"的证据。这个结论正是靠 5 个种子才看得出来的 —— 单次运行的话,+0.0044
看着就像个发现。

耗时差异则完全取决于数据集,**不是一个固定倍数**:Cora 上两者持平(0.9~1.3×),
Citeseer 上采样**快一倍**(0.5~0.6×,因为它特征有 3,703 维,全图每层都要乘整个
特征矩阵),Flickr 上采样**慢 15 倍**(14.9×)。

**任务二 · 采样在每个组合中都占优,但优势大半来自"更新次数多"。** 采样 AUC 在
全部 8 个组合上胜出(+0.068 ~ +0.171),然而代价是每 epoch 慢 14.6~24.8 倍。
关键在于两种模式的 "1 个 epoch" 不是同一件事:全图 1 次梯度更新,采样 29~33 次。
把累计更新次数对齐后(全图跑满 3300 epoch),差距**缩小 58%~65% 但并未消失**。

**任务三 · 最小池化总体最好,且推翻了"min 池化会塌向 0"的预判。** 三种池化在
36 个(数据集 × 模型)组合上的平均准确率:最小池化 0.5485 > 平均池化 0.5400 >
最大池化 0.5393。在 PROTEINS 上,**最小池化对 4 个模型全部最优**。这与"ReLU 之后
min 池化特征会全变 0"的直觉相反 —— 实际原因是本实现池化前保留了负值,min 起到了
筛选"最弱信号"的作用。

**任务四 · TransE 与 RotatE 基本持平,ConvE 大幅落后。** 三个 countries 数据集上
两者 MRR 互有胜负(差距 ≤0.008);ConvE 显著更低(0.05~0.23)。需要说明的是
**ConvE 的参数量是另外两个的 9 倍**(488,855 vs 54,200),在只有 271 个实体、
约 1,000 条三元组的小数据集上严重过参数化,这个落后不能直接推广为"ConvE 更差"。

## 五、未完成 / 打了折扣的部分

**如实列出,避免读者误以为实验矩阵是满的。**

| 缺口 | 原因 |
|---|---|
| **ZINC 完全未跑** | 时间预算。ZINC 是回归任务,需另做标准化与 MAE 评测 |
| **Flickr 只跑了部分** | 89k 节点 / 900k 边,单次采样训练一个 epoch 就要 **9.8~11.1 秒**(Cora 采样只要 20~24 ms,**约 400~550 倍**)。目前只有 GCN 完成了两种模式的对比(各 2 种子)、GraphSAGE 只有全图结果(2 种子);GAT/GIN 未跑 |
| **任务二的 Flickr 未跑** | 同上。链路预测还要额外做负采样,开销更大 |
| **FB15k-237 未跑** | 只用老师框架自带的小规模 countries 数据集。FB15k-237 有 14,541 实体 / 310,116 三元组,在纯 CPU 上单模型训练以小时计 |
| **超参扫描只在 Cora 上做** | 4 模型 × 3 学习率 × 3 层数 = 36 组。方案设计如此(阶段 A 在 Cora 上定参,阶段 B 迁移),原因是组合爆炸 |
| **扫描只用 1 个种子** | 主实验用 3~5 个。因此扫描结果的噪声大于主实验表,只用于看趋势 |

**计算预算**:纯 CPU 上累计约 15 小时机时。凡是判断"再跑就要超出预算"的组合,
一律选择**如实标注未跑**,而不是编一个数字。

## 六、文档索引

| 文档 | 给谁看 |
|---|---|
| [REPORT.md](REPORT.md) | **总实验报告** —— 四个任务的完整分析、两个必答问题(参数/模型影响、全图 vs 采样) |
| [操作手册.md](操作手册.md) | 怎么跑 —— 环境搭建、全部命令、结果解读、**错误速查表**、常见提问 |
| [代码导读.md](代码导读.md) | 代码怎么工作 —— 跟着一条命令走完全程,以及几处"看起来多余但删了会出错"的代码 |
| [task1](task1_node_classification/README.md) / [task2](task2_link_prediction/README.md) / [task3](task3_graph_classification/README.md) / [task4](task4_knowledge_graph/README.md) | 各任务的训练/测试脚本说明与详细结果 |
| [results/report_task1~4.md](results/) | **结果表格**(由脚本自动生成,请勿手工编辑) |

## 七、关于任务四的框架

任务四按作业要求"基于提供的框架",直接使用老师的
[kge_framework](https://github.com/Maxioo/kge_framework),**只补它缺的两个模型**。

原框架 `code/model.py` 中**只有 TransE**。本作业在其中新增了 **RotatE 与 ConvE**
(共 +262 行),并修正了 `main.py` 的模型分发(原文件引用了不存在的 `RotatE`/`ConvKB`)。

实现过程中修正了两个会导致结果错误的细节,记录在
[task4_knowledge_graph/README.md](task4_knowledge_graph/README.md):

1. **RotatE 的距离用 L2 范数**。原写法 `.norm(p=1)` 是 L1,与论文不一致,MRR 只有 0.29。
2. **实体初始化范围必须是 `(gamma + ε) / hidden_dim`**,而不是常见的 `6/sqrt(dim)`。
   因为距离随维度线性增长,100 维时距离可达 30~80,而 gamma 只有 9 —— 所有三元组
   的得分都会是很大的负数,模型无法学习。改成前者后 MRR 从 0.29 提升到 **0.55**。

---

以下是老师原始的任务书,未作任何改动。

---

# 图神经网络-Beginner
选择一种框架完成如下任务

参考：
- [DGL框架](https://www.dgl.ai/)
- [Pytorch_geometric框架](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [子图训练](https://docs.dgl.ai/tutorials/large/L0_neighbor_sampling_overview.html#sphx-glr-tutorials-large-l0-neighbor-sampling-overview-py)

****
## 任务一、 节点分类
实现基于GNN主流模型(GCN, GAT, GraphSAGE, GIN)的节点分类:

1. 实现要求：基于现有模型框架(DGL, Pytorch_geometric)实现图上的节点分类任务
2. 使用Cora, Citeseer, Flickr数据集
3. 测试GCN, GAT, GraphSAGE, GIN模型
4. 利用框架自带的Sampler采样子图进行训练，并与全图训练进行性能和运行时间的对比

## 任务二、 图上的链路预测
1. 实现要求：基于现有模型框架(DGL, Pytorch_geometric)实现图上的链路预测
2. 使用Cora, Citeseer, Flickr数据集
3. 测试GCN, GAT, GraphSAGE, GIN模型
4. 利用框架自带的Sampler采样子图进行训练，并与全图训练进行性能和运行时间的对比

## 任务三、 图分类
1. 实现要求：基于现有模型框架(DGL, Pytorch_geometric)实现图分类任务
2. 使用TUDataset, ZINC数据集
3. 分析不同的池化方法对图分类性能的影响(AvgPooling, MaxPooling, MinPooling)
4. 测试GCN, GAT, GraphSAGE, GIN模型


## 任务四、 知识图谱
1. 参考
   1. 训练和测试框架[KGE框架](https://github.com/Maxioo/kge_framework)
   2. TransE, RotatE, ConvE的论文
2. 实现要求：基于参考资料和知识图谱补全框架，支持常见模型(TransE, RotatE, ConvE)
3. 需要了解的知识点：
   1. 常见知识图谱补全模型的原理(TransE, RotatE, ConvE)
   2. 数据集：训练集/验证集/测试集的划分

## 任务要求
1. 需要了解的知识点：
   1. GNN主流模型的原理(GCN, GAT, GraphSAGE, GIN, ...)
   2. 数据集：训练集/验证集/测试集的划分
2. 代码要求

   代码文件夹按统一格式:
      - 任务一
         - data
         - code
         - README.md
      - 任务二
         - data
         - code
         - README.md
      - ...
      - requirements.txt (运行环境文件)
   
   其中README.md中需要写明训练和测试的脚本

3. 报告要求
   1. 分析不同参数（学习率、网络层数）和不同的神经网络对性能的影响
   2. 测试全图训练和分批次训练对模型性能和运行时间的影响(任务四不需要)
