# 任务二 · 图上的链路预测

基于 PyTorch Geometric 实现 GCN / GAT / GraphSAGE / GIN 在 Cora、Citeseer 上的
链路预测,并对比**全图训练**与**子图采样训练**的性能和运行时间。

---

## 一、环境

与任务一相同,见 [task1 的 README](../task1_node_classification/README.md#一环境)。
关键点:`NeighborLoader` / `LinkNeighborLoader` 依赖 `pyg-lib` 或 `torch-sparse`,
两者都不是纯 Python 包,不装会在**采样模式**下抛 `ImportError`。

```bash
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.14.0+cpu.html
```

---

## 二、数据与划分

| 数据集 | 节点 | 边 | 特征 | 训练 / 验证 / 测试 监督边 |
|---|---|---|---|---|
| Cora | 2,708 | 10,556 | 1,433 | 8,448 / 1,055 / 1,056 |
| Citeseer | 3,327 | 9,104 | 3,703 | 7,282 / 910 / 911 |

划分由 `torch_geometric.transforms.RandomLinkSplit` 生成,验证/测试各占 10%:

```python
RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                add_negative_train_samples=True, neg_sampling_ratio=1.0,
                split_labels=False)
```

正负样本比例 **1 : 1**。`is_undirected=True` 会先去掉重边 —— 无向图里同一条边
会出现两次,不去重的话它会同时落进正样本和负样本。

**训练集的正负样本是固定的一批**,不每个 epoch 重新采样。这是为了让全图与采样
两种模式见到**完全相同**的监督信号,否则差异会混进"模式对比"里。

---

## 三、目录结构

```
task2_link_prediction/
├── data/                    # 数据集(自动下载)
├── models/                  # 训练好的权重(.pt)
├── README.md
└── code/
    ├── train.py             # 训练脚本
    ├── test.py              # 测试脚本(载入权重,不训练)
    └── report.py            # 结果汇总
```

---

## 四、训练脚本

[`code/train.py`](code/train.py)

```bash
# 单次运行
python task2_link_prediction/code/train.py --dataset Cora --model GCN --mode full

# 主实验:Cora + Citeseer × 4 模型 × 全图/采样 × 3 个种子
python task2_link_prediction/code/train.py --all --seeds 3

# 训练并保存权重,供 test.py 评测
python task2_link_prediction/code/train.py --all --save-model task2_link_prediction/models
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | Cora | Cora / Citeseer |
| `--model` | GCN | GCN / GAT / GraphSAGE / GIN |
| `--mode` | full | `full` 全图,`sampler` 子图采样 |
| `--val-ratio` / `--test-ratio` | 0.1 / 0.1 | 验证/测试边占比 |
| `--epochs` / `--patience` | 150 / 30 | 最大 epoch 与早停耐心值 |
| `--batch-size` | 256 | **仅采样模式**:每批监督边数 |
| `--neighbors` | 10 10 | 每层采样邻居数 |
| `--save-model` | — | 存权重的目录 |

---

## 五、测试脚本

[`code/test.py`](code/test.py)**不训练**,只载入权重在测试集上评测。

```bash
python task2_link_prediction/code/test.py \
       --checkpoint task2_link_prediction/models/Cora_GCN_full_seed0.pt
python task2_link_prediction/code/test.py --checkpoints task2_link_prediction/models
```

`eval_link_prediction` 会用 checkpoint 里存的**同一组划分参数**重建
`RandomLinkSplit`。这点很重要:TUDataset 那种随机划分如果换了种子,测试边集就
不是同一批了,数字没有意义。链路预测的划分同样依赖种子。

---

## 六、评测口径

| 指标 | 说明 |
|---|---|
| **AUC** | 正负边得分的 ROC 曲线下面积。早停与选优以它为准 —— 不受正负比例影响,比准确率稳健 |
| **AP** | 平均精度(PR 曲线下面积),在正样本稀少时比 AUC 更能反映实际表现 |
| **MRR** | 平均倒数排名 |
| **Hits@K** | 排名 ≤ K 的正样本比例 |

**排序口径**:每个正样本边与**全部负样本边**一起排名,而不是只跟同批次内的少量
负样本比。后者会显著低估难度 —— 若一批只有 256 条负样本,随机猜的 Hits@10 就有
4%,报出来的数字会虚高。本项目的实现见 `common/experiments.py` 的
`link_prediction_metrics`。

---

## 七、实验设计中的关键决定

### 7.1 两种模式唯一的差异是"批次从哪来"

与任务一共用同一份训练循环([`common/trainer.py`](../common/trainer.py))。
全图模式把整图与全部 8,448 条监督边打包成一个批次;采样模式用
`LinkNeighborLoader` 分批。优化器、学习率、早停规则、评测函数全部相同。

`LinkNeighborLoader` 的 `neg_sampling_ratio=0.0` —— 使用预先固定的负样本,
不每轮重采。

### 7.2 采样 batch size 用 256 而非 64

实测 Cora 上 `batch_size=64` 会让采样模式慢到 **1886 ms/epoch**
(132 步 × 14 ms,其中很大一部分是每批的固定开销),按 200 epoch × 4 模型 ×
3 种子 × 2 数据集估要跑约 3 小时。放大到 256 后步数降到 33,总耗时降一个数量级。

**不破坏对比公平性**:两种模式监督的边集完全相同,每个 epoch 仍然把全部
8,448 条监督边各用一次,变的只是梯度更新次数(132 → 33),而这个数会被记录
在结果表里。

### 7.3 必须切到种子边(与任务一同源的坑)

`LinkNeighborLoader` 输出的子图包含被采样到的邻居节点,计算损失时只能用
**种子边**(`edge_label_index` / `edge_label`),不能拿子图里所有边当监督。
不切不会报错,只会静默把监督量算错。

---

## 八、结论摘要

> 详细表格见 [`results/report_task2.md`](../results/report_task2.md)。

### 8.1 采样模式在 AUC 上全面胜出 —— 但**约六成是假象**

跑同样 100 个 epoch,采样模式在 8 个(数据集 × 模型)组合上**全部**优于全图:

| 数据集 | 模型 | 全图 AUC | 采样 AUC | 差距 |
|---|---|---|---|---|
| Cora | GraphSAGE | 0.6531 | 0.8240 | **+0.171** |
| Cora | GIN | 0.6297 | 0.7452 | +0.116 |
| Cora | GCN | 0.7594 | 0.8598 | +0.100 |
| Cora | GAT | 0.6631 | 0.7312 | +0.068 |
| Citeseer | GraphSAGE | 0.6865 | 0.7932 | +0.107 |
| Citeseer | GCN | 0.7664 | 0.8496 | +0.083 |
| Citeseer | GIN | 0.6947 | 0.7707 | +0.076 |
| Citeseer | GAT | 0.6481 | 0.7227 | +0.075 |

**但这个对比本身是不公平的**,而且不公平的方向恰好有利于采样:

| | 1 个 epoch | 累计梯度更新(100 epoch) |
|---|---|---|
| 全图 | **1 次**更新 | 100 次 |
| 采样 | **33 次**更新 | 3,300 次 |

采样模式做了 **33 倍**的优化。所以这 +0.10 里混着"优化得多"这个因素。

### 8.2 等步数对照:差距缩小六成,但没有消失

把两者的**累计梯度更新次数**对齐(全图跑满 3300 epoch):

| 模型 | 全图(100 ep) | 全图(3300 ep,等步数) | 采样(100 ep) | 原差距 | 等步数后 | 缩小 |
|---|---|---|---|---|---|---|
| GCN | 0.7594 | 0.8181 | 0.8598 | +0.1004 | +0.0417 | **58%** |
| GraphSAGE | 0.6531 | 0.7640 | 0.8240 | +0.1709 | +0.0601 | **65%** |

**结论:采样优势里约 60% 来自"多做优化"这个混淆因素,剩下约 40% 是真实的。**

剩下这部分可能来自采样本身的**正则化效应** —— 每个批次只看一个子图,等价于对
邻居做随机丢弃,能抑制对训练边的过拟合。注意全图模式的最佳轮次出现在第 310 步
(GCN)就再没改善,而它总共跑了 3300 步 —— 后 90% 的时间是在过拟合。

复现:`python task2_link_prediction/code/equal_steps.py`

### 8.3 运行时间:采样慢 15~25 倍

| 数据集 | 每 epoch 耗时倍数 | 梯度步数倍数 |
|---|---|---|
| Cora | 22~25× | 33× |
| Citeseer | 15~17× | 29× |

和任务一在 Flickr 上的发现一致:**采样省的是显存,不是时间**。它只在
"图大 + 特征宽 + 训练节点占比小"时才省时间。

### 8.4 模型对比

四个模型里,**GCN 的 AUC 最高**(Cora 0.86 / Citeseer 0.85),GraphSAGE 次之,
GIN 与 GAT 较弱。注意这与任务一的结论不同 —— 节点分类上四个模型差距很小,
链路预测上差距明显,说明链路预测这个任务对模型的**结构归纳偏置**更敏感。

---

## 九、本任务未覆盖的部分

如实说明:

- **未使用 Flickr 数据集**。任务书要求使用 Cora / Citeseer / Flickr,但本任务
  只覆盖前两个。原因是 Flickr 有 **899,756 条边**,链路预测要在其中约 18 万条
  监督边上做训练与评测;在纯 CPU 上单次运行(4 模型 × 2 模式 × 多种子)
  预计超过 10 小时,超出本次作业的计算预算。
  任务一中 Flickr 的结果(见 [task1 README](../task1_node_classification/README.md))
  可以作为"大图上的表现"的参考。
- **未做超参扫描**。任务二的超参影响未单独扫描,学习率沿用任务一在 Cora 上扫出的
  结论。原因是扫描的成本会随边数增长,而本任务的可分析维度(模式 × 模型)已经
  足够回答任务书的问题。
