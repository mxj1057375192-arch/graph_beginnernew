# 任务一 · 节点分类

基于 PyTorch Geometric 实现 GCN / GAT / GraphSAGE / GIN 四个模型在 Cora、Citeseer、
Flickr 三个数据集上的节点分类,并对比**全图训练**与**子图采样训练**的性能和运行时间。

---

## 一、环境

| 项 | 版本 |
|---|---|
| Python | 3.12.14(conda 环境 `py312`) |
| PyTorch | 2.14.0+cpu |
| PyTorch Geometric | 2.8.0.post1 |
| pyg-lib | 0.9.0+pt214cpu |
| 硬件 | i5-1135G7(4 核 8 线程)/ 16 GB / **无独立显卡,纯 CPU** |

> **关于 pyg-lib**:`NeighborLoader` 依赖 `pyg-lib` 或 `torch-sparse`,两者都不是
> 纯 Python 包。不装会在运行时抛 `ImportError: 'NeighborSampler' requires either
> 'pyg-lib' or 'torch-sparse'` —— 注意这个错误只在**采样模式**下才出现,全图模式
> 不受影响。安装:
>
> ```bash
> pip install pyg-lib -f https://data.pyg.org/whl/torch-2.14.0+cpu.html
> ```

完整依赖见仓库根目录的 [requirements.txt](../requirements.txt)。

---

## 二、数据集

三个数据集都由 `common/datasets.py` 自动下载到本目录的 `data/` 下。
规模为**实际加载后统计**的结果,非照抄文档:

| 数据集 | 节点 | 边 | 特征 | 类别 | 训练 / 验证 / 测试 |
|---|---|---|---|---|---|
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 / 500 / 1,000 |
| Citeseer | 3,327 | 9,104 | 3,703 | 6 | 120 / 500 / 1,000 |
| Flickr | 89,250 | 899,756 | 500 | 7 | 44,625 / 22,312 / 22,312 |

**划分方式**:Cora 与 Citeseer 使用 Planetoid 的公开划分;Flickr 使用其自带的
`role.json` 划分。三者都不在本项目里重新切分,便于与文献对比。

**特征预处理**(见 `common/experiments.py` 的 `FEATURE_TREATMENT`):

| 数据集 | 处理方式 | 理由 |
|---|---|---|
| Cora / Citeseer | 行归一化(`NormalizeFeatures`) | 稀疏词袋向量,文献标准做法 |
| Flickr | **列标准化(z-score)** | 原始词频计数,量纲跨度 0 ~ 1827 |

**两种训练模式应用完全相同的预处理** —— 否则预处理差异会混进"模式对比"里,
成为一个不外显的混淆因素。

> ⚠️ **Flickr 这一步曾经漏做,后果很隐蔽。** 不处理时模型连续 **35 个 epoch
> 卡在多数类(0.424)** 上不动,而 `patience=25` 会在第 26 轮就掐断 —— 正好死在
> 模型开始学习**之前**。表现出来是:`best_epoch=0`、三个随机种子给出完全相同的
> 4 位小数准确率。这类"看起来跑通了"的失败比报错难查得多。
>
> 实测对照(`python -m common.diag_flickr`):
>
> | 特征处理 | 前 5 轮 val | 收敛后 |
> |---|---|---|
> | 原始计数 | 0.424(卡住 35 轮) | 0.4847 |
> | 行归一化 | 0.489 | 0.4883 |
> | **列标准化** | **0.480(立刻正常)** | **0.5176** |

---

## 三、目录结构

```
task1_node_classification/
├── data/                    # 数据集(自动下载,已在 .gitignore 中排除)
├── models/                  # 训练好的权重(.pt),由 --save-model 产生
├── README.md                # 本文件
└── code/
    ├── train.py             # 训练脚本
    ├── test.py              # 测试脚本(载入权重,不训练)
    └── report.py            # 结果汇总,生成表格
```

`code/` 下的脚本可以**从任意目录启动**(内部按 `__file__` 反推仓库根),
无需先 `cd` 到仓库根,也不依赖 `PYTHONPATH`。

---

## 四、训练脚本

[`code/train.py`](code/train.py)

```bash
# 单次运行
python task1_node_classification/code/train.py --dataset Cora --model GCN --mode full

# 主对比实验:全部数据集 × 4 模型 × 全图/采样 × 5 个种子
python task1_node_classification/code/train.py --all --seeds 5

# 只跑部分数据集(Flickr 比另外两个慢三个数量级,建议单独跑)
python task1_node_classification/code/train.py --all --datasets Cora Citeseer --seeds 5
python task1_node_classification/code/train.py --all --datasets Flickr --seeds 3 \
       --epochs 100 --patience 25 --batch-size 256

# 超参扫描(报告要求分析 lr 与层数的影响)
python task1_node_classification/code/train.py --sweep lr
python task1_node_classification/code/train.py --sweep layers

# 训练并把「验证集最优」权重存下来,供 test.py 评测
python task1_node_classification/code/train.py --all --seeds 5 \
       --save-model task1_node_classification/models
```

主要参数:

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | Cora | Cora / Citeseer / Flickr |
| `--model` | GCN | GCN / GAT / GraphSAGE / GIN |
| `--mode` | full | `full` 全图训练,`sampler` 子图采样训练 |
| `--layers` | 2 | 层数。采样模式下 `--neighbors` 长度会同步 |
| `--lr` | 0.01 | 学习率 |
| `--hidden` | 64 | 隐藏层维度 |
| `--epochs` / `--patience` | 300 / 50 | 最大 epoch 与早停耐心值 |
| `--batch-size` | 64 | **仅采样模式**的每批种子节点数 |
| `--neighbors` | 10 10 | 每层采样邻居数(fan-out) |
| `--seeds` | 5 | `--all` 时每个组合重复的种子数 |
| `--save-model` | — | 存权重的目录 |

---

## 五、测试脚本

[`code/test.py`](code/test.py)**不训练**,只载入 [train.py](code/train.py) 存下的权重,
在测试集上评测。

```bash
# 评测单个权重文件
python task1_node_classification/code/test.py \
       --checkpoint task1_node_classification/models/Cora_GCN_full_seed0.pt

# 评测整个目录下的全部权重(打印汇总表)
python task1_node_classification/code/test.py \
       --checkpoints task1_node_classification/models
```

**为什么测试集不参与任何选择**:训好的权重是「**验证集**上最优」的那一组,不是
最后一个 epoch 的。测试集只用于报告最终数字,不参与早停、不参与挑超参。否则测试集
就事实上变成了验证集,报出来的数字会系统性偏乐观。

验证方式:对同一个权重,[test.py](code/test.py) 报出的测试准确率与 [train.py](code/train.py)
训练结束时记录的一致(如 Cora/GCN/full/seed0 两者都是 0.8130),说明两条路径口径相同。

---

## 六、结果汇总

[`code/report.py`](code/report.py) 从仓库根的 `results/runs.jsonl` 汇总出表格:

```bash
python task1_node_classification/code/report.py
# 输出到控制台,同时写入 results/report_task1.md
```

**报告里的每一个数字都由这个脚本从原始记录生成,不手工抄写。** 手工抄写有三个
必然后果:抄错、漏掉失败的 run、以及无法回答"这个数是怎么来的"。

---

## 七、实验设计中的几个关键决定

### 7.1 两个模式如何做到可比

全图训练和采样训练**共用同一份训练循环代码**
([`common/trainer.py`](../common/trainer.py)),唯一差异是"数据怎么分批":
全图模式返回 `[整图]`,采样模式返回 `NeighborLoader`。优化器、学习率、早停规则、
评测函数、线程数全部相同。

这是**结构性保证**,不是靠人记得。如果两边各写一套循环,迟早会出现"一边用了学习率
调度、另一边没有"这类不对称,而这类偏差不会报错,只会让结论悄悄失真。

### 7.2 "1 个 epoch"在两种模式下不是同一件事

| | 1 个 epoch | 梯度更新次数 |
|---|---|---|
| 全图训练 | 遍历全部 N 个训练节点 | **1** |
| 采样训练 | 遍历全部 N 个训练节点 | **ceil(N / batch_size)** |

`NeighborLoader(input_nodes=train_mask)` 在一个 epoch 内会把每个训练节点各当一次
种子,所以**监督量是对齐的**(都是 N 个节点),但**梯度更新次数不对齐** ——
采样那边多做了几十到几百倍。

本项目因此对每次 run 都记录 `grad_steps_per_epoch`,并写进结果表。不列出这一列,
读者会把"优化次数多"误读成"采样方法更好"。

### 7.3 必须切种子节点(最容易静默出错的一处)

`NeighborLoader` 输出的子图**包含所有被采样到的邻居**,而只有前 `batch_size` 个是
种子节点。计算损失必须切:

```python
logits = logits[:batch.batch_size]
y = batch.y[:batch.batch_size]
```

**不切不会报错**,只会静默地把监督量算错(把邻居节点也当成了监督目标),让"每
epoch 遍历的监督量"虚高,破坏对比的可比性。这是本项目最容易踩的坑。

### 7.4 Flickr 的参数与其他两个数据集不同

| | Cora / Citeseer | Flickr |
|---|---|---|
| epochs / patience | 300 / 50 | 100 / 25 |
| 种子数 | 5 | 3 |
| 采样 batch size | 64 | 256 |

- **epochs 与种子数**:Flickr 是 89k 节点,收敛比 140 个训练节点的 Cora 快得多;
  而单次运行成本高两个数量级,纯 CPU 预算下取 3 个种子。
- **batch size**:实测每批有约 **7.5 ms 的固定开销**(子图构造、张量拼接、优化器
  调用),与批量大小无关。`batch_size=64` 时它占单步耗时的 **58%**。
  放大 batch **不破坏对比公平性** —— 每个 epoch 仍然恰好遍历一遍全部训练节点,
  变的只是梯度更新次数(698 → 175),而这个数会被记录下来。
  完整的 batch size 标定见 `results/bench_sampler*.json`。

### 7.5 GAT 在 Flickr 上需要更低的学习率

其余三个模型在 Flickr 上用 `lr=0.01`,**GAT 单独用 `lr=0.002`**。

原因不是调参偏好,而是一个会被早停掩盖的现象。实测四档学习率的验证曲线
(`python -m common.diag_gat`),**曲线形状完全相同**:先卡在多数类(0.42)
十几到二十几轮,然后才开始上升,区别只是**何时脱离平台期**:

| lr | 脱离平台期的轮次 | 40 轮后 |
|---|---|---|
| 0.01 | ~28 | 0.4803 |
| 0.005 | ~12 | 0.4975 |
| **0.002** | **~8** | **0.5068** |
| 0.001 | ~4 | 0.5047 |

`patience=15` 恰好落在 lr=0.01 的平台期中间(第 16 轮),于是早停在**模型开始
学习之前**就触发了 —— 表现为 `best_epoch=1`、测试准确率精确等于多数类占比
0.4234。这类失败不会报错,只会给出一个"看起来像结论"的错数字。

GCN 没有这个现象(标准化后验证集从第 0 轮就是 0.480,平滑上升)。差别在注意力:
GAT 对 500 维输入做 softmax 加权,初始 logits 方差偏大,需要更小的步长才能爬出来。

> **一般性教训**:早停的 `patience` 必须大于"模型可能原地不动的轮数"。
> 若验证指标在最初几轮最好、之后下降,先怀疑**早停掐断了恢复过程**,
> 而不要急着下"这个模型不行"的结论。

---

## 八、结论摘要

> 详细表格见 [`results/report_task1.md`](../results/report_task1.md)。

**1. 两种训练模式在精度上等价。** Cora 上采样平均高 +0.0044,但标准差是 ±0.0083;
Citeseer 上 GCN 反而低 0.0084。**所有差异都在一个标准差以内**,不构成"谁更好"的
证据。这个结论正是靠 5 个种子才看得出来的 —— 单次运行的话,+0.0044 看着就像个发现。

**2. 运行时间的差异取决于数据集,不是一个固定倍数。**

| 数据集 | 采样/全图的每 epoch 耗时 | 梯度步数倍数 |
|---|---|---|
| Cora | 0.9× ~ 1.3×(基本持平) | 3× |
| Citeseer | **0.5× ~ 0.6×(采样快一倍)** | 2× |
| Flickr | **14.9×(采样慢 15 倍)** | 175× |

Citeseer 上采样快,是因为它**特征 3703 维**(Cora 只有 1433):全图训练每层要乘整个
3327×3703 的特征矩阵,采样只碰子图内那几百个节点。Cora 上样本太少(140 个训练节点),
3 个批次就覆盖全图,采样省不出什么,还多付了子图构造的固定开销。

**3. 层数不是越多越好。** Cora 上 2 层普遍最优,加深后缓慢退化(过平滑)。
**GIN 是断崖式崩溃**:2 层 0.789 → 4 层 0.320(7 分类,已接近瞎猜),原因是 GIN 的
sum 聚合没有归一化,层数一多邻域规模的指数增长会让所有节点表示趋同。

**4. 学习率**:GCN / GAT / GIN 都是 0.01 最优,只有 GraphSAGE 偏好 0.005。

---

## 九、本任务未覆盖的部分

**如实说明,避免读者误以为矩阵是满的。**

### 9.1 Flickr 只完成了一部分(重要)

任务书要求使用 Cora、Citeseer、Flickr 三个数据集。**Flickr 的矩阵是残缺的:**

| 模型 | 全图训练 | 子图采样训练 |
|---|---|---|
| GCN | ✅ 2 个种子(0.5114) | ✅ 2 个种子(0.4979) |
| GraphSAGE | ✅ 2 个种子(0.5188) | ❌ 未跑 |
| GAT | ❌ 未跑 | ❌ 未跑 |
| GIN | ❌ 未跑 | ❌ 未跑 |

**只有 GCN 完成了两种模式的完整对比**,其余模型缺失。已有数字:

- Flickr 上的准确率(0.50~0.52)显著低于 Cora(0.81),符合"图越大越难"的预期
- GCN 的两种模式差距 −0.0135,大于 Cora/Citeseer 上的差距,但**只有 2 个种子**,
  不足以判断是真实效应还是噪声

原因是计算预算:Flickr 有 89,250 节点 / 899,756 边,单次采样训练的一个 epoch
就要 **9.8~11.1 秒**(Cora 采样只要 20~24 ms,**约 400~550 倍**)。跑满
4 模型 × 2 模式 × 2 种子在纯 CPU 上还需约 1.5 小时,超出本次作业的时间预算。

**保留的部分仍然能回答问题**:GCN 在两种模式下的完整对比已经足以说明"大图上
采样训练反而更慢"这一现象(见第八节),GraphSAGE 的全图结果也提供了第二个
模型在 Flickr 上的精度参考。**但四模型的横向对比在 Flickr 上是不完整的**,
不应据此得出"某个模型在 Flickr 上更好"的结论。

### 9.2 其他

- **Flickr 的超参扫描未做**。扫描只在 Cora 上进行,再把结论迁移到其他数据集
  (方案设计如此)。Cora 上单次运行仅数秒,而 Flickr 上仅一个组合就要数分钟,
  在纯 CPU 上做完整网格不可行。
- **扫描只用了 1 个种子**,而主实验用 5 个。因此报告表 2 的噪声大于表 1,
  只用于看趋势,不宜逐位比较。
- **未做多种子 × 多超参的联合网格**。原因是组合爆炸:4 模型 × 3 数据集 ×
  2 模式 × 3 lr × 3 层数 × 5 种子 = 1080 次运行。

