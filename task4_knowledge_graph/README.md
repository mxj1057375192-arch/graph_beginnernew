# 任务四 · 知识图谱补全

在 [kge_framework](https://github.com/Maxioo/kge_framework) 的基础上,支持
TransE / RotatE / ConvE 三种知识图谱补全模型。

> **本任务与任务一、二不同**:任务书明确说明任务四**不需要**做"全图训练 vs
> 分批次训练"的对比(任务要求第 3.2 条)。知识图谱补全天然按三元组 mini-batch
> 训练,不存在"全图"这个对应物。本任务的考点是**三个模型的实现与对比**。

---

## 一、环境

在任务一的环境基础上,另需 `scikit-learn`(框架用它算 AUC/AP)。

```bash
pip install scikit-learn
```

**纯 CPU 运行。** 框架原代码默认开启 CUDA,本仓库运行时统一不加 `-c` 参数。

---

## 二、目录结构

```
task4_knowledge_graph/
├── README.md                    # 本文件
├── code/
│   ├── run_experiments.py       # 批量实验入口(调用框架)
│   └── report.py                # 结果汇总
├── models/                      # 训练产物与日志
└── kge_framework/               # 老师提供的参考框架(见下)
    ├── code/                    # args / dataloader / main / model / utils
    └── data/                    # 8 个数据集,随框架提供
```

### 关于 kge_framework

**框架是老师提供的参考实现,本任务在其上做了两类改动:**

**(1) 兼容性修补**(不改任何训练逻辑,详见各文件内的 `[修补]` 注释):

| 文件 | 问题 | 现象 |
|---|---|---|
| `code/utils.py` | 日志文件名含冒号 | Windows 下 `OSError: [Errno 22]` 无法启动 |
| `code/dataloader.py` | 用了 `np.in1d` | NumPy 2.0 已移除该函数,`AttributeError` |
| `code/model.py` | 关系类型分桶越界 | 小数据集(关系数 < 4)评测时 `IndexError` |
| `code/main.py` | `'%d'` 打印浮点学习率 | 输出 `learning_rate = 0`,看着像训练失效 |

**(2) 新增两个模型**(任务四的实际工作量):

| 文件 | 新增内容 |
|---|---|
| `code/model.py` | `RotatE` 与 `ConvE` 两个类 |
| `code/main.py` | 修正 import,把 `ConvKB` 分支换成 `ConvE`,补 `else` 报错分支 |

新增的两个类**严格遵循框架原有的接口**,因此可以直接复用框架已有的负采样、
filtered MRR / HITS@K 评测、关系类型拆解等全部设施:

```python
__init__(model_name, nentity, nrelation, args)
loss(positive_score, negative_score)   -> 标量
forward(px, nx, py, ny)                -> 标量
predict(x)                             -> [N] 个得分,x 形状 [N, 3]
```

> 框架原 `main.py` 里本来就写了 `RotatE` 和 `ConvKB` 两个分支,但**这两个类在
> 框架里根本不存在** —— 一旦用 `-model RotatE` 启动就是 `NameError`。
> 这次补上了 RotatE 与 ConvE(ConvKB 不属于任务要求,未实现)。

---

## 三、数据集

框架自带 8 个数据集(`kge_framework/data/`)。本任务使用其中三个 countries 数据集:

| 数据集 | 实体 | 关系 | 训练 / 验证 / 测试 三元组 |
|---|---|---|---|
| countries_S1 | 271 | 2 | 1,111 / 24 / 24 |
| countries_S2 | 271 | 2 | 1,063 / 24 / 24 |
| countries_S3 | 271 | 2 | 985 / 24 / 24 |

**这组数据集很小**(测试集只有 24 条三元组),因此 MRR 的**噪声很大** ——
每条测试三元组的排名变化就值 1/24 ≈ 0.042。表中数字只用于**横向比较三个模型**,
不宜当作精确的性能刻画。

⚠️ 另外,三个数据集都**只有 2 个关系**,所以框架按关系类型拆分的
"N-N / N-1 / 1-N / 1-1"四个桶里大部分是空的 —— 这正是上面提到的 `IndexError`
的来源(原代码假定四个桶都有数据)。

### 缺失文件的补充

框架只提供了 `train.txt` / `valid.txt` / `test.txt`,但它的 `build_data` 需要
`entity2id.txt` 与 `relation2id.txt`。本仓库的
[`code/prepare_data.py`](../task4_knowledge_graph/code/) 负责生成它们:

```bash
python task4_knowledge_graph/code/prepare_data.py
```

注意原始 txt 里的格式是 `<id>\t<name>`,而框架读的是 `<name>\t<id>`,
脚本会做这个转换。已有这两个文件的数据集会跳过。

---

## 四、训练与测试脚本

### 训练(也是评测)

[`code/run_experiments.py`](code/run_experiments.py)

```bash
# 全部组合:3 数据集 × 3 模型
python task4_knowledge_graph/code/run_experiments.py

# 只跑一个数据集 / 一个模型
python task4_knowledge_graph/code/run_experiments.py \
       --datasets countries_S1 --models TransE

# 估时:只跑 3 个 epoch,不写结果
python task4_knowledge_graph/code/run_experiments.py --epochs 3 --no-record

# 打印将执行的命令而不运行
python task4_knowledge_graph/code/run_experiments.py --dry-run
```

也可以直接调用框架本体(单个模型、单个数据集):

```bash
cd task4_knowledge_graph/kge_framework/code
python main.py -model RotatE --data_path ../data/countries_S1 \
       --epochs 300 --batch_size 200 --hidden_dim 100 \
       -cpu 1 -save ../models/countries_S1_RotatE
```

### 测试

框架的评测与训练在同一个 `main()` 里:训练过程中按 `--valid_steps` 在验证集上
评测,取 **MRR 最高**的权重存为 `best/`,最后用该权重在测试集上评测一次
(`--do_test 1`,默认开启)。因此**训练脚本即评测脚本**,测试结果写在
`{save_path}/*test*.log` 里。

`run_experiments.py` 会从日志里把这些指标解析出来,写进 `results/runs.jsonl`。

**测试集不参与选优** —— 框架用验证集 MRR 挑权重(`if metrics['MRR'] > max_val_value`),
测试集只在最后评测一次。

### 结果汇总

```bash
python task4_knowledge_graph/code/report.py
# 输出到控制台,同时写入 results/report_task4.md
```

---

## 五、三个模型的原理与实现

### TransE(框架自带)

把关系看作**平移**:``h + r ≈ t``,打分 ``score = gamma - ||h + r - t||_1``。
表达能力强在 1-1 关系,但对**对称关系**无能为力 —— 若 r 是对称的,会强制 h = t。

### RotatE(本次实现)

把关系看作**复平面上的旋转**:``h ∘ r ≈ t``,其中 ``r`` 的模长固定为 1,
只保留相位。打分 ``score = gamma - ||h ∘ r - t||_1``。

实现要点:

- 复数按**实部/虚部拼接**成实数向量,所以实体维度是 ``2 * hidden_dim``,
  而关系只存 ``hidden_dim`` 个相位角。
- 关系的初始化**必须是 ``[-π, π]`` 上的均匀分布**,不能沿用 xavier ——
  关系存的是相位角,初始值超出这个范围没有意义。
- 相比 TransE,旋转可以表达**对称 / 反对称 / 逆关系 / 关系组合**四种模式,
  这是它相对 TransE 的主要优势。

### ConvE(本次实现)

把头实体与关系嵌入拼成 2D 矩阵后做**二维卷积**,再与尾实体做内积:

```
h [B, d] ─┐
          ├─ 各自 reshape 成 (H, W) 上下拼接 -> [B, 1, 2H, W]
r [B, d] ─┘
          └─ Conv2d -> BatchNorm -> ReLU -> Dropout -> Flatten -> Linear
             -> [B, d] 的特征;得分 = <特征, t> + bias[t]
```

与 TransE / RotatE 这类"距离型"打分不同,ConvE 是**相似度型**:卷积能捕捉头实体
与关系之间的二维局部交互,表达能力更强,代价是**参数量大得多**。

实现要点:

- 嵌入维度必须能分解为二维(建议取 10 的倍数,如 100 / 200),
  否则 `view` 会失败。构造函数里对这一点做了显式检查。
- 本实现按**逐三元组**打分(框架的 `test_step` 依赖这个接口),
  而不是 ConvE 论文里的 1-N 批量打分。数学上等价,只是不如 1-N 高效。
- 尾实体偏置 `entity_bias` 是论文的标准配置。

**注意**:在 countries 这种只有 271 个实体、2 个关系的小数据集上,ConvE 的
参数量优势换不来收益 —— 它需要更多数据才能体现卷积的价值。这一点会在结果里
体现出来。

---

## 六、CPU 上的性能调优

框架默认 `--cpu_num 10`,会开 10 个 DataLoader 工作进程。在只有一千多条三元组的
数据集上,**进程启动开销远大于实际计算** —— 实测同一组合从 203 秒降到 53 秒:

```bash
-cpu 1     # 本仓库的默认值
```

另外注意框架的训练长度由 `--epochs` 控制,**`--max_steps` 不生效**
(`main.py` 的循环上界是 `range(args.epochs)`)。框架默认 `--epochs 300`,
不显式设置会跑很久。

---

## 七、实验配置

| 参数 | 值 | 说明 |
|---|---|---|
| `--epochs` | 300 | |
| `--batch_size` | 200 | 训练集约 1,000 条 → 每 epoch 约 6 个批次 |
| `--hidden_dim` | 100 | ConvE 需要能分解为二维,100 = 10 × 10 |
| `--neg_ratio` | 8 | 每个正样本配 8 个负样本 |
| `--lr` | 0.001 | 框架默认 5e-6 过小,在千级三元组上几乎不移动 |
| `-cpu` | 1 | 见上一节 |
| `--test_batch_size` | 8 | 每个测试批次要对全部 271 个实体打分 |

---

## 八、结论摘要

> 详细表格见 [`results/report_task4.md`](../results/report_task4.md)。

### 8.1 结果总览(filtered MRR)

| 数据集 | TransE | RotatE | ConvE |
|---|---|---|---|
| countries_S1 | 0.5220 | **0.5283** | 0.2327 |
| countries_S2 | **0.4450** | 0.4400 | 0.1639 |
| countries_S3 | 0.1456 | **0.1731** | 0.0497 |

**TransE 与 RotatE 基本持平**(S1/S2 上差距都在 0.006 以内,S3 上 RotatE 领先)。

### 8.2 ⚠️ 先看噪声:测试集只有 24 条三元组

**在解读上面的数字之前必须知道**:三个数据集的测试集都只有 **24 条三元组**,
所以 MRR 的**最小分辨率是 1/24 ≈ 0.042**。

这意味着:

- S1 上 TransE 与 RotatE 的差距(0.006)约等于 **0.15 条三元组** —— 完全是噪声。
- S3 上两者的差距(0.0275)约等于 **0.66 条三元组** —— 同样不到一条。
- **不能下"RotatE 比 TransE 好"的结论。**

表中的数字只适合看**量级**和**明显差异**,不适合做精细比较。

### 8.3 ConvE 显著落后,原因能定量解释

ConvE 在三个数据集上都只有另外两个模型的一半不到。这不是实现错误,而是
**参数量与数据规模严重不匹配**:

| 模型 | 参数量 | 每个训练三元组分到的参数 |
|---|---|---|
| TransE | 54,200 | 48.8 |
| RotatE | 54,400 | 49.0 |
| **ConvE** | **488,855** | **440.0** |

在 `hidden_dim=100` 下,ConvE 的参数量是另外两个的 **9 倍**(主要来自卷积展平后的
全连接层,4608 × 100 ≈ 46 万)。而训练集只有 **985~1,111 条三元组** ——
每个训练样本要撑起 **440 个参数**,过拟合几乎是必然的。

ConvE 是为 FB15k-237(310,116 条三元组)那种规模设计的。**在千级数据集上,
它的容量优势反而变成了负担** —— 这是"模型选择要匹配数据规模"的一个直接例证。

此外还有一层结构性原因:ConvE 论文的原始训练协议是 **1-N 打分**(每个
(头,关系) 对同时与**全部**实体比较),而本框架的训练循环是"逐三元组 + 采样负样本"
(负样本比例 1:8)。1-N 协议下每个正样本能带来 271 倍的梯度信号,本框架下只有 8 倍。
**ConvE 在这里没有拿到它设计时假设的训练信号密度。**(此条为分析,未做对照实验验证。)

### 8.4 一个有趣的行为差异:RotatE 的排序更"两极"

| 数据集 | 模型 | HITS@1 | HITS@3 | HITS@10 |
|---|---|---|---|---|
| S1 | TransE | 0.2708 | **0.7292** | **0.8333** |
| S1 | RotatE | **0.4583** | 0.5000 | 0.7917 |
| S2 | TransE | 0.2083 | **0.6875** | **0.8125** |
| S2 | RotatE | **0.2917** | 0.4792 | 0.7292 |

**RotatE 的 HITS@1 明显更高,但 HITS@3 明显更低。** 两个模型的 MRR 几乎相同,
但**排序分布的形状完全不同**:RotatE 要么把正确答案排在第 1 位,要么排得很靠后;
TransE 更"中庸",正确答案常常落在第 2~3 位。

这与两者的打分机制吻合:RotatE 用**旋转**(角度匹配)来判断,是"对得上/对不上"
式的强判别;TransE 用**平移距离**,是连续的接近程度。MRR 这一个数字看不出
这种差异,HITS@K 的多档位才能暴露出来 —— 这也是报告里同时列出 @1/@3/@10 的原因。

---

## 八·补、实现过程中修正的一个关键问题

### RotatE 的初始化范围必须按 hidden_dim 反比缩放

初次实现时用了常规的初始化(`6/sqrt(dim)`),结果 RotatE 的 MRR 只有 **0.2886**,
不到 TransE(0.5237)的六成。原因:

RotatE 的打分是「**每维算模长、再对维度求和**」,所以距离的期望值**随维度线性增长**。
100 维下距离可达 30~80,而 `gamma` 固定为 9 —— 所有得分都是大负数、`logsigmoid`
饱和、梯度消失。

按 RotatE 官方实现改为 `embedding_range = (gamma + epsilon) / hidden_dim`
(epsilon=2 为官方常量)后,**MRR 从 0.2886 提升到 0.5460,反超 TransE**。

另外关系嵌入存的是**相位角**,初始化必须落在 `[-π, π]` 上,不能沿用 xavier ——
角度超出这个范围没有意义。

---

## 九、本任务未覆盖的部分

- **未使用 FB15k-237**。框架自带了 fb15k / fb15k-237 / wn18 / wn18rr / YAGO3-10
  等大型数据集,但 FB15k-237 有 **14,541 个实体、310,116 条三元组**,而框架的
  评测要对每条测试三元组给**全部实体**打分(14k × 2 方向),在纯 CPU 上评测一轮
  就要数十分钟,完整训练不可行。因此只使用三个 countries 数据集。
- **未实现 ConvKB**。框架 `main.py` 里有一个 `ConvKB` 分支,但任务书只要求
  TransE / RotatE / ConvE 三个模型,故未实现(分支已改为 ConvE)。
- **未做多种子重复**。这三个数据集的测试集只有 24 条三元组,多种子能降低的方差
  有限,而每个组合的训练成本不低。表中数字为单次运行结果。
