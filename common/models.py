"""GCN / GAT / GraphSAGE / GIN 的统一封装。

结构上分成两部分,这样四种模型在三个任务里可以复用同一套代码:

    GNNEncoder           把节点特征编码成节点嵌入(不含池化)
      ├─ NodeClassifier  节点分类    : 嵌入 -> 逐节点线性头
      ├─ LinkPredictor   链路预测    : 嵌入 -> 两端点积
      └─ GraphClassifier 图分类/回归 : 嵌入 -> 池化 -> 图级线性头

关于归一化层的选择(影响实验有效性,不是随意决定):
    本模块用 LayerNorm 而非 BatchNorm。BatchNorm 的统计量依赖批次构成 ——
    全图训练时统计的是整张图,采样训练时统计的是每个子图,于是**同一个模型
    在两种训练模式下的行为并不一致**。任务一、任务二的核心就是比较这两种
    模式,若用 BatchNorm,测得的差异里会混入"归一化统计量不同"这一因素,
    对比就不再干净。LayerNorm 逐节点独立计算,两种模式下行为一致。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

GNN_MODELS = ("GCN", "GAT", "GraphSAGE", "GIN")


# ---------------------------------------------------------------------------
# 池化
# ---------------------------------------------------------------------------
def pool_nodes(h: Tensor, batch: Tensor, mode: str) -> Tensor:
    """把节点嵌入聚合成图级表示。

    Args:
        h: ``[N, D]`` 节点嵌入。
        batch: ``[N]`` 每个节点所属的图编号。
        mode: ``'mean'`` / ``'max'`` / ``'min'`` / ``'sum'``。

    Note:
        PyG **没有** ``global_min_pool``(只有 mean/max/add)。最小值池化用
        ``-global_max_pool(-h)`` 等价实现。
    """
    if mode == "mean":
        return global_mean_pool(h, batch)
    if mode == "max":
        return global_max_pool(h, batch)
    if mode == "min":
        # 取负 -> 最大池化 -> 再取负 = 最小池化
        return -global_max_pool(-h, batch)
    if mode == "sum":
        return global_add_pool(h, batch)
    raise ValueError(f"未知池化方式:{mode}(可选 mean/max/min/sum)")


POOLING_MODES = ("mean", "max", "min")


# ---------------------------------------------------------------------------
# 编码器
# ---------------------------------------------------------------------------
class GNNEncoder(nn.Module):
    """把节点特征编码为节点嵌入,不做任何池化。

    Args:
        name: ``'GCN'`` / ``'GAT'`` / ``'GraphSAGE'`` / ``'GIN'``。
        in_channels: 输入节点特征维度。
        hidden_channels: 隐藏层维度。
        num_layers: 层数(>=1)。层数是对比实验的一个自变量。
        dropout: dropout 比例。
        num_embeddings: 若不为 None,则在最前面加一个节点特征 embedding 层。
            ZINC 的节点特征是原子类型编号(单维整数),**不能直接当数值用**,
            必须先 embedding;此时 ``in_channels`` 被忽略。
        gnn_kwargs: 透传给底层卷积的额外参数。
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.5,
        num_embeddings: int | None = None,
        **gnn_kwargs,
    ) -> None:
        super().__init__()
        if name not in GNN_MODELS:
            raise ValueError(f"未知模型 {name},可选:{GNN_MODELS}")
        if num_layers < 1:
            raise ValueError("num_layers 至少为 1")

        self.name = name
        self.num_layers = num_layers
        self.dropout = dropout

        # 可选的输入 embedding(ZINC 这类节点特征为离散编号的数据集需要)
        if num_embeddings is not None:
            self.node_emb = nn.Embedding(num_embeddings, hidden_channels)
            first_in = hidden_channels
        else:
            self.node_emb = None
            first_in = in_channels

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            cin = first_in if i == 0 else hidden_channels
            self.convs.append(self._make_conv(name, cin, hidden_channels, **gnn_kwargs))
            self.norms.append(nn.LayerNorm(hidden_channels))

    @staticmethod
    def _make_conv(name: str, cin: int, cout: int, **kwargs) -> nn.Module:
        if name == "GCN":
            return GCNConv(cin, cout, **kwargs)
        if name == "GraphSAGE":
            return SAGEConv(cin, cout, **kwargs)
        if name == "GAT":
            # 多头但 concat=False,保证输出维度仍是 cout,便于各模型统一对比。
            # heads 可通过 gnn_kwargs 覆盖。
            kwargs.setdefault("heads", 4)
            kwargs.setdefault("concat", False)
            return GATConv(cin, cout, **kwargs)
        if name == "GIN":
            # GIN 的卷积核是一个 MLP(作用在"自身 + 邻居求和"上)
            mlp = nn.Sequential(
                nn.Linear(cin, cout),
                nn.ReLU(),
                nn.Linear(cout, cout),
            )
            return GINConv(mlp, **kwargs)
        raise ValueError(name)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.node_emb is not None:
            # ZINC 的 x 形如 [N, 1],存的是原子类型编号
            x = self.node_emb(x.view(-1).long())
        elif not x.is_floating_point():
            # 整数特征直接进卷积时,各个卷积核的失败方式**并不一致**:
            #   GCN   —— gcn_norm(..., dtype=x.dtype) 取到 int64,deg.pow_(-0.5) 抛错
            #   GAT / GraphSAGE —— 线性层 matmul 抛错
            #   GIN   —— **不抛错**。GINConv.forward 的 `out + (1+eps)*x_r` 里
            #            eps 是 float,类型提升规则会把整数静默转成浮点,于是
            #            模型把"类别编号"当成数值大小在用,结果无意义。
            # 最后一种不报错却算错,所以在这里统一拦下,不给它静默通过的机会。
            raise TypeError(
                f"节点特征 dtype 是 {x.dtype}(整数),但没有配置 embedding 层。"
                f"若这些整数是类别编号(如 ZINC 的原子类型),请通过 num_embeddings "
                f"走 embedding;若是数值特征,请先 .float()。"
            )

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            # 最后一层后不再 dropout,是常见做法
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    @property
    def out_channels(self) -> int:
        return self.norms[0].normalized_shape[0]


# ---------------------------------------------------------------------------
# 任务头
# ---------------------------------------------------------------------------
class NodeClassifier(nn.Module):
    """节点分类:嵌入 -> 逐节点线性分类头。

    GIN 在节点级任务上同样不接池化层 —— 池化只属于图级任务。
    """

    def __init__(self, encoder: GNNEncoder, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.head = nn.Linear(encoder.out_channels, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self.encoder(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)


class GraphClassifier(nn.Module):
    """图分类 / 图回归:嵌入 -> 池化 -> 图级线性头。

    Args:
        out_channels: 分类任务为类别数;回归任务(ZINC)为 1。
        pooling: ``'mean'`` / ``'max'`` / ``'min'``。任务三要对比不同池化的影响。
        regression: True 时输出标量且不加任何激活(ZINC 用 MAE 评价)。
    """

    def __init__(
        self,
        encoder: GNNEncoder,
        out_channels: int,
        pooling: str = "mean",
        dropout: float = 0.5,
        regression: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"未知池化方式:{pooling}")
        self.encoder = encoder
        self.pooling = pooling
        self.regression = regression
        self.dropout = dropout
        self.head = nn.Linear(encoder.out_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        h = self.encoder(x, edge_index)
        h = pool_nodes(h, batch, self.pooling)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.head(h)
        if self.regression:
            return out.view(-1)
        return out


class LinkPredictor(nn.Module):
    """链路预测:嵌入 -> 两端点积打分。

    用点积解码器是链路预测的标准做法,也便于把"编码器"和"解码器"解耦 ——
    这样不同 GNN 编码器的对比不会被解码器差异污染。
    """

    def __init__(self, encoder: GNNEncoder, dropout: float = 0.5) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor) -> Tensor:
        h = self.encoder(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        src, dst = edge_label_index
        # 点积:得分越高越可能是真实边
        return (h[src] * h[dst]).sum(dim=-1)


# ---------------------------------------------------------------------------
# 便捷构造
# ---------------------------------------------------------------------------
def build_model(
    task: str,
    name: str,
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
    out_channels: int,
    dropout: float = 0.5,
    pooling: str = "mean",
    num_embeddings: int | None = None,
    **gnn_kwargs,
) -> nn.Module:
    """按任务类型装配 编码器 + 合适的任务头。

    Args:
        task: ``'node'`` / ``'link'`` / ``'graph'`` / ``'graph_reg'``。
    """
    encoder = GNNEncoder(
        name=name,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        num_embeddings=num_embeddings,
        **gnn_kwargs,
    )
    if task == "node":
        return NodeClassifier(encoder, out_channels, dropout=dropout)
    if task == "link":
        return LinkPredictor(encoder, dropout=dropout)
    if task == "graph":
        return GraphClassifier(encoder, out_channels, pooling=pooling, dropout=dropout)
    if task == "graph_reg":
        return GraphClassifier(
            encoder, out_channels=1, pooling=pooling, dropout=dropout, regression=True
        )
    raise ValueError(f"未知任务类型:{task}")


def count_parameters(model: nn.Module) -> int:
    """可训练参数量 —— 报告里应给出,便于说明对比是否在同等模型容量下进行。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
