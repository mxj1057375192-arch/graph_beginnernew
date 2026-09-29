"""可运行的实验单元:把"跑一次任务"封装成一个函数。

任务脚本(code/*.py)与冒烟测试都调用这里的函数,**保证两者跑的是同一套
流程** —— 否则冒烟测试通过、正式实验却失败,前期工作就白做了。

每个实验函数返回一个 :class:`common.results.RunRecord`,由调用方决定是否
落盘(``common.results.record``)。
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from common.datasets import load_node_dataset
from common.models import build_model, count_parameters
from common.results import RunRecord
from common.trainer import (
    full_graph_batches,
    node_classification_metrics,
    run_training,
)
from common.utils import set_seed


# ---------------------------------------------------------------------------
# 特征预处理
# ---------------------------------------------------------------------------
#: 每个数据集适用的特征处理方式。
#:
#: **这不是可有可无的细节,而是决定实验成败的一步。** Flickr 上曾因为漏做
#: 处理,模型连续 35 个 epoch 卡在多数类(0.424)上不动,早停于是在模型开始
#: 学习**之前**就把它掐断了 —— 表现为 ``best_epoch=0``、三个随机种子给出完全
#: 相同的准确率。实测对照(common/diag_flickr.py):
#:
#:     特征处理        前 5 轮 val              收敛后
#:     原始计数        0.424(卡住 35 轮)      0.4847
#:     行归一化        0.489                    0.4883
#:     列标准化        0.480(立刻正常)        0.5176
#:
#: Cora/Citeseer 是稀疏词袋向量,行归一化是文献标准做法;Flickr 是原始计数,
#: 量纲跨度 0~1827,列标准化能同时解决尺度与各维方差差异。
FEATURE_TREATMENT = {
    "Cora": "rownorm",
    "Citeseer": "rownorm",
    "Flickr": "standardize",
}


def _prepare_features(data: Any, dataset: str, verbose: bool = False) -> Any:
    """按数据集类型做特征预处理。两种训练模式使用**完全相同**的处理 —— 否则
    预处理差异会混进"模式对比"里,成为不外显的混淆因素。
    """
    how = FEATURE_TREATMENT.get(dataset, "none")
    if how == "none":
        return data

    if how == "rownorm":
        from torch_geometric.transforms import NormalizeFeatures

        out = NormalizeFeatures()(data)
        desc = "行归一化"
    elif how == "standardize":
        out = data.clone()
        mu = out.x.mean(dim=0, keepdim=True)
        sd = out.x.std(dim=0, keepdim=True).clamp(min=1e-9)
        out.x = (out.x - mu) / sd
        desc = "列标准化 (z-score)"
    else:
        raise ValueError(f"未知的特征处理方式:{how}")

    if verbose:
        print(f"    特征处理:{desc}({dataset})")
    return out


# ---------------------------------------------------------------------------
# 节点分类(任务一)
# ---------------------------------------------------------------------------
def run_node_classification(
    *,
    dataset: str,
    model_name: str,
    mode: str,
    seed: int = 0,
    num_layers: int = 2,
    hidden_channels: int = 64,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    max_epochs: int = 300,
    patience: int = 50,
    batch_size: int = 64,
    num_neighbors: tuple[int, ...] = (10, 10),
    task_dir: str = "task1",
    verbose: bool = True,
    normalize_features: bool = True,
    save_path: str | None = None,
) -> RunRecord:
    """跑一次节点分类实验,返回结果记录。

    Args:
        mode: ``'full'``(全图训练)或 ``'sampler'``(子图采样训练)。
        normalize_features: 是否对特征做行归一化。Cora/Citeseer 是词袋特征,
            必须归一化;两种模式必须**同样**应用,否则就是隐藏的不公平因素。
        num_neighbors: 每层的采样邻居数,长度需与 ``num_layers`` 一致。
        save_path: 若给出,把验证集最优权重存到这里,供 :func:`eval_node_classification`
            单独做测试用(对应任务书"训练脚本"与"测试脚本"分开的要求)。

    Note:
        两个模式共用同一个 ``eval_fn`` 与同一套早停规则,唯一差异是批次来源。
        这是对比公平性的结构性保证(见 common/trainer.py 的说明)。
    """
    set_seed(seed)

    data = load_node_dataset(task_dir, dataset)[0]
    if normalize_features:
        data = _prepare_features(data, dataset, verbose=verbose)

    masks = {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}
    num_classes = int(data.y.max()) + 1

    model = build_model(
        "node", model_name,
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=num_classes,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- 评测:两个模式共用 ---
    def eval_fn(m: torch.nn.Module) -> dict[str, float]:
        m.eval()
        with torch.no_grad():
            logits = m(data.x, data.edge_index)
        return node_classification_metrics(logits, data.y, masks)

    # --- 批次产生:两个模式唯一的差异 ---
    if mode == "full":
        epoch_batches = full_graph_batches(data)

        def step_fn(m: torch.nn.Module, batch: Any):
            logits = m(batch.x, batch.edge_index)
            loss = F.cross_entropy(logits[batch.train_mask], batch.y[batch.train_mask])
            return loss, int(batch.train_mask.sum())

    elif mode == "sampler":
        from torch_geometric.loader import NeighborLoader

        def epoch_batches():
            loader = NeighborLoader(
                data,
                num_neighbors=list(num_neighbors),
                batch_size=batch_size,
                input_nodes=data.train_mask,
                shuffle=True,
            )
            return loader, len(loader)

        def step_fn(m: torch.nn.Module, batch: Any):
            logits = m(batch.x, batch.edge_index)
            # 关键:NeighborLoader 的子图包含所有被采样到的节点,但只有前
            # batch_size 个是种子节点。必须切到种子,否则会:
            #   (1) 在非种子节点上也算损失,监督量口径与全图模式不一致
            #   (2) 让"每 epoch 遍历的监督量"虚高,破坏对比的可比性
            # 不切不会报错,只会静默地把数字算错 —— 这是本项目最容易踩的坑。
            logits = logits[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            return F.cross_entropy(logits, y), int(batch.batch_size)
    else:
        raise ValueError(f"未知训练模式:{mode}(可选 full/sampler)")

    if verbose:
        print(f"  [{dataset}/{model_name}/{mode}] 层数={num_layers} lr={lr} "
              f"hidden={hidden_channels} 参数量={count_parameters(model):,}")

    t0 = time.perf_counter()
    result = run_training(
        model, optimizer, epoch_batches, step_fn, eval_fn,
        max_epochs=max_epochs, patience=patience, monitor="val",
        grad_clip=5.0, verbose=verbose,
    )

    # 用最优权重在测试集上取最终指标(选优只看验证集,测试集仅用于报告)
    final = result.curve[result.best_epoch] if result.curve else {}

    # --- 存最优权重,供 test.py 单独评测 ---
    # 存的是**验证集最优**时的权重而非最后一个 epoch 的权重:测试集只用于
    # 报告,不能参与任何选择。这一点必须在测试脚本里也保持一致。
    if save_path and result.best_state is not None:
        from pathlib import Path

        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": result.best_state,
            "meta": {
                "task": "task1", "dataset": dataset, "model": model_name,
                "mode": mode, "seed": seed, "num_layers": num_layers,
                "hidden_channels": hidden_channels, "dropout": dropout,
                "num_classes": num_classes, "in_channels": int(data.num_features),
                "normalize_features": normalize_features,
                "best_epoch": result.best_epoch, "best_val": result.best_val,
            },
        }, p)
        if verbose:
            print(f"    最优权重已存至 {p}")

    if verbose:
        print(f"    -> best_epoch={result.best_epoch} val={result.best_val:.4f} "
              f"test={final.get('val_test', float('nan')):.4f} "
              f"steps/ep={result.grad_steps_per_epoch} {result.sec_per_epoch * 1000:.1f}ms/ep")

    return RunRecord(
        task="task1",
        dataset=dataset,
        model=model_name,
        mode=mode,
        seed=seed,
        num_layers=num_layers,
        lr=lr,
        hidden_dim=hidden_channels,
        extra_hparams={
            "weight_decay": weight_decay,
            "dropout": dropout,
            "batch_size": batch_size if mode == "sampler" else 0,
            "num_neighbors": list(num_neighbors) if mode == "sampler" else [],
            "max_epochs": max_epochs,
            "patience": patience,
            "params": count_parameters(model),
            "node_features": int(data.num_features),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
        },
        grad_steps_per_epoch=result.grad_steps_per_epoch,
        epochs_run=result.epochs_run,
        best_epoch=result.best_epoch,
        sec_per_epoch=result.sec_per_epoch,
        total_sec=result.total_sec,
        train_sec=result.train_sec,
        metrics={
            "val": result.best_val,
            "test": float(final.get("val_test", float("nan"))),
            "train": float(final.get("val_train", float("nan"))),
        },
        status="ok" if result.converged else "max_epochs",
        note="" if result.converged else f"跑满 {max_epochs} epoch 早停未触发,未收敛",
    )


# ---------------------------------------------------------------------------
# 链路预测(任务二)
# ---------------------------------------------------------------------------
@torch.no_grad()
def link_prediction_metrics(scores: Tensor, labels: Tensor,
                            ks: tuple[int, ...] = (1, 3, 10, 50)) -> dict[str, float]:
    """由正负边得分算 AUC / AP / MRR / Hits@K。

    排序口径:对每个正样本边,把它与**全部负样本边**的得分一起排名。
    这是链路预测的标准做法 —— 若只跟同批次内的少量负样本比,难度会被大幅
    低估,报出来的 MRR 会虚高。
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = labels.detach().cpu().numpy()
    s = scores.detach().cpu().numpy()
    out = {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
    }

    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    if pos.numel() and neg.numel():
        # 排名 = 得分不低于该正样本的负样本数 + 1(含自身)
        ranks = (neg.unsqueeze(0) >= pos.unsqueeze(1)).sum(dim=1).float() + 1.0
        out["mrr"] = float((1.0 / ranks).mean())
        for k in ks:
            out[f"hits@{k}"] = float((ranks <= k).float().mean())
    return out


def _supervision_edges(split: Any) -> tuple[Tensor, Tensor]:
    """取出一个划分里的监督边与标签。

    ``RandomLinkSplit`` 的字段名取决于 ``split_labels`` 参数:为 True 时验证/测试
    划分给出 ``pos_edge_label_index`` 与 ``neg_edge_label_index`` 两个字段,为
    False 时才给统一的 ``edge_label_index``/``edge_label``。两种都要能处理,
    否则换个参数就静默取到 None。
    """
    if getattr(split, "pos_edge_label_index", None) is not None:
        pos = split.pos_edge_label_index
        neg = split.neg_edge_label_index
        ei = torch.cat([pos, neg], dim=1)
        y = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))])
        return ei, y
    if getattr(split, "edge_label_index", None) is not None:
        return split.edge_label_index, split.edge_label.float()
    raise ValueError("该划分里既没有 pos/neg_edge_label_index 也没有 edge_label_index")


def run_link_prediction(
    *,
    dataset: str,
    model_name: str,
    mode: str,
    seed: int = 0,
    num_layers: int = 2,
    hidden_channels: int = 64,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    max_epochs: int = 300,
    patience: int = 50,
    batch_size: int = 64,
    num_neighbors: tuple[int, ...] = (10, 10),
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    task_dir: str = "task2",
    verbose: bool = True,
    save_path: str | None = None,
) -> RunRecord:
    """跑一次链路预测实验,全图训练与子图采样训练共用同一套流程。

    Args:
        mode: ``'full'`` 或 ``'sampler'``。

    Note:
        两种模式见到的**监督边完全相同**(同一份 ``edge_label_index`` /
        ``edge_label``),唯一差异是前向传播时用整图还是采样子图。若让采样模式
        每轮重新生成负样本、全图模式用固定负样本,两者的监督信号就不可比了。
    """
    from torch_geometric.transforms import RandomLinkSplit

    set_seed(seed)

    data = load_node_dataset(task_dir, dataset)[0]
    data = _prepare_features(data, dataset, verbose=verbose)

    # 无向图:RandomLinkSplit 必须先去掉重边,否则同一条边会在正负样本里各出现一次
    splitter = RandomLinkSplit(
        num_val=val_ratio, num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,   # 训练用固定的正+负样本,两模式共享
        neg_sampling_ratio=1.0,
        split_labels=False,                # 统一给 edge_label_index/edge_label
    )
    train_data, val_data, test_data = splitter(data)
    train_ei, train_el = _supervision_edges(train_data)

    model = build_model(
        "link", model_name,
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=hidden_channels,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- 评测:两个模式共用 ---
    @torch.no_grad()
    def _score_split(split) -> dict[str, float]:
        model.eval()
        ei, el = _supervision_edges(split)
        scores = model(split.x, split.edge_index, ei)
        return link_prediction_metrics(scores, el)

    def eval_fn(m: torch.nn.Module) -> dict[str, float]:
        v = _score_split(val_data)
        t = _score_split(test_data)
        # 早停以验证集 AUC 为准(monitor="auc")—— AUC 不受正负样本比例影响,
        # 比准确率稳健。trainer 会把这里返回的每个键再加 "val_" 前缀写进曲线,
        # 所以最终读的是 val_test_auc / val_val_ap 这类名字。
        out = {"auc": v["auc"]}
        out.update({f"val_{k}": x for k, x in v.items()})
        out.update({f"test_{k}": x for k, x in t.items()})
        return out

    def step_fn(m: torch.nn.Module, batch: Any):
        ei = batch.edge_label_index
        y = batch.edge_label.float()
        scores = m(batch.x, batch.edge_index, ei)
        loss = F.binary_cross_entropy_with_logits(scores, y)
        return loss, int(y.numel())

    if mode == "full":
        def epoch_batches():
            # 整图 + 全部监督边一次算完,梯度更新 1 次/epoch
            return [_FullGraphBatch(train_data, train_ei, train_el)], 1
    elif mode == "sampler":
        from torch_geometric.loader import LinkNeighborLoader

        def epoch_batches():
            loader = LinkNeighborLoader(
                train_data,
                num_neighbors=list(num_neighbors),
                batch_size=batch_size,
                edge_label_index=train_ei,
                edge_label=train_el,
                neg_sampling_ratio=0.0,  # 用已有的固定负样本,不每轮重采
                shuffle=True,
            )
            return loader, len(loader)
    else:
        raise ValueError(f"未知训练模式:{mode}(可选 full/sampler)")

    if verbose:
        n_pos = int((train_el > 0.5).sum())
        print(f"  [{dataset}/{model_name}/{mode}] 层数={num_layers} lr={lr} "
              f"hidden={hidden_channels} 参数量={count_parameters(model):,}")
        print(f"    监督边 训练 {train_ei.size(1):,}(正 {n_pos:,} / 负 "
              f"{train_ei.size(1) - n_pos:,})")

    t0 = time.perf_counter()
    result = run_training(
        model, optimizer, epoch_batches, step_fn, eval_fn,
        max_epochs=max_epochs, patience=patience, monitor="auc",
        grad_clip=5.0, verbose=verbose,
    )
    final = result.curve[result.best_epoch] if result.curve else {}

    if verbose:
        print(f"    -> best_epoch={result.best_epoch} "
              f"val_auc={result.best_val:.4f} test_auc={final.get('val_test_auc', float('nan')):.4f} "
              f"steps/ep={result.grad_steps_per_epoch} {result.sec_per_epoch * 1000:.1f}ms/ep")

    # 存「验证集最优」权重,供 test.py 单独评测
    if save_path and result.best_state is not None:
        from pathlib import Path

        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": result.best_state,
            "meta": {
                "task": "task2", "dataset": dataset, "model": model_name,
                "mode": mode, "seed": seed, "num_layers": num_layers,
                "hidden_channels": hidden_channels, "dropout": dropout,
                "in_channels": int(data.num_features),
                "val_ratio": val_ratio, "test_ratio": test_ratio,
                "best_epoch": result.best_epoch, "best_val": result.best_val,
            },
        }, p)
        if verbose:
            print(f"    最优权重已存至 {p}")

    return RunRecord(
        task="task2", dataset=dataset, model=model_name, mode=mode, seed=seed,
        num_layers=num_layers, lr=lr, hidden_dim=hidden_channels,
        extra_hparams={
            "weight_decay": weight_decay, "dropout": dropout,
            "batch_size": batch_size if mode == "sampler" else 0,
            "num_neighbors": list(num_neighbors) if mode == "sampler" else [],
            "max_epochs": max_epochs, "patience": patience,
            "params": count_parameters(model),
            "num_nodes": int(data.num_nodes), "num_edges": int(data.num_edges),
            "n_train_edges": int(train_ei.size(1)),
            "n_val_edges": int(_supervision_edges(val_data)[0].size(1)),
            "n_test_edges": int(_supervision_edges(test_data)[0].size(1)),
        },
        grad_steps_per_epoch=result.grad_steps_per_epoch,
        epochs_run=result.epochs_run, best_epoch=result.best_epoch,
        sec_per_epoch=result.sec_per_epoch,
        total_sec=result.total_sec, train_sec=result.train_sec,
        metrics={
            "val": result.best_val,
            "test": float(final.get("val_test_auc", float("nan"))),
            "test_ap": float(final.get("val_test_ap", float("nan"))),
            "test_mrr": float(final.get("val_test_mrr", float("nan"))),
            "test_hits@10": float(final.get("val_test_hits@10", float("nan"))),
            "test_hits@50": float(final.get("val_test_hits@50", float("nan"))),
            "metric": "AUC",
        },
        status="ok" if result.converged else "max_epochs",
        note="" if result.converged else f"跑满 {max_epochs} epoch 早停未触发,未收敛",
    )


class _FullGraphBatch:
    """把整图与全部监督边伪装成一个"批次",让全图模式复用同一套 step_fn。

    这样两个模式的 ``step_fn`` 可以写成完全相同的代码,差异只剩"批次从哪来",
    与任务一的做法保持一致。
    """

    __slots__ = ("x", "edge_index", "edge_label_index", "edge_label")

    def __init__(self, data: Any, ei: Tensor, el: Tensor) -> None:
        self.x = data.x
        self.edge_index = data.edge_index
        self.edge_label_index = ei
        self.edge_label = el


def eval_link_prediction(
    *, checkpoint: str, task_dir: str = "task2", verbose: bool = True
) -> RunRecord:
    """加载训练好的权重,只做测试(任务二的"测试脚本"入口)。

    必须用**与训练时完全相同**的划分参数重建 ``RandomLinkSplit``(种子、比例),
    否则测试边集与训练时不是同一批,数字没有意义。这些参数都存在 checkpoint 里。
    """
    from torch_geometric.transforms import RandomLinkSplit

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]

    set_seed(meta["seed"])
    data = load_node_dataset(task_dir, meta["dataset"])[0]
    data = _prepare_features(data, meta["dataset"])

    splitter = RandomLinkSplit(
        num_val=meta["val_ratio"], num_test=meta["test_ratio"],
        is_undirected=True, add_negative_train_samples=True,
        neg_sampling_ratio=1.0, split_labels=False,
    )
    _, val_data, test_data = splitter(data)

    model = build_model(
        "link", meta["model"], in_channels=meta["in_channels"],
        hidden_channels=meta["hidden_channels"], num_layers=meta["num_layers"],
        out_channels=meta["hidden_channels"], dropout=meta["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out: dict[str, float] = {}
    with torch.no_grad():
        for name, split in (("val", val_data), ("test", test_data)):
            ei, el = _supervision_edges(split)
            out.update({f"{name}_{k}": v for k, v in
                        link_prediction_metrics(model(split.x, split.edge_index, ei), el).items()})

    if verbose:
        print(f"  [{meta['dataset']}/{meta['model']}/{meta['mode']}] 载入 {checkpoint}")
        print(f"    训练时最佳 epoch={meta['best_epoch']} val_auc={meta['best_val']:.4f}")
        print(f"    测试集 AUC={out['test_auc']:.4f}  AP={out['test_ap']:.4f}  "
              f"MRR={out['test_mrr']:.4f}  Hits@10={out['test_hits@10']:.4f}")

    return RunRecord(
        task="task2", dataset=meta["dataset"], model=meta["model"], mode=meta["mode"],
        seed=meta["seed"], num_layers=meta["num_layers"], lr=0.0,
        hidden_dim=meta["hidden_channels"], tag="test",
        extra_hparams={"checkpoint": str(checkpoint),
                       "params": count_parameters(model)},
        metrics={"val": out["val_auc"], "test": out["test_auc"],
                 "test_ap": out["test_ap"], "test_mrr": out["test_mrr"],
                 "test_hits@10": out["test_hits@10"],
                 "test_hits@50": out["test_hits@50"], "metric": "AUC"},
        epochs_run=0, best_epoch=meta["best_epoch"],
        status="ok", note=f"由 {checkpoint} 载入权重评测",
    )


def eval_node_classification(
    *,
    checkpoint: str,
    task_dir: str = "task1",
    verbose: bool = True,
) -> RunRecord:
    """加载已训练权重,只做评测 —— 对应任务书要求的"测试脚本"。

    与 :func:`run_node_classification` 的关系:后者训练 + 评测一起做,本函数
    **不训练**,直接读权重跑测试集。两者用同一套评测口径(同一份 eval_fn),
    所以结果可以互相印证。

    注意这里评测的是**验证集最优**时的权重(存的时候就已经选好了),测试集
    不参与任何选择 —— 否则测试集就变成了验证集,报出来的数字会偏乐观。
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]

    set_seed(meta["seed"])
    data = load_node_dataset(task_dir, meta["dataset"])[0]
    if meta.get("normalize_features"):
        # 必须与训练时用**同一个**函数,否则测试看到的输入分布与训练不一致,
        # 权重再对也没用
        data = _prepare_features(data, meta["dataset"])

    model = build_model(
        "node", meta["model"],
        in_channels=meta["in_channels"],
        hidden_channels=meta["hidden_channels"],
        num_layers=meta["num_layers"],
        out_channels=meta["num_classes"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    masks = {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    m = node_classification_metrics(logits, data.y, masks)

    if verbose:
        print(f"  [{meta['dataset']}/{meta['model']}/{meta['mode']}] "
              f"载入 {checkpoint}")
        print(f"    训练时最佳 epoch={meta['best_epoch']} "
              f"val={meta['best_val']:.4f}")
        print(f"    测试集准确率 = {m['test']:.4f}"
              f"  (val={m['val']:.4f} train={m['train']:.4f})")

    return RunRecord(
        task="task1", dataset=meta["dataset"], model=meta["model"],
        mode=meta["mode"], seed=meta["seed"], num_layers=meta["num_layers"],
        lr=0.0, hidden_dim=meta["hidden_channels"], tag="test",
        extra_hparams={"checkpoint": str(checkpoint),
                       "params": count_parameters(model)},
        metrics={"test": m["test"], "val": m["val"], "train": m["train"],
                 "metric": "accuracy"},
        epochs_run=0, best_epoch=meta["best_epoch"],
        status="ok", note=f"由 {checkpoint} 载入权重评测",
    )


# ---------------------------------------------------------------------------
# 图分类 / 图回归(任务三)
# ---------------------------------------------------------------------------
TU_DATASETS = ("MUTAG", "PROTEINS", "ENZYMES")


def _tu_split(n: int, seed: int, frac=(0.8, 0.1, 0.1)) -> dict[str, list[int]]:
    """给 TUDataset 做 80/10/10 随机划分。

    TUDataset **自带没有官方划分**,必须自己切。用带种子的生成器而非全局随机
    状态,否则同一份代码在不同调用顺序下会切出不同的集合,实验不可复现。
    """
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_train = int(n * frac[0])
    n_val = int(n * frac[1])
    return {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }


def run_graph_classification(
    *,
    dataset: str,
    model_name: str,
    pooling: str = "mean",
    seed: int = 0,
    num_layers: int = 3,
    hidden_channels: int = 64,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    max_epochs: int = 200,
    patience: int = 50,
    batch_size: int = 64,
    task_dir: str = "task3",
    verbose: bool = True,
    save_path: str | None = None,
) -> RunRecord:
    """跑一次图级任务(MUTAG / PROTEINS / ENZYMES 分类,ZINC 回归)。

    与任务一不同,这里**没有全图/采样两种模式** —— 图级任务天然就是按
    mini-batch 训练的(每个批次若干张独立的小图)。所以本函数只负责
    池化方式的对比,那是任务三的核心考点。

    Args:
        pooling: ``'mean'`` / ``'max'`` / ``'min'``。任务三要对比三者。
        dataset: ZINC 时自动切到**回归**模式:单标量输出、MAE 评价、
            使用官方划分而非随机切分。
    """
    from torch_geometric.loader import DataLoader as GraphLoader

    set_seed(seed)
    regression = dataset == "ZINC"

    # --- 取数据 ---
    # 标签的形状问题**不在数据集上就地修改**,而是在用到时压平(见 step_fn /
    # _eval_split)。原因有两个,都是 PyG 的 InMemoryDataset 直接访问内部存储
    # `.data` 引起的:
    #   1. 会发 UserWarning(输出里混进源码行,很难看);
    #   2. 更实质的是 —— `ds[v]` 切出的子集,其 `.data` 仍指向**完整数据集**,
    #      在子集上写 `s.data.y = ...` 实际改的是全集。当前对 TU 数据集无害
    #      (y 本来就是 [N]),但一旦将来需要改子集标签就会改错对象,且不报错。
    if regression:
        from common.datasets import load_zinc

        splits = {s: load_zinc(task_dir, subset=True, split=s)
                  for s in ("train", "val", "test")}
        num_classes = 1
        batch_size = 128 if batch_size == 64 else batch_size
    else:
        from common.datasets import load_tu

        ds = load_tu(task_dir, dataset)
        num_classes = int(ds.y.max()) + 1
        idx = _tu_split(len(ds), seed)
        splits = {k: ds[v] for k, v in idx.items()}

    loaders = {
        k: GraphLoader(v, batch_size=batch_size,
                       shuffle=(k == "train"))
        for k, v in splits.items()
    }

    sample = splits["train"][0]
    in_channels = int(sample.num_features)

    # --- ZINC 的特征必须走 embedding ---
    # ZINC 的 x 是 [N,1] 的**原子类型编号**(int64),不是可直接使用的连续特征,
    # 直接喂进卷积会出两种后果:
    #
    #   GCN / GAT / GraphSAGE —— GCNConv 内部 gcn_norm(..., dtype=x.dtype) 取到
    #       int64,度数张量成了 Long,deg.pow_(-0.5) 抛 RuntimeError。属于
    #       "响亮地失败",容易发现。
    #   GIN —— 不报错。GINConv.forward 里有 `out + (1 + self.eps) * x_r`,而
    #       eps 是 float,按 PyTorch 的类型提升规则会把 Long **静默提升**为
    #       Float。于是模型把"原子类型编号"当成数值大小在使用(碳=0、氮=1…),
    #       而编号顺序本身是任意的、无度量含义 —— 结果无意义却不报错。
    #
    # 后者才是真正需要防的一类错误,也是冒烟矩阵这次挖到的最有价值的一条。
    num_embeddings = None
    if regression and not sample.x.is_floating_point():
        # 用 `s.x`(PyG 推荐的堆叠属性访问)而不是 `s.data.x` —— 后者直接读
        # 内部存储,会发 UserWarning
        num_embeddings = int(max(int(s.x.max()) for s in splits.values())) + 1

    # 回归用 task="graph_reg":build_model 会据此把输出维度钉成 1 并让任务头
    # 不加激活。**不能**把 regression=True 直接透传 —— 它会顺着 **gnn_kwargs
    # 一路流进 GNNEncoder → MessagePassing.__init__,报
    # "unexpected keyword argument 'regression'"。
    model = build_model(
        "graph_reg" if regression else "graph", model_name,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=num_classes,
        dropout=dropout,
        pooling=pooling,
        num_embeddings=num_embeddings,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def _eval_split(loader) -> float:
        model.eval()
        total, n = 0.0, 0
        for b in loader:
            out = model(b.x, b.edge_index, b.batch)
            # TUDataset 的 y 有时是 [1]、有时是 [1,1];ZINC 的 y 是浮点回归目标。
            # 统一在批次上压平,不在数据集上就地改标签(见上面的说明)。
            y = b.y.view(-1)
            if regression:
                total += float(F.l1_loss(out, y.float(), reduction="sum"))
            else:
                total += int((out.argmax(-1) == y.long()).sum())
            n += b.num_graphs
        if regression:
            # 早停约定"越大越好",而 MAE 越小越好 —— 取负号。记录时会翻回正数。
            return -total / max(n, 1)
        return total / max(n, 1)

    def eval_fn(m: torch.nn.Module) -> dict[str, float]:
        return {k: _eval_split(ld) for k, ld in loaders.items()}

    def epoch_batches():
        return loaders["train"], len(loaders["train"])

    def step_fn(m: torch.nn.Module, batch: Any):
        out = m(batch.x, batch.edge_index, batch.batch)
        y = batch.y.view(-1)  # 同上:在批次上压平,不就地改数据集
        if regression:
            loss = F.l1_loss(out, y.float())
        else:
            loss = F.cross_entropy(out, y.long())
        return loss, int(batch.num_graphs)

    if verbose:
        print(f"  [{dataset}/{model_name}/{pooling}] 层数={num_layers} lr={lr} "
              f"hidden={hidden_channels} 参数量={count_parameters(model):,}"
              f"{'  [回归/MAE]' if regression else ''}")

    t0 = time.perf_counter()
    result = run_training(
        model, optimizer, epoch_batches, step_fn, eval_fn,
        max_epochs=max_epochs, patience=patience, monitor="val",
        warmup_epochs=0,  # 图级任务每 epoch 本身就是多批次,不存在"单批次预热"
        grad_clip=5.0, verbose=verbose,
    )

    final = result.curve[result.best_epoch] if result.curve else {}
    sign = -1.0 if regression else 1.0  # 回归的验证指标是负 MAE,翻回正数

    # 存「验证集最优」权重,供 test.py 单独评测
    if save_path and result.best_state is not None:
        from pathlib import Path

        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": result.best_state,
            "meta": {
                "task": "task3", "dataset": dataset, "model": model_name,
                "pooling": pooling, "seed": seed, "regression": regression,
                "num_layers": num_layers, "hidden_channels": hidden_channels,
                "dropout": dropout, "num_classes": num_classes,
                "in_channels": in_channels, "num_embeddings": num_embeddings,
                "batch_size": batch_size,
                "best_epoch": result.best_epoch, "best_val": result.best_val,
            },
        }, p)
        if verbose:
            print(f"    最优权重已存至 {p}")

    if verbose:
        unit = "MAE" if regression else "acc"
        print(f"    -> best_epoch={result.best_epoch} val_{unit}={sign * result.best_val:.4f} "
              f"test_{unit}={sign * final.get('val_test', float('nan')):.4f} "
              f"{result.sec_per_epoch * 1000:.1f}ms/ep")

    n_train = len(splits["train"])
    return RunRecord(
        task="task3",
        dataset=dataset,
        model=model_name,
        mode=pooling,  # 图级任务的"模式"维度就是池化方式
        seed=seed,
        num_layers=num_layers,
        lr=lr,
        hidden_dim=hidden_channels,
        extra_hparams={
            "pooling": pooling,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "params": count_parameters(model),
            "regression": regression,
            "n_train_graphs": n_train,
            "n_val_graphs": len(splits["val"]),
            "n_test_graphs": len(splits["test"]),
            "in_channels": in_channels,
            "num_classes": num_classes,
        },
        grad_steps_per_epoch=result.grad_steps_per_epoch,
        epochs_run=result.epochs_run,
        best_epoch=result.best_epoch,
        sec_per_epoch=result.sec_per_epoch,
        total_sec=result.total_sec,
        train_sec=result.train_sec,
        metrics={
            "val": sign * result.best_val,
            "test": sign * float(final.get("val_test", float("nan"))),
            "train": sign * float(final.get("val_train", float("nan"))),
            "metric": "MAE" if regression else "accuracy",
        },
        status="ok" if result.converged else "max_epochs",
        note="" if result.converged else f"跑满 {max_epochs} epoch 早停未触发,未收敛",
    )


def eval_graph_classification(
    *, checkpoint: str, task_dir: str = "task3", verbose: bool = True
) -> RunRecord:
    """加载训练好的权重,只做测试(任务三的"测试脚本"入口)。

    TUDataset 的划分是**随机切的**(`_tu_split`),所以必须用存进 checkpoint 的
    种子重新切出同一份划分,否则评测的图集与训练时不是同一批。ZINC 用官方划分,
    不依赖种子。
    """
    from torch_geometric.loader import DataLoader as GraphLoader

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    regression = bool(meta["regression"])

    set_seed(meta["seed"])
    # 与 run_graph_classification 保持完全一致:不在数据集上就地改标签
    if regression:
        from common.datasets import load_zinc

        splits = {s: load_zinc(task_dir, subset=True, split=s)
                  for s in ("train", "val", "test")}
    else:
        from common.datasets import load_tu

        ds = load_tu(task_dir, meta["dataset"])
        idx = _tu_split(len(ds), meta["seed"])
        splits = {k: ds[v] for k, v in idx.items()}

    model = build_model(
        "graph_reg" if regression else "graph", meta["model"],
        in_channels=meta["in_channels"], hidden_channels=meta["hidden_channels"],
        num_layers=meta["num_layers"], out_channels=meta["num_classes"],
        dropout=meta["dropout"], pooling=meta["pooling"],
        num_embeddings=meta.get("num_embeddings"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out: dict[str, float] = {}
    with torch.no_grad():
        for name, ds in splits.items():
            loader = GraphLoader(ds, batch_size=meta["batch_size"], shuffle=False)
            total, n = 0.0, 0
            for b in loader:
                pred = model(b.x, b.edge_index, b.batch)
                y = b.y.view(-1)
                if regression:
                    total += float(F.l1_loss(pred, y.float(), reduction="sum"))
                else:
                    total += int((pred.argmax(-1) == y.long()).sum())
                n += b.num_graphs
            out[name] = total / max(n, 1)

    unit = "MAE" if regression else "accuracy"
    if verbose:
        print(f"  [{meta['dataset']}/{meta['model']}/{meta['pooling']}] 载入 {checkpoint}")
        print(f"    训练时最佳 epoch={meta['best_epoch']}")
        print(f"    测试集 {unit} = {out['test']:.4f}"
              f"  (val={out['val']:.4f} train={out['train']:.4f})")

    return RunRecord(
        task="task3", dataset=meta["dataset"], model=meta["model"],
        mode=meta["pooling"], seed=meta["seed"], num_layers=meta["num_layers"],
        lr=0.0, hidden_dim=meta["hidden_channels"], tag="test",
        extra_hparams={"checkpoint": str(checkpoint),
                       "pooling": meta["pooling"], "regression": regression,
                       "params": count_parameters(model)},
        metrics={"val": out["val"], "test": out["test"], "train": out["train"],
                 "metric": unit},
        epochs_run=0, best_epoch=meta["best_epoch"],
        status="ok", note=f"由 {checkpoint} 载入权重评测",
    )
