"""统一训练循环 —— 全图训练与子图采样训练共用同一份代码。

## 为什么必须共用

任务一、任务二的核心是"全图训练 vs 采样训练"的对比。要让这个对比成立,
两个模式**除了"数据怎么分批"之外,其余一切都必须相同**:优化器、学习率、
早停规则、epoch 上限、评测时机、线程数。

如果两边各写一套训练循环,就迟早会出现"一边用了学习率调度、另一边没有"
这类不对称,而这类偏差**不会报错**,只会让结论悄悄失真。共用一个函数是
把公平性变成结构性保证,而不是靠人记得。

## 两个模式的工作量口径

关键事实(容易搞错):

* 全图训练:1 个 epoch = **1 次梯度更新**,监督全部 N 个训练节点。
* 采样训练:1 个 epoch = ceil(N/B) 次梯度更新,每个批次 B 个种子节点。
  ``NeighborLoader(input_nodes=train_mask)`` 在一个 epoch 内会把全部训练
  节点各当一次种子,所以**遍历的监督量相同**,都是 N。

因此"跑相同的 epoch 数"在**监督量**上是对齐的,但在**梯度更新次数**上
并不对齐 —— 采样那边多做了 ceil(N/B) 倍更新,在 Flickr 上可达 87 倍。
这不是错误,而是采样训练的真实特性,但必须在结果里如实记录 ``n_steps``,
否则会把"优化次数多"误读成"采样方法好"。

本模块因此对每次 run 都记录 ``grad_steps_per_epoch`` 和 ``n_steps``。
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    """一次训练的完整结果。字段命名与 results.RunRecord 对齐。"""

    best_epoch: int = -1
    best_val: float = float("nan")
    test_at_best: float = float("nan")
    epochs_run: int = 0
    grad_steps_per_epoch: int = 0
    n_steps: int = 0  # 累计梯度更新次数
    n_node_visits: int = 0  # 累计被监督的节点次数
    sec_per_epoch: float = 0.0
    train_sec: float = 0.0
    total_sec: float = 0.0
    converged: bool = False  # 早停是否真正触发过
    status: str = "ok"
    curve: list[dict[str, float]] = field(default_factory=list)
    best_state: dict[str, Any] | None = None


class EarlyStopping:
    """基于验证指标的早停。

    指标统一约定为**越大越好**(准确率、AUC)。若指标是损失(越小越好),
    调用方应先取负号。
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.best_epoch = -1
        self.counter = 0
        self.improved_ever = False

    def step(self, value: float, epoch: int) -> bool:
        """传入本 epoch 的验证指标,返回 True 表示应停止。"""
        if value > self.best + self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
            self.improved_ever = True
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------
def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_batches: Callable[[], tuple[Iterable[Any], int]],
    step_fn: Callable[[nn.Module, Any], tuple[Tensor, int]],
    eval_fn: Callable[[nn.Module], dict[str, float]],
    *,
    max_epochs: int,
    patience: int,
    monitor: str,
    warmup_epochs: int = 1,
    grad_clip: float | None = None,
    verbose: bool = True,
    log_every: int = 10,
) -> TrainResult:
    """通用训练循环。

    Args:
        epoch_batches: 无参函数,每次调用返回 ``(批次可迭代对象, 本 epoch 的
            梯度步数)``。全图模式返回 ``([整图], 1)``,采样模式返回
            ``(loader, len(loader))``。**这是两个模式唯一的差异所在。**
        step_fn: ``(model, batch) -> (loss, 该批次的监督节点数)``。
        eval_fn: ``(model) -> {指标名: 值}``,在验证集上评测;必须是越大越好的
            指标。全图与采样模式使用**同一个** eval_fn,保证评测口径一致。
        monitor: 用哪个指标做早停与选优。
        warmup_epochs: 前若干 epoch 的耗时不计入统计。首个 epoch 包含惰性
            分配、内存池预热等一次性开销,直接计入会严重高估每 epoch 耗时。
        grad_clip: 梯度裁剪阈值(None 表示不裁剪)。

    Returns:
        TrainResult,其中 ``best_state`` 是验证指标最优时的权重副本。
    """
    stopper = EarlyStopping(patience)
    result = TrainResult()
    best_state: dict[str, Any] | None = None

    timer = time.perf_counter()
    timed_epochs = 0
    timed_sec = 0.0
    n_steps = 0
    n_visits = 0

    for epoch in range(max_epochs):
        model.train()
        batches, steps_this_epoch = epoch_batches()
        if epoch == 0:
            result.grad_steps_per_epoch = steps_this_epoch

        epoch_loss = 0.0
        epoch_visits = 0
        epoch_start = time.perf_counter()

        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss, n_sup = step_fn(model, batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += float(loss.detach())
            epoch_visits += n_sup
            n_steps += 1

        epoch_sec = time.perf_counter() - epoch_start

        # --- 评测(两个模式共用同一个 eval_fn)---
        metrics = eval_fn(model)
        val = metrics[monitor]

        n_visits += epoch_visits
        result.epochs_run = epoch + 1
        result.n_steps = n_steps
        result.n_node_visits = n_visits

        # 跳过 warmup 的计时
        if epoch >= warmup_epochs:
            timed_epochs += 1
            timed_sec += epoch_sec

        result.curve.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(steps_this_epoch, 1),
            "val": val,
            "epoch_sec": epoch_sec,
            "n_steps": n_steps,
            "n_node_visits": n_visits,
            **{f"val_{k}": v for k, v in metrics.items()},
        })

        # --- 早停与最优权重 ---
        if val > stopper.best:
            best_state = copy.deepcopy(model.state_dict())
            result.best_epoch = epoch
            result.best_val = val
            # 记录该权重下的完整指标(含测试集),但选优只看验证集
            result.test_at_best = float("nan")

        if stopper.step(val, epoch):
            if verbose:
                print(f"    早停于 epoch {epoch}(最佳 {stopper.best_epoch},"
                      f"{monitor}={stopper.best:.4f},patience={patience})")
            break

        if verbose and (epoch % log_every == 0 or epoch == max_epochs - 1):
            print(f"    epoch {epoch:4d}  loss={epoch_loss / max(steps_this_epoch, 1):.4f}"
                  f"  {monitor}={val:.4f}  {epoch_sec:.3f}s/epoch")

    result.total_sec = time.perf_counter() - timer
    result.train_sec = timed_sec
    result.sec_per_epoch = timed_sec / timed_epochs if timed_epochs else 0.0
    result.best_state = best_state

    # 早停真正触发过 => 收敛;跑满上限未触发 => 未收敛,如实标注
    result.converged = (stopper.counter >= patience) if patience > 0 else True
    if not result.converged:
        result.status = "max_epochs"

    return result


# ---------------------------------------------------------------------------
# 批次产出器:两个模式唯一的差异
# ---------------------------------------------------------------------------
def full_graph_batches(data: Any):
    """全图模式:每个 epoch 只有一个批次,就是整张图。

    返回一个 ``() -> (批次可迭代对象, 步数)`` 的工厂,与 run_training 约定一致。
    注意步数恒为 1 —— 这正是全图与采样在"梯度更新次数"上的差异来源。
    """

    def make():
        return [data], 1

    return make


def sampled_batches(loader_factory: Callable[[], Iterable[Any]]):
    """采样模式:每个 epoch 由 loader 产出多个子图批次。

    ``loader_factory`` 每次调用应返回一个**新的**迭代器(通常就是
    ``NeighborLoader`` 对象本身,它可重复迭代)。
    """

    def make():
        it = loader_factory()
        try:
            n = len(it)  # NeighborLoader 支持 __len__
        except TypeError:
            n = -1
        return it, n

    return make


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
@torch.no_grad()
def accuracy(logits: Tensor, y: Tensor, mask: Tensor) -> float:
    """节点/图级分类准确率。mask 为布尔掩码。"""
    pred = logits[mask].argmax(dim=-1)
    return float((pred == y[mask]).float().mean())


@torch.no_grad()
def node_classification_metrics(logits: Tensor, y: Tensor,
                                masks: dict[str, Tensor]) -> dict[str, float]:
    """一次算出多个划分上的准确率。"""
    out = {}
    for name, m in masks.items():
        if bool(m.any()):
            out[name] = accuracy(logits, y, m)
    return out
