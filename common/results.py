"""实验结果落盘与汇总。

设计原则:每次 run 追加一行 JSON 到 results/runs.jsonl,报告的表格与曲线
由脚本从这些原始记录汇总生成 —— 不手工抄数字。

这样做的三个好处:
  1. 可复现:每个数字都能追溯到产生它的那次 run
  2. 可增量:跑完一个组合就落一次盘,中断不丢结果
  3. 可对比:汇总时按 数据集/模型/模式 分组,自动算 mean ± std

用法::

    from common.results import RunRecord, record, Stopwatch

    with Stopwatch() as sw:
        ...训练...
        record(RunRecord(
            task="task1", dataset="Cora", model="GCN", mode="full",
            seed=0, num_layers=2, lr=0.01, hidden_dim=64,
            epochs_run=200, best_epoch=137, grad_steps_per_epoch=1,
            sec_per_epoch=sw.lap(), total_sec=sw.elapsed,
            metrics={"acc": 0.812, "f1": 0.805},
        ))
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RUNS_FILE = RESULTS_DIR / "runs.jsonl"


# --------------------------------------------------------------------------
# 计时
# --------------------------------------------------------------------------
class Stopwatch:
    """累计式秒表。可反复 ``lap()`` 取分段耗时,``elapsed`` 取总耗时。

    用作上下文管理器时,退出后 ``elapsed`` 即为总耗时。
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.elapsed = 0.0
        self.laps: list[float] = []

    def lap(self) -> float:
        """返回距上次 lap 的秒数,并累计到 laps。"""
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        self.laps.append(dt)
        return dt

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self._t0
        return self.elapsed

    def __enter__(self) -> "Stopwatch":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


# --------------------------------------------------------------------------
# 单次 run 的记录
# --------------------------------------------------------------------------
@dataclass
class RunRecord:
    """一次训练 run 的完整记录。

    字段分为四类:标识(是哪次实验)、超参、耗时、指标。
    ``status`` 用于诚实标注未能正常完成的 run —— 见下方说明。
    """

    # --- 标识 ---
    task: str  # 'task1' | 'task2' | 'task3' | 'task4'
    dataset: str
    model: str
    mode: str  # 'full'(全图) | 'sampler'(子图采样)
    seed: int

    # --- 超参 ---
    num_layers: int
    lr: float
    hidden_dim: int = 0
    # 实验分组标记,例如 'main' / 'sweep_lr' / 'sweep_layers'。
    # 必须显式区分:超参扫描和主实验都会在 Cora 上、用同一批种子产生记录,
    # 若不加标记,汇总时两者会混进同一个分组,均值就是两个不同实验的平均,
    # 而且**不会报错**。字段放在这里是为了满足 dataclass 的默认值顺序约束。
    tag: str = "main"
    extra_hparams: dict[str, Any] = field(default_factory=dict)

    # --- 训练规模与耗时 ---
    # grad_steps_per_epoch 必须记录:全图训练 1 epoch = 1 次梯度更新,
    # 采样训练 1 epoch = 几百次。不写明这一点,"epoch 数"的对比就是误导。
    grad_steps_per_epoch: int = 0
    epochs_run: int = 0
    best_epoch: int = -1
    sec_per_epoch: float = 0.0
    total_sec: float = 0.0
    train_sec: float = 0.0  # 纯训练耗时,不含数据加载/评测

    # --- 指标 ---
    metrics: dict[str, float] = field(default_factory=dict)

    # --- 状态 ---
    # 'ok'      正常完成
    # 'timeout' 超出预算被中止(如实记录,不编造数字)
    # 'diverged' 训练发散(loss 变 NaN 等)
    # 'failed'  抛异常
    status: str = "ok"
    note: str = ""

    timestamp: float = field(default_factory=time.time)
    cuda: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# --------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------
def record(rec: RunRecord, path: Path | None = None) -> None:
    """把一次 run 追加写入 runs.jsonl(UTF-8,每行一条)。"""
    path = path or RUNS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(rec.to_json() + "\n")


def load_runs(path: Path | None = None) -> list[dict[str, Any]]:
    """读回全部 run 记录。文件不存在时返回空列表。"""
    path = path or RUNS_FILE
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------
# 延迟导入 pandas:仅在做汇总时才需要它,训练脚本不需要付出导入代价。
def to_dataframe(path: Path | None = None):
    import pandas as pd

    return pd.DataFrame(load_runs(path))


def summarize(
    by: tuple[str, ...] = ("task", "dataset", "model", "mode"),
    metric_keys: tuple[str, ...] = (),
    path: Path | None = None,
    include_status: tuple[str, ...] = ("ok",),
) -> "Any":
    """按 ``by`` 分组汇总,数值指标算 mean / std / count。

    Args:
        by: 分组键,例如 ``("dataset", "model", "mode")``。
        metric_keys: 需要展开成列的 metric 名,例如 ``("acc", "f1")``。
        include_status: 只统计这些状态的 run。默认排除 timeout/failed,
            **但被排除的数量会出现在结果的 ``n_excluded`` 列里** —— 不能
            让失败的 run 静默消失,那等于选择性报告。
    """
    import pandas as pd

    df = to_dataframe(path)
    if df.empty:
        return df

    if metric_keys:
        for k in metric_keys:
            df[k] = df["metrics"].apply(lambda d, k=k: d.get(k))

    total = len(df)
    df = df[df["status"].isin(include_status)]
    n_excluded = total - len(df)

    value_cols = list(metric_keys) + [
        c for c in ("total_sec", "sec_per_epoch", "grad_steps_per_epoch", "best_epoch")
        if c in df.columns
    ]

    grouped = df.groupby(list(by), dropna=False)[value_cols].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
    grouped["n_excluded"] = n_excluded
    return grouped.reset_index()


# --------------------------------------------------------------------------
# 环境快照
# --------------------------------------------------------------------------
def environment_snapshot() -> dict[str, Any]:
    """记录运行环境,便于报告里说明软硬件条件(以及日后复现)。"""
    snap: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["cuda_available"] = torch.cuda.is_available()
        snap["num_threads"] = torch.get_num_threads()
    except ImportError:
        snap["torch"] = None
    try:
        import torch_geometric

        snap["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        snap["torch_geometric"] = None
    try:
        import numpy

        snap["numpy"] = numpy.__version__
    except ImportError:
        snap["numpy"] = None
    return snap


if __name__ == "__main__":
    print(json.dumps(environment_snapshot(), indent=2, ensure_ascii=False))
