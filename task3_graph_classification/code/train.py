"""任务三:图分类 / 图回归 —— 训练与评测入口。

与任务一、二不同,图级任务**没有"全图 vs 采样"的对比** —— 数据本身就是一堆
独立的小图,天然按 mini-batch 训练。本任务的核心考点是**池化方式**:

* AvgPooling(``mean``)
* MaxPooling(``max``)
* MinPooling(``min``)

以及 ZINC 上的**回归**任务(指标是 MAE,不是准确率)。

用法示例::

    # 单次运行
    python code/train.py --dataset MUTAG --model GCN --pooling mean

    # 主实验:全部数据集 × 4 模型 × 3 池化
    python code/train.py --all --seeds 3

    # ZINC 单独跑(比另外三个数据集慢两个数量级)
    python code/train.py --all --datasets ZINC --seeds 1 --epochs 60

    # 训练并保存「验证集最优」权重
    python code/train.py --all --save-model models
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.experiments import run_graph_classification  # noqa: E402
from common.models import GNN_MODELS, POOLING_MODES  # noqa: E402
from common.results import RunRecord, record  # noqa: E402
from common.utils import banner, fix_console_encoding, human_time, setup  # noqa: E402

TASK = "task3"

#: 分类数据集(TUDataset)与回归数据集(ZINC)。两者评价指标不同,分开列。
TU_DATASETS = ("MUTAG", "PROTEINS", "ENZYMES")
REG_DATASETS = ("ZINC",)
ALL_DATASETS = TU_DATASETS + REG_DATASETS

#: 各数据集默认 epoch 上限。ZINC 是 1 万张图、每 epoch 79 个批次,比 TU 数据集
#: 慢两个数量级,纯 CPU 上必须压轮次。
EPOCHS_BY_DATASET = {"ZINC": 60}
DEFAULT_EPOCHS = 120


def run_one(dataset: str, model_name: str, pooling: str, seed: int,
            args: argparse.Namespace, tag: str = "single") -> RunRecord:
    save_path = None
    if args.save_model:
        save_path = str(Path(args.save_model)
                        / f"{dataset}_{model_name}_{pooling}_seed{seed}.pt")
    try:
        rec = run_graph_classification(
            dataset=dataset, model_name=model_name, pooling=pooling, seed=seed,
            num_layers=args.layers, hidden_channels=args.hidden, lr=args.lr,
            weight_decay=args.weight_decay, dropout=args.dropout,
            max_epochs=args.epochs or EPOCHS_BY_DATASET.get(dataset, DEFAULT_EPOCHS),
            patience=args.patience, batch_size=args.batch_size,
            task_dir=TASK, verbose=args.verbose, save_path=save_path,
        )
        rec.tag = tag
    except Exception as e:  # noqa: BLE001
        print(f"    ✗ 失败:{type(e).__name__}: {e}")
        if args.traceback:
            traceback.print_exc()
        rec = RunRecord(
            task=TASK, dataset=dataset, model=model_name, mode=pooling, seed=seed,
            num_layers=args.layers, lr=args.lr, hidden_dim=args.hidden,
            tag=tag, status="failed", note=f"{type(e).__name__}: {e}",
        )
    record(rec)
    return rec


def run_matrix(datasets, models, poolings, seeds, args, tag: str = "main") -> None:
    cells = [(d, m, p, s) for d in datasets for m in models
             for p in poolings for s in seeds]
    print(f"\n共 {len(cells)} 个 run"
          f"(数据集 {len(datasets)} × 模型 {len(models)}"
          f" × 池化 {len(poolings)} × 种子 {len(seeds)})")

    t0 = time.perf_counter()
    failed = 0
    for i, (d, m, p, s) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {d}/{m}/{p}/seed={s}", flush=True)
        rec = run_one(d, m, p, s, args, tag=tag)
        if rec.status == "failed":
            failed += 1

    elapsed = time.perf_counter() - t0
    print(f"\n完成 {len(cells) - failed}/{len(cells)},耗时 {human_time(elapsed)}")
    if failed:
        print(f"  有 {failed} 个 run 失败 —— 详见 results/runs.jsonl 中 status=failed 的记录")
    print("  汇总:python code/report.py")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="任务三:图分类/图回归(4 模型 × 3 池化;ZINC 为回归)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="MUTAG", choices=ALL_DATASETS)
    p.add_argument("--model", default="GCN", choices=GNN_MODELS)
    p.add_argument("--pooling", default="mean", choices=POOLING_MODES)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--all", action="store_true", help="跑主实验")
    p.add_argument("--datasets", nargs="+", default=None, choices=ALL_DATASETS)
    p.add_argument("--seeds", type=int, default=3)

    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=0,
                   help="0 表示按数据集取默认值(ZINC=%d,其余=%d)"
                        % (EPOCHS_BY_DATASET["ZINC"], DEFAULT_EPOCHS))
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64,
                   help="每批图数。ZINC 会自动提到 128")

    p.add_argument("--save-model", default=None, metavar="DIR")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", dest="verbose", action="store_false")
    p.add_argument("--traceback", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    fix_console_encoding()
    args = build_parser().parse_args(argv)

    setup(seed=args.seed, num_threads=args.threads)
    banner(f"任务三 · 图分类   [{args.dataset} / {args.model} / {args.pooling}]")

    if args.all:
        run_matrix(args.datasets or list(ALL_DATASETS), list(GNN_MODELS),
                   list(POOLING_MODES), list(range(args.seeds)), args)
    else:
        run_one(args.dataset, args.model, args.pooling, args.seed, args)
        print("\n  汇总:python code/report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
