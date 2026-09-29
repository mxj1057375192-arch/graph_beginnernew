"""任务二:链路预测 —— 训练与评测入口。

用法示例::

    # 单次运行
    python code/train.py --dataset Cora --model GCN --mode full

    # 主对比实验:全部数据集 × 4 模型 × 全图/采样 × 多种子
    python code/train.py --all --seeds 3

    # 训练并保存「验证集最优」权重,供 test.py 评测
    python code/train.py --all --save-model models

结果追加写入仓库根目录的 results/runs.jsonl,报告表格由
``python code/report.py`` 自动生成,不手工抄数字。
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

from common.datasets import NODE_DATASETS  # noqa: E402
from common.experiments import run_link_prediction  # noqa: E402
from common.models import GNN_MODELS  # noqa: E402
from common.results import RunRecord, record  # noqa: E402
from common.utils import banner, fix_console_encoding, human_time, setup  # noqa: E402

TASK = "task2"
MODES = ("full", "sampler")

#: 默认数据集。**不含 Flickr** —— 链路预测要在 89.9 万条边上做监督,单次运行
#: 在纯 CPU 上要数小时。本任务只覆盖 Cora 与 Citeseer,理由已写进 README。
DEFAULT_DATASETS = ("Cora", "Citeseer")


def run_one(dataset: str, model_name: str, mode: str, seed: int,
            args: argparse.Namespace, tag: str = "single") -> RunRecord:
    """跑一次并落盘。异常不吞掉,如实记为 failed 后继续后续组合。"""
    save_path = None
    if args.save_model:
        save_path = str(Path(args.save_model)
                        / f"{dataset}_{model_name}_{mode}_seed{seed}.pt")
    try:
        rec = run_link_prediction(
            dataset=dataset, model_name=model_name, mode=mode, seed=seed,
            num_layers=args.layers, hidden_channels=args.hidden, lr=args.lr,
            weight_decay=args.weight_decay, dropout=args.dropout,
            max_epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, num_neighbors=tuple(args.neighbors),
            val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            task_dir=TASK, verbose=args.verbose, save_path=save_path,
        )
        rec.tag = tag
    except Exception as e:  # noqa: BLE001
        print(f"    ✗ 失败:{type(e).__name__}: {e}")
        if args.traceback:
            traceback.print_exc()
        rec = RunRecord(
            task=TASK, dataset=dataset, model=model_name, mode=mode, seed=seed,
            num_layers=args.layers, lr=args.lr, hidden_dim=args.hidden,
            tag=tag, status="failed", note=f"{type(e).__name__}: {e}",
        )
    record(rec)
    return rec


def run_matrix(datasets, models, modes, seeds, args, tag: str = "main") -> None:
    cells = [(d, m, md, s) for d in datasets for m in models
             for md in modes for s in seeds]
    print(f"\n共 {len(cells)} 个 run"
          f"(数据集 {len(datasets)} × 模型 {len(models)}"
          f" × 模式 {len(modes)} × 种子 {len(seeds)})")

    t0 = time.perf_counter()
    failed = 0
    for i, (d, m, md, s) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {d}/{m}/{md}/seed={s}", flush=True)
        rec = run_one(d, m, md, s, args, tag=tag)
        if rec.status == "failed":
            failed += 1

    elapsed = time.perf_counter() - t0
    print(f"\n完成 {len(cells) - failed}/{len(cells)},耗时 {human_time(elapsed)}")
    if failed:
        print(f"  有 {failed} 个 run 失败 —— 详见 results/runs.jsonl 中 status=failed 的记录")
    print("  汇总:python code/report.py")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="任务二:链路预测(GCN/GAT/GraphSAGE/GIN,全图 vs 子图采样)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="Cora", choices=NODE_DATASETS)
    p.add_argument("--model", default="GCN", choices=GNN_MODELS)
    p.add_argument("--mode", default="full", choices=MODES)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--all", action="store_true", help="跑主对比实验")
    p.add_argument("--datasets", nargs="+", default=None, choices=NODE_DATASETS,
                   help=f"--all 时只跑这些数据集(默认 {list(DEFAULT_DATASETS)})")
    p.add_argument("--seeds", type=int, default=3)

    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=30)
    # 批大小默认 256 而非 PyG 示例常用的 64。实测 Cora 上 64 会让采样模式慢到
    # 1886 ms/epoch(132 步 × 14 ms,其中很大一部分是每批的固定开销),
    # 200 epoch × 4 模型 × 3 种子 × 2 数据集要跑约 3 小时。放大到 256 后步数
    # 降到 33,总耗时降一个数量级。
    # **这不破坏对比公平性**:两种模式监督的边集完全相同,每个 epoch 仍然把
    # 全部 8,448 条监督边各用一次,变的只是梯度更新次数(132 → 33),
    # 而这个数会记进结果表并对读者透明。
    p.add_argument("--batch-size", type=int, default=256,
                   help="仅采样模式:每批监督边数")
    p.add_argument("--neighbors", type=int, nargs="+", default=[10, 10])
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)

    p.add_argument("--save-model", default=None, metavar="DIR",
                   help="把每个组合「验证集最优」的权重存到该目录")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", dest="verbose", action="store_false")
    p.add_argument("--traceback", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    fix_console_encoding()
    args = build_parser().parse_args(argv)

    if len(args.neighbors) != args.layers:
        args.neighbors = [args.neighbors[0]] * args.layers

    setup(seed=args.seed, num_threads=args.threads)
    banner(f"任务二 · 链路预测   [{args.dataset} / {args.model} / {args.mode}]")

    if args.all:
        run_matrix(args.datasets or list(DEFAULT_DATASETS), list(GNN_MODELS),
                   list(MODES), list(range(args.seeds)), args)
    else:
        run_one(args.dataset, args.model, args.mode, args.seed, args)
        print("\n  汇总:python code/report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
