"""任务一:节点分类 —— 训练与评测入口。

用法示例::

    # 单次运行
    python code/train.py --dataset Cora --model GCN --mode full

    # 主对比实验:3 数据集 × 4 模型 × 2 模式 × 多种子
    python code/train.py --all

    # 超参扫描(报告需要分析 lr 与层数的影响)
    python code/train.py --sweep lr
    python code/train.py --sweep layers

    # 冒烟自检:每个组合只跑 3 个 epoch
    python code/train.py --smoke

结果追加写入仓库根目录的 results/runs.jsonl,报告的表格由
``python code/report.py`` 自动生成,不手工抄数字。
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

# 让本脚本无论从哪个目录启动都能 import 到 common/
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.datasets import NODE_DATASETS  # noqa: E402
from common.experiments import run_node_classification  # noqa: E402
from common.models import GNN_MODELS  # noqa: E402
from common.results import RunRecord, record  # noqa: E402
from common.utils import banner, fix_console_encoding, human_time, setup  # noqa: E402

TASK = "task1"
MODES = ("full", "sampler")


# ---------------------------------------------------------------------------
# 单次运行
# ---------------------------------------------------------------------------
def run_one(
    dataset: str, model_name: str, mode: str, seed: int, args: argparse.Namespace,
    tag: str = "single",
) -> RunRecord:
    """跑一次并落盘。异常不吞掉,如实记为 failed 后继续后续组合。

    ``tag`` 把不同来源的 run 区分开 —— 它们都会在 Cora 上、用相同的种子产生
    记录,不区分就会在汇总时被平均到一起。三种取值:

    * ``main``   —— `--all` 主实验
    * ``sweep_*`` —— 超参扫描
    * ``single`` —— 单独跑一个组合(§默认值)

    **默认值是 ``single`` 而不是 ``main``**:临时跑一次做验证是常事,若默认
    归入 main,它就会静默混进主实验的均值里。本项目实际发生过 —— 一次
    `--save-model` 的验证运行让 Cora/GCN/全图的"5 个种子"变成了 6 条记录。
    """
    # --save-model 给的是目录:矩阵模式下每个组合各存一份,互不覆盖
    save_path = None
    if args.save_model:
        save_path = str(Path(args.save_model)
                        / f"{dataset}_{model_name}_{mode}_seed{seed}.pt")

    try:
        rec = run_node_classification(
            dataset=dataset,
            model_name=model_name,
            mode=mode,
            seed=seed,
            num_layers=args.layers,
            hidden_channels=args.hidden,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            num_neighbors=tuple(args.neighbors),
            task_dir=TASK,
            verbose=args.verbose,
            save_path=save_path,
        )
        rec.tag = tag
    except Exception as e:  # noqa: BLE001
        # 任何组合失败都要如实记录,不能让它在汇总时静默消失
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


# ---------------------------------------------------------------------------
# 实验矩阵
# ---------------------------------------------------------------------------
def run_matrix(
    datasets: list[str], models: list[str], modes: list[str],
    seeds: list[int], args: argparse.Namespace, tag: str = "main",
) -> None:
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


# ---------------------------------------------------------------------------
# 超参扫描
# ---------------------------------------------------------------------------
def run_sweep(which: str, args: argparse.Namespace) -> None:
    """超参扫描。

    按方案的设计,**只在 Cora 上扫描,再把每个模型的最优设置迁移到其他数据集**。
    完整网格(模型 × 数据集 × 模式 × lr × 层数 × 种子)在上千个 run 量级,
    在 CPU 上不可行;而 Cora 上单次运行仅数秒,扫描代价可忽略。
    """
    sweep_dataset = args.sweep_dataset
    print(f"\n超参扫描:{which}(仅在 {sweep_dataset} 上,结果将迁移到其他数据集)")

    if which == "lr":
        values = args.lr_grid
        for lr in values:
            args.lr = lr
            print(f"\n--- lr = {lr} ---")
            run_matrix([sweep_dataset], list(GNN_MODELS), list(MODES), [args.seed],
                       args, tag=f"sweep_{which}")
    elif which == "layers":
        for nl in args.layer_grid:
            args.layers = nl
            # 采样层数必须与 num_neighbors 长度一致
            args.neighbors = [args.neighbors[0]] * nl
            print(f"\n--- 层数 = {nl}(采样邻居数同步为 {args.neighbors})---")
            run_matrix([sweep_dataset], list(GNN_MODELS), list(MODES), [args.seed],
                       args, tag=f"sweep_{which}")
    else:
        raise ValueError(f"未知的扫描类型:{which}(可选 lr / layers)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="任务一:节点分类(GCN/GAT/GraphSAGE/GIN,全图 vs 子图采样)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 选择跑什么
    p.add_argument("--dataset", default="Cora", choices=NODE_DATASETS)
    p.add_argument("--model", default="GCN", choices=GNN_MODELS)
    p.add_argument("--mode", default="full", choices=MODES)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--all", action="store_true",
                   help="跑主对比实验:全部数据集 × 全部模型 × 两种模式")
    p.add_argument("--datasets", nargs="+", default=None, choices=NODE_DATASETS,
                   help="--all 时只跑这些数据集。用于把耗时差几个数量级的"
                        "数据集分开跑(如先跑完 Cora/Citeseer 再单独跑 Flickr)")
    p.add_argument("--models", nargs="+", default=None, choices=GNN_MODELS,
                   help="--all 时只跑这些模型。用于给个别模型单独指定学习率 —— "
                        "见 README 中 GAT 在 Flickr 上需要更低 lr 的说明")
    p.add_argument("--seeds", type=int, default=5,
                   help="--all 时每个组合重复的种子数(应对 Cora 140 节点划分的高方差)")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟自检:每个组合只跑 3 个 epoch")
    p.add_argument("--sweep", choices=("lr", "layers"), default=None,
                   help="超参扫描")
    p.add_argument("--sweep-dataset", default="Cora")

    # 超参
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--neighbors", type=int, nargs="+", default=[10, 10],
                   help="每层采样邻居数,长度须与 --layers 一致")

    p.add_argument("--lr-grid", type=float, nargs="+",
                   default=[0.01, 0.005, 0.001])
    p.add_argument("--layer-grid", type=int, nargs="+", default=[2, 3, 4])

    p.add_argument("--save-model", default=None, metavar="DIR",
                   help="把每个组合「验证集最优」的权重存到该目录,"
                        "供 test.py 单独评测")
    p.add_argument("--threads", type=int, default=None,
                   help="CPU 线程数。两种训练模式必须一致,否则运行时对比无效")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", dest="verbose", action="store_false")
    p.add_argument("--traceback", action="store_true",
                   help="失败时打印完整堆栈(调试用)")
    return p


def main(argv: list[str] | None = None) -> int:
    fix_console_encoding()
    args = build_parser().parse_args(argv)

    # 层数与采样邻居数必须匹配,否则 NeighborLoader 会报错
    if len(args.neighbors) != args.layers:
        args.neighbors = [args.neighbors[0]] * args.layers

    setup(seed=args.seed, num_threads=args.threads)

    banner(f"任务一 · 节点分类   [{args.dataset} / {args.model} / {args.mode}]")

    if args.smoke:
        args.epochs, args.patience = 3, 10_000
        run_matrix(NODE_DATASETS, list(GNN_MODELS), list(MODES), [0], args)
    elif args.sweep:
        run_sweep(args.sweep, args)
    elif args.all:
        run_matrix(args.datasets or list(NODE_DATASETS),
                   args.models or list(GNN_MODELS),
                   list(MODES), list(range(args.seeds)), args)
    else:
        run_one(args.dataset, args.model, args.mode, args.seed, args)
        print("\n  汇总:python code/report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
