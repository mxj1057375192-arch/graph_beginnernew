"""任务四:知识图谱补全 —— 批量运行实验并汇总结果。

## 为什么用子进程而不是 import

kge_framework 的 `main.py` 是一个整体脚本:读数据、建模型、训练、评测、写日志
全在一个 `main()` 里。与其把它拆开(改动大、容易引入偏差),不如**原样调用** ——
每个组合起一个独立进程,跑完从它自己的日志里把指标读出来。

好处是框架代码保持原样(除了必要的 Windows / NumPy 2.0 兼容性修补),
实验结果可追溯到一条完整的日志。

## 指标来源

框架用 `log_metrics('Test', step, metrics)` 把 filtered MRR / MR / HITS@1/3/10
写进日志文件,格式固定::

    Test MRR at step 19: 0.014445

本脚本用正则从日志里取这些值,写进仓库根的 ``results/runs.jsonl``,
与任务一、二、三的结果并排,报告的表格由 `report.py` 统一生成。

用法::

    # 全部组合
    python code/run_experiments.py

    # 只跑一个数据集 / 一个模型
    python code/run_experiments.py --datasets countries_S1 --models TransE

    # 先估时间:每个组合只跑 3 个 epoch
    python code/run_experiments.py --epochs 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
# CODE_DIR 就是 code/ 本身,往上一层是 task4_knowledge_graph/,再上一层才是仓库根。
# 写成 parents[2] 会退到仓库根之外(上层目录),import common 直接失败。
REPO_ROOT = CODE_DIR.parents[1]
FRAMEWORK_CODE = CODE_DIR.parent / "kge_framework" / "code"
FRAMEWORK_DATA = CODE_DIR.parent / "kge_framework" / "data"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.results import RunRecord, record  # noqa: E402
from common.utils import banner, fix_console_encoding, human_time  # noqa: E402

TASK = "task4"

DEFAULT_DATASETS = ("countries_S1", "countries_S2", "countries_S3")
DEFAULT_MODELS = ("TransE", "RotatE", "ConvE")

#: 从日志里取顶层指标。注意要锚定行尾,否则 `MRR_N-N_head` 这类细分指标
#: 也会被 `MRR` 匹配到,把平均 MRR 覆盖成某个关系子类的值。
METRIC_RE = re.compile(
    r"Test (MRR|MR|HITS@\d+) at step (\d+): ([0-9.]+)\s*$", re.MULTILINE)


def count_triples(path: Path) -> int:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def parse_log_text(text: str) -> dict[str, float]:
    """从日志内容里取出最后一次 Test 评测的指标。"""
    out: dict[str, float] = {}
    last_step = -1
    for name, step, val in METRIC_RE.findall(text):
        # 只认最后一次评测(step 最大的那批),避免把 valid 阶段的残留读进来
        s = int(step)
        if s >= last_step:
            out[name] = float(val)
            last_step = s
    return out


def parse_log(log_path: Path) -> dict[str, float]:
    """从框架写的日志文件里取指标。文件不存在时返回空字典。"""
    try:
        return parse_log_text(log_path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return {}


def run_one(dataset: str, model: str, args: argparse.Namespace) -> RunRecord:
    data_path = FRAMEWORK_DATA / dataset
    # 必须用**绝对路径**:子进程的 cwd 是框架的 code/ 目录,相对路径会被框架
    # 按它的 cwd 解析,日志就写到别处去了,而本进程按自己的 cwd 去找 —— 结果是
    # 训练明明成功,却一个指标都读不出来。
    save_path = (Path(args.save_dir) / f"{dataset}_{model}").resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", "main.py",
        "-model", model,
        "--data_path", str(data_path),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--test_batch_size", str(args.test_batch_size),
        "--hidden_dim", str(args.hidden_dim),
        "--neg_ratio", str(args.neg_ratio),
        "-lr", str(args.lr),
        # 框架默认 cpu_num=10。数据集只有一千多条三元组时,10 个 DataLoader
        # 工作进程的启动开销远大于实际计算 —— 实测同一组合从 203s 降到 53s。
        "-cpu", str(args.cpu_num),
        "-save", str(save_path),
    ]
    if not args.cuda:
        cmd = [c for c in cmd]  # 不加 -c 即为 CPU

    print(f"  $ {' '.join(cmd[2:])}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(FRAMEWORK_CODE),
                          capture_output=True, text=True, errors="replace")
    elapsed = time.perf_counter() - t0

    logs = sorted(save_path.glob("*test*.log")) + sorted(save_path.glob("*train*.log"))
    metrics = parse_log(logs[-1]) if logs else {}
    if not metrics:
        # 兜底:框架的 logging 同时挂了一个 StreamHandler(输出到 stderr),
        # 所以即使日志文件没找到,也能从进程输出里把指标捞回来
        metrics = parse_log_text(proc.stderr or "") or parse_log_text(proc.stdout or "")

    if proc.returncode != 0 or not metrics:
        tail = (proc.stderr or proc.stdout or "")[-400:]
        print(f"    ✗ 失败(returncode={proc.returncode})")
        if tail:
            print(f"      {tail.strip()[:300]}")
        return RunRecord(
            task=TASK, dataset=dataset, model=model, mode="full", seed=args.seed,
            num_layers=0, lr=args.lr, hidden_dim=args.hidden_dim,
            tag=args.tag, status="failed",
            note=f"returncode={proc.returncode}; 日志 {logs[-1].name if logs else '无'}",
        )
    print(f"    -> MRR={metrics.get('MRR', float('nan')):.4f} "
          f"HITS@1={metrics.get('HITS@1', float('nan')):.4f} "
          f"HITS@10={metrics.get('HITS@10', float('nan')):.4f}  "
          f"耗时 {human_time(elapsed)}")

    return RunRecord(
        task=TASK, dataset=dataset, model=model, mode="full", seed=args.seed,
        num_layers=0, lr=args.lr, hidden_dim=args.hidden_dim, tag=args.tag,
        extra_hparams={
            "epochs": args.epochs, "batch_size": args.batch_size,
            "neg_ratio": args.neg_ratio, "cpu_num": args.cpu_num,
            "n_train_triples": count_triples(data_path / "train.txt"),
            "n_valid_triples": count_triples(data_path / "valid.txt"),
            "n_test_triples": count_triples(data_path / "test.txt"),
            "n_entity": count_triples(data_path / "entity2id.txt"),
            "n_relation": count_triples(data_path / "relation2id.txt"),
            "log": str(logs[-1]) if logs else "",
        },
        total_sec=elapsed, train_sec=elapsed,
        metrics={
            "MRR": metrics.get("MRR", float("nan")),
            "MR": metrics.get("MR", float("nan")),
            "HITS@1": metrics.get("HITS@1", float("nan")),
            "HITS@3": metrics.get("HITS@3", float("nan")),
            "HITS@10": metrics.get("HITS@10", float("nan")),
            "test": metrics.get("MRR", float("nan")),  # 与其他任务对齐的"主指标"
            "metric": "MRR",
        },
        status="ok", note=f"{args.epochs} epoch, batch {args.batch_size}, lr {args.lr}",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="任务四:TransE / RotatE / ConvE 批量实验")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--test-batch-size", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=100)
    ap.add_argument("--neg-ratio", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--cpu-num", type=int, default=1,
                    help="DataLoader 工作进程数。框架默认 10,在千级三元组上纯属开销")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--save-dir",
                    default=str(CODE_DIR.parent / "models"),
                    help="框架的 save_path,日志与 checkpoint 落在这里")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将执行的命令,不真的运行")
    ap.add_argument("--no-record", action="store_true",
                    help="不写入 runs.jsonl(用于估时)")
    args = ap.parse_args(argv)

    fix_console_encoding()
    banner("任务四 · 知识图谱补全(TransE / RotatE / ConvE)")

    cells = [(d, m) for d in args.datasets for m in args.models]
    print(f"\n共 {len(cells)} 个组合"
          f"(数据集 {len(args.datasets)} × 模型 {len(args.models)})"
          f"  epochs={args.epochs} batch={args.batch_size} "
          f"neg_ratio={args.neg_ratio} lr={args.lr}\n")

    t0 = time.perf_counter()
    failed = 0
    for i, (d, m) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {d}/{m}", flush=True)
        if args.dry_run:
            continue
        rec = run_one(d, m, args)
        if rec.status == "failed":
            failed += 1
        if not args.no_record:
            record(rec)

    print(f"\n完成 {len(cells) - failed}/{len(cells)},"
          f"总耗时 {human_time(time.perf_counter() - t0)}")
    if not args.no_record:
        print("  汇总:python code/report.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
