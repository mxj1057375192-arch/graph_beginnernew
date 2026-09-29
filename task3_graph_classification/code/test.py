"""任务三:图分类 / 图回归 —— 测试脚本(不训练)。

* 训练:[train.py](train.py) —— 训练并保存「验证集最优」权重
* 测试:本脚本 —— 载入权重,只在测试集上评测

**注意 TUDataset 的划分是随机切的**,所以测试时用 checkpoint 里存的种子重新
切出同一份划分;否则评测的图集与训练时不是同一批,数字没有意义。ZINC 用官方
划分,不受影响。

用法::

    python code/test.py --checkpoint models/MUTAG_GCN_mean_seed0.pt
    python code/test.py --checkpoints models/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.experiments import eval_graph_classification  # noqa: E402
from common.results import RunRecord, record  # noqa: E402
from common.utils import banner, fix_console_encoding, human_time  # noqa: E402

TASK = "task3"


def _test_one(path: Path, data_dir: str, quiet: bool) -> RunRecord:
    try:
        rec = eval_graph_classification(checkpoint=str(path), task_dir=data_dir,
                                        verbose=not quiet)
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {path.name}: {type(e).__name__}: {e}")
        rec = RunRecord(task=TASK, dataset="?", model="?", mode="?", seed=-1,
                        num_layers=0, lr=0.0, tag="test",
                        status="failed", note=f"{type(e).__name__}: {e}")
    record(rec)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="任务三:载入权重并评测(不训练)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="单个权重文件(.pt)")
    src.add_argument("--checkpoints", help="权重所在目录,评测其中全部 .pt")
    ap.add_argument("--data-dir", default=TASK)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    fix_console_encoding()
    banner("任务三 · 测试(载入权重,不训练)")

    if args.checkpoint:
        files = [Path(args.checkpoint)]
    else:
        d = Path(args.checkpoints)
        if not d.is_dir():
            print(f"  ✗ 不是目录:{d}")
            return 1
        files = sorted(d.glob("*.pt"))
        if not files:
            print(f"  ✗ {d} 下没有 .pt 文件 —— 先用 train.py --save-model {d} 训练")
            return 1

    t0 = time.perf_counter()
    results = [_test_one(p, args.data_dir, args.quiet) for p in files]
    elapsed = time.perf_counter() - t0

    ok = [r for r in results if r.status == "ok"]
    if len(files) > 1 and ok:
        print(f"\n  {'权重文件':<42}{'指标':>8}{'测试值':>10}")
        print("  " + "-" * 62)
        for r, p in zip(results, files):
            if r.status != "ok":
                print(f"  {p.name:<42}{'失败':>8}")
                continue
            print(f"  {p.name:<42}{r.metrics['metric']:>8}{r.metrics['test']:>10.4f}")

    print(f"\n  耗时 {human_time(elapsed)};结果已追加到 results/runs.jsonl(tag=test)")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
