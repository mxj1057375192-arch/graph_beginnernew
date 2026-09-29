"""任务一:节点分类 —— 测试脚本(不训练)。

任务书要求"README.md 中需要写明训练和测试的脚本",所以训练与测试是两条
独立的入口:

* 训练:[train.py](train.py) —— 训练并保存**验证集最优**权重
* 测试:本脚本 —— 载入权重,只在测试集上评测

## 为什么测试集不能参与任何选择

保存的权重是"验证集上最优"的那一组,不是最后一个 epoch 的。测试集只用来
报告最终数字,不参与模型选择、不参与早停、不参与挑超参。否则测试集就事实上
变成了验证集,报出来的数字会系统性偏乐观 —— 这是实验设计里最容易犯、也最
难被读者察觉的错误。

用法::

    # 评测单个权重文件
    python code/test.py --checkpoint models/Cora_GCN_full_seed0.pt

    # 评测整个目录下的全部权重(会打印汇总表)
    python code/test.py --checkpoints models/

    # 若权重不在默认位置,指定数据目录
    python code/test.py --checkpoints models/ --data-dir task1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.experiments import eval_node_classification  # noqa: E402
from common.results import RunRecord, record  # noqa: E402
from common.utils import banner, fix_console_encoding, human_time  # noqa: E402

TASK = "task1"


def _test_one(path: Path, data_dir: str, quiet: bool) -> RunRecord:
    try:
        rec = eval_node_classification(checkpoint=str(path), task_dir=data_dir,
                                       verbose=not quiet)
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {path.name}: {type(e).__name__}: {e}")
        rec = RunRecord(task=TASK, dataset="?", model="?", mode="?",
                        seed=-1, num_layers=0, lr=0.0, tag="test",
                        status="failed", note=f"{type(e).__name__}: {e}")
    record(rec)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="任务一:载入权重并评测(不训练)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="单个权重文件(.pt)")
    src.add_argument("--checkpoints", help="权重所在目录,评测其中全部 .pt")
    ap.add_argument("--data-dir", default=TASK,
                    help="数据集目录名(默认 task1)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    # 必须在任何 print 之前调用,否则 Windows 控制台(cp936)会把中文打成乱码
    fix_console_encoding()
    banner("任务一 · 测试(载入权重,不训练)")

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

    import time

    t0 = time.perf_counter()
    results = [_test_one(p, args.data_dir, args.quiet) for p in files]
    elapsed = time.perf_counter() - t0

    ok = [r for r in results if r.status == "ok"]
    if len(files) > 1 and ok:
        print(f"\n  {'权重文件':<44}{'测试准确率':>12}")
        print("  " + "-" * 56)
        for r, p in zip(results, files):
            if r.status != "ok":
                print(f"  {p.name:<44}{'失败':>12}")
                continue
            print(f"  {p.name:<44}{r.metrics['test']:>12.4f}")
        vals = [r.metrics["test"] for r in ok]
        print(f"\n  共 {len(ok)}/{len(results)} 个成功,"
              f"测试准确率 {sum(vals) / len(vals):.4f} ± "
              f"{(sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5:.4f}")

    print(f"\n  耗时 {human_time(elapsed)};结果已追加到 results/runs.jsonl")
    print("  注意:这些记录 tag=test,汇总时不会与训练记录混在一起")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
