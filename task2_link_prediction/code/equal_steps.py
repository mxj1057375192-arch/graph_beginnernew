"""等梯度步数对照实验。

## 为什么需要这个实验

任务二的主要结论是"采样训练在链路预测上明显更好"(Cora 上 GCN 高 +0.11 AUC)。
但这个数字**不能直接当作采样方法的优势**,因为:

* 全图训练 1 个 epoch = **1 次**梯度更新
* 采样训练 1 个 epoch = **33 次**(8448 条监督边 / 每批 256 条)

跑同样 100 个 epoch,采样模式实际做了 **3300 次**优化,全图只做了 **100 次**。
所以那个 +0.11 里混着"优化得多"这个因素,不是纯粹的方法差异。

## 本实验做什么

把两者的**累计梯度更新次数**对齐:让全图模式跑满 3300 个 epoch
(1 步/epoch),与采样模式的 3300 步持平,再看差距还剩多少。

* 差距缩到 0  → 采样的优势全部来自更新次数,方法本身没有优势
* 差距不变   → 更新时间不是主因,采样确实有别的价值
* 部分缩小   → 两者都有(本项目的实测结果属于这一类)

## 代价

全图模式每 epoch 只要 20~30 ms,Cora 上跑 3300 epoch 约 1~2 分钟,
远比采样模式便宜 —— 这是一个**低成本、高信息量**的对照。

用法::

    python code/equal_steps.py
    python code/equal_steps.py --datasets Cora --models GCN GraphSAGE
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.experiments import run_link_prediction  # noqa: E402
from common.models import GNN_MODELS  # noqa: E402
from common.results import record  # noqa: E402
from common.utils import banner, fix_console_encoding, human_time, setup  # noqa: E402

TASK = "task2"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="链路预测:等梯度步数对照")
    ap.add_argument("--datasets", nargs="+", default=["Cora", "Citeseer"])
    ap.add_argument("--models", nargs="+", default=list(GNN_MODELS))
    ap.add_argument("--target-steps", type=int, default=3300,
                    help="要对齐的累计梯度步数(采样模式 100 epoch x 33 步)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="eqsteps")
    args = ap.parse_args(argv)

    fix_console_encoding()
    setup(seed=args.seed, verbose=False)
    banner(f"链路预测 · 等梯度步数对照(全图跑满 {args.target_steps} 步)")

    t0 = time.perf_counter()
    for ds in args.datasets:
        for model in args.models:
            print(f"  {ds}/{model} 全图,{args.target_steps} epoch ...",
                  end="", flush=True)
            r = run_link_prediction(
                dataset=ds, model_name=model, mode="full", seed=args.seed,
                num_layers=2, hidden_channels=64, lr=0.01,
                # 全图 1 步/epoch,所以 epoch 数就是步数
                max_epochs=args.target_steps, patience=args.target_steps,
                batch_size=args.batch_size, task_dir=TASK, verbose=False,
            )
            r.tag = args.tag
            record(r)
            print(f" AUC={r.metrics['test']:.4f} "
                  f"best_epoch={r.best_epoch}(第 {r.best_epoch + 1} 步)")

    print(f"\n  总耗时 {human_time(time.perf_counter() - t0)}")
    print("  汇总:python code/report.py(结果会出现在报告的「等梯度步数对照」表)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
