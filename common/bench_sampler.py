"""采样训练的批次大小标定。

## 为什么需要这个

冒烟矩阵实测出 Flickr 上采样模式比全图模式慢 10~25 倍:

    Flickr/GCN/full      362.8 ms/epoch      1 步/epoch
    Flickr/GCN/sampler  8949.9 ms/epoch    698 步/epoch

注意**单步并不慢**(8949.9 / 698 = 12.8 ms),慢在步数。而步数由
``batch_size`` 决定:batch 越小,Python 层要构造的子图越多,固定开销
(子图采样、张量拼接、优化器调用)被摊薄的次数就越少。

因此 "batch_size 默认 64"(PyG 官方示例的取值)在 Cora 那种小图上无所谓,
在 Flickr 这种 89k 节点图上会直接把实验预算推高一到两个数量级。

本脚本实测不同 batch_size 下的每-epoch 耗时,为正式实验选定取值。
**注意这不影响对比公平性**:每个 epoch 仍然恰好遍历一遍全部训练节点,
"1 epoch = 一遍训练集"这个单位没有变,变的只是每个 epoch 里梯度更新的
次数 —— 而后者本来就会被记进结果表并对读者透明。

用法::

    python -m common.bench_sampler --dataset Flickr --models GCN GAT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def bench(
    dataset: str,
    model_name: str,
    batch_sizes: list[int],
    *,
    epochs: int = 3,
    num_layers: int = 2,
    hidden: int = 32,
    task_dir: str = "task1",
) -> list[dict[str, Any]]:
    from common.experiments import run_node_classification

    out: list[dict[str, Any]] = []
    for bs in batch_sizes:
        rec: dict[str, Any] = {"dataset": dataset, "model": model_name,
                               "batch_size": bs}
        t0 = time.perf_counter()
        try:
            r = run_node_classification(
                dataset=dataset, model_name=model_name, mode="sampler", seed=0,
                num_layers=num_layers, hidden_channels=hidden,
                batch_size=bs, max_epochs=epochs, patience=10_000,
                task_dir=task_dir, verbose=False,
            )
            rec.update({
                "ok": True,
                "sec_per_epoch": r.sec_per_epoch,
                "grad_steps_per_epoch": r.grad_steps_per_epoch,
                "ms_per_step": r.sec_per_epoch / max(r.grad_steps_per_epoch, 1) * 1000,
                "wall_sec": time.perf_counter() - t0,
            })
        except Exception as e:  # noqa: BLE001
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
        out.append(rec)
        status = (f"{rec['sec_per_epoch']:6.2f} s/epoch  "
                  f"{rec['grad_steps_per_epoch']:5d} 步  "
                  f"{rec['ms_per_step']:7.2f} ms/步"
                  if rec["ok"] else f"失败: {rec['error'][:70]}")
        print(f"  {dataset}/{model_name:<10} bs={bs:<5} {status}")
    return out


def main(argv: list[str] | None = None) -> int:
    from common.utils import banner, fix_console_encoding, setup

    ap = argparse.ArgumentParser(description="采样批次大小标定")
    ap.add_argument("--dataset", default="Flickr")
    ap.add_argument("--models", nargs="+", default=["GCN"])
    ap.add_argument("--batch-sizes", nargs="+", type=int,
                    default=[64, 128, 256, 512, 1024])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--task-dir", default="task1")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "bench_sampler.json"))
    args = ap.parse_args(argv)

    fix_console_encoding()
    setup(seed=0)
    banner(f"采样批次标定:{args.dataset}")

    results: list[dict[str, Any]] = []
    for m in args.models:
        results += bench(args.dataset, m, args.batch_sizes,
                         epochs=args.epochs, hidden=args.hidden,
                         task_dir=args.task_dir)

    print(f"\n  {'模型':<12}{'batch':>7}{'s/epoch':>11}{'步/epoch':>10}"
          f"{'ms/步':>10}{'相对 bs=64':>12}")
    print("  " + "-" * 62)
    for m in args.models:
        rows = [r for r in results if r["model"] == m and r.get("ok")]
        if not rows:
            continue
        base = rows[0]["sec_per_epoch"]
        for r in rows:
            print(f"  {r['model']:<12}{r['batch_size']:>7}{r['sec_per_epoch']:>11.2f}"
                  f"{r['grad_steps_per_epoch']:>10}{r['ms_per_step']:>10.2f}"
                  f"{r['sec_per_epoch'] / base:>11.2f}x")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"args": vars(args), "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  明细已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
