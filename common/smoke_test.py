"""冒烟测试 / 预算标定。

前期工作里最有价值的一步。它做两件事:

1. **暴露问题**:对每个「数据集 × 模型 × 训练模式」组合只跑极少 epoch,
   验证能跑通、张量形状正确、指标算得出来。目的是在投入正式实验之前把所有
   坑一次性挖出来 —— 尤其是那些**不报错但静默算错**的问题(例如采样批次
   未按种子节点切片)。
2. **标定预算**:记录每个组合的实测每-epoch 耗时。正式实验要跑多少组合、
   总共要多久,应当基于实测而非估计。

用法::

    python -m common.smoke_test                  # 全部组合
    python -m common.smoke_test --datasets Cora  # 只测指定数据集
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_OUT = REPO_ROOT / "results" / "smoke.json"


# ---------------------------------------------------------------------------
# 数据集体检
# ---------------------------------------------------------------------------
def validate_datasets(datasets: list[str], task_dir: str = "task1") -> list[dict[str, Any]]:
    """加载每个数据集并检查其字段是否符合后续代码的假设。

    这里专门检查几类**已知容易出问题**的地方,而不是笼统地"能加载就算过"。
    """
    from common.datasets import describe, load_node_dataset

    reports = []
    for name in datasets:
        rep: dict[str, Any] = {"dataset": name, "ok": False}
        try:
            ds = load_node_dataset(task_dir, name)
            info = describe(ds, name)
            rep.update(info)
            d = ds[0]

            issues: list[str] = []

            # --- 标签形状:分类任务要求 y 是 [N] 的类别索引 ---
            # Flickr 的 class_map 存的是长度 7 的 0/1 列表,PyG 直接转张量会
            # 得到 [N, 7]。若如此,CrossEntropyLoss 会报错或静默算错。
            if d.y.dim() > 1:
                issues.append(
                    f"y 形状为 {tuple(d.y.shape)},是多标签而非类别索引;"
                    f"分类任务需先 argmax(-1) 转换"
                )
            if not hasattr(d, "train_mask") or d.train_mask is None:
                issues.append("缺少 train_mask,无法划分训练集")

            # --- 边索引范围 ---
            if d.edge_index.numel() and int(d.edge_index.max()) >= d.num_nodes:
                issues.append("edge_index 含越界节点编号")

            rep["issues"] = issues
            rep["y_shape"] = list(d.y.shape)
            rep["y_dtype"] = str(d.y.dtype)
            rep["ok"] = len(issues) == 0
        except Exception as e:  # noqa: BLE001
            rep["error"] = f"{type(e).__name__}: {e}"
            rep["traceback"] = traceback.format_exc()[-800:]
        reports.append(rep)
    return reports


# ---------------------------------------------------------------------------
# 冒烟矩阵
# ---------------------------------------------------------------------------
def smoke_matrix(
    datasets: list[str],
    models: list[str],
    modes: list[str],
    *,
    epochs: int = 2,
    hidden: int = 32,
    task_dir: str = "task1",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """对每个组合跑极少量 epoch,记录是否通过与实测耗时。"""
    from common.experiments import run_node_classification
    from common.utils import set_seed

    results: list[dict[str, Any]] = []
    total = len(datasets) * len(models) * len(modes)
    i = 0

    for ds in datasets:
        for model_name in models:
            for mode in modes:
                i += 1
                cell = f"{ds}/{model_name}/{mode}"
                if verbose:
                    print(f"[{i}/{total}] {cell} ...", end="", flush=True)

                rec: dict[str, Any] = {"dataset": ds, "model": model_name, "mode": mode}
                t0 = time.perf_counter()
                try:
                    # 只跑 2 个 epoch、patience 设成不可能触发 —— 目的不是收敛,
                    # 而是验证流程通畅并测出单 epoch 的真实成本
                    r = run_node_classification(
                        dataset=ds, model_name=model_name, mode=mode, seed=0,
                        num_layers=2, hidden_channels=hidden,
                        max_epochs=epochs, patience=10_000,
                        task_dir=task_dir, verbose=False,
                    )
                    rec.update({
                        "ok": True,
                        "sec_per_epoch": r.sec_per_epoch,
                        "grad_steps_per_epoch": r.grad_steps_per_epoch,
                        "params": r.extra_hparams["params"],
                        "num_nodes": r.extra_hparams["num_nodes"],
                        "num_edges": r.extra_hparams["num_edges"],
                        "wall_sec": time.perf_counter() - t0,
                    })
                    if verbose:
                        print(f" ok  {r.sec_per_epoch * 1000:8.1f} ms/epoch"
                              f"  steps/ep={r.grad_steps_per_epoch}")
                except Exception as e:  # noqa: BLE001
                    rec.update({
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-1500:],
                        "wall_sec": time.perf_counter() - t0,
                    })
                    if verbose:
                        print(f" 失败: {type(e).__name__}: {str(e)[:90]}")
                results.append(rec)
    return results


# ---------------------------------------------------------------------------
# 预算外推
# ---------------------------------------------------------------------------
def smoke_matrix_graph(
    datasets: list[str],
    models: list[str],
    poolings: list[str],
    *,
    epochs: int = 3,
    hidden: int = 32,
    task_dir: str = "task3",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """任务三:图级数据集 × 模型 × 池化方式。

    这一组冒烟的意义和任务一不同 —— 图级任务没有"全图 vs 采样"的对比,
    它要验证的是:

      1. ZINC 确实走**回归**分支(标量输出 + MAE),而不是被当成分类
      2. 三种池化都能跑通 —— 特别是 **min pooling**,它在 ReLU 之后会
         大面积塌向 0,需要确认拿到的是"真实发现"而不是实现错误
      3. TUDataset 的 y 形状在压平后确实变成 [N] 的类别索引
    """
    from common.experiments import run_graph_classification

    results: list[dict[str, Any]] = []
    total = len(datasets) * len(models) * len(poolings)
    i = 0

    for ds in datasets:
        for model_name in models:
            for pool in poolings:
                i += 1
                cell = f"{ds}/{model_name}/{pool}"
                if verbose:
                    print(f"[{i}/{total}] {cell} ...", end="", flush=True)

                rec: dict[str, Any] = {"dataset": ds, "model": model_name,
                                       "pooling": pool}
                t0 = time.perf_counter()
                try:
                    r = run_graph_classification(
                        dataset=ds, model_name=model_name, pooling=pool, seed=0,
                        num_layers=3, hidden_channels=hidden,
                        max_epochs=epochs, patience=10_000,
                        task_dir=task_dir, verbose=False,
                    )
                    rec.update({
                        "ok": True,
                        "sec_per_epoch": r.sec_per_epoch,
                        "grad_steps_per_epoch": r.grad_steps_per_epoch,
                        "params": r.extra_hparams["params"],
                        "n_train_graphs": r.extra_hparams["n_train_graphs"],
                        "regression": r.extra_hparams["regression"],
                        "metric_name": r.metrics["metric"],
                        "val": r.metrics["val"],
                        "wall_sec": time.perf_counter() - t0,
                    })
                    if verbose:
                        print(f" ok  {r.sec_per_epoch * 1000:8.1f} ms/epoch"
                              f"  {r.metrics['metric']}={r.metrics['val']:.4f}"
                              f"  steps/ep={r.grad_steps_per_epoch}")
                except Exception as e:  # noqa: BLE001
                    rec.update({
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-1500:],
                        "wall_sec": time.perf_counter() - t0,
                    })
                    if verbose:
                        print(f" 失败: {type(e).__name__}: {str(e)[:90]}")
                results.append(rec)
    return results


def project_budget(
    smoke: list[dict[str, Any]],
    plan: dict[str, dict[str, int]],
    seeds: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """由实测每-epoch 耗时外推完整实验的墙钟时间。

    Args:
        smoke: :func:`smoke_matrix` 的返回值。
        plan: ``{数据集: {"epochs": N, "seeds": M}}``。
        seeds: 可选,覆盖 plan 里每个数据集的种子数。
    """
    out = []
    for cell in smoke:
        if not cell.get("ok"):
            continue
        p = plan.get(cell["dataset"])
        if not p:
            continue
        n_seeds = (seeds or {}).get(cell["dataset"], p.get("seeds", 1))
        epochs = p.get("epochs", 200)
        sec = cell["sec_per_epoch"] * epochs * n_seeds
        out.append({
            "dataset": cell["dataset"],
            "model": cell["model"],
            "mode": cell["mode"],
            "epochs": epochs,
            "seeds": n_seeds,
            "est_sec": sec,
        })
    return out


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s}s"
    h, m = divmod(m, 60)
    return f"{h}h{m}m"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    from common.models import GNN_MODELS
    from common.utils import banner, fix_console_encoding, setup

    ap = argparse.ArgumentParser(description="冒烟测试与预算标定")
    ap.add_argument("--task", default="task1", choices=["task1", "task3"],
                    help="task1=节点分类(全图/采样),task3=图分类/回归(池化)")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=list(GNN_MODELS))
    ap.add_argument("--modes", nargs="+", default=["full", "sampler"],
                    help="仅 task1 使用")
    ap.add_argument("--poolings", nargs="+", default=["mean", "max", "min"],
                    help="仅 task3 使用")
    # 默认 3 而非 2:训练循环会把首个 epoch 当作预热排除在计时之外,
    # 跑 2 个 epoch 实际只有 1 个被计时,单 epoch 噪声会让预算外推失真。
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--task-dir", default=None)
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    # 数据集与目录按任务取默认值
    if args.datasets is None:
        args.datasets = (["Cora", "Citeseer", "Flickr"] if args.task == "task1"
                         else ["MUTAG", "PROTEINS", "ENZYMES", "ZINC"])
    if args.task_dir is None:
        args.task_dir = args.task
    if args.out is None:
        args.out = str(SMOKE_OUT) if args.task == "task1" else \
            str(SMOKE_OUT.with_name(f"smoke_{args.task}.json"))

    setup(seed=0)

    report: dict[str, Any] = {"args": vars(args)}

    # ---- 第一步:数据集体检(目前只覆盖节点级数据集)----
    if not args.skip_validate and args.task == "task1":
        banner("一、数据集体检")
        reports = validate_datasets(args.datasets, args.task_dir)
        report["datasets"] = reports
        for r in reports:
            if "error" in r:
                print(f"  ✗ {r['dataset']:10} 加载失败:{r['error']}")
            elif r["ok"]:
                print(f"  ✓ {r['dataset']:10} y 形状 {r['y_shape']} {r['y_dtype']}")
            else:
                print(f"  ! {r['dataset']:10} 存在问题:")
                for iss in r["issues"]:
                    print(f"      - {iss}")
        bad = [r["dataset"] for r in reports if "error" in r]
        if bad:
            print(f"\n  以下数据集无法加载,冒烟测试将跳过:{bad}")

    # ---- 第二步:冒烟矩阵 ----
    if args.task == "task1":
        banner("二、冒烟矩阵:节点分类 × 训练模式")
        results = smoke_matrix(
            args.datasets, args.models, args.modes,
            epochs=args.epochs, hidden=args.hidden, task_dir=args.task_dir,
        )
        key = "mode"
    else:
        banner("二、冒烟矩阵:图分类/回归 × 池化方式")
        results = smoke_matrix_graph(
            args.datasets, args.models, args.poolings,
            epochs=args.epochs, hidden=args.hidden, task_dir=args.task_dir,
        )
        key = "pooling"
    report["matrix"] = results

    # ---- 汇总 ----
    banner("三、结果")
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    print(f"  通过 {len(ok)}/{len(results)}")
    if bad:
        print("\n  失败组合:")
        for r in bad:
            print(f"    ✗ {r['dataset']}/{r['model']}/{r[key]}: "
                  f"{r.get('error', '')[:110]}")

    if ok:
        if args.task == "task1":
            print(f"\n  {'组合':<28}{'ms/epoch':>10}{'步数/epoch':>11}{'参数量':>12}")
            print("  " + "-" * 62)
        else:
            print(f"\n  {'组合':<28}{'ms/epoch':>10}{'步数/epoch':>11}"
                  f"{'指标':>9}{'参数量':>12}")
            print("  " + "-" * 72)
        for r in sorted(ok, key=lambda x: -x["sec_per_epoch"]):
            cell = f"{r['dataset']}/{r['model']}/{r[key]}"
            if args.task == "task1":
                print(f"  {cell:<28}{r['sec_per_epoch'] * 1000:>10.1f}"
                      f"{r['grad_steps_per_epoch']:>11}{r['params']:>12,}")
            else:
                print(f"  {cell:<28}{r['sec_per_epoch'] * 1000:>10.1f}"
                      f"{r['grad_steps_per_epoch']:>11}"
                      f"{r['metric_name']:>9}"
                      f"{r['params']:>12,}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  明细已写入 {args.out}")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
