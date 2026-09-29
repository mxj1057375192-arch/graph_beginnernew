"""任务一:结果汇总与报告表格生成。

## 为什么表格必须由脚本生成

报告里的每一个数字都来自 ``results/runs.jsonl`` 里的一次真实 run。手工抄写
有三个必然的后果:抄错、抄漏失败的 run、以及无法回答"这个数是怎么来的"。
所以这里只做一件事 —— 把原始记录汇总成 markdown 表格。

## 三个表各回答什么

* **表 1 · 主对比** —— 全图 vs 采样,同时给精度和耗时。这是任务一的核心问题。
  注意 **梯度步数** 一列:全图 1 epoch 只有 1 次更新,采样有几百次。不列出
  这一列,读者会把"优化次数多"误读成"采样方法更好"。
* **表 2 · 超参影响** —— lr 与层数各自的影响(只在 Cora 上扫描,再迁移)。
* **表 3 · 未完成的 run** —— 失败/超时的组合如实列出,不静默丢弃。

用法::

    python code/report.py
    python code/report.py --out results/report_task1.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.results import RUNS_FILE, load_runs  # noqa: E402
from common.utils import fix_console_encoding, human_time  # noqa: E402

TASK = "task1"


# ---------------------------------------------------------------------------
# 小组件
# ---------------------------------------------------------------------------
def _mean_std(s, digits: int = 4) -> str:
    """把一组数格式化成 ``mean ± std``。只有一个样本时不显示 std。"""
    s = s.dropna()
    if len(s) == 0:
        return "—"
    if len(s) == 1:
        return f"{s.iloc[0]:.{digits}f}"
    return f"{s.mean():.{digits}f} ± {s.std():.{digits}f}"


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(无数据)_\n"
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def _pivot(df, index: list[str], col: str):
    return df.pivot_table(index=index, columns=col, values="test", aggfunc="mean")


_STATS_CACHE: dict[str, dict[str, int]] = {}


def _dataset_stats(name: str, task_dir: str = "task1") -> dict[str, int]:
    """实际加载数据集取划分规模。

    **不能拿 ``num_nodes`` 冒充训练节点数** —— Cora 有 2708 个节点但只有
    140 个训练节点,Citeseer 3327 个节点只有 120 个。把总数写成训练数会让
    读者完全误判任务难度,而且这个错误不会触发任何异常。
    """
    if name in _STATS_CACHE:
        return _STATS_CACHE[name]
    from common.datasets import load_node_dataset

    d = load_node_dataset(task_dir, name)[0]
    st = {
        "num_nodes": int(d.num_nodes),
        "num_edges": int(d.num_edges),
        "num_features": int(d.num_features),
        "n_train": int(d.train_mask.sum()),
        "n_val": int(d.val_mask.sum()),
        "n_test": int(d.test_mask.sum()),
        "num_classes": int(d.y.max()) + 1,
    }
    _STATS_CACHE[name] = st
    return st


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="任务一结果汇总")
    ap.add_argument("--runs", default=str(RUNS_FILE))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "report_task1.md"))
    args = ap.parse_args(argv)

    fix_console_encoding()

    import pandas as pd

    runs = load_runs(Path(args.runs))
    rows = [r for r in runs if r.get("task") == TASK]
    if not rows:
        print(f"  runs.jsonl 中没有 {TASK} 的记录,先跑 code/train.py")
        return 1

    df = pd.DataFrame(rows)
    # 兼容早期没有 tag 字段的记录
    if "tag" not in df.columns:
        df["tag"] = "main"
    df["tag"] = df["tag"].fillna("main")
    df["test"] = df["metrics"].apply(lambda d: (d or {}).get("test"))
    df["val"] = df["metrics"].apply(lambda d: (d or {}).get("val"))

    # `max_epochs` **不是失败**:训练正常完成、指标算得出来,只是早停没触发。
    # 把它当异常排除会静默丢掉有效数据(本项目就出现过因此少算一个种子)。
    # 只有真正异常的 failed / timeout / diverged 才排除。
    VALID = ("ok", "max_epochs")
    ok = df[df["status"].isin(VALID)]
    bad = df[~df["status"].isin(VALID)]
    main = ok[ok["tag"] == "main"]
    sweep_lr = ok[ok["tag"] == "sweep_lr"]
    sweep_layers = ok[ok["tag"] == "sweep_layers"]

    md: list[str] = ["# 任务一 · 节点分类 实验结果\n",
                     "> 本文件由 `task1_node_classification/code/report.py` "
                     "从 `results/runs.jsonl` 自动生成,**请勿手工编辑**。\n"]

    # ---------------- 表 1:主对比 ----------------
    md.append("\n## 表 1 · 全图训练 vs 子图采样训练\n")
    if main.empty:
        md.append("_(尚无主实验记录,请先运行 `python code/train.py --all`)_\n")
    else:
        datasets = sorted(main["dataset"].unique())
        models = [m for m in ("GCN", "GAT", "GraphSAGE", "GIN")
                  if m in set(main["model"])]
        for ds in datasets:
            sub = main[main["dataset"] == ds]
            md.append(f"\n### {ds}\n")
            try:
                st = _dataset_stats(ds)
                md.append(
                    f"节点 {st['num_nodes']:,} · 边 {st['num_edges']:,} · "
                    f"特征 {st['num_features']:,} 维 · 类别 {st['num_classes']} · "
                    f"划分 训练 {st['n_train']:,} / 验证 {st['n_val']:,} / "
                    f"测试 {st['n_test']:,}\n")
            except Exception as e:  # noqa: BLE001
                md.append(f"_(数据集规模读取失败:{e})_\n")
            head = ["模型", "模式", "测试准确率", "lr", "每 epoch 梯度步数",
                    "每 epoch 耗时", "总耗时", "最佳 epoch", "种子数"]
            body = []
            for m in models:
                for mode, label in (("full", "全图"), ("sampler", "采样")):
                    g = sub[(sub["model"] == m) & (sub["mode"] == mode)]
                    if g.empty:
                        continue
                    body.append([
                        m, label,
                        _mean_std(g["test"]),
                        # lr 必须列出来:Flickr 上 GAT 用了比其余模型更低的学习率
                        # (见 README 第 7.5 节),不写读者会以为四个模型同参
                        f"{g['lr'].mean():g}",
                        f"{g['grad_steps_per_epoch'].mean():.0f}",
                        f"{g['sec_per_epoch'].mean() * 1000:.1f} ms",
                        human_time(g["total_sec"].sum()),
                        f"{g['best_epoch'].mean():.0f}",
                        len(g),
                    ])
            md.append(_md_table(head, body))

        # ---- 采样相对全图的倍数 ----
        md.append("\n**采样相对全图的倍数**(同一模型内比较)\n")
        head = ["数据集", "模型", "精度变化", "每 epoch 耗时倍数", "梯度步数倍数"]
        body = []
        for ds in datasets:
            sub = main[main["dataset"] == ds]
            for m in models:
                f = sub[(sub["model"] == m) & (sub["mode"] == "full")]
                s = sub[(sub["model"] == m) & (sub["mode"] == "sampler")]
                if f.empty or s.empty:
                    continue
                da = s["test"].mean() - f["test"].mean()
                body.append([
                    ds, m,
                    f"{da:+.4f}",
                    f"{s['sec_per_epoch'].mean() / max(f['sec_per_epoch'].mean(), 1e-9):.1f}×",
                    f"{s['grad_steps_per_epoch'].mean() / max(f['grad_steps_per_epoch'].mean(), 1):.0f}×",
                ])
        md.append(_md_table(head, body))

    # ---------------- 表 2:超参影响 ----------------
    md.append("\n## 表 2 · 超参影响(仅在 Cora 上扫描)\n")
    sw = pd.concat([sweep_lr, sweep_layers]) if not (
        sweep_lr.empty and sweep_layers.empty) else pd.DataFrame()
    if not sw.empty:
        n_seed = sw["seed"].nunique()
        md.append(
            f"\n扫描在**只在 Cora** 上进行,每个组合 **{n_seed} 个种子**"
            f"(主实验是 {main['seed'].nunique() if not main.empty else '多'} 个),"
            f"因此表 2 的数字**噪声大于表 1**,"
            f"只用于看趋势(哪个 lr/层数更好),不宜逐位比较。\n"
            f"选出的超参随后被迁移到 Citeseer 与 Flickr。\n")
    if sweep_lr.empty and sweep_layers.empty:
        md.append("_(尚无扫描记录,请先运行 `python code/train.py --sweep lr` "
                  "与 `--sweep layers`)_\n")

    if not sweep_lr.empty:
        md.append("\n### 2.1 学习率的影响(层数固定为 2)\n")
        head = ["模型", "模式"] + [f"lr={v}" for v in sorted(sweep_lr["lr"].unique())]
        body = []
        for m in [x for x in ("GCN", "GAT", "GraphSAGE", "GIN")
                  if x in set(sweep_lr["model"])]:
            for mode, label in (("full", "全图"), ("sampler", "采样")):
                g = sweep_lr[(sweep_lr["model"] == m) & (sweep_lr["mode"] == mode)]
                if g.empty:
                    continue
                cells = []
                for lr in sorted(g["lr"].unique()):
                    gg = g[g["lr"] == lr]
                    cells.append(_mean_std(gg["test"]))
                body.append([m, label] + cells)
        md.append(_md_table(head, body))
        # 每个模型的最优 lr
        best = (sweep_lr.groupby(["model", "lr"])["test"].mean()
                .reset_index().sort_values("test", ascending=False)
                .groupby("model").first())
        md.append("\n**各模型的最优学习率**\n")
        md.append(_md_table(["模型", "最优 lr", "测试准确率"],
                            [[m, r["lr"], f"{r['test']:.4f}"]
                             for m, r in best.iterrows()]))

    if not sweep_layers.empty:
        md.append("\n### 2.2 层数的影响(lr 固定为默认值)\n")
        head = ["模型", "模式"] + [f"{int(v)} 层"
                                   for v in sorted(sweep_layers["num_layers"].unique())]
        body = []
        for m in [x for x in ("GCN", "GAT", "GraphSAGE", "GIN")
                  if x in set(sweep_layers["model"])]:
            for mode, label in (("full", "全图"), ("sampler", "采样")):
                g = sweep_layers[(sweep_layers["model"] == m)
                                 & (sweep_layers["mode"] == mode)]
                if g.empty:
                    continue
                cells = []
                for nl in sorted(g["num_layers"].unique()):
                    cells.append(_mean_std(g[g["num_layers"] == nl]["test"]))
                body.append([m, label] + cells)
        md.append(_md_table(head, body))

    # ---------------- 覆盖情况 ----------------
    # 自动列出缺失的组合。矩阵残缺时读者最容易被误导 —— 表格里只有 4 行,
    # 没人会注意到本该有 8 行。
    md.append("\n## 表 3 · 实验矩阵覆盖情况\n")
    if not main.empty:
        all_models = [m for m in ("GCN", "GAT", "GraphSAGE", "GIN")
                      if m in set(main["model"])]
        head = ["数据集", "模式"] + all_models + ["缺失数"]
        body = []
        missing_total = 0
        for ds in sorted(main["dataset"].unique()):
            for mode, label in (("full", "全图"), ("sampler", "采样")):
                cells, miss = [], 0
                for m in all_models:
                    g = main[(main["dataset"] == ds) & (main["model"] == m)
                             & (main["mode"] == mode)]
                    if g.empty:
                        cells.append("❌")
                        miss += 1
                    else:
                        cells.append(f"✅ {len(g)}")
                missing_total += miss
                body.append([ds, label] + cells + [miss])
        md.append(_md_table(head, body))
        if missing_total:
            md.append(f"\n**有 {missing_total} 个组合缺失。** 表中 ✅ 后的数字是该组合"
                      f"实际完成的种子数,各组合的种子数可能不同 —— 比较时请注意。\n")
        else:
            md.append("\n矩阵完整,无缺失组合。\n")

    # ---------------- 未完成的 run ----------------
    md.append("\n## 表 4 · 未正常完成的 run\n")
    if bad.empty:
        md.append("\n全部 run 均正常完成。\n")
    else:
        head = ["数据集", "模型", "模式", "种子", "状态", "说明"]
        body = [[r["dataset"], r["model"], r["mode"], r["seed"],
                 r["status"], str(r.get("note", ""))[:70]]
                for _, r in bad.iterrows()]
        md.append(_md_table(head, body))
        md.append(f"\n共 {len(bad)} 个,占全部 {len(df)} 个 run 的 "
                  f"{len(bad) / len(df) * 100:.0f}%。\n")

    text = "\n".join(md)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\n  已写入 {args.out}")
    print(f"  (原始记录 {len(df)} 条,正常 {len(ok)} 条,异常 {len(bad)} 条)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
