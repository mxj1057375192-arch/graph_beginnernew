"""任务三:结果汇总与报告表格生成。

核心表是**池化方式的对比**(mean / max / min),这是任务三的考点。
ZINC 是回归任务,指标为 MAE(越小越好),其余为准确率(越大越好),
两类分开成表,不混在一起比较。

用法::

    python code/report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.results import RUNS_FILE, load_runs  # noqa: E402
from common.utils import fix_console_encoding  # noqa: E402

TASK = "task3"
MODELS = ("GCN", "GAT", "GraphSAGE", "GIN")
POOLINGS = ("mean", "max", "min")
POOL_CN = {"mean": "平均池化", "max": "最大池化", "min": "最小池化"}


def _mean_std(s, digits: int = 4) -> str:
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="任务三结果汇总")
    ap.add_argument("--runs", default=str(RUNS_FILE))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "report_task3.md"))
    args = ap.parse_args(argv)

    fix_console_encoding()

    import pandas as pd

    rows = [r for r in load_runs(Path(args.runs)) if r.get("task") == TASK]
    if not rows:
        print(f"  runs.jsonl 中没有 {TASK} 的记录,先跑 code/train.py")
        return 1

    df = pd.DataFrame(rows)
    if "tag" not in df.columns:
        df["tag"] = "main"
    df["tag"] = df["tag"].fillna("main")
    df["test"] = df["metrics"].apply(lambda d: (d or {}).get("test"))
    df["metric"] = df["metrics"].apply(lambda d: (d or {}).get("metric"))
    df["regression"] = df["extra_hparams"].apply(
        lambda d: bool((d or {}).get("regression")))

    # `max_epochs` **不是失败**:训练正常完成、指标算得出来,只是早停没触发。
    # 把它当异常排除会静默丢掉有效数据。只有 failed / timeout / diverged 才排除。
    VALID = ("ok", "max_epochs")
    ok = df[df["status"].isin(VALID)]
    main = ok[ok["tag"] == "main"]
    bad = df[~df["status"].isin(VALID)]
    cls = main[~main["regression"]]
    reg = main[main["regression"]]

    md = ["# 任务三 · 图分类 实验结果\n",
          "> 本文件由 `task3_graph_classification/code/report.py` 从 "
          "`results/runs.jsonl` 自动生成,**请勿手工编辑**。\n",
          "\n**指标**:MUTAG / PROTEINS / ENZYMES 为**准确率**(越大越好);"
          "ZINC 是**回归**任务,指标为 **MAE**(越小越好),两者分表,不混比。\n"]

    # ---------------- 分类 ----------------
    md.append("\n## 表 1 · 池化方式对比(分类任务,准确率)\n")
    if cls.empty:
        md.append("_(尚无记录)_\n")
    else:
        for ds in sorted(cls["dataset"].unique()):
            sub = cls[cls["dataset"] == ds]
            md.append(f"\n### {ds}\n")
            try:
                h = sub["extra_hparams"].iloc[0]
                md.append(f"图数 训练 {int(h['n_train_graphs']):,} / "
                          f"验证 {int(h['n_val_graphs']):,} / "
                          f"测试 {int(h['n_test_graphs']):,} · "
                          f"类别 {int(h['num_classes'])}\n")
            except Exception:  # noqa: BLE001
                pass
            head = ["模型"] + [POOL_CN[p] for p in POOLINGS] + ["最优池化"]
            body = []
            for m in MODELS:
                cells, vals = [], {}
                for p in POOLINGS:
                    g = sub[(sub["model"] == m) & (sub["mode"] == p)]
                    if g.empty:
                        cells.append("—")
                        continue
                    cells.append(_mean_std(g["test"]))
                    vals[p] = g["test"].mean()
                best = max(vals, key=vals.get) if vals else "—"
                body.append([m] + cells + [POOL_CN.get(best, "—")])
            md.append(_md_table(head, body))

        # 汇总:每种池化的平均表现
        md.append("\n**各池化方式在所有(数据集 × 模型)组合上的平均准确率**\n")
        head = ["池化方式", "平均准确率", "组合数"]
        body = []
        for p in POOLINGS:
            g = cls[cls["mode"] == p]
            if g.empty:
                continue
            body.append([POOL_CN[p], f"{g['test'].mean():.4f}", len(g)])
        md.append(_md_table(head, body))

    # ---------------- 回归 ----------------
    md.append("\n## 表 2 · ZINC 回归(MAE,**越小越好**)\n")
    if reg.empty:
        md.append("_(尚无记录)_\n")
    else:
        head = ["模型"] + [POOL_CN[p] for p in POOLINGS] + ["最优池化"]
        body = []
        for m in MODELS:
            cells, vals = [], {}
            for p in POOLINGS:
                g = reg[(reg["model"] == m) & (reg["mode"] == p)]
                if g.empty:
                    cells.append("—")
                    continue
                cells.append(_mean_std(g["test"]))
                vals[p] = g["test"].mean()
            best = min(vals, key=vals.get) if vals else "—"
            body.append([m] + cells + [POOL_CN.get(best, "—")])
        md.append(_md_table(head, body))

    # ---------------- 未完成 ----------------
    md.append("\n## 表 3 · 未正常完成的 run\n")
    if bad.empty:
        md.append("\n全部 run 均正常完成。\n")
    else:
        md.append(_md_table(
            ["数据集", "模型", "池化", "种子", "状态", "说明"],
            [[r["dataset"], r["model"], r["mode"], r["seed"], r["status"],
              str(r.get("note", ""))[:70]] for _, r in bad.iterrows()]))
        md.append(f"\n共 {len(bad)} 个,占全部 {len(df)} 个 run 的 "
                  f"{len(bad) / len(df) * 100:.0f}%。\n")

    text = "\n".join(md)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\n  已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
