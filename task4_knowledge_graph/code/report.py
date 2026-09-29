"""任务四:知识图谱补全 —— 结果汇总与报告表格生成。

指标是框架计算的 **filtered** MRR / MR / HITS@1/3/10:
评测某个候选实体时,会把"训练/验证/测试集中已存在的其它真三元组"的得分
压到最低,排除它们对排名的干扰。这是知识图谱补全的标准口径。

用法::

    python code/report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
# CODE_DIR 就是 code/ 本身,往上一层是 task4_knowledge_graph/,再上一层才是仓库根
REPO_ROOT = CODE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.results import RUNS_FILE, load_runs  # noqa: E402
from common.utils import fix_console_encoding  # noqa: E402

TASK = "task4"
MODELS = ("TransE", "RotatE", "ConvE")
KEYS = ("MRR", "MR", "HITS@1", "HITS@3", "HITS@10")


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(无数据)_\n"
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="任务四结果汇总")
    ap.add_argument("--runs", default=str(RUNS_FILE))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "report_task4.md"))
    args = ap.parse_args(argv)

    fix_console_encoding()

    import pandas as pd

    rows = [r for r in load_runs(Path(args.runs)) if r.get("task") == TASK]
    if not rows:
        print(f"  runs.jsonl 中没有 {TASK} 的记录,先跑 code/run_experiments.py")
        return 1

    df = pd.DataFrame(rows)
    for k in KEYS:
        df[k] = df["metrics"].apply(lambda d, k=k: (d or {}).get(k))

    ok = df[df["status"] == "ok"]
    bad = df[df["status"] != "ok"]

    md = ["# 任务四 · 知识图谱补全 实验结果\n",
          "> 本文件由 `task4_knowledge_graph/code/report.py` 从 "
          "`results/runs.jsonl` 自动生成,**请勿手工编辑**。\n",
          "\n**指标口径**:filtered MRR / MR / HITS@K。评测候选实体时会把"
          "训练/验证/测试集中已存在的其它真三元组排除在外,避免它们挤占排名。\n",
          "**MRR ↑ 越大越好;MR ↓ 越小越好。**\n"]

    md.append("\n## 表 1 · 各数据集上的模型对比\n")
    if ok.empty:
        md.append("_(尚无记录)_\n")
    else:
        for ds in sorted(ok["dataset"].unique()):
            sub = ok[ok["dataset"] == ds]
            md.append(f"\n### {ds}\n")
            try:
                h = sub["extra_hparams"].iloc[0]
                md.append(f"{int(h['n_entity'])} 实体 · {int(h['n_relation'])} 关系 · "
                          f"训练 {int(h['n_train_triples'])} 三元组 · "
                          f"验证 {int(h['n_valid_triples'])} · "
                          f"测试 {int(h['n_test_triples'])}\n")
            except Exception:  # noqa: BLE001
                pass
            head = ["模型", "MRR ↑", "MR ↓", "HITS@1", "HITS@3", "HITS@10",
                    "epoch", "batch", "lr"]
            body = []
            for m in MODELS:
                g = sub[sub["model"] == m]
                if g.empty:
                    continue
                r = g.iloc[0]
                ep = r.get("extra_hparams", {}) or {}
                body.append([
                    m, f"{r['MRR']:.4f}", f"{r['MR']:.2f}",
                    f"{r['HITS@1']:.4f}", f"{r['HITS@3']:.4f}",
                    f"{r['HITS@10']:.4f}",
                    ep.get("epochs", "—"), ep.get("batch_size", "—"),
                    f"{r['lr']:g}",
                ])
            md.append(_md_table(head, body))
            if body:
                best = max(body, key=lambda x: float(x[1]))
                md.append(f"\n该数据集上 MRR 最高:**{best[0]}**({best[1]})\n")

    md.append("\n## 表 2 · 未正常完成的 run\n")
    if bad.empty:
        md.append("\n全部 run 均正常完成。\n")
    else:
        md.append(_md_table(
            ["数据集", "模型", "状态", "说明"],
            [[r["dataset"], r["model"], r["status"], str(r.get("note", ""))[:60]]
             for _, r in bad.iterrows()]))

    text = "\n".join(md)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\n  已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
