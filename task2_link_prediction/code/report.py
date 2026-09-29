"""任务二:结果汇总与报告表格生成。

表格全部由本脚本从 ``results/runs.jsonl`` 生成,不手工抄写。

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
from common.utils import fix_console_encoding, human_time  # noqa: E402

TASK = "task2"
MODELS = ("GCN", "GAT", "GraphSAGE", "GIN")


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
    ap = argparse.ArgumentParser(description="任务二结果汇总")
    ap.add_argument("--runs", default=str(RUNS_FILE))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "report_task2.md"))
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
    for k in ("test", "test_ap", "test_mrr", "test_hits@10", "test_hits@50"):
        df[k] = df["metrics"].apply(lambda d, k=k: (d or {}).get(k))

    # `max_epochs` **不是失败**:训练正常完成、指标算得出来,只是早停没触发
    # (验证指标跑满上限时仍在改善)。把它当成异常排除会**静默丢掉有效数据** ——
    # 本项目就出现过 Cora/GCN/采样 因此从 3 个种子缩成 2 个,等步数对照表整个
    # 消失。只有真正异常的 failed / timeout / diverged 才排除。
    VALID = ("ok", "max_epochs")
    ok = df[df["status"].isin(VALID)]
    main = ok[ok["tag"] == "main"]
    bad = df[~df["status"].isin(VALID)]
    n_unconverged = int((ok["status"] == "max_epochs").sum())

    md = ["# 任务二 · 链路预测 实验结果\n",
          "> 本文件由 `task2_link_prediction/code/report.py` 从 `results/runs.jsonl` "
          "自动生成,**请勿手工编辑**。\n",
          "\n**评测口径**:AUC / AP 用正负边得分直接算;MRR 与 Hits@K 的口径是"
          "「每个正样本边与**全部**负样本边一起排名」,而不是只跟同批次内的少量"
          "负样本比 —— 后者会显著低估难度,让 MRR 虚高。\n"]

    md.append("\n## 表 1 · 全图训练 vs 子图采样训练\n")
    if main.empty:
        md.append("_(尚无主实验记录,请先运行 `python code/train.py --all`)_\n")
    else:
        for ds in sorted(main["dataset"].unique()):
            sub = main[main["dataset"] == ds]
            md.append(f"\n### {ds}\n")
            try:
                h = sub["extra_hparams"].iloc[0]
                md.append(f"监督边 训练 {int(h['n_train_edges']):,} / "
                          f"验证 {int(h['n_val_edges']):,} / "
                          f"测试 {int(h['n_test_edges']):,}"
                          f"(原图 {int(h['num_edges']):,} 条边)\n")
            except Exception:  # noqa: BLE001
                pass
            head = ["模型", "模式", "测试 AUC", "AP", "MRR", "Hits@10",
                    "每 epoch 步数", "每 epoch 耗时", "种子数"]
            body = []
            for m in MODELS:
                for mode, label in (("full", "全图"), ("sampler", "采样")):
                    g = sub[(sub["model"] == m) & (sub["mode"] == mode)]
                    if g.empty:
                        continue
                    body.append([
                        m, label, _mean_std(g["test"]), _mean_std(g["test_ap"]),
                        _mean_std(g["test_mrr"]), _mean_std(g["test_hits@10"]),
                        f"{g['grad_steps_per_epoch'].mean():.0f}",
                        f"{g['sec_per_epoch'].mean() * 1000:.1f} ms", len(g),
                    ])
            md.append(_md_table(head, body))

        md.append("\n**采样相对全图的倍数**(同一模型内比较)\n")
        head = ["数据集", "模型", "AUC 变化", "每 epoch 耗时倍数", "梯度步数倍数"]
        body = []
        for ds in sorted(main["dataset"].unique()):
            sub = main[main["dataset"] == ds]
            for m in MODELS:
                f = sub[(sub["model"] == m) & (sub["mode"] == "full")]
                s = sub[(sub["model"] == m) & (sub["mode"] == "sampler")]
                if f.empty or s.empty:
                    continue
                body.append([
                    ds, m, f"{s['test'].mean() - f['test'].mean():+.4f}",
                    f"{s['sec_per_epoch'].mean() / max(f['sec_per_epoch'].mean(), 1e-9):.1f}×",
                    f"{s['grad_steps_per_epoch'].mean() / max(f['grad_steps_per_epoch'].mean(), 1):.0f}×",
                ])
        md.append(_md_table(head, body))

    # ---------------- 等梯度步数对照 ----------------
    if n_unconverged:
        md.append(f"\n> ⚠️ **注意**:表 1 的 {len(main)} 条记录中有 **{n_unconverged}** 条"
                  f"跑满 epoch 上限时早停仍未触发(`status=max_epochs`)。"
                  f"这些 run 在预算耗尽时**验证指标仍在改善**,说明它们**被预算限制了**、"
                  f"并未收敛。表中的数字是「最佳验证轮」的值,对它们是**保守的**。\n")

    eq = ok[ok["tag"] == "eqsteps"]
    md.append("\n## 表 2 · 等梯度步数对照(排除\"更新次数\"这个混淆因素)\n")
    if eq.empty:
        md.append("_(未跑,可用 `code/equal_steps.py` 生成)_\n")
    else:
        md.append(
            "\n任务一二里反复强调,\"1 个 epoch\"在两种模式下**不是同一件事**:全图 1 次"
            "梯度更新,采样几十次。上表里采样模式 AUC 明显更高,**其中一部分只是"
            "\"多做了几十倍优化\"的假象**,而非采样方法本身的优势。\n"
            "\n本表把两者的**累计梯度更新次数**对齐(全图跑满 3300 epoch,与采样模式"
            "100 epoch × 33 步持平),看差距还剩多少:\n")
        head = ["数据集", "模型", "全图(100 ep)", "全图(3300 ep,等步数)",
                "采样(100 ep)", "原差距", "等步数后差距", "缩小"]
        body = []
        for ds in sorted(eq["dataset"].unique()):
            for m in MODELS:
                e = eq[(eq["dataset"] == ds) & (eq["model"] == m)]
                f = main[(main["dataset"] == ds) & (main["model"] == m)
                         & (main["mode"] == "full")]
                s = main[(main["dataset"] == ds) & (main["model"] == m)
                         & (main["mode"] == "sampler")]
                if e.empty or f.empty or s.empty:
                    continue
                fv, sv, ev = f["test"].mean(), s["test"].mean(), e["test"].mean()
                d0, d1 = sv - fv, sv - ev
                body.append([
                    ds, m, f"{fv:.4f}", f"{ev:.4f}", f"{sv:.4f}",
                    f"{d0:+.4f}", f"{d1:+.4f}",
                    f"{(1 - d1 / d0) * 100:.0f}%" if d0 else "—",
                ])
        md.append(_md_table(head, body))
        md.append(
            "\n**怎么读**:最后一列是差距缩小的比例。若缩到 0,说明采样没有任何"
            "真实优势,全部来自更新次数;若几乎不缩,说明更新时间不是主因。\n")

    md.append("\n## 表 3 · 未正常完成的 run\n")
    if bad.empty:
        md.append("\n全部 run 均正常完成。\n")
    else:
        md.append(_md_table(
            ["数据集", "模型", "模式", "种子", "状态", "说明"],
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
