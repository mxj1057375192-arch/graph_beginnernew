"""诊断:Flickr 上模型塌缩到多数类,是否是特征尺度的问题。

## 现象

Flickr 上 GCN 全图训练得到 test=0.4234,而训练集里最大类占比 0.4216 —— 两者
几乎相等,说明模型只是**把全部节点预测成类 6**,没有学到任何结构信息。
更直接的证据:3 个不同随机种子给出**完全相同**的 4 位小数准确率 —— 塌缩到
常数预测之后,初始化不再影响结果。

## 怀疑

Flickr 的特征是原始词频计数,实测范围 0 ~ 1827。而 Cora/Citeseer 在
`run_node_classification` 里做了行归一化,Flickr 没有。特征尺度差三个数量级会
让第一层的输出直接饱和。

本脚本对比三种特征处理,验证这个怀疑。

用法::

    python -m common.diag_flickr
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.datasets import load_node_dataset  # noqa: E402
from common.models import build_model  # noqa: E402
from common.trainer import node_classification_metrics, run_training  # noqa: E402
from common.utils import fix_console_encoding, setup  # noqa: E402

EPOCHS = 60


def _treat(data, how: str):
    """按 `how` 处理特征。返回 (处理后的 data, 说明)。"""
    d = data.clone()
    if how == "raw":
        return d, "不处理(当前行为)"
    if how == "rownorm":
        # 行归一化:每个节点的特征向量除以自身 L1 范数
        s = d.x.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        d.x = d.x / s
        return d, "行归一化 (L1,等价于 NormalizeFeatures)"
    if how == "standardize":
        # 按列标准化到零均值单位方差
        mu = d.x.mean(dim=0, keepdim=True)
        sd = d.x.std(dim=0, keepdim=True).clamp(min=1e-9)
        d.x = (d.x - mu) / sd
        return d, "列标准化 (z-score)"
    if how == "l2norm":
        s = d.x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
        d.x = d.x / s
        return d, "行 L2 归一化"
    raise ValueError(how)


def run_one(how: str, epochs: int = EPOCHS, seed: int = 0, verbose: bool = False):
    from common.utils import set_seed

    set_seed(seed)
    data = load_node_dataset("task1", "Flickr")[0]
    data, desc = _treat(data, how)
    masks = {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}
    num_classes = int(data.y.max()) + 1

    model = build_model("node", "GCN", in_channels=data.num_features,
                        hidden_channels=64, num_layers=2,
                        out_channels=num_classes, dropout=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    @torch.no_grad()
    def eval_fn(m):
        m.eval()
        return node_classification_metrics(m(data.x, data.edge_index), data.y, masks)

    def batches():
        return [data], 1

    def step_fn(m, b):
        logits = m(b.x, b.edge_index)
        return F.cross_entropy(logits[b.train_mask], b.y[b.train_mask]), \
            int(b.train_mask.sum())

    r = run_training(model, opt, batches, step_fn, eval_fn,
                     max_epochs=epochs, patience=10_000, monitor="val",
                     grad_clip=5.0, verbose=verbose)
    final = r.curve[r.best_epoch]
    return {
        "how": how, "desc": desc,
        "train_acc": final.get("val_train", float("nan")),
        "val_acc": final.get("val_val", float("nan")),
        "test_acc": final.get("val_test", float("nan")),
        "best_epoch": r.best_epoch,
    }


def main() -> int:
    fix_console_encoding()
    setup(seed=0, verbose=False)
    print(f"Flickr 特征处理对照实验(GCN 全图,{EPOCHS} epoch)\n")
    print(f"  {'处理方式':<40}{'训练准确率':>10}{'验证':>9}{'测试':>9}")
    print("  " + "-" * 68)
    out = []
    for how in ("raw", "rownorm", "l2norm", "standardize"):
        r = run_one(how)
        out.append(r)
        print(f"  {r['desc']:<40}{r['train_acc']:>10.4f}"
              f"{r['val_acc']:>9.4f}{r['test_acc']:>9.4f}")
    print(f"\n  参考:训练集最大类占比 0.4216 —— 测试准确率接近这个数就说明模型在瞎猜")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
