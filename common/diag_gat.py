"""诊断:Flickr 上 GAT 塌缩到多数类,是否由学习率过高引起。

## 现象

Flickr 全图训练,GCN 用标准化的特征得到 test=0.5110,而 GAT 只有 0.4234 ——
后者恰好等于训练集最大类占比(0.4216),且 ``best_epoch=1``(第 1 轮就最好)。
说明 GAT 在前两轮就把表示学坏了,之后一路退化,patience 到点后早停。

## 与本项目已修的另一处问题的区别

之前 GCN 塌缩是因为**特征没标准化**(量级 0~1827);修完后 GCN 正常。
GAT 这次是在**同样的标准化特征**上塌缩 —— 所以成因不同,不能套用同一个解释。

GAT 与前三个模型的差别在注意力:GATConv 每层对邻居做 softmax 加权,
4 个头、``concat=False``(即四头结果取平均)。输入 500 维、89k 节点时,
注意力 logits 的方差容易偏大,softmax 饱和 -> 梯度消失/爆炸。典型对策是
**降低学习率**。

本脚本在若干学习率下各跑一小段,打印验证曲线,看哪一档能稳定收敛。

用法::

    python -m common.diag_gat
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
from common.experiments import _prepare_features  # noqa: E402
from common.models import build_model  # noqa: E402
from common.trainer import node_classification_metrics, run_training  # noqa: E402
from common.utils import fix_console_encoding, set_seed, setup  # noqa: E402

EPOCHS = 40


def run_one(lr: float, seed: int = 0, epochs: int = EPOCHS, verbose: bool = False):
    set_seed(seed)
    data = _prepare_features(load_node_dataset("task1", "Flickr")[0], "Flickr")
    masks = {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}

    model = build_model("node", "GAT", in_channels=data.num_features,
                        hidden_channels=64, num_layers=2,
                        out_channels=int(data.y.max()) + 1, dropout=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

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

    return run_training(model, opt, batches, step_fn, eval_fn,
                        max_epochs=epochs, patience=10_000, monitor="val",
                        grad_clip=5.0, verbose=verbose)


def main() -> int:
    fix_console_encoding()
    setup(seed=0, verbose=False)

    # 参考:训练集最大类占比,val 长期贴着这个数就说明模型在瞎猜
    print(f"Flickr / GAT / 全图,{EPOCHS} epoch(无早停)\n")
    print("  参考:多数类占比 0.4216 —— 验证准确率贴住它就是塌缩\n")

    rows = []
    for lr in (0.01, 0.005, 0.002, 0.001):
        r = run_one(lr)
        vs = [c["val_val"] for c in r.curve]
        best = r.curve[r.best_epoch]
        rows.append((lr, r, vs))
        head = " ".join(f"{vs[i]:.3f}" for i in range(0, min(len(vs), 40), 4))
        print(f"  lr={lr:<6} best_ep={r.best_epoch:3d} "
              f"test={best.get('val_test', float('nan')):.4f}  "
              f"val 每 4 轮: {head}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
