"""任务四:数据准备 —— 为 kge_framework 补齐缺失的 id 映射文件。

## 背景

kge_framework 的 ``code/utils.py: build_data()`` 要求每个数据集目录下同时存在:

    entity2id.txt     名称 -> 编号
    relation2id.txt   名称 -> 编号

但仓库里**只有 FB15k-237 带这两个文件**;它自带的 countries_S1/S2/S3、FB15k、
wn18、wn18rr、YAGO3-10 都只有 ``entities.dict`` / ``relations.dict``,而且列序
相反(编号在前、名称在后):

    entities.dict    0<TAB>nepal          # 编号 名称
    entity2id.txt    nepal<TAB>0          # 名称 编号   <- 框架实际要的

因此那些数据集**开箱即用不了**,一跑就报
``FileNotFoundError: data/xxx/entity2id.txt``。

本脚本把 .dict 转成框架要的 .txt,属于**数据准备**而非改动框架逻辑。

用法::

    python code/prepare_data.py                  # 处理全部缺失的数据集
    python code/prepare_data.py countries_S1     # 只处理指定数据集
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import fix_console_encoding  # noqa: E402

TASK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TASK_DIR / "kge_framework" / "data"


def read_dict(path: Path) -> list[tuple[str, int]]:
    """读 '编号<TAB>名称' 格式的 .dict 文件。"""
    pairs: list[tuple[str, int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            pairs.append((parts[1], int(parts[0])))
    return pairs


def write_id_file(pairs: list[tuple[str, int]], path: Path) -> None:
    """写成 '名称<TAB>编号',与 FB15k-237 的 entity2id.txt 格式一致。"""
    with open(path, "w", encoding="utf-8") as f:
        for name, idx in pairs:
            f.write(f"{name}\t{idx}\n")


def prepare(dataset: str, force: bool = False) -> dict[str, object]:
    """为单个数据集生成 entity2id.txt / relation2id.txt。"""
    folder = DATA_DIR / dataset
    if not folder.is_dir():
        return {"dataset": dataset, "status": "skip", "reason": "目录不存在"}

    # 已有的不覆盖(尤其 FB15k-237,它自带官方版本,不应改动)
    e_out, r_out = folder / "entity2id.txt", folder / "relation2id.txt"
    if e_out.exists() and r_out.exists() and not force:
        return {"dataset": dataset, "status": "skip", "reason": "已存在"}

    e_dict, r_dict = folder / "entities.dict", folder / "relations.dict"
    if not e_dict.exists() or not r_dict.exists():
        return {"dataset": dataset, "status": "skip", "reason": "缺少 entities.dict/relations.dict"}

    ents = read_dict(e_dict)
    rels = read_dict(r_dict)
    write_id_file(ents, e_out)
    write_id_file(rels, r_out)
    return {
        "dataset": dataset,
        "status": "ok",
        "entities": len(ents),
        "relations": len(rels),
    }


def main(argv: list[str] | None = None) -> int:
    fix_console_encoding()
    ap = argparse.ArgumentParser(description="为 kge_framework 补齐 entity2id/relation2id")
    ap.add_argument("datasets", nargs="*", help="留空则处理全部")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    args = ap.parse_args(argv)

    datasets = args.datasets or sorted(
        d.name for d in DATA_DIR.iterdir() if d.is_dir()
    )

    print(f"数据目录:{DATA_DIR}\n")
    print(f"  {'数据集':<14}{'结果':<8}说明")
    print("  " + "-" * 58)
    n_ok = 0
    for name in datasets:
        r = prepare(name, force=args.force)
        if r["status"] == "ok":
            n_ok += 1
            print(f"  {name:<14}{'✓ 已生成':<8}实体 {r['entities']} 个,关系 {r['relations']} 个")
        else:
            print(f"  {name:<14}{'- 跳过':<8}{r.get('reason', '')}")

    print(f"\n完成,新生成 {n_ok} 个数据集的 id 映射文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
