"""数据集加载。

各任务的数据集分别放在自己的 ``data/`` 下(任务书要求的目录格式)。Cora /
Citeseer / Flickr 同时被任务一和任务二使用 —— 这里**有意各存一份**,不做
跨任务共享,以换取"每个任务文件夹都能独立运行"这一性质,便于逐个评审。

直接运行本模块会下载全部数据集并打印实测规模::

    python -m common.datasets
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 数据集清单:各任务用到哪些数据
# ---------------------------------------------------------------------------
TASKS: dict[str, str] = {
    "task1": "task1_node_classification",
    "task2": "task2_link_prediction",
    "task3": "task3_graph_classification",
    "task4": "task4_knowledge_graph",
}

# 任务书指定的数据集
NODE_DATASETS = ["Cora", "Citeseer", "Flickr"]
TU_DATASETS = ["MUTAG", "PROTEINS", "ENZYMES"]


def task_dir(task: str) -> Path:
    """接受 'task1' 或完整目录名,返回任务目录。"""
    return REPO_ROOT / TASKS.get(task, task)


def data_root(task: str) -> Path:
    """返回某任务的 data/ 目录,不存在则创建。"""
    p = task_dir(task) / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 节点级数据集(任务一、任务二)
# ---------------------------------------------------------------------------
def load_planetoid(task: str, name: str):
    """Cora / Citeseer。返回 PyG Planetoid 数据集对象(内含单张图)。"""
    from torch_geometric.datasets import Planetoid

    assert name in ("Cora", "Citeseer"), f"Planetoid 只支持 Cora/Citeseer,收到 {name}"
    root = data_root(task)
    _ensure_planetoid_raw(root, name)
    return Planetoid(root=str(root), name=name)


# ---------------------------------------------------------------------------
# Planetoid 手动下载兜底
# ---------------------------------------------------------------------------
# 背景:PyG 2.8 的 Planetoid.download() 用 fsspec 的 fs.cp() 远程复制文件,
# 而该函数会先对源 URL 做一次"是否目录"的探测(HTTP 列目录)。对本网络下的
# https://github.com/kimiyoung/planetoid/raw/... 这次探测会以
# "OSError: [WinError 121] 信号灯超时时间已到" 失败,导致完全无法下载。
#
# 本机实测(2026-09-27)的下载源表现:
#   raw.githubusercontent.com  间歇性超时,同一文件几分钟内可下、随后又挂
#                              (ind.cora.x 先成功 22119B,稍后同一个 URL 超时)
#   gitee.com 镜像             稳定,0.7~0.9s,文件大小与上游一致
# 因此把 gitee 放在首位,raw.githubusercontent 仅作备用。
PLANETOID_MIRRORS = [
    "https://gitee.com/mirrors_Kimiyoung/planetoid/raw/master/data",
    "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data",
]

_PLANETOID_SUFFIXES = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]


def _download_file(url: str, dest: Path, retries: int = 3, timeout: int = 25) -> None:
    """下载单个文件到 dest,失败则重试。已存在且非空则跳过。"""
    import urllib.error
    import urllib.request

    if dest.exists() and dest.stat().st_size > 0:
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise ValueError("返回内容为空")
            # 先写临时文件再改名,避免下载中断留下半截文件被误判为已完成
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, OSError, ValueError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"下载失败 {url}\n  最后一次错误:{last_err}")


def _ensure_planetoid_raw(root: Path, name: str) -> None:
    """确保 Planetoid 的 8 个原始文件存在于 raw/ 下,缺失则手动补齐。"""
    raw_dir = root / name / "raw"
    prefix = name.lower()
    missing = [f"ind.{prefix}.{s}" for s in _PLANETOID_SUFFIXES
               if not (raw_dir / f"ind.{prefix}.{s}").exists()]

    if not missing:
        return

    print(f"  {name}: 需补下 {len(missing)} 个文件(PyG 自带下载在本网络不可用,改用手动下载)")
    errors: list[str] = []
    for fname in missing:
        ok = False
        for base in PLANETOID_MIRRORS:
            try:
                _download_file(f"{base}/{fname}", raw_dir / fname)
                ok = True
                break
            except RuntimeError as e:
                errors.append(f"{base}/{fname}: {str(e).splitlines()[0]}")
        if not ok:
            raise RuntimeError(
                f"{name} 的 {fname} 从所有镜像均下载失败:\n  " + "\n  ".join(errors[-3:])
            )
    print(f"  {name}: 原始文件已就绪 -> {raw_dir}")


def load_flickr(task: str):
    """Flickr(GraphSAINT 版本)。

    单张大图(8.9 万节点 / 89 万边),是任务书里唯一真正需要子图采样才能
    高效训练的数据集 —— 也正是"全图 vs 采样"对比最有说服力的地方。
    """
    from torch_geometric.datasets import Flickr

    root = data_root(task)
    _ensure_flickr_raw(root)
    return Flickr(root=str(root / "Flickr"))


# ---------------------------------------------------------------------------
# Flickr 手动下载
# ---------------------------------------------------------------------------
# 背景:PyG 的 Flickr 从 Google Drive 取 4 个文件。在国内网络下 Google Drive
# 不可直连,且即便挂了代理,drive.usercontent.google.com 传输大文件时也会
# 反复断流(实测:PyG 自带的 urllib 下载停在 2.4MB,curl 一次停在 167KB)。
# 因此这里实现**断点续传**:每次断了就从已下载的字节数继续,直到文件完整。
FLICKR_FILES = {
    "adj_full.npz": "1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy",
    "feats.npy": "1join-XdvX3anJU_MLVtick7MgeAQiWIZ",
    "class_map.json": "1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9",
    "role.json": "1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7",
}
_GOOGLE_DRIVE = "https://drive.usercontent.google.com/download"


def _download_resumable(
    url: str, dest: Path, max_attempts: int = 40, timeout: int = 60
) -> None:
    """断点续传下载。

    ## 为什么必须区分 206 和 200

    续传的前提是服务端**真的**支持 Range 请求,并以 206 Partial Content 回答。
    但服务端完全可能忽略 Range 头,直接返回 **200 + 完整内容**(代理抖动、
    网盘限速时很常见)。若此时仍以追加模式写入,会:

      1. 把整份内容追加到已有的半截文件后面,文件体积翻倍甚至膨胀到几十倍;
      2. 200 响应没有 Content-Range,而 Content-Length 分支又要求 ``not have``,
         于是 ``total`` 始终为 None,两个 break 条件都不成立 —— **循环空转
         满 max_attempts 轮**,每轮都再追加一整份。

    对一个 357MB 的文件,最坏情况就是 40 × 357MB ≈ 14GB 的无效写入。

    因此这里按 HTTP 状态码分流:206 才追加;200 一律**截断重写**。
    """
    import urllib.error
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    total: int | None = None
    last_err: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total is not None and have >= total:
            break

        headers = {"Range": f"bytes={have}-"} if have else {}
        req = urllib.request.Request(url, headers=headers)
        completed = False

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None)
                cr = resp.headers.get("Content-Range")
                cl = resp.headers.get("Content-Length")

                if status == 206 and cr and "/" in cr:
                    # 服务端遵守了 Range:从 have 处续传
                    total = int(cr.rsplit("/", 1)[-1])
                    mode = "ab"
                else:
                    # 服务端忽略了 Range,返回的是完整内容:
                    # 必须截断重写,并从头开始计数
                    total = int(cl) if cl else None
                    mode = "wb"
                    have = 0

                with open(tmp, mode) as f:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        if total is not None and f.tell() > total:
                            raise ValueError(
                                f"收到内容超过声明大小({f.tell()} > {total}),已中止"
                            )
                completed = True  # 读到了流末尾,这一轮传输完整结束

        except (urllib.error.URLError, OSError, ValueError) as e:
            last_err = e
            time.sleep(min(1.0 * attempt, 5.0))
            continue

        if completed:
            # 传输完整结束。若服务端给了 total 则核对大小;没给也无法再续,
            # 一律接受 —— 再循环一轮只会重复下载。
            if total is None or tmp.stat().st_size >= total:
                break
        else:
            time.sleep(0.5)

    got = tmp.stat().st_size if tmp.exists() else 0
    if total is not None and got > total:
        raise RuntimeError(f"下载内容超过预期:{dest.name} 得到 {got} 字节,应为 {total} 字节")
    if total is not None and got >= total:
        tmp.replace(dest)
        return
    if total is None and got > 0 and last_err is None:
        # 服务端未提供长度信息,但已完整传输完毕
        tmp.replace(dest)
        return

    raise RuntimeError(
        f"下载未完成:{dest.name} 已获取 {got} 字节"
        + (f" / 共 {total} 字节" if total else "")
        + f",尝试 {max_attempts} 轮。最后一次错误:{last_err}"
    )


def _is_valid_dataset_file(path: Path) -> bool:
    """校验已下载文件能否真正被解析。

    只检查"文件存在且非空"是不够的 —— 下载中断会留下一个**非空的半截文件**
    (实测卡在 167KB 的 adj_full.npz),它会被误判为已完成,直到训练时才炸。
    这里直接尝试解析,把问题拦在下载阶段。
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        if path.suffix == ".npz":
            import numpy as np

            with np.load(path) as z:
                z.files  # 触发真正的 ZIP 目录读取
            return True
        if path.suffix == ".npy":
            import numpy as np

            np.load(path, mmap_mode="r")
            return True
        if path.suffix == ".json":
            import json

            json.loads(path.read_text(encoding="utf-8"))
            return True
    except Exception:  # noqa: BLE001 —— 任何解析失败都说明文件不完整
        return False
    return path.stat().st_size > 0


def _ensure_flickr_raw(root: Path) -> None:
    """确保 Flickr 的 4 个原始文件完整,缺失或损坏则手动补齐。"""
    raw_dir = root / "Flickr" / "raw"
    bad = [n for n in FLICKR_FILES if not _is_valid_dataset_file(raw_dir / n)]
    if not bad:
        return

    print(f"  Flickr: 需下载 {len(bad)} 个文件(Google Drive,支持断点续传)")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in bad:
        dest = raw_dir / name
        if dest.exists():
            dest.unlink()  # 丢弃损坏的半截文件,从头下
        url = f"{_GOOGLE_DRIVE}?id={FLICKR_FILES[name]}&confirm=t&export=download"
        print(f"    {name} ...", end="", flush=True)
        _download_resumable(url, dest)
        print(f" {dest.stat().st_size / 1e6:.1f} MB ✓", flush=True)
    print(f"  Flickr: 原始文件已就绪 -> {raw_dir}")


def load_node_dataset(task: str, name: str):
    """按名字加载节点级数据集(Cora / Citeseer / Flickr)。"""
    if name in ("Cora", "Citeseer"):
        dataset = load_planetoid(task, name)
    elif name == "Flickr":
        dataset = load_flickr(task)
    else:
        raise ValueError(f"未知的节点级数据集:{name}")
    return dataset


# ---------------------------------------------------------------------------
# 图级数据集(任务三)
# ---------------------------------------------------------------------------
def load_tu(task: str, name: str):
    """TUDataset 中的一个(如 MUTAG / PROTEINS / ENZYMES)。"""
    from torch_geometric.datasets import TUDataset

    root = data_root(task) / "TU"
    _ensure_tu_raw(root, name)
    return TUDataset(root=str(root), name=name)


# ---------------------------------------------------------------------------
# TUDataset 手动下载兜底
# ---------------------------------------------------------------------------
# 背景:PyG 用 fsspec 的 fs.cp(..., extract=True) 下载并解压。实测即使在
# 网络完全正常的情况下(同一 URL 用 curl 取稳定返回 200),fsspec 仍会因
# 瞬时抖动抛出 FileNotFoundError —— 错误信息极具误导性,看起来像"文件不存在",
# 实际是"这次连接没成功"。因此改为自己下载 + 解压,并带重试。
TU_URL = "https://www.chrsmrrs.com/graphkerneldatasets"


def _ensure_tu_raw(root: Path, name: str) -> None:
    """确保 TUDataset 的原始文件已解压到 raw/,缺失则手动补齐。"""
    import zipfile

    raw_dir = root / name / "raw"
    if raw_dir.exists() and any(raw_dir.glob(f"{name}_*.txt")):
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {name}: 手动下载并解压 ...", end="", flush=True)

    zpath = raw_dir / f"{name}.zip"
    _download_file(f"{TU_URL}/{name}.zip", zpath, retries=4, timeout=90)

    # 解压:zip 内可能有一层与数据集同名的目录,需要摊平到 raw/ 下,
    # 否则 PyG 找不到 {name}_*.txt
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():
            if member.endswith("/"):
                continue
            parts = Path(member).parts
            # 去掉与数据集同名的最外层目录
            if len(parts) > 1 and parts[0].lower() == name.lower():
                parts = parts[1:]
            if not parts:
                continue
            dest = raw_dir.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(dest, "wb") as out:
                out.write(src.read())

    zpath.unlink(missing_ok=True)  # 解压后删掉 zip,避免 PyG 再次解压
    print(f" 完成({len(list(raw_dir.glob(f'{name}_*.txt')))} 个数据文件)")


def load_zinc(task: str, subset: bool = True, split: str = "train"):
    """ZINC 分子图数据集。

    注意这是**回归**任务(目标是约束对数溶解度),评价用 MAE 而非准确率;
    且节点只有原子类型、没有连续特征,模型首层需要 embedding。

    Args:
        subset: True 时只用 1 万张图的子集(作业在 CPU 上跑,必须用子集)。
    """
    from torch_geometric.datasets import ZINC

    return ZINC(root=str(data_root(task) / "ZINC"), subset=subset, split=split)


# ---------------------------------------------------------------------------
# 规模描述(用于校验下载完整性)
# ---------------------------------------------------------------------------
def describe(dataset: Any, name: str = "") -> dict[str, Any]:
    """返回数据集的实际规模,用于确认下载完整、字段符合预期。

    **实测而非照抄文档** —— 文档里的数字和当前 PyG 版本实际给到的可能不一致,
    而实验报告里的所有描述都必须基于真实加载到的数据。
    """
    info: dict[str, Any] = {"name": name, "num_graphs": len(dataset)}

    try:
        d = dataset[0]
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        return info

    info["num_nodes"] = int(d.num_nodes)
    info["num_edges"] = int(d.num_edges)
    info["num_node_features"] = int(dataset.num_node_features)
    info["num_edge_features"] = int(dataset.num_edge_features)
    info["num_classes"] = int(getattr(dataset, "num_classes", 0))

    # 单图数据集 vs 多图数据集,字段含义不同
    info["is_single_graph"] = len(dataset) == 1
    if info["is_single_graph"]:
        info["has_train_mask"] = bool(hasattr(d, "train_mask") and d.train_mask is not None)
        if info["has_train_mask"]:
            info["n_train"] = int(d.train_mask.sum())
            info["n_val"] = int(d.val_mask.sum())
            info["n_test"] = int(d.test_mask.sum())
        info["avg_degree"] = round(2 * info["num_edges"] / max(info["num_nodes"], 1), 2)
    else:
        # 多图:统计节点/边数的分布,以及标签是否为连续值(判断回归/分类)
        import torch

        n = [int(g.num_nodes) for g in dataset[:200]]
        info["nodes_min"] = min(n)
        info["nodes_max"] = max(n)
        info["nodes_mean"] = round(sum(n) / len(n), 1)
        y = d.y
        info["label_is_float"] = bool(torch.is_floating_point(y))
        info["label_shape"] = list(y.shape)

    return info


def format_info(info: dict[str, Any]) -> str:
    """把 describe 的结果排成一行,便于日志里扫读。"""
    if "error" in info:
        return f"  {info['name']:10} ✗ {info['error']}"

    if info["is_single_graph"]:
        s = (f"  {info['name']:10} 单图  {info['num_nodes']:>7,} 节点 "
             f"{info['num_edges']:>9,} 边  度均值 {info['avg_degree']:>5.2f}  "
             f"特征 {info['num_node_features']:>4} 维  类别 {info['num_classes']}")
        if info.get("has_train_mask"):
            s += f"  划分 {info['n_train']}/{info['n_val']}/{info['n_test']}"
    else:
        s = (f"  {info['name']:10} {info['num_graphs']:>6,} 张图  "
             f"节点数 {info['nodes_min']}~{info['nodes_max']}"
             f"(均 {info['nodes_mean']})  "
             f"特征 {info['num_node_features']:>4} 维  类别 {info['num_classes']}")
        if info.get("label_is_float"):
            s += f"  ⚠ 标签为浮点 → 回归任务 {info['label_shape']}"
    return s


# ---------------------------------------------------------------------------
# 全量下载
# ---------------------------------------------------------------------------
def download_all(verbose: bool = True) -> list[dict[str, Any]]:
    """下载全部数据集并校验。返回每项的规模信息。

    每个数据集**独立容错**:某一个失败不会中断其余下载。早期版本没有这层
    保护,PROTEINS 的一次瞬时失败直接让后续的 ZINC 根本没被尝试下载。
    """

    def _one(label: str, loader, name: str) -> dict[str, Any]:
        if verbose:
            print(f"{label} 下载 {name} ...", flush=True)
        try:
            return describe(loader(), name)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  ✗ {name} 失败:{type(e).__name__}: {str(e)[:120]}", flush=True)
            return {"name": name, "error": f"{type(e).__name__}: {e}"}

    results: list[dict[str, Any]] = []

    for name in NODE_DATASETS:
        results.append(_one("[任务一]", lambda n=name: load_node_dataset("task1", n), name))
    for name in NODE_DATASETS:
        results.append(_one("[任务二]", lambda n=name: load_node_dataset("task2", n), f"{name}(task2)"))
    for name in TU_DATASETS:
        results.append(_one("[任务三]", lambda n=name: load_tu("task3", n), name))
    for split in ("train", "val", "test"):
        results.append(_one("[任务三]", lambda s=split: load_zinc("task3", subset=True, split=s),
                            f"ZINC/{split}"))

    return results


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    from common.utils import banner, fix_console_encoding

    fix_console_encoding()
    banner("数据集下载与校验")
    infos = download_all()
    banner("实测规模")
    for info in infos:
        print(format_info(info))
    print()
    print("完成。以上数字为实际加载所得,报告中的数据集描述请以此为准。")
