"""公共工具:控制台编码、随机种子、线程数、日志。

每个任务脚本开头调用一次 :func:`setup` 即可(它会做控制台编码修复等一次性设置)。
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# 控制台编码
# --------------------------------------------------------------------------
def fix_console_encoding() -> None:
    """把 stdout/stderr 切到 UTF-8。

    Windows 控制台默认是 GBK(cp936),直接 print 中文会乱码甚至抛
    UnicodeEncodeError。与其要求调用方每次记得设 ``PYTHONIOENCODING=utf-8``,
    不如在代码里自我修复 —— 脚本被任何人用任何方式启动都能正常输出中文。
    """
    for stream in (sys.stdout, sys.stderr):
        # reconfigure 需要 Python 3.7+;被重定向到非文本流时可能没有该方法
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # 流已被关闭或不支持重配置,忽略即可 —— 不该因此让脚本挂掉
                pass


# --------------------------------------------------------------------------
# 随机种子
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定所有随机源。

    注意:即便固定了种子,CPU 上部分算子的并行归约顺序仍可能引入微小非确定性。
    因此实验结论要建立在多种子重复的 mean ± std 上,而不是单次精确复现。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# 线程数
# --------------------------------------------------------------------------
def setup_threads(num_threads: int | None = None) -> int:
    """设置 torch 的 CPU 线程数并返回生效值。

    本机 i5-1135G7 是 4 核 8 线程。实测 Cora 全图 GCN 的每 epoch 耗时:

        线程数   1      2      3      4      6      7      8
        ms     26.3   20.4   18.3   17.6   16.3   16.2   17.8

    7 线程最快(即逻辑核数减一),故取此为默认值。注意 1 线程仅比 7 线程慢
    1.62 倍 —— 说明该负载受内存带宽与调度开销主导,并非计算受限,这与
    "多加线程就能大幅加速"的直觉不符。

    **重要**:任务一要比较"运行时间",两种训练模式必须使用**相同**的线程数,
    否则线程数差异会淹没真正要测量的效应。调用方不要在中途改动它。
    """
    import torch

    if num_threads is None:
        # 实测最优:留一个逻辑核给数据加载与系统
        num_threads = max(1, (os.cpu_count() or 4) - 1)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # 只能在并行工作启动前设置,重复调用会抛错 —— 忽略
        pass
    return torch.get_num_threads()


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
def setup(seed: int = 0, num_threads: int | None = None, verbose: bool = True) -> dict[str, Any]:
    """任务脚本的一站式初始化。返回环境信息字典,便于记入结果。"""
    fix_console_encoding()
    set_seed(seed)

    info: dict[str, Any] = {"seed": seed}
    try:
        info["num_threads"] = setup_threads(num_threads)
    except ImportError:
        info["num_threads"] = None

    if verbose:
        from common.results import environment_snapshot

        snap = environment_snapshot()
        print(f"Python {snap['python']} | torch {snap['torch']} | "
              f"PyG {snap['torch_geometric']} | CUDA {snap['cuda_available']}")
        print(f"CPU 线程数 {info['num_threads']} | 随机种子 {seed}")
    return info


def banner(title: str, char: str = "=", width: int = 72) -> None:
    """打印醒目的分节标题,便于在长日志里定位。"""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def human_time(seconds: float) -> str:
    """把秒数格式化成易读形式。"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m"
