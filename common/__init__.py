"""四个任务共享的工具包。

各任务脚本通过下面的方式把仓库根目录加入 sys.path 后即可 ``import common``::

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # code/ -> taskX/ -> 仓库根

    from common.utils import setup
"""

__all__ = ["datasets", "metrics", "models", "results", "trainer", "utils"]
