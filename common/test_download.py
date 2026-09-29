"""断点续传下载的回归测试。

对应一个真实修复的缺陷:服务端**忽略 Range 头**、直接返回 200 + 完整内容时
(代理抖动、网盘限速时常见),旧实现仍以追加模式写入,导致

  * 整份内容被追加到半截文件后面,文件体积翻倍;
  * 200 响应没有 Content-Range,而 Content-Length 分支又要求 ``not have``,
    于是 ``total`` 恒为 None,两个 break 条件都不成立 —— 循环空转满
    ``max_attempts`` 轮,每轮再追加一整份。357MB 的文件最坏约 40 × 357MB ≈ 14GB。

本测试用**本地 HTTP 服务器**复现两种服务端行为,验证修复后两条路径都正确。

运行::

    python -m common.test_download
"""

from __future__ import annotations

import http.server
import os
import pathlib
import re
import socketserver
import sys
import tempfile
import threading
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.datasets import _download_resumable  # noqa: E402
from common.utils import fix_console_encoding  # noqa: E402

PAYLOAD = os.urandom(300_000)
PREFIX = 120_000  # 预置的"已下载"字节数


def _serve(directory: pathlib.Path) -> tuple[socketserver.TCPServer, int]:
    handler = type(
        "H",
        (http.server.SimpleHTTPRequestHandler,),
        {"__init__": lambda self, *a, **k: http.server.SimpleHTTPRequestHandler.__init__(
            self, *a, directory=str(directory), **k)},
    )

    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    srv = Server(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_ignores_range() -> bool:
    """服务端忽略 Range,返回 200 + 全量内容 —— 旧实现在这里会膨胀。"""
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "data.bin").write_bytes(PAYLOAD)
    srv, port = _serve(d)
    try:
        dest = d / "out.bin"
        dest.with_suffix(".bin.part").write_bytes(PAYLOAD[:PREFIX])

        _download_resumable(f"http://127.0.0.1:{port}/data.bin", dest,
                            max_attempts=40, timeout=20)

        got = dest.read_bytes()
        ok = len(got) == len(PAYLOAD) and got == PAYLOAD
        print(f"  [忽略 Range] 得到 {len(got):,} 字节,期望 {len(PAYLOAD):,}"
              f"  -> {'通过' if ok else '失败:文件被重复追加!'}")
        return ok
    finally:
        srv.shutdown()


def test_honours_range() -> bool:
    """服务端遵守 Range,返回 206 —— 验证续传能力没有被改坏。"""
    d = pathlib.Path(tempfile.mkdtemp())
    seen: list[str | None] = []

    class RangeHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            rng = self.headers.get("Range")
            seen.append(rng)
            if rng:
                start = int(re.match(r"bytes=(\d+)-", rng).group(1))
                body = PAYLOAD[start:]
                self.send_response(206)
                self.send_header("Content-Range",
                                 f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
            else:
                body = PAYLOAD
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # 静音
            pass

    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    srv = Server(("127.0.0.1", 0), RangeHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        dest = d / "out.bin"
        dest.with_suffix(".bin.part").write_bytes(PAYLOAD[:PREFIX])

        _download_resumable(f"http://127.0.0.1:{port}/data.bin", dest,
                            max_attempts=5, timeout=20)

        got = dest.read_bytes()
        resumed = any(r and r.startswith(f"bytes={PREFIX}-") for r in seen)
        ok = got == PAYLOAD and resumed
        print(f"  [遵守 Range] 收到请求头 {seen},结果 {len(got):,} 字节"
              f"  -> {'通过(确实续传)' if ok else '失败:未走续传'}")
        return ok
    finally:
        srv.shutdown()


def test_no_content_length() -> bool:
    """服务端不给 Content-Length(分块传输)—— 应能正常完成而非空转。"""
    d = pathlib.Path(tempfile.mkdtemp())

    class ChunkedHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(0, len(PAYLOAD), 8192):
                chunk = PAYLOAD[i:i + 8192]
                self.wfile.write(f"{len(chunk):x}\r\n".encode())
                self.wfile.write(chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

        def log_message(self, *a):
            pass

    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    srv = Server(("127.0.0.1", 0), ChunkedHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        dest = d / "out.bin"
        _download_resumable(f"http://127.0.0.1:{port}/data.bin", dest,
                            max_attempts=3, timeout=20)
        got = dest.read_bytes()
        ok = got == PAYLOAD
        print(f"  [无 Content-Length] 得到 {len(got):,} 字节"
              f"  -> {'通过' if ok else '失败'}")
        return ok
    finally:
        srv.shutdown()


def main() -> int:
    fix_console_encoding()
    print("断点续传下载回归测试\n")
    results = [
        ("忽略 Range(返回 200 全量)", test_ignores_range()),
        ("遵守 Range(返回 206 续传)", test_honours_range()),
        ("无 Content-Length(分块)", test_no_content_length()),
    ]
    print()
    passed = sum(1 for _, ok in results)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n{passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
