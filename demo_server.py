# -*- coding: utf-8 -*-
"""本地 Web 演示：浏览器上传图像，返回判定报告。

零第三方依赖，基于 Python 内置 http.server。

用法：
    python demo_server.py [--port 8088]

然后浏览器打开 http://127.0.0.1:8088
仅监听本机回环地址，不做任何对外暴露。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aigc_forensics.detector import ImageAnalysisError, analyze_image, report_to_json  # noqa: E402

PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>AIGC-Forensics 检测演示</title>
<style>
 body {{ font-family: system-ui, 'Microsoft YaHei', sans-serif; max-width: 720px;
        margin: 40px auto; padding: 0 16px; color: #222; line-height: 1.6; }}
 .card {{ border: 1px solid #ccc; border-radius: 10px; padding: 18px 22px; margin: 14px 0; }}
 .verdict {{ font-size: 20px; font-weight: 500; }}
 .bar {{ background: #eee; border-radius: 6px; height: 14px; margin: 8px 0; }}
 .fill {{ background: #185FA5; border-radius: 6px; height: 14px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 td, th {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
 th {{ background: #f2f2f2; }}
 .muted {{ color: #666; font-size: 13px; }}
</style></head><body>
<h1>AIGC-Forensics</h1>
<p>AI 生成图像多信号融合取证检测演示（元数据 / 频域 / 噪声残差 / 尺寸格式）。</p>
<div class="card"><form method="post" enctype="multipart/form-data">
 <input type="file" name="file" accept="image/*" required>
 <button type="submit">开始分析</button>
</form></div>
{report}
<p class="muted">本工具为启发式取证辅助，输出为概率倾向而非事实认定；
社交平台转存、重压缩、截图、裁剪均会显著影响结果。</p>
</body></html>"""


def _render_report(report: dict) -> str:
    score_pct = round(report["score"] * 100)
    rows = ""
    for s in report["signals"]:
        rows += (
            "<tr><th>%s</th><td>score = %.2f（置信度 %.2f）</td><td>%s</td></tr>"
            % (html.escape(s["label"]), s["score"], s["confidence"], html.escape(s["explanation"]))
        )
    body = (
        '<div class="card"><div class="verdict">%s</div>'
        '<div class="bar"><div class="fill" style="width:%d%%"></div></div>'
        '<div class="muted">score = %.2f ｜ %s</div>'
        '<table><tr><th>信号</th><th>得分</th><th>解释</th></tr>%s</table></div>'
        % (html.escape(report["verdict_text"]), score_pct, report["score"],
           html.escape(report["file"]), rows)
    )
    return body


def _parse_multipart(body: bytes, boundary: str) -> tuple:
    """返回 (第一个带 filename 的 part 载荷, 原始文件名)；未找到返回 (None, '')。"""
    delim = b"--" + boundary.encode("latin-1")
    for chunk in body.split(delim):
        head, sep, payload = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        m = re.search(rb'filename="([^"]*)"', head)
        if m and payload.endswith(b"\r\n"):
            return payload[:-2], m.group(1).decode("utf-8", "replace")
    return None, ""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, content: str, ctype: str) -> None:
        data = content.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send(200, PAGE.format(report=""), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            ctype = self.headers.get("Content-Type", "")
            m = re.search(r'boundary=([^;]+)', ctype)
            if not m:
                self._send(400, "缺少 multipart boundary", "text/plain; charset=utf-8")
                return
            payload, upload_name = _parse_multipart(body, m.group(1).strip('"'))
            if payload is None:
                self._send(400, "未找到上传的文件字段", "text/plain; charset=utf-8")
                return
            # 从原始文件名继承后缀，供 analyze_image 的类型白名单校验使用。
            suffix = Path(upload_name).suffix.lower() or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(payload)
                tmp_path = tmp.name
            try:
                report = analyze_image(tmp_path)
                report["file"] = "上传的图像"
            except ImageAnalysisError as exc:
                self._send(400, PAGE.format(report="<div class='card'>分析失败：%s</div>" % html.escape(str(exc))),
                           "text/html; charset=utf-8")
                return
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            accept = self.headers.get("Accept", "")
            if "application/json" in accept:
                self._send(200, report_to_json(report), "application/json; charset=utf-8")
            else:
                self._send(200, PAGE.format(report=_render_report(report)), "text/html; charset=utf-8")
        except Exception as exc:  # 兜底：服务器不得因单个请求崩溃
            self._send(500, "服务器内部错误: %s" % html.escape(repr(exc)), "text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:  # 精简控制台日志
        print("[demo] " + fmt % args)


def main() -> int:
    parser = argparse.ArgumentParser(description="AIGC-Forensics 本地 Web 演示")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("演示服务已启动: http://127.0.0.1:%d （Ctrl+C 退出）" % args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
