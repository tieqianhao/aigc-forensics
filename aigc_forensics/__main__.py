# -*- coding: utf-8 -*-
"""命令行入口。

用法：
    python -m aigc_forensics detect <image_path> [--json]
    python -m aigc_forensics batch  <dir_path>   [--json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .detector import (
    SUPPORTED_SUFFIXES,
    ImageAnalysisError,
    analyze_image,
    report_to_json,
    report_to_text,
)


def _cmd_detect(args: argparse.Namespace) -> int:
    try:
        report = analyze_image(args.image)
    except ImageAnalysisError as exc:
        if args.json:
            print('{"error": "%s"}' % exc)
        else:
            print("错误: %s" % exc, file=sys.stderr)
        return 1
    print(report_to_json(report) if args.json else report_to_text(report))
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    if not directory.is_dir():
        print("错误: 目录不存在: %s" % directory, file=sys.stderr)
        return 1

    files = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file()
    )
    results = []
    for p in files:
        try:
            results.append(analyze_image(str(p)))
        except ImageAnalysisError as exc:
            results.append({"file": str(p), "error": str(exc)})

    if args.json:
        import json

        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            if "error" in r:
                print("%s  ->  错误: %s" % (r["file"], r["error"]))
            else:
                print("%s  ->  %s (score = %.2f)" % (r["file"], r["verdict_text"], r["score"]))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="aigc_forensics",
        description="AI 生成图像多信号融合取证检测器（元数据 / 频域 / 噪声残差 / 尺寸格式）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_detect = sub.add_parser("detect", help="分析单张图像")
    p_detect.add_argument("image", help="图像文件路径")
    p_detect.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_detect.set_defaults(func=_cmd_detect)

    p_batch = sub.add_parser("batch", help="批量分析目录下全部支持格式图像")
    p_batch.add_argument("directory", help="目录路径")
    p_batch.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_batch.set_defaults(func=_cmd_batch)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
