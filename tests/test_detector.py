# -*- coding: utf-8 -*-
"""AIGC-Forensics 测试套件。

运行：python -m pytest tests/ -v   或   python tests/test_detector.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aigc_forensics.detector import ImageAnalysisError, analyze_image  # noqa: E402
from tests.make_samples import generate_all  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent / "samples"
AI_MARKERS = ("very_likely_ai", "likely_ai")
REAL_MARKERS = ("very_likely_real", "likely_real")


def run_all():
    generate_all(SAMPLE_DIR)
    failures = []
    stats = []

    for p in sorted(SAMPLE_DIR.glob("ai_like_*.png")):
        r = analyze_image(str(p))
        stats.append((p.name, r["score"], r["verdict"]))
        if r["verdict"] not in AI_MARKERS or r["score"] < 0.60:
            failures.append("ai_like 样本未判为 AI: %s -> %.2f / %s" % (p.name, r["score"], r["verdict"]))

    for p in sorted(SAMPLE_DIR.glob("real_like_*.jpg")):
        r = analyze_image(str(p))
        stats.append((p.name, r["score"], r["verdict"]))
        if r["verdict"] not in REAL_MARKERS or r["score"] > 0.40:
            failures.append("real_like 样本未判为真实: %s -> %.2f / %s" % (p.name, r["score"], r["verdict"]))

    for p in sorted(SAMPLE_DIR.glob("ai_tagged_*.png")):
        r = analyze_image(str(p))
        stats.append((p.name, r["score"], r["verdict"]))
        if r["verdict"] != "very_likely_ai" or r["score"] < 0.90:
            failures.append("ai_tagged 样本未触发元数据短路: %s -> %.2f / %s" % (p.name, r["score"], r["verdict"]))

    # 报告结构完整性：四路信号、五级判定字段、免责声明齐全。
    r = analyze_image(str(SAMPLE_DIR / "ai_like_000.png"))
    names = [s["name"] for s in r["signals"]]
    if names != ["metadata", "frequency", "noise", "heuristic"]:
        failures.append("信号名不齐: %s" % names)
    for field in ("file", "score", "verdict", "verdict_text", "signals", "disclaimer", "version"):
        if field not in r:
            failures.append("报告缺字段: %s" % field)
    json.loads(json.dumps(r, ensure_ascii=False))  # JSON 序列化不应报错

    # 异常输入优雅处理：不崩溃、返回明确错误。
    bad = SAMPLE_DIR / "corrupt.png"
    bad.write_bytes(b"not an image at all")
    try:
        analyze_image(str(bad))
        failures.append("损坏文件未被拒绝")
    except ImageAnalysisError:
        pass

    print("=" * 72)
    print("样本判定结果：")
    for name, score, verdict in stats:
        print("  %-24s score = %.3f  ->  %s" % (name, score, verdict))
    if failures:
        print("\n测试未通过（%d 项）：" % len(failures))
        for f in failures:
            print("  [FAIL] " + f)
        return 1
    print("\n全部测试通过（15 个样本 + 结构完整性 + 异常处理）。")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
