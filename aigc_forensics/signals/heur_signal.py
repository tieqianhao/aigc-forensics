# -*- coding: utf-8 -*-
"""信号四：尺寸与格式启发式。

原理：主流扩散模型按固定分辨率桶输出（如 512x512、1024x1024 及 SDXL 的
常见构图桶，宽高均为 64 的倍数）；真实拍摄照片分辨率由传感器决定，
几乎不会恰好落在这些桶上。此外生成工具默认输出 PNG / WebP 且不写 EXIF。

局限：这是四路信号中最弱的一路——用户手动裁剪、缩放即可改变分辨率，
仅作辅助证据。典型分辨率桶列表随模型演进需要更新。
"""

from __future__ import annotations

from typing import Any, Dict

# SD / SDXL / Flux 等常见输出分辨率桶（宽高完全相等时命中精确项）。
TYPICAL_SIZES = {
    (512, 512), (768, 768), (1024, 1024),
    (1024, 768), (768, 1024),
    (1152, 896), (896, 1152),
    (1216, 832), (832, 1216),
    (1344, 768), (768, 1344),
    (1536, 640), (640, 1536),
}


def analyze(img, raw: bytes) -> Dict[str, Any]:
    w, h = img.size
    fmt = (img.format or "").upper()

    if (w, h) in TYPICAL_SIZES:
        size_score = 0.75
        size_note = "分辨率 %dx%d 命中主流扩散模型输出桶" % (w, h)
    elif w % 64 == 0 and h % 64 == 0 and min(w, h) >= 512:
        size_score = 0.55
        size_note = "宽高均为 64 的倍数，符合模型分辨率桶特征"
    else:
        size_score = 0.25
        size_note = "分辨率 %dx%d 不符合典型生成输出特征" % (w, h)

    fmt_bonus = 0.0
    if fmt in ("PNG", "WEBP"):
        fmt_bonus = 0.15
        fmt_note = "%s 格式（生成工具默认输出格式）" % fmt
    else:
        fmt_note = "%s 格式" % (fmt or "未知")

    score = min(0.90, 0.25 + size_score * 0.6 + fmt_bonus)
    explanation = "%s；%s。本路为最弱信号，仅作辅助。" % (size_note, fmt_note)
    evidence = {"width": w, "height": h, "format": fmt or None, "size_note": size_note}
    return {
        "name": "heuristic",
        "label": "尺寸与格式启发式",
        "score": score,
        "confidence": 0.40,
        "evidence": evidence,
        "explanation": explanation,
    }
