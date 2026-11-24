# -*- coding: utf-8 -*-
"""信号一：EXIF / 元数据分析。

原理：真实手机 / 相机照片几乎必然携带完整拍摄元数据（制造商、型号、拍摄时间），
而生成式工具输出的图像通常不携带这些字段。反过来，部分生成工具会在 XMP /
C2PA（内容凭证）中主动声明 AI 来源，这是确定性的强 AIGC 证据。

局限：社交平台转存、截图、隐私抹除同样会移除 EXIF，因此「无 EXIF」只作弱证据。
"""

from __future__ import annotations

import re
from typing import Any, Dict

from PIL import Image

# 元数据中以字节形式出现的 AI 来源标记（匹配时统一转小写）。
# TrainedAlgorithmicMedia 是 IPTC 数字源类型标准中对 AI 生成媒体的正式术语。
AI_METADATA_MARKERS = [
    b"c2pa",
    b"trainedalgorithmicmedia",
    b"aigc",
    b"adobe firefly",
    b"dall-e",
    b"dall\xe2\x80\xa2e",
    b"midjourney",
    b"stable diffusion",
    b"stable-diffusion",
    b"generative-ai",
    b"generative ai",
    b"ai-generated",
    b"ai_generated",
    b"comfyui",
    b"automatic1111",
    b"invokeai",
    b"runwayml",
    b"google imagen",
    b"novelai",
]

# 相机 EXIF 标签编号（TIFF 主 IFD）。
TAG_MAKE = 271
TAG_MODEL = 272
TAG_DATETIME = 306
TAG_SOFTWARE = 305


def _scan_ai_markers(raw: bytes) -> list:
    raw_lower = raw.lower()
    return [m.decode("ascii", "ignore") for m in AI_METADATA_MARKERS if m in raw_lower]


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").strip()
    return str(value).strip() if value is not None else ""


def analyze(img: Image.Image, raw: bytes) -> Dict[str, Any]:
    """返回统一信号结构：score 越高越倾向 AIGC。"""
    exif = img.getexif()
    make = _decode(exif.get(TAG_MAKE))
    model = _decode(exif.get(TAG_MODEL))
    datetime_original = _decode(exif.get(TAG_DATETIME))
    software = _decode(exif.get(TAG_SOFTWARE))

    markers = _scan_ai_markers(raw)
    software_hit = [m for m in AI_METADATA_MARKERS if m.decode("ascii", "ignore") in software.lower()]

    has_camera_identity = bool(make) and bool(model)
    has_datetime = bool(datetime_original)
    full_camera_exif = has_camera_identity and has_datetime

    if markers or software_hit:
        score, confidence = 0.95, 0.90
        explanation = (
            "元数据中明确出现 AI 生成来源标记（%s），这是确定性证据。"
            % "、".join(sorted(set(markers + software_hit))[:4])
        )
    elif full_camera_exif:
        score, confidence = 0.10, 0.80
        explanation = "携带完整相机 EXIF（%s %s，拍摄时间 %s），是真实拍摄照片的强证据。" % (make, model, datetime_original)
    elif has_camera_identity or has_datetime:
        score, confidence = 0.25, 0.50
        explanation = "携带部分相机 EXIF（Make=%s / Model=%s / 时间=%s），倾向真实拍摄。" % (
            make or "缺失", model or "缺失", datetime_original or "缺失"
        )
    else:
        score, confidence = 0.55, 0.30
        explanation = "无任何相机 EXIF。生成工具输出、社交平台转存、截图均会出现此状态，仅作弱证据。"

    evidence = {
        "make": make or None,
        "model": model or None,
        "datetime": datetime_original or None,
        "software": software or None,
        "ai_markers_found": markers,
        "full_camera_exif": full_camera_exif,
    }
    return {
        "name": "metadata",
        "label": "元数据信号",
        "score": score,
        "confidence": confidence,
        "evidence": evidence,
        "explanation": explanation,
    }
