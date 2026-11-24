# -*- coding: utf-8 -*-
"""AIGC-Forensics：AI 生成图像多信号融合取证检测器。

四路独立信号（元数据 / 频域 / 噪声残差 / 尺寸格式）各自输出
score（AIGC 倾向，0-1）、confidence（该信号本次判定的可靠度 0-1）与证据，
融合引擎按「权重 x 置信度」加权平均得到总分，并给出五级判定与逐信号解释。

本工具为启发式辅助检测：输出是概率倾向，不是事实认定。
社交平台转存、重压缩、截图、裁剪都会显著影响结果。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, UnidentifiedImageError

from .signals import exif_signal, freq_signal, heur_signal, noise_signal

VERSION = "0.1.0"

# 每路信号的权重（来源：信号区分度与可被后处理破坏的程度，元数据确定性最高）。
FUSION_WEIGHTS = {
    "metadata": 0.35,
    "frequency": 0.25,
    "noise": 0.25,
    "heuristic": 0.15,
}

# 五级判定阈值（显式表，score >= 下界即判定）。
VERDICT_TABLE = [
    (0.80, "very_likely_ai", "极可能为 AI 生成"),
    (0.60, "likely_ai", "较可能为 AI 生成"),
    (0.40, "uncertain", "无法判定"),
    (0.20, "likely_real", "较可能为真实拍摄"),
    (0.00, "very_likely_real", "极可能为真实拍摄"),
]

DISCLAIMER = (
    "本工具为启发式取证辅助，输出为概率倾向而非事实认定；"
    "社交平台转存、重压缩、截图、裁剪均会显著影响结果。"
)

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class ImageAnalysisError(Exception):
    """输入无法作为图像分析（损坏 / 非图像 / 不支持的格式）。"""


def _verdict(score: float):
    for lower, key, text in VERDICT_TABLE:
        if score >= lower:
            return key, text
    return VERDICT_TABLE[-1][1], VERDICT_TABLE[-1][2]


def fuse(signals: List[Dict[str, Any]]) -> float:
    """置信度加权融合；元数据出现确定性 AI 标记时直接抬高下限。"""
    total_weight = 0.0
    weighted_sum = 0.0
    for s in signals:
        w = FUSION_WEIGHTS[s["name"]] * s["confidence"]
        total_weight += w
        weighted_sum += w * s["score"]
    score = weighted_sum / total_weight if total_weight > 0 else 0.5

    meta = next(s for s in signals if s["name"] == "metadata")
    if meta["score"] >= 0.90 and meta["confidence"] >= 0.80:
        score = max(score, 0.93)
    return round(min(1.0, max(0.0, score)), 4)


def analyze_image(path: str) -> Dict[str, Any]:
    """分析单张图像，返回完整报告 dict。分析失败抛 ImageAnalysisError。"""
    p = Path(path)
    if not p.exists():
        raise ImageAnalysisError("文件不存在: %s" % path)
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ImageAnalysisError("不支持的文件类型: %s" % p.suffix)

    raw = p.read_bytes()
    try:
        img = Image.open(p)
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageAnalysisError("无法解码图像: %s（%s）" % (path, exc)) from exc

    signals = [
        exif_signal.analyze(img, raw),
        freq_signal.analyze(img, raw),
        noise_signal.analyze(img, raw),
        heur_signal.analyze(img, raw),
    ]
    score = fuse(signals)
    key, text = _verdict(score)

    return {
        "file": str(p),
        "tool": "AIGC-Forensics",
        "version": VERSION,
        "score": score,
        "verdict": key,
        "verdict_text": text,
        "signals": signals,
        "disclaimer": DISCLAIMER,
    }


def report_to_text(report: Dict[str, Any]) -> str:
    """渲染人类可读报告（终端展示用）。"""
    lines = [
        "文件: %s" % report["file"],
        "判定: %s（score = %.2f）" % (report["verdict_text"], report["score"]),
        "",
    ]
    for s in report["signals"]:
        lines.append("【%s】score = %.2f，置信度 = %.2f" % (s["label"], s["score"], s["confidence"]))
        lines.append("  %s" % s["explanation"])
        lines.append("")
    lines.append("声明: %s" % report["disclaimer"])
    return "\n".join(lines)


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
