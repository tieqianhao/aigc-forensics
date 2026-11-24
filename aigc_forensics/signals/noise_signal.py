# -*- coding: utf-8 -*-
"""信号三：传感器噪声残差分析。

原理：真实数码照片经过传感器成像，带有特征性的高频噪声；生成图像由扩散
模型 / GAN 解码输出，整体更为平滑，高频残差能量显著偏低
（典型取证文献中以残差统计区分相机来源与合成来源，如 PRNU 类方法的简化形态）。

实现为免训练统计量：
1. residual_std：原图与 3x3 中值滤波结果的差值标准差（高通残差强度）；
2. jpeg_grid_ratio：8 像素网格边界梯度 / 块内梯度之比，检测 JPEG 压缩痕迹。

局限：JPEG 压缩本身会注入块伪影、扭曲噪声统计，因此检测到强 JPEG 网格时
本路信号自动降权；强降噪、多轮转存同样会压低残差，造成假阳性。
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def _jpeg_grid_ratio(g: np.ndarray) -> float:
    """JPEG 8x8 块边界梯度相对块内梯度的倍数，接近 1 表示无明显块结构。"""
    dx = np.abs(np.diff(g, axis=1))
    if dx.shape[1] < 16:
        return 1.0
    boundary = dx[:, 7::8].mean()
    inner = dx[:, [c for c in range(dx.shape[1]) if c % 8 != 7][: min(1000, dx.shape[1])]].mean()
    if inner <= 0:
        return 1.0
    return float(boundary / inner)


def analyze(img, raw: bytes) -> Dict[str, Any]:
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    if min(h, w) < 48:
        return {
            "name": "noise",
            "label": "噪声残差信号",
            "score": 0.5,
            "confidence": 0.15,
            "evidence": {"skipped": "图像过小，噪声统计不可靠"},
            "explanation": "图像尺寸过小，噪声残差统计不可靠，本路信号未参与判定。",
        }

    denoised = cv2.medianBlur(g.astype(np.float32), 3)
    residual = g - denoised
    rstd = float(residual.std())

    grid = _jpeg_grid_ratio(g)
    fmt = (img.format or "").upper()
    is_jpeg = fmt == "JPEG"
    strong_jpeg = is_jpeg and grid >= 1.35

    # 显式规则表：残差越低越平滑，越倾向生成图。
    if rstd < 1.5:
        score = 0.80
    elif rstd < 3.0:
        score = 0.60
    elif rstd < 6.0:
        score = 0.40
    else:
        score = 0.25

    confidence = 0.60
    if strong_jpeg:
        # JPEG 块伪影会污染残差统计：向不确定方向收缩并降权。
        score = 0.5 + (score - 0.5) * 0.4
        confidence = 0.30

    explanation = (
        "高通残差标准差 %.2f（低于 1.5 提示异常平滑，生成图常见；真实照片通常 3 以上）。"
        % rstd
    )
    if strong_jpeg:
        explanation += "检测到强 JPEG 压缩网格（%.2f 倍），本路信号已降权。" % grid

    evidence = {
        "residual_std": round(rstd, 3),
        "jpeg_grid_ratio": round(grid, 3),
        "is_jpeg": is_jpeg,
        "strong_jpeg": bool(strong_jpeg),
    }
    return {
        "name": "noise",
        "label": "噪声残差信号",
        "score": score,
        "confidence": confidence,
        "evidence": evidence,
        "explanation": explanation,
    }
