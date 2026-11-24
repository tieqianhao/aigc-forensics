# -*- coding: utf-8 -*-
"""信号二：FFT 频域网格伪影检测。

原理：GAN 上采样层与扩散模型 VAE 解码器在生成图像中留下周期性的网格伪影。
在二维傅里叶幅度谱上，这些伪影表现为沿坐标轴方向、等间隔分布的谱峰
（Frank et al., 2020, "Leveraging Frequency Analysis for Deep Fake Image
Recognition", ICML Workshop）。真实照片的频谱能量在方向上接近均匀。

实现为免训练统计量：
1. axial_excess：坐标轴带内高频能量占比相对「均匀分布基线」的倍数；
2. spectral_kurtosis：log 幅度谱的峰度，周期性峰会把分布拉尖。

局限：强 JPEG 压缩会部分抹平谱峰；阈值基于小样本经验标定，非大规模评测结论。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.stats import kurtosis

# 轴带半宽（像素频率轴上 |du| < AXIS_BAND 视作轴向）。
AXIS_BAND = 3
# 中心低频屏蔽半径（排除直流与整体构图带来的低频差异）。
DC_RADIUS = 24
# 奈奎斯特角窗半径：谱四角（DC 对角位置）附近视为棋盘伪影区。
NYQ_RADIUS = 12


def _region_masks(A: np.ndarray):
    h, w = A.shape
    cy, cx = h // 2, w // 2
    du = np.abs(np.arange(w) - cx)
    dv = np.abs(np.arange(h) - cy)
    high = (du[None, :] > DC_RADIUS) & (dv[:, None] > DC_RADIUS)
    axial = ((du[None, :] < AXIS_BAND) | (dv[:, None] < AXIS_BAND)) & high
    # fftshift 后 DC 在中心，奈奎斯特频率位于四角（距中心 w/2, h/2）。
    nyq = ((du[None, :] > w // 2 - NYQ_RADIUS) & (dv[:, None] > h // 2 - NYQ_RADIUS)) & high
    return high, axial, nyq


def _excess(A: np.ndarray, mask: np.ndarray, high: np.ndarray) -> float:
    """mask 区域能量占比相对「均匀分布基线」的倍数。"""
    total = float((A ** 2)[high].sum())
    if total <= 0 or mask.sum() == 0:
        return 1.0
    ratio = float((A ** 2)[mask].sum()) / total
    baseline = float(mask.sum()) / float(high.sum())
    if baseline <= 0:
        return 1.0
    return ratio / baseline


def _axial_excess(A: np.ndarray) -> float:
    """轴向高频能量占比相对均匀基线的倍数。"""
    h, w = A.shape
    cy, cx = h // 2, w // 2
    du = np.abs(np.arange(w) - cx)
    dv = np.abs(np.arange(h) - cy)
    high = (du[None, :] > DC_RADIUS) & (dv[:, None] > DC_RADIUS)
    if high.sum() == 0:
        return 1.0
    axial = ((du[None, :] < AXIS_BAND) | (dv[:, None] < AXIS_BAND)) & high
    total = float((A ** 2)[high].sum())
    if total <= 0:
        return 1.0
    ratio = float((A ** 2)[axial].sum()) / total
    baseline = float(axial.sum()) / float(high.sum())
    if baseline <= 0:
        return 1.0
    return ratio / baseline


def analyze(img, raw: bytes) -> Dict[str, Any]:
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    if min(h, w) < 64:
        return {
            "name": "frequency",
            "label": "频域信号",
            "score": 0.5,
            "confidence": 0.15,
            "evidence": {"skipped": "图像过小，频谱统计不可靠"},
            "explanation": "图像尺寸过小，频域统计不可靠，本路信号未参与判定。",
        }

    spectrum = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    amplitude = np.log1p(np.abs(spectrum))
    # 区域能量统计必须用线性功率谱：log 压缩动态范围会把尖峰稀释进噪声底。
    power = np.abs(spectrum).astype(np.float64) ** 2

    high, axial, nyq = _region_masks(amplitude)
    excess_ax = _excess(power, axial, high)
    excess_nyq = _excess(power, nyq, high)
    cy, cx = h // 2, w // 2
    du = np.abs(np.arange(w) - cx)
    dv = np.abs(np.arange(h) - cy)
    high_mask = (du[None, :] > DC_RADIUS) & (dv[:, None] > DC_RADIUS)
    k = float(kurtosis(amplitude[high_mask], fisher=False)) if high_mask.sum() > 100 else 3.0

    # 显式规则表（线性功率域标定）：取轴向峰与奈奎斯特棋盘峰两档中较强者，
    # 再加 log 谱峰度加分。线性域下棋盘尖峰 excess 可达十倍以上，
    # 真实照片（噪声均匀 / JPEG 高频衰减）通常不超过 1.5。
    def _tier(excess: float) -> float:
        if excess >= 3.0:
            return 0.85
        if excess >= 1.8:
            return 0.65
        if excess >= 1.2:
            return 0.45
        return 0.30

    s1 = max(_tier(excess_ax), _tier(excess_nyq))
    if k >= 10.0:
        bonus = 0.10
    elif k >= 6.0:
        bonus = 0.05
    else:
        bonus = 0.0
    score = min(0.95, s1 + bonus)

    confidence = 0.70 if min(h, w) >= 512 else 0.50
    if max(excess_ax, excess_nyq) < 3.0 and k < 6.0:
        confidence = min(confidence, 0.55)

    explanation = (
        "轴向谱能量为均匀基线的 %.1f 倍、奈奎斯特角窗能量为基线的 %.1f 倍"
        "（3.0 及以上分别提示网格周期内容 / 棋盘伪影；周期纹理内容也可能触发，见局限），"
        "log 幅度谱峰度 %.1f。阈值来自小样本经验标定。"
        % (excess_ax, excess_nyq, k)
    )
    evidence = {
        "axial_excess": round(excess_ax, 3),
        "nyquist_excess": round(excess_nyq, 3),
        "spectral_kurtosis": round(k, 2),
        "image_size": [w, h],
    }
    return {
        "name": "frequency",
        "label": "频域信号",
        "score": score,
        "confidence": confidence,
        "evidence": evidence,
        "explanation": explanation,
    }
