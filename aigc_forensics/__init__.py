# -*- coding: utf-8 -*-
"""AIGC-Forensics：AI 生成图像多信号融合取证检测器。"""

from .detector import analyze_image, report_to_json, report_to_text, ImageAnalysisError, VERSION

__all__ = ["analyze_image", "report_to_json", "report_to_text", "ImageAnalysisError", "VERSION"]
__version__ = VERSION
