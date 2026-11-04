"""Pipeline preprocessing citra display seven segment.

Urutan: decode -> crop ROI -> resize (opsional) -> grayscale -> CLAHE ->
blur -> threshold -> invert -> dilasi (menyambung celah antar-segmen).
Semua parameter dicatat ke labels.csv agar eksperimen bisa direproduksi.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class PreprocessParams:
    roi_x: int = 0
    roi_y: int = 0
    roi_w: int = 0          # 0 = pakai seluruh lebar
    roi_h: int = 0          # 0 = pakai seluruh tinggi
    target_height: int = 0  # 0 = tanpa resize; mis. 64 untuk menyeragamkan
    grayscale: bool = True
    clahe: bool = False
    clahe_clip: float = 2.0
    blur: int = 0           # ukuran kernel Gaussian (0 = mati, ganjil)
    threshold: str = "none"  # none | otsu | adaptive | fixed
    threshold_value: int = 127
    invert: bool = False
    dilate: int = 0         # iterasi dilasi (kernel 3x3)

    def to_dict(self) -> dict:
        return asdict(self)


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("file bukan gambar yang valid")
    return img


def crop_roi(img: np.ndarray, p: PreprocessParams) -> np.ndarray:
    h, w = img.shape[:2]
    if p.roi_w <= 0 or p.roi_h <= 0:
        return img
    x0, y0 = max(0, p.roi_x), max(0, p.roi_y)
    x1, y1 = min(w, x0 + p.roi_w), min(h, y0 + p.roi_h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI di luar gambar ({w}x{h})")
    return img[y0:y1, x0:x1]


def preprocess(img: np.ndarray, p: PreprocessParams) -> np.ndarray:
    out = crop_roi(img, p)
    if p.target_height and out.shape[0] != p.target_height:
        scale = p.target_height / out.shape[0]
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    needs_gray = p.grayscale or p.clahe or p.threshold != "none"
    if needs_gray and out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if p.clahe:
        out = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(8, 8)).apply(out)
    if p.blur and p.blur > 1:
        k = p.blur if p.blur % 2 == 1 else p.blur + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    if p.threshold == "otsu":
        _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif p.threshold == "adaptive":
        out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 5)
    elif p.threshold == "fixed":
        _, out = cv2.threshold(out, p.threshold_value, 255, cv2.THRESH_BINARY)
    if p.invert:
        out = cv2.bitwise_not(out)
    if p.dilate > 0:
        # Pada citra teks-gelap-di-latar-terang, "menebalkan" teks = erosi piksel putih.
        # Kita asumsikan teks gelap (setelah invert bila perlu), jadi pakai erode.
        out = cv2.erode(out, np.ones((3, 3), np.uint8), iterations=p.dilate)
    return out


def to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("gagal encode PNG")
    return buf.tobytes()
