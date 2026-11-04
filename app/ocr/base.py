"""Interface (kontrak) untuk semua engine OCR.

Setiap engine (Tesseract, CNN per-digit, CRNN-CTC) wajib mewarisi `OCREngine`
supaya aplikasi web bisa memanggilnya dengan cara yang sama dan waktu proses
bisa dibandingkan secara adil (Bab 4 skripsi).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OCRResult:
    text: str                      # hasil akhir yang sudah dibersihkan, mis. "123.4"
    confidence: float | None = None  # 0..1 bila engine menyediakan
    raw: str = ""                  # keluaran mentah engine (untuk debugging)
    extra: dict[str, Any] = field(default_factory=dict)


class EngineUnavailable(RuntimeError):
    """Dilempar bila engine belum siap (model belum ada / dependency hilang)."""


class OCREngine(ABC):
    #: id unik, dipakai di form web dan di kolom `engine` pada labels.csv
    name: str = "base"
    #: label yang ramah dibaca manusia
    label: str = "Base engine"

    def available(self) -> tuple[bool, str]:
        """Kembalikan (siap?, alasan). Dipanggil untuk mengisi dropdown engine."""
        return True, "ok"

    @abstractmethod
    def predict(self, image: np.ndarray) -> OCRResult:
        """Terima citra hasil preprocessing (uint8, 1 kanal ATAU 3 kanal BGR)
        yang sudah di-crop ke ROI display, kembalikan `OCRResult`."""
        raise NotImplementedError


ALLOWED_CHARS = "0123456789."


def clean_digits(s: str) -> str:
    """Buang semua karakter selain digit dan titik."""
    return "".join(ch for ch in s if ch in ALLOWED_CHARS)
