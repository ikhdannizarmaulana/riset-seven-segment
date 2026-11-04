"""Engine baseline: Tesseract via pytesseract dengan whitelist digit + titik.

Variabel lingkungan:
  TESS_LANG  : bahasa/traineddata (default "eng"). Setelah fine-tuning, salin
               mis. `ssd_ft.traineddata` ke folder tessdata lalu set TESS_LANG=ssd_ft.
  TESS_PSM   : page segmentation mode (default 7 = satu baris teks).
  TESSDATA_PREFIX : folder tessdata kustom (opsional, dibaca Tesseract sendiri).
  TESSERACT_CMD : path lengkap ke program tesseract (opsional). Berguna di Windows bila
               tesseract tidak ada di PATH, mis. C:\\Program Files\\Tesseract-OCR\\tesseract.exe.
               Bila kosong, engine mencari di PATH lalu di lokasi instalasi umum Windows.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytesseract

from .base import ALLOWED_CHARS, EngineUnavailable, OCREngine, OCRResult, clean_digits

# Lokasi instalasi umum Tesseract di Windows (installer UB Mannheim / Chocolatey).
_WINDOWS_CANDIDATES = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
]


def find_tesseract() -> str | None:
    """Cari program tesseract: env TESSERACT_CMD → PATH → lokasi umum Windows."""
    env = os.getenv("TESSERACT_CMD")
    if env and Path(env).is_file():
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    if sys.platform.startswith("win"):
        for c in _WINDOWS_CANDIDATES:
            if c.is_file():
                return str(c)
    return None


def _install_hint() -> str:
    if sys.platform.startswith("win"):
        return ("pasang Tesseract: winget install UB-Mannheim.TesseractOCR "
                "(atau set TESSERACT_CMD ke path tesseract.exe)")
    if sys.platform == "darwin":
        return "pasang Tesseract: brew install tesseract"
    return "pasang Tesseract: sudo apt install tesseract-ocr"


class TesseractEngine(OCREngine):
    name = "tesseract"
    label = "Tesseract (baseline / fine-tuned)"

    def __init__(self, lang: str | None = None, psm: int | None = None):
        self.lang = lang or os.getenv("TESS_LANG", "eng")
        self.psm = int(psm or os.getenv("TESS_PSM", "7"))

    def available(self) -> tuple[bool, str]:
        cmd = find_tesseract()
        if cmd is None:
            return False, f"program tesseract tidak ditemukan ({_install_hint()})"
        pytesseract.pytesseract.tesseract_cmd = cmd
        try:
            langs = pytesseract.get_languages(config="")
        except Exception as e:  # pragma: no cover
            return False, f"gagal memanggil tesseract: {e}"
        if self.lang not in langs:
            return False, f"traineddata '{self.lang}' tidak ada (tersedia: {langs})"
        return True, f"lang={self.lang}, psm={self.psm}"

    def _config(self) -> str:
        return f"--oem 1 --psm {self.psm} -c tessedit_char_whitelist={ALLOWED_CHARS}"

    def predict(self, image: np.ndarray) -> OCRResult:
        ok, why = self.available()
        if not ok:
            raise EngineUnavailable(why)
        cfg = self._config()
        raw = pytesseract.image_to_string(image, lang=self.lang, config=cfg)
        data = pytesseract.image_to_data(
            image, lang=self.lang, config=cfg, output_type=pytesseract.Output.DICT
        )
        confs = [float(c) for c, t in zip(data["conf"], data["text"])
                 if str(t).strip() and float(c) >= 0]
        conf = (sum(confs) / len(confs) / 100.0) if confs else None
        return OCRResult(
            text=clean_digits(raw),
            confidence=conf,
            raw=raw.strip(),
            extra={"lang": self.lang, "psm": self.psm, "config": cfg},
        )
