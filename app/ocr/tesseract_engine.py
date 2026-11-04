"""Engine baseline: Tesseract via pytesseract dengan whitelist digit + titik.

Variabel lingkungan:
  TESS_LANG  : bahasa/traineddata (default "eng"). Setelah fine-tuning, salin
               mis. `ssd_ft.traineddata` ke folder tessdata lalu set TESS_LANG=ssd_ft.
  TESS_PSM   : page segmentation mode (default 7 = satu baris teks).
  TESSDATA_PREFIX : folder tessdata kustom (opsional, dibaca Tesseract sendiri).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytesseract

from .base import ALLOWED_CHARS, EngineUnavailable, OCREngine, OCRResult, clean_digits


class TesseractEngine(OCREngine):
    name = "tesseract"
    label = "Tesseract (baseline / fine-tuned)"

    def __init__(self, lang: str | None = None, psm: int | None = None):
        self.lang = lang or os.getenv("TESS_LANG", "eng")
        self.psm = int(psm or os.getenv("TESS_PSM", "7"))

    def available(self) -> tuple[bool, str]:
        if shutil.which("tesseract") is None:
            return False, "binary tesseract tidak ditemukan (brew install tesseract)"
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
