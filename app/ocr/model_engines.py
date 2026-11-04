"""Slot plugin untuk model hasil training (Colab): CNN per-digit dan CRNN-CTC.

STATUS: STUB. Interface sudah final, isi `predict()` dilengkapi setelah model
selesai di-training dan diekspor ke ONNX (disarankan) atau TorchScript (.pt).

Cara mengaktifkan nanti:
  1. Ekspor model dari Colab -> `models/cnn_digit.onnx` / `models/crnn_ctc.onnx`
     (atau set env CNN_MODEL_PATH / CRNN_MODEL_PATH).
  2. `pip install onnxruntime` (atau `torch` untuk .pt) di venv app.
  3. Lengkapi bagian bertanda TODO di bawah, terutama ukuran input dan
     urutan karakter (CHARSET) yang HARUS sama persis dengan saat training.
  4. Restart server — engine otomatis muncul "siap" di dropdown.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import EngineUnavailable, OCREngine, OCRResult

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Index 0 = blank CTC. Wajib identik dengan charset saat training CRNN.
CTC_CHARSET = ["<blank>"] + list("0123456789.")
# Kelas CNN per-digit (10 digit; tambahkan '.'/kosong bila dilatih begitu).
CNN_CLASSES = list("0123456789")


def ctc_greedy_decode(logits: np.ndarray, charset: list[str] = CTC_CHARSET) -> tuple[str, float]:
    """Decode greedy CTC. logits: (T, C) skor/prob per timestep.

    Aturan CTC: ambil argmax per timestep, gabungkan yang berulang, buang blank (index 0).
    Mengembalikan (teks, rata-rata probabilitas karakter terpilih).
    """
    if logits.ndim != 2:
        raise ValueError("logits harus berdimensi (T, C)")
    x = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    best = probs.argmax(axis=1)
    out, scores, prev = [], [], -1
    for t, k in enumerate(best):
        if k != prev and k != 0:
            out.append(charset[k])
            scores.append(probs[t, k])
        prev = k
    return "".join(out), float(np.mean(scores)) if scores else 0.0


class _ModelFileEngine(OCREngine):
    env_var = ""
    default_file = ""

    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path or os.getenv(self.env_var, MODELS_DIR / self.default_file))
        self._session = None

    def available(self) -> tuple[bool, str]:
        if not self.model_path.exists():
            return False, f"model belum ada: {self.model_path.name} (stub — lihat ocr/model_engines.py)"
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False, "onnxruntime belum terpasang (pip install onnxruntime)"
        return True, f"model={self.model_path.name}"

    def _load(self):
        ok, why = self.available()
        if not ok:
            raise EngineUnavailable(why)
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        return self._session


class CNNDigitEngine(_ModelFileEngine):
    name = "cnn_digit"
    label = "CNN per-digit (segmentasi + klasifikasi)"
    env_var = "CNN_MODEL_PATH"
    default_file = "cnn_digit.onnx"
    input_size = (28, 28)  # TODO: samakan dengan ukuran input saat training

    def segment_digits(self, image: np.ndarray) -> list[np.ndarray]:
        """TODO: pisahkan ROI menjadi potongan per digit.
        Opsi: (a) proyeksi vertikal + ambang kolom kosong, (b) connected components
        setelah dilasi agar segmen satu digit menyatu, (c) pembagian lebar tetap bila
        jumlah digit display diketahui. Kembalikan list citra dari kiri ke kanan."""
        raise NotImplementedError("segment_digits() belum diimplementasi")

    def predict(self, image: np.ndarray) -> OCRResult:
        self._load()
        # TODO: crops = self.segment_digits(image)
        #       batch = resize tiap crop ke self.input_size, normalisasi 0..1, shape (N,1,H,W)
        #       logits = self._session.run(None, {input_name: batch})[0]
        #       text = "".join(CNN_CLASSES[i] for i in logits.argmax(1))
        raise NotImplementedError("CNNDigitEngine.predict() masih stub — lengkapi setelah model ada")


class CRNNCTCEngine(_ModelFileEngine):
    name = "crnn_ctc"
    label = "CRNN + CTC (end-to-end)"
    env_var = "CRNN_MODEL_PATH"
    default_file = "crnn_ctc.onnx"
    input_size = (32, 128)  # (H, W) TODO: samakan dengan training

    def predict(self, image: np.ndarray) -> OCRResult:
        self._load()
        # TODO: x = resize ke self.input_size (jaga rasio + padding), normalisasi,
        #       shape (1,1,H,W) float32
        #       out = self._session.run(None, {input_name: x})[0]  # (T,1,C) atau (1,T,C)
        #       text, conf = ctc_greedy_decode(out.squeeze())
        #       return OCRResult(text=text, confidence=conf, raw=text)
        raise NotImplementedError("CRNNCTCEngine.predict() masih stub — lengkapi setelah model ada")
