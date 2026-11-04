"""Registry engine OCR. Tambah engine baru cukup dengan mendaftarkannya di sini."""
from .base import EngineUnavailable, OCREngine, OCRResult
from .model_engines import CNNDigitEngine, CRNNCTCEngine
from .tesseract_engine import TesseractEngine

ENGINES: dict[str, OCREngine] = {
    e.name: e for e in (TesseractEngine(), CNNDigitEngine(), CRNNCTCEngine())
}


def get_engine(name: str) -> OCREngine:
    if name not in ENGINES:
        raise KeyError(f"engine tidak dikenal: {name}")
    return ENGINES[name]


def list_engines() -> list[dict]:
    out = []
    for e in ENGINES.values():
        ok, why = e.available()
        out.append({"name": e.name, "label": e.label, "available": ok, "status": why})
    return out


__all__ = ["ENGINES", "get_engine", "list_engines", "OCREngine", "OCRResult", "EngineUnavailable"]
