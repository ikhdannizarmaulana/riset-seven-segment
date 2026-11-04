"""Uji otomatis MVP: .venv/bin/python -m pytest -q tests/"""
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
os.environ["APP_DATA_DIR"] = tempfile.mkdtemp(prefix="ssd_test_")  # jangan kotori data/ asli

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from ocr.model_engines import ctc_greedy_decode  # noqa: E402
from tools.make_test_image import render  # noqa: E402

client = TestClient(app)


def _png(text="123.4", style="lcd"):
    buf = io.BytesIO()
    render(text, style=style).save(buf, format="PNG")
    return buf.getvalue()


def test_engines_listed():
    d = client.get("/api/engines").json()
    names = {e["name"]: e for e in d["engines"]}
    assert set(names) == {"tesseract", "cnn_digit", "crnn_ctc"}
    assert names["cnn_digit"]["available"] is False  # stub, model belum ada


def test_preprocess_roi():
    r = client.post("/api/preprocess", files={"image": ("t.png", _png(), "image/png")},
                    data={"roi_x": 10, "roi_y": 10, "roi_w": 100, "roi_h": 80, "clahe": "true",
                          "threshold": "otsu"})
    assert r.status_code == 200, r.text
    assert r.json()["output_size"] == [100, 80]


def test_predict_and_label_roundtrip():
    r = client.post("/api/predict", files={"image": ("t.png", _png(), "image/png")},
                    data={"engine": "tesseract", "threshold": "otsu"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d["text"]) <= set("0123456789.")
    r2 = client.post("/api/label", data={
        "image_file": d["image_file"], "crop_file": d["crop_file"], "label": "123.4",
        "predicted": d["text"], "engine": "tesseract", "device": "sintetis",
        "lighting": "normal", "session": "UJI"})
    assert r2.status_code == 200, r2.text
    s = client.get("/api/summary").json()
    assert s["total"] == 1 and s["by_lighting"] == {"normal": 1}


def test_stub_engine_returns_503():
    r = client.post("/api/predict", files={"image": ("t.png", _png(), "image/png")},
                    data={"engine": "crnn_ctc"})
    assert r.status_code == 503


def test_label_validation():
    r = client.post("/api/label", data={"image_file": "x.png", "label": "12a"})
    assert r.status_code == 400


def test_ctc_greedy_decode():
    # timestep: blank,1,1,blank,2,'.',blank,3  -> "12.3"
    idx = [0, 2, 2, 0, 3, 11, 0, 4]
    logits = np.full((len(idx), 12), -5.0)
    logits[np.arange(len(idx)), idx] = 5.0
    text, conf = ctc_greedy_decode(logits)
    assert text == "12.3" and conf > 0.9
