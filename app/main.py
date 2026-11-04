"""MVP web app OCR seven segment (FastAPI).

Jalankan:  .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Buka:      http://localhost:8000  (laptop)  /  http://<IP-laptop>:8000 (HP, satu Wi-Fi)
"""
from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import storage
from ocr import EngineUnavailable, get_engine, list_engines
from preprocess import PreprocessParams, decode_image, preprocess, to_png_bytes

APP_DIR = Path(__file__).resolve().parent
STATIC = APP_DIR / "static"

app = FastAPI(title="Seven Segment OCR — MVP Skripsi", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
storage.ensure_dirs()


def _params(roi_x: int, roi_y: int, roi_w: int, roi_h: int, target_height: int,
            grayscale: bool, clahe: bool, clahe_clip: float, blur: int,
            threshold: str, threshold_value: int, invert: bool, dilate: int) -> PreprocessParams:
    if threshold not in {"none", "otsu", "adaptive", "fixed"}:
        raise HTTPException(400, f"threshold tidak valid: {threshold}")
    return PreprocessParams(roi_x, roi_y, roi_w, roi_h, target_height, grayscale, clahe,
                            clahe_clip, blur, threshold, threshold_value, invert, dilate)


def _b64png(img) -> str:
    return "data:image/png;base64," + base64.b64encode(to_png_bytes(img)).decode()


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/summary", response_class=HTMLResponse)
def summary_page():
    return FileResponse(STATIC / "summary.html")


@app.get("/api/engines")
def engines():
    return {"engines": list_engines(), "lighting_options": storage.LIGHTING_OPTIONS}


@app.post("/api/preprocess")
async def api_preprocess(
    image: UploadFile = File(...),
    roi_x: int = Form(0), roi_y: int = Form(0), roi_w: int = Form(0), roi_h: int = Form(0),
    target_height: int = Form(0), grayscale: bool = Form(True), clahe: bool = Form(False),
    clahe_clip: float = Form(2.0), blur: int = Form(0), threshold: str = Form("none"),
    threshold_value: int = Form(127), invert: bool = Form(False), dilate: int = Form(0),
):
    p = _params(roi_x, roi_y, roi_w, roi_h, target_height, grayscale, clahe, clahe_clip,
                blur, threshold, threshold_value, invert, dilate)
    try:
        img = decode_image(await image.read())
        t0 = time.perf_counter()
        out = preprocess(img, p)
        ms = (time.perf_counter() - t0) * 1000
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"preview": _b64png(out), "preprocess_ms": round(ms, 2),
            "original_size": [img.shape[1], img.shape[0]], "output_size": [out.shape[1], out.shape[0]]}


@app.post("/api/predict")
async def api_predict(
    image: UploadFile = File(...),
    engine: str = Form("tesseract"),
    roi_x: int = Form(0), roi_y: int = Form(0), roi_w: int = Form(0), roi_h: int = Form(0),
    target_height: int = Form(0), grayscale: bool = Form(True), clahe: bool = Form(False),
    clahe_clip: float = Form(2.0), blur: int = Form(0), threshold: str = Form("none"),
    threshold_value: int = Form(127), invert: bool = Form(False), dilate: int = Form(0),
):
    p = _params(roi_x, roi_y, roi_w, roi_h, target_height, grayscale, clahe, clahe_clip,
                blur, threshold, threshold_value, invert, dilate)
    data = await image.read()
    try:
        img = decode_image(data)
        t0 = time.perf_counter()
        proc = preprocess(img, p)
        pre_ms = (time.perf_counter() - t0) * 1000
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        eng = get_engine(engine)
    except KeyError as e:
        raise HTTPException(400, str(e))
    t1 = time.perf_counter()
    try:
        res = eng.predict(proc)
    except EngineUnavailable as e:
        raise HTTPException(503, f"engine '{engine}' belum siap: {e}")
    except NotImplementedError as e:
        raise HTTPException(501, f"engine '{engine}': {e}")
    ocr_ms = (time.perf_counter() - t1) * 1000

    # Simpan gambar asli + crop agar bisa dilabeli. File tanpa label boleh dihapus berkala.
    image_id = uuid.uuid4().hex[:12]
    ext = Path(image.filename or "x.jpg").suffix.lower() or ".jpg"
    img_name = f"{image_id}{ext}"
    (storage.IMG_DIR / img_name).write_bytes(data)
    crop_name = f"{image_id}_roi.png"
    cv2.imwrite(str(storage.CROP_DIR / crop_name), preprocess(img, PreprocessParams(
        p.roi_x, p.roi_y, p.roi_w, p.roi_h, grayscale=False)))

    return {
        "image_file": img_name, "crop_file": crop_name, "engine": engine,
        "text": res.text, "raw": res.raw,
        "confidence": None if res.confidence is None else round(res.confidence, 4),
        "preprocess_ms": round(pre_ms, 2), "ocr_ms": round(ocr_ms, 2),
        "total_ms": round(pre_ms + ocr_ms, 2),
        "preview": _b64png(proc), "params": p.to_dict(), "extra": res.extra,
    }


@app.post("/api/label")
def api_label(
    image_file: str = Form(...), crop_file: str = Form(""), label: str = Form(...),
    predicted: str = Form(""), engine: str = Form(""), confidence: str = Form(""),
    ocr_ms: str = Form(""), preprocess_ms: str = Form(""),
    device: str = Form(""), lighting: str = Form(""), session: str = Form(""),
    notes: str = Form(""), params_json: str = Form("{}"),
    roi_x: int = Form(0), roi_y: int = Form(0), roi_w: int = Form(0), roi_h: int = Form(0),
):
    label = label.strip()
    if not label or any(c not in "0123456789." for c in label):
        raise HTTPException(400, "label wajib berisi digit/titik saja")
    if not (storage.IMG_DIR / Path(image_file).name).exists():
        raise HTTPException(404, "image_file tidak ditemukan; lakukan prediksi dulu")
    row = storage.append_label(dict(
        image_file=Path(image_file).name, crop_file=Path(crop_file).name if crop_file else "",
        label=label, predicted=predicted, engine=engine, confidence=confidence,
        ocr_ms=ocr_ms, preprocess_ms=preprocess_ms, device=device.strip(),
        lighting=lighting.strip(), session=session.strip(), notes=notes.strip(),
        roi_x=roi_x, roi_y=roi_y, roi_w=roi_w, roi_h=roi_h, preprocess_params=params_json,
    ))
    return {"saved": True, "row": row, "total": len(storage.read_labels())}


@app.get("/api/summary")
def api_summary():
    return storage.summarize()


@app.get("/api/labels.csv")
def download_csv():
    if not storage.LABELS_CSV.exists():
        raise HTTPException(404, "belum ada data")
    return FileResponse(storage.LABELS_CSV, media_type="text/csv", filename="labels.csv")
