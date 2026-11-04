"""Penyimpanan dataset: gambar asli, crop ROI, dan data/labels.csv."""
from __future__ import annotations

import csv
import json
import os
import threading
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("APP_DATA_DIR", APP_DIR / "data"))
IMG_DIR = DATA_DIR / "images"
CROP_DIR = DATA_DIR / "crops"
LABELS_CSV = DATA_DIR / "labels.csv"

FIELDS = [
    "timestamp", "image_file", "crop_file", "label", "predicted", "engine",
    "confidence", "ocr_ms", "preprocess_ms", "correct",
    "device", "lighting", "session", "notes",
    "roi_x", "roi_y", "roi_w", "roi_h", "preprocess_params",
]

# Pilihan default (bisa diubah sesuai desain eksperimen di Bab 3)
LIGHTING_OPTIONS = ["terang", "normal", "redup", "silau_pantulan", "malam_backlight"]

_lock = threading.Lock()


def ensure_dirs() -> None:
    for d in (DATA_DIR, IMG_DIR, CROP_DIR):
        d.mkdir(parents=True, exist_ok=True)


def append_label(row: dict) -> dict:
    ensure_dirs()
    row = {k: row.get(k, "") for k in FIELDS}
    row["timestamp"] = row["timestamp"] or datetime.now().isoformat(timespec="seconds")
    if isinstance(row["preprocess_params"], dict):
        row["preprocess_params"] = json.dumps(row["preprocess_params"], separators=(",", ":"))
    row["correct"] = int(str(row["label"]) == str(row["predicted"]))
    with _lock:
        new = not LABELS_CSV.exists() or LABELS_CSV.stat().st_size == 0
        with LABELS_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    return row


def read_labels() -> list[dict]:
    if not LABELS_CSV.exists():
        return []
    with LABELS_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def summarize() -> dict:
    rows = read_labels()
    by = {k: Counter(r[k] or "(kosong)" for r in rows) for k in ("lighting", "device", "session", "engine")}
    cross = defaultdict(Counter)
    for r in rows:
        cross[r["lighting"] or "(kosong)"][r["device"] or "(kosong)"] += 1
    # Akurasi prediksi saat pelabelan (indikatif, BUKAN hasil uji formal Bab 4)
    acc = defaultdict(lambda: {"n": 0, "exact": 0, "edits": 0, "chars": 0, "ms": 0.0})
    for r in rows:
        key = (r["engine"] or "(kosong)", r["lighting"] or "(kosong)")
        a = acc[key]
        a["n"] += 1
        a["exact"] += int(r["label"] == r["predicted"])
        a["edits"] += levenshtein(r["predicted"], r["label"])
        a["chars"] += max(1, len(r["label"]))
        try:
            a["ms"] += float(r["ocr_ms"] or 0)
        except ValueError:
            pass
    acc_rows = [
        {"engine": e, "lighting": l, "n": a["n"],
         "exact_match": round(a["exact"] / a["n"], 4),
         "cer": round(a["edits"] / a["chars"], 4),
         "avg_ocr_ms": round(a["ms"] / a["n"], 1)}
        for (e, l), a in sorted(acc.items())
    ]
    return {
        "total": len(rows),
        "by_lighting": dict(by["lighting"]),
        "by_device": dict(by["device"]),
        "by_session": dict(by["session"]),
        "by_engine": dict(by["engine"]),
        "lighting_x_device": {k: dict(v) for k, v in cross.items()},
        "accuracy": acc_rows,
    }
