# Riset Seven Segment — OCR Angka Seven Segment

Skripsi: *Analisis Komparatif Tesseract Fine-Tuned, CNN Per-Digit, dan CRNN-CTC untuk Pengenalan Angka Seven Segment pada Variasi Kondisi Citra.*

## Struktur

| Folder/File | Isi |
|---|---|
| `survei_literatur_seven_segment_AMR.md` | Survei literatur |
| `strategi_data.md` | Strategi pengadaan data & daftar dataset publik |
| `pilot/` | Studi kelayakan: generator data sintetis, baseline Tesseract, CNN per-digit, CRNN-CTC (lihat `pilot/README_pilot.md`) |
| `pipeline/` | Penyatuan dataset publik ke satu manifest (`unify.py`) + generator degradasi (`degrade.py`) |
| `labeling/` | Label ulang 858 foto pompa BBM (dataset SachaIZADI), lihat di bawah |
| `app/` | Web app FastAPI untuk pengumpulan data & demo tiga engine OCR (lihat `app/README_app.md`) |

## Yang tidak ikut di repo

- `data/`: dataset publik (unduh ulang sesuai `strategi_data.md`).
- `pilot/data/`, `pipeline/degraded/`: bisa digenerate ulang dengan skrip.
- `*.traineddata`: unduh dari `github.com/tesseract-ocr/tessdata` dan `github.com/Shreeshrii/tessdata_ssd` ke `pilot/tessdata_extra/`.
- `.venv/`: buat ulang sesuai README masing-masing folder.

## Label ulang foto pompa BBM (`labeling/`)

- 858 foto dilabeli ulang secara buta oleh AI vision, lalu 102 foto sulit dibaca ulang dengan zoom.
- Label asli dataset (`original_labels.csv`) ternyata **dibulatkan ke atas** (layar `128.28` → label `129`).
- Hasil final: `labeling/labels_final.csv`

| status_vs_original | jumlah |
|---|---|
| match | 688 |
| orig_rounded_up | 93 |
| unreadable (masih ada `?`) | 74 |
| mismatch | 3 |

`usable_for_training = true` untuk 779 foto. Data mentah: `labeling/out/batch_*.jsonl` (pass 1) dan `labeling/out/recheck_*.jsonl` (pass 2).
