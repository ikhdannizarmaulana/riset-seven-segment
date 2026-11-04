# Riset Seven Segment — Aplikasi OCR Angka Seven Segment

Repo ini adalah bagian praktik dari skripsi:

> **Analisis Komparatif Tesseract Fine-Tuned, CNN Per-Digit, dan CRNN-CTC untuk Pengenalan Angka Seven Segment pada Variasi Kondisi Citra**

## Ini skripsi tentang apa?

Banyak alat ukur menampilkan angka dalam format **seven segment** (LCD/LED): pompa BBM, meter listrik/air, timbangan, tensimeter. Angka itu masih sering dicatat manual. OCR umum seperti Tesseract bawaan gagal membacanya, karena segmennya terputus, titik desimalnya kecil, dan foto nyata sering silau, buram, miring, atau terpotong.

Skripsi ini membandingkan **tiga metode** untuk membaca angka seven segment dari foto:

| Metode | Cara kerja singkat |
|---|---|
| **Tesseract fine-tuned** | OCR open-source (LSTM) yang dilatih ulang khusus untuk huruf seven segment |
| **CNN per-digit** | Layar dipotong per digit, lalu tiap digit diklasifikasi (0–9 + titik desimal) oleh CNN kecil |
| **CRNN-CTC** | CNN + BiLSTM membaca satu baris angka sekaligus tanpa memotong per digit (loss CTC) |

Perbandingan dilakukan pada **variasi kondisi citra**: bersih, silau (glare), buram (blur), miring, redup, dan terpotong bingkai. Metrik yang dipakai: exact match, CER, akurasi per digit, akurasi titik desimal, dan waktu proses.

Hasilnya berupa **aplikasi web**. Foto layar diambil dari HP, aplikasi mendeteksi layar secara otomatis, lalu ketiga metode membaca angkanya dan hasilnya tampil berdampingan.

## Status

| Bagian | Status |
|---|---|
| Pilot pada data sintetis (3 metode) | ✅ Selesai: `pilot/` |
| Penyatuan dataset publik | ✅ Selesai: `pipeline/` |
| Label ulang 858 foto pompa BBM | ✅ Selesai: `labeling/` |
| Aplikasi web (MVP, Tesseract saja, ROI manual) | ✅ Jalan: `app/` |
| Deteksi layar otomatis, CNN dan CRNN di app, evaluasi batch | 🚧 Dikerjakan, lihat [PRD](docs/PRD.md) dan [Issues](https://github.com/ikhdannizarmaulana/riset-seven-segment/issues) |

Rencana lengkap ada di **[docs/PRD.md](docs/PRD.md)**. Pekerjaan dipecah menjadi 11 tiket per tahap di **GitHub Issues**. Urutan dan dependensinya ada di tiket [#12 Roadmap](https://github.com/ikhdannizarmaulana/riset-seven-segment/issues/12).

## Cara pakai

Aplikasi dan pipeline didukung di **Windows 10/11, macOS, dan Linux**. Setiap push diuji otomatis di ketiga sistem lewat GitHub Actions (`.github/workflows/ci.yml`).

### 🪟 Windows (paling cepat)

Buka **PowerShell**, lalu jalankan:

```powershell
git clone https://github.com/ikhdannizarmaulana/riset-seven-segment.git
cd riset-seven-segment
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_app.ps1   # sekali saja
powershell -ExecutionPolicy Bypass -File scripts\windows\run_app.ps1     # setiap kali mau pakai
```

Apa yang dilakukan `setup_app.ps1`:
1. **Cek Python 3.12.** Kalau belum ada, dipasang lewat `winget`. Setelah itu tutup PowerShell, buka lagi, dan jalankan skripnya ulang.
2. **Cek Tesseract.** Kalau belum ada, dipasang lewat `winget install UB-Mannheim.TesseractOCR`. App mencari `C:\Program Files\Tesseract-OCR\tesseract.exe` secara otomatis, jadi PATH tidak perlu diatur. Kalau lokasinya lain, set `$env:TESSERACT_CMD = "D:\...\tesseract.exe"`.
3. **Buat `app\.venv`** dan pasang paket.
4. **Jalankan uji otomatis.** Kalau hasilnya `6 passed`, instalasi berhasil.

`run_app.ps1` menampilkan alamat untuk laptop (`http://localhost:8000`) dan untuk HP (`http://<IP-laptop>:8000`). Saat pertama dijalankan, **Windows Firewall akan bertanya: pilih "Allow" untuk jaringan Private**. Tanpa izin ini, HP tidak bisa membuka aplikasinya.

Catatan Windows:
- **Git:** kalau `git` belum ada, pasang dengan `winget install Git.Git`, atau unduh ZIP repo dari GitHub.
- **Pilot:** jalankan `scripts\windows\run_pilot.ps1`. Kalau laptop punya **GPU NVIDIA** dan PyTorch versi CUDA terpasang, training otomatis memakai GPU (`cuda`); kalau tidak, memakai CPU (lebih lambat, tetap jalan). Di Mac, training memakai GPU Apple (`mps`).
- **Path:** manifest dan label memakai path relatif dengan `/`, jadi bisa dipakai di Windows apa adanya.
- **Folder:** hindari folder dengan karakter non-ASCII (misalnya huruf beraksen) di path repo. OpenCV di Windows bisa gagal membaca gambar di folder seperti itu.
- **Fine-tuning Tesseract (#5):** `tesstrain` butuh `make` dan bash, jadi di Windows jalankan lewat **WSL2 (Ubuntu)** atau Docker. Detailnya ada di tiket #5 dan #17.

### macOS / Linux

#### Prasyarat
- Python 3.12 (app) / 3.9+ (pilot)
- Tesseract 5.x: `brew install tesseract` (macOS) atau `sudo apt install tesseract-ocr` (Ubuntu)
- Opsional: [`uv`](https://github.com/astral-sh/uv) untuk membuat venv lebih cepat

```bash
git clone https://github.com/ikhdannizarmaulana/riset-seven-segment.git
cd riset-seven-segment
```

#### 1. Menjalankan aplikasi web

```bash
cd app
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
#   tanpa uv: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

- Laptop: <http://localhost:8000>
- HP (Wi-Fi yang sama): `http://<IP-laptop>:8000`. Cek IP laptop dengan `ipconfig getifaddr en0` (macOS), `hostname -I` (Linux), atau `ipconfig` (Windows, lihat "IPv4 Address").
- Alur: **Foto/Unggah → pilih area layar (ROI) → pilih engine → Baca → koreksi bila salah → Simpan**.
- Ringkasan data ada di `/summary`, unduh label di `/api/labels.csv`, dokumentasi API di `/docs`.
- Uji otomatis: `.venv/bin/python -m pytest -q tests/`

> Saat ini engine **CNN dan CRNN di app masih nonaktif** (stub) sampai model ONNX dimasukkan ke `app/models/`. Lihat tiket [#8](https://github.com/ikhdannizarmaulana/riset-seven-segment/issues/8).

Detail arsitektur ada di [`app/README_app.md`](app/README_app.md).

#### 2. Menjalankan pilot (data sintetis, ±10 menit)

```bash
cd pilot
python3 -m venv .venv && .venv/bin/pip install numpy pillow opencv-python-headless pytesseract torch
.venv/bin/python generator.py --out data/test  --per-condition 300 --seed 2026
.venv/bin/python generator.py --out data/train --n 12000 --mixed --seed 1
OMP_THREAD_LIMIT=1 .venv/bin/python eval_tesseract.py
.venv/bin/python train_cnn.py --epochs 6
.venv/bin/python train_crnn.py --epochs 10
.venv/bin/python make_report.py          # hasil → results.csv, results.json
```

Sebelum menjalankan `eval_tesseract.py`, unduh traineddata `ssd`, `7seg`, dan `ssd_plus` dari [Shreeshrii/tessdata_ssd](https://github.com/Shreeshrii/tessdata_ssd), lalu `eng` dari [tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata), ke folder `pilot/tessdata_extra/`.

Hasil pilot (sintetis, exact match): Tesseract `7seg` 0,618, CNN per-digit 0,995, CRNN-CTC 0,988. Angka ini **batas atas yang optimistis**; keterbatasannya dijelaskan di [`pilot/README_pilot.md`](pilot/README_pilot.md).

### Dataset & label (semua OS)

#### 3. Dataset

Dataset publik tidak di-commit. Cara mengunduhnya, beserta lisensi tiap dataset, ada di [`strategi_data.md`](strategi_data.md). Simpan ke folder `data/`, lalu satukan manifest-nya:

```bash
cd pipeline && ../pilot/.venv/bin/python unify.py   # → pipeline/unified/manifest.csv
# Windows: cd pipeline; ..\pilot\.venv\Scripts\python.exe unify.py
```

#### 4. Label foto pompa BBM (`labeling/`)

- 858 foto dataset SachaIZADI dilabeli ulang secara buta oleh AI vision. Sebanyak 102 foto sulit dibaca ulang dengan zoom.
- Temuan: label asli dataset **tidak konsisten**. Umumnya desimal dibuang (`66.02` → `66`), tetapi 93 foto dinaikkan +1 (`128.28` → `129`). Jadi label asli tidak bisa dipakai mentah-mentah.
- Hasil final ada di `labeling/labels_final.csv`. Kolom `usable_for_training` bernilai `true` untuk 779 foto.

| status_vs_original | jumlah |
|---|---|
| match | 688 |
| orig_rounded_up | 93 |
| unreadable (masih ada `?`) | 74 |
| mismatch | 3 |

## Struktur repo

| Folder/File | Isi |
|---|---|
| `docs/PRD.md` | Kebutuhan produk aplikasi: tujuan, user story, acceptance criteria, metrik, risiko |
| `app/` | Web app FastAPI + HTML: ambil foto, OCR, koreksi label, ringkasan |
| `pilot/` | Generator data sintetis, baseline Tesseract, training CNN per-digit & CRNN-CTC |
| `pipeline/` | Penyatuan dataset publik (`unify.py`) dan generator degradasi citra (`degrade.py`) |
| `labeling/` | Label ulang foto pompa BBM (pass 1 `out/batch_*.jsonl`, pass 2 `out/recheck_*.jsonl`) |
| `tools/` | Utilitas: ekstraksi frame video, split data per grup |
| `scripts/windows/` | Skrip PowerShell untuk Windows: setup app, jalankan app, jalankan pilot |
| `.github/workflows/ci.yml` | Uji otomatis di Windows, macOS, dan Linux setiap push |
| `survei_literatur_seven_segment_AMR.md` | Survei 22 paper terkait (2018–2026) |
| `strategi_data.md` | Verifikasi dataset publik dan lisensinya |

Yang tidak ikut di repo: `data/` (dataset publik), `pilot/data/` dan `pipeline/degraded/` (bisa digenerate ulang), `*.traineddata`, `.venv/`.

## Kontribusi

1. Ambil tiket di [Issues](https://github.com/ikhdannizarmaulana/riset-seven-segment/issues), lalu buat branch `tahap-N/<deskripsi-singkat>`.
2. Cocokkan pekerjaan dengan acceptance criteria di tiket dan [PRD](docs/PRD.md).
3. Buka Pull Request dengan `Closes #<nomor>`. Sertakan output perintah yang benar-benar dijalankan (metrik, hasil pytest).

## Lisensi & atribusi data

Kode di repo ini untuk keperluan riset akademik. Dataset pihak ketiga mengikuti lisensinya masing-masing (lihat `strategi_data.md`). Dataset [SachaIZADI/Seven-Segment-OCR](https://github.com/SachaIZADI/Seven-Segment-OCR) tidak memiliki file lisensi, jadi fotonya tidak disertakan di repo ini, dan label turunannya hanya untuk evaluasi riset.
