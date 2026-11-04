# Pilot / Studi Kelayakan — Pengenalan Angka Seven Segment

Konteks: skripsi *"Analisis Komparatif Tesseract Fine-Tuned, CNN Per-Digit, dan CRNN-CTC untuk Pengenalan Angka Seven Segment pada Variasi Kondisi Citra"*.

Tujuan pilot: membuktikan bahwa seluruh pipeline (generator data → baseline Tesseract → model deep learning → metrik per kondisi) **bisa dijalankan di mesin lokal**. Semua angka di bawah berasal dari eksekusi nyata pada 4 Nov 2026 (log di `logs/`, angka mentah di `results.json` / `results.csv`).

## Lingkungan

- macOS, Apple M4 (10 core), Python 3.9.6 (venv `.venv`)
- Tesseract 5.5.4 (leptonica 1.87.0), dipasang via `brew install tesseract` (hanya `eng` + `osd`)
- torch 2.8.0 (MPS tersedia), opencv-python-headless 5.0.0, numpy 2.0.2, Pillow 11.3.0, pytesseract
- Traineddata tambahan dari `github.com/Shreeshrii/tessdata_ssd`: `ssd.traineddata`, `7seg.traineddata`, `ssd_plus.traineddata` → `tessdata_extra/` (unduhan berhasil)
- Font DSEG v0.46 (`github.com/keshikan/DSEG`, lisensi OFL) → `fonts/`, hanya dipakai untuk uji generalisasi (OOD)

## Cara menjalankan

```bash
cd ~/riset-seven-segment/pilot
python3 -m venv .venv && .venv/bin/pip install numpy pillow opencv-python-headless pytesseract torch
brew install tesseract            # bila belum ada
# (tessdata_extra/ dan fonts/ sudah berisi file unduhan)

.venv/bin/python generator.py --out data/test  --per-condition 300 --seed 2026   # 2400 citra uji (≈1 detik)
.venv/bin/python generator.py --out data/train --n 12000 --mixed --seed 1        # 12000 citra latih (≈4 detik)
OMP_THREAD_LIMIT=1 .venv/bin/python eval_tesseract.py                            # ≈100 detik total, 4 varian
.venv/bin/python train_cnn.py --epochs 6                                         # CNN per-digit (62.6 s latih, MPS)
.venv/bin/python train_crnn.py --epochs 10                                       # CRNN-CTC (120.9 s latih, MPS)
OMP_THREAD_LIMIT=1 .venv/bin/python eval_ood_dseg.py --per-condition 50          # uji OOD font DSEG7
.venv/bin/python make_report.py                                                  # results.json/csv + contoh_gambar/
```

## Isi folder

| File | Fungsi |
|---|---|
| `generator.py` | Generator sintetis: 7 segmen + titik desimal digambar sebagai poligon (OpenCV, anti-alias). Variasi: 1–6 digit, titik desimal (p=0,5), LCD (latar abu-hijau, segmen gelap) vs LED (latar gelap, merah/hijau + glow), kemiringan (italic), tebal segmen, dan degradasi per kondisi. Metadata per citra (label, kondisi, parameter, bbox tiap digit) → `labels.jsonl` |
| `metrics.py` | Exact match, CER (Levenshtein/panjang label, termasuk titik), akurasi per digit (jumlah digit GT yang cocok dalam alignment Levenshtein optimal setelah titik dibuang ÷ jumlah digit GT) |
| `eval_tesseract.py` | Baseline Tesseract (`--oem 1 --psm 7`, whitelist `0123456789.`) |
| `train_cnn.py` | CNN per-digit (3 blok conv, 1,41 jt parameter), 2 head: digit 0–9 + titik desimal |
| `train_crnn.py` | CRNN-CTC (CNN + BiLSTM, 0,56 jt parameter), input 32×256, tanpa segmentasi |
| `eval_ood_dseg.py` | Uji OOD: citra dirender dengan font DSEG7 (bentuk glyph berbeda dari data latih) |
| `make_report.py` | Menggabungkan hasil + menyalin contoh citra |
| `results.json`, `results.csv`, `results_ood_dseg.csv` | Hasil akhir; prediksi per citra di `results/*_predictions.jsonl` |
| `contoh_gambar/` | 10 contoh (1 per kondisi + 2 DSEG); nama file memuat label |
| `logs/` | Output konsol asli setiap tahap |

Kondisi uji (300 citra/kondisi, total 2400; LCD/LED acak ~50:50): `clean`, `noise` (Gaussian σ 12–30 + JPEG q 20–60), `blur` (Gaussian / motion), `perspective` (rotasi ±12° + jitter sudut 4–10%), `dim` (gamma 1,6–2,6 × kecerahan 0,3–0,6), `glare` (elips putih blur, α 0,35–0,75), `ghost` (segmen mati terlihat, α 0,15–0,35), `combined` (2–3 degradasi acak).

## Hasil (set uji in-distribution, n=2400)

Varian Tesseract: `eng_raw` = eng pada citra mentah; `eng_pre`/`ssd_pre`/`7seg_pre` = prapemrosesan sederhana (grayscale, inversi bila tepi gelap, resize tinggi 80 px, Otsu, padding putih).

**Exact match per kondisi**

| metode | clean | noise | blur | perspective | dim | glare | ghost | combined | ALL |
|---|---|---|---|---|---|---|---|---|---|
| Tesseract eng_raw | 0.180 | 0.157 | 0.193 | 0.137 | 0.173 | 0.063 | 0.137 | 0.073 | 0.139 |
| Tesseract eng_pre | 0.180 | 0.203 | 0.177 | 0.157 | 0.110 | 0.040 | 0.187 | 0.107 | 0.145 |
| Tesseract ssd_pre | 0.690 | 0.753 | 0.683 | 0.547 | 0.423 | 0.143 | 0.710 | 0.347 | 0.537 |
| Tesseract 7seg_pre | 0.820 | 0.850 | 0.743 | 0.633 | 0.523 | 0.110 | 0.867 | 0.397 | 0.618 |
| CNN per-digit | 1.000 | 1.000 | 1.000 | 0.990 | 1.000 | 0.997 | 1.000 | 0.973 | 0.995 |
| CRNN-CTC | 1.000 | 1.000 | 1.000 | 0.960 | 1.000 | 0.997 | 1.000 | 0.947 | 0.988 |

**Ringkasan keseluruhan (ALL) + latency**

| metode | CER | akurasi per digit | latency rata-rata (ms/citra) | median (ms) |
|---|---|---|---|---|
| Tesseract eng_raw | 0.5267 | 0.5219 | 53.4 | 52.3 |
| Tesseract eng_pre | 0.5025 | 0.5461 | 52.3 | 52.0 |
| Tesseract ssd_pre | 0.2320 | 0.8366 | 25.2 | 24.6 |
| Tesseract 7seg_pre | 0.2040 | 0.8331 | 24.7 | 24.1 |
| CNN per-digit | 0.0016 | 0.9992 | 3.89 | 3.11 |
| CRNN-CTC | 0.0037 | 0.9980 | 8.39 | 5.92 |

Latency: CPU, 1 thread (Tesseract `OMP_THREAD_LIMIT=1`; torch `set_num_threads(1)`), 1 citra per panggilan, termasuk baca file + prapemrosesan. Tesseract: 80 citra/varian (10/kondisi), angka termasuk overhead `pytesseract` yang memanggil proses `tesseract` baru setiap citra (tesseract via API C/tesserocr akan lebih cepat). CNN/CRNN: 80 citra merata semua kondisi. CNN per-digit memakai bbox oracle (lihat keterbatasan), jadi biaya segmentasi tidak termasuk.

CER dan akurasi per digit per kondisi lengkap ada di `results.csv` dan `logs/report.log`.

**Analisis galat Tesseract (dari `results/tesseract_predictions.jsonl`):**
- `7seg_pre` kondisi `clean`: 54 dari 300 salah; **41 dari 54 hanya salah karena titik desimal hilang** (mis. `204.84` → `20484`); sisanya antara lain digit tunggal yang menghasilkan string kosong.
- `7seg_pre` kondisi `dim`: LCD hanya 33/154 benar, LED 124/146. Penyebabnya heuristik prapemrosesan: LCD yang diredupkan punya tepi gelap sehingga ikut di-*invert* secara keliru. Ini menunjukkan baseline Tesseract sangat bergantung pada prapemrosesan.
- `glare` paling berat untuk semua varian Tesseract (EM 0,04–0,14).

## Uji generalisasi OOD: font DSEG7 (n=400, 50/kondisi)

Model CNN/CRNN **tidak pernah melihat** glyph DSEG saat latih (hanya poligon dari generator.py).

| metode | clean | noise | blur | perspective | dim | glare | ghost | combined | ALL EM | ALL CER | ALL digit acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tesseract eng_pre | 0.160 | 0.180 | 0.160 | 0.120 | 0.100 | 0.000 | 0.120 | 0.080 | 0.115 | 0.5187 | 0.5445 |
| Tesseract ssd_pre | 0.380 | 0.580 | 0.520 | 0.380 | 0.220 | 0.080 | 0.500 | 0.300 | 0.370 | 0.2938 | 0.8229 |
| Tesseract 7seg_pre | 0.440 | 0.500 | 0.500 | 0.420 | 0.400 | 0.060 | 0.440 | 0.300 | 0.383 | 0.2797 | 0.7898 |
| CNN per-digit | 1.000 | 1.000 | 0.980 | 0.780 | 1.000 | 1.000 | 1.000 | 0.860 | 0.953 | 0.0154 | 0.9876 |
| CRNN-CTC | 0.960 | 0.960 | 0.900 | 0.760 | 0.880 | 0.920 | 0.860 | 0.840 | 0.885 | 0.0356 | 0.9807 |

Saat gaya font berganti, CNN/CRNN turun (paling jelas di `perspective` dan `combined`), sedangkan CNN per-digit tetap dibantu bbox oracle.

## Keterbatasan (penting dibaca sebelum mengutip angka)

1. **Sintetis ≠ nyata.** Semua data dibuat program. Angka ~99% CNN/CRNN adalah **batas atas yang optimistis**: data latih dan uji berasal dari generator yang sama (in-distribution). Validasi pada **foto nyata** (meteran listrik/air, timbangan, termometer, SPBU, dsb.) wajib dilakukan sebelum menarik kesimpulan.
2. **CNN per-digit memakai segmentasi oracle** (bbox diketahui dari generator). Pada foto nyata, segmentasi digit (proyeksi kolom/kontur/detektor) akan menjadi sumber galat tambahan dan harus diukur. CRNN-CTC tidak membutuhkan segmentasi.
3. **Tesseract di pilot ini belum di-fine-tune sendiri.** `ssd`/`7seg` adalah model hasil fine-tuning pihak ketiga (Shreeshrii) dengan font lain, bukan model yang dilatih pada data skripsi. Fine-tuning Tesseract dengan `tesstrain` (lstmtraining dari `eng`/`ssd`) belum dicoba.
4. Prapemrosesan Tesseract masih naif (heuristik inversi gagal pada LCD redup). Perbandingan yang adil perlu prapemrosesan yang sama-sama dioptimalkan.
5. Ukuran uji kecil (300/kondisi; OOD 50/kondisi). Belum ada interval kepercayaan, belum ada pengulangan dengan seed berbeda. Model DL dilatih sekali (seed 0).
6. Latency diukur di Apple M4 (CPU 1 thread). Di perangkat target (Raspberry Pi/HP Android) angkanya akan berbeda. Tesseract terbebani overhead *subprocess* dari pytesseract.
7. Percobaan pertama CRNN di CPU (~60 detik/epoch) terhenti oleh batas waktu tool setelah 6 epoch. Versi final memakai MPS untuk CNN/LSTM dan menghitung loss CTC di CPU (~12 detik/epoch).

## Kesimpulan kelayakan

**Layak secara teknis.** Generator, ketiga keluarga metode, metrik per kondisi, dan pengukuran latency semuanya berjalan lokal dalam kurang dari ~10 menit komputasi total. Pilot ini juga memberi sinyal awal: Tesseract `eng` tanpa adaptasi buruk untuk seven segment (EM 0,14), model khusus seven segment menaikkannya (EM 0,54–0,62), dan model DL kecil yang dilatih pada domain yang sama mendekati sempurna pada data sintetis.

## Rekomendasi untuk skripsi

1. **Kumpulkan dataset foto nyata** (target ≥500–1000 citra, beberapa jenis perangkat LCD/LED, dianotasi label string + kondisi). Pakai sebagai set uji utama; data sintetis untuk latih/pra-latih.
2. **Fine-tune Tesseract sungguhan** (tesstrain, start dari `eng` dan/atau `ssd`) pada data sintetis yang sama dengan CNN/CRNN agar perbandingan adil sesuai judul ("Tesseract Fine-Tuned").
3. **Ganti bbox oracle dengan segmentasi nyata** untuk CNN per-digit (atau deteksi ROI display + segmentasi digit), lalu laporkan galat end-to-end.
4. Perlakukan **titik desimal** sebagai metrik terpisah (dalam pilot ini menjadi sumber galat utama Tesseract).
5. Rancang eksperimen: train sintetis → uji nyata; sintetis + sebagian nyata → uji nyata; ablation per kondisi; ulangi ≥3 seed; laporkan mean ± std dan uji signifikansi (mis. McNemar untuk exact match).
6. Seragamkan prapemrosesan (deteksi polaritas LCD/LED yang lebih andal, CLAHE untuk kondisi redup) dan laporkan dengan/tanpa prapemrosesan.
7. Ukur latency di perangkat target bila skripsi mengklaim aplikasi real-time/edge.
