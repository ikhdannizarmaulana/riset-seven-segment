# `tools/` — Alat Bantu Pembagian Data (`split_by_group.py`) dan Ekstraksi Frame (`extract_frames.sh`)

> **Isi dokumen:** dua alat kecil. `split_by_group.py` membagi data menjadi train/val/test **per kelompok** supaya tidak ada kebocoran data. `extract_frames.sh` memotong video menjadi foto-foto (frame).
>
> **Konteks penting:** kedua alat ini dirancang ketika rencana awal masih **mengumpulkan data sendiri** (beberapa alat ukur, beberapa sesi foto/video, dengan metadata `alat_id` dan `sesi_id`). Keputusan saat ini: **tidak ada pengumpulan data baru**. Data latih = sintetis + split train SachaIZADI. Foto HP peneliti (30–50) hanya untuk *smoke test end-to-end*. Jadi kedua alat ini **belum dipakai** di pipeline utama. Konsep anti-bocornya tetap dipakai, dan pembagian SachaIZADI dikerjakan di tiket **#1**. Status per 5 November 2026.

---

## 1. `split_by_group.py`: pembagian data anti-bocor

### 1.1 APA
Skrip ini membaca `metadata.csv` (satu baris per foto), mengelompokkan foto berdasarkan kolom tertentu (default `alat_id` + `sesi_id`), lalu menempatkan **seluruh foto dalam satu kelompok ke split yang sama**: `train`, `val`, `test`, atau `test_unseen`.

### 1.2 KENAPA: masalah kebocoran data (*data leakage*)
Misalnya kita merekam video layar meteran selama 10 detik lalu mengambil 20 frame. Ke-20 frame itu **hampir identik**. Kalau dibagi acak per foto, sebagian masuk data latih dan sebagian masuk data uji. Model cukup **mengingat** wajah layar itu untuk menjawab benar di data uji. Akurasinya terlihat tinggi, padahal model belum tentu bisa membaca layar yang belum pernah dilihatnya.

Analoginya seperti ujian yang soalnya sama dengan latihan kemarin, hanya difotokopi ulang. Nilai tinggi tidak membuktikan siswa paham.

Solusinya: bagi **per sesi**, bukan per foto. Satu sesi (satu alat, satu waktu pengambilan) masuk **satu** split saja.

**`test_unseen` (alat tak terlihat):** opsi `--test-alat` memasukkan **seluruh foto dari alat tertentu** ke split `test_unseen`. Fungsinya untuk menguji: "apakah model bisa membaca alat yang **belum pernah** dilihat sama sekali saat latih?" Ini uji generalisasi yang lebih keras daripada test biasa.

### 1.3 BAGAIMANA (langkah demi langkah)
1. Baca CSV. Wajib ada kolom `file`, `label`, `kondisi`, `alat_id`, dan semua kolom di `--group`.
2. Kelompokkan baris berdasarkan nilai kolom `--group`, misalnya (`ALAT03`, `S012`).
3. Kelompok yang `alat_id`-nya ada di `--test-alat` langsung diberi label `test_unseen`.
4. Kelompok sisanya diurutkan, lalu **diacak dengan seed** (`--seed`, default 42), sehingga urutannya sama setiap kali dijalankan.
5. Kelompok diisi berurutan ke `train` sampai kuota tercapai, lalu `val`, dan sisanya `test`. Kuota dihitung dari **jumlah foto**, bukan jumlah kelompok. Pengecekan dilakukan **sebelum** kelompok ditambahkan, jadi satu kelompok besar bisa membuat train sedikit melebihi kuotanya.
6. Tulis CSV baru (`--out`) dengan tambahan kolom `split`.
7. Cetak **tabel jumlah per split × kondisi**, lalu dua cek:
   - `grup bocor ke >1 split: N`, yang **harus 0**.
   - `label string test yang juga ada di train: X dari Y`. Ini informasi saja: angka yang sama (misalnya `12.30`) bisa muncul di dua sesi berbeda, dan itu bukan kebocoran gambar.

### 1.4 Cara menjalankan
```bash
python3 tools/split_by_group.py metadata.csv --out metadata_split.csv \
    --group alat_id,sesi_id --ratio 0.7,0.15,0.15 --seed 42 [--test-alat ALAT07,ALAT08]
```
Hanya memakai *standard library* Python, tanpa instalasi tambahan.

| Argumen | Default | Arti |
|---|---|---|
| `csv` (posisi) | — | File metadata input |
| `--out` | wajib | File output (input + kolom `split`) |
| `--group` | `alat_id,sesi_id` | Kolom kunci kelompok |
| `--ratio` | `0.7,0.15,0.15` | Proporsi train, val, test. **Rencana skripsi saat ini 60/20/20**, jadi pakai `--ratio 0.6,0.2,0.2`. |
| `--seed` | 42 | Seed pengacakan |
| `--test-alat` | kosong | Daftar `alat_id` untuk `test_unseen` |

### 1.5 Hasil uji cepat (dijalankan saat menulis dokumen ini)
CSV buatan: 3 alat × 2 sesi × 3 foto = 18 baris, `--ratio 0.6,0.2,0.2 --test-alat A3`. Output skrip:
```
split             clean  total
test_unseen           6  6
train                 9  9
val                   3  3
grup bocor ke >1 split: 0
```
Pelajaran dari uji ini: dengan hanya 4 kelompok biasa (A1 dan A2, masing-masing 2 sesi), split `test` **kosong**, karena kelompok dibagi utuh. **Kalau jumlah kelompok sedikit, proporsi tidak bisa tepat**, jadi selalu cek tabel ringkasannya.

### 1.6 Keterbatasan dan catatan untuk tiket #1 (SachaIZADI)
- **Kolom `alat_id` di-*hardcode*.** Kolom ini dipakai di kode untuk `--test-alat` walaupun `--group` diganti. Kolom `kondisi` juga wajib ada. SachaIZADI tidak punya kolom-kolom ini, jadi **skrip ini tidak bisa langsung dipakai** untuk `labeling/labels_final.csv` atau `pipeline/unified/manifest.csv`.
- **Rencana untuk #1:** pakai `split_group` dari `pipeline/unified/manifest.csv` (grup dHash, 856 grup untuk 858 foto) sebagai kunci kelompok. Ada dua cara: (a) tambahkan kolom tiruan (`alat_id` = `split_group`, `kondisi` = `quality`) lalu jalankan skrip ini, atau (b) tulis versi yang membaca manifest langsung. Hanya foto `usable_for_training=True` (779) yang dibagi ±60/20/20. Pilihan antara (a) dan (b) **belum diputuskan**.
- **Tidak ada stratifikasi** (penyeimbangan proporsi per kategori). Proporsi HQ/MQ/LQ atau kondisi di setiap split bisa sedikit berbeda dari keseluruhan data, hanya karena kebetulan acak. Cek tabel ringkasan, lalu laporkan sebaran per split di Bab 3.

---

## 2. `extract_frames.sh`: memotong video menjadi frame

### 2.1 APA
Skrip ini mengubah **satu video satu sesi** menjadi file foto JPEG dengan nama yang membawa identitas alat dan sesi:
```
raw/ALAT03/S012/ALAT03_S012_f0001.jpg, ALAT03_S012_f0002.jpg, ...
```

### 2.2 KENAPA
- Merekam video lalu mengambil frame jauh lebih cepat daripada memotret satu per satu.
- **Nama file berisi `ALAT` dan `SESI`**, sehingga metadata `alat_id`/`sesi_id` untuk `split_by_group.py` bisa dibuat otomatis dan setiap frame tidak kehilangan asal-usulnya.
- **fps rendah (default 2 frame per detik)** mengurangi frame yang hampir identik.
- Aturan yang tertulis di skrip: **SATU sesi = SATU split.** Frame dari video yang sama tidak boleh dipisah ke train dan test (alasannya di §1.2).

### 2.3 BAGAIMANA
```bash
./tools/extract_frames.sh VIDEO.mp4 ALAT03 S012 [fps=2]
```
1. Buat folder `raw/ALAT03/S012/` (relatif terhadap folder tempat perintah dijalankan).
2. Jalankan `ffmpeg -vf fps=2 -q:v 2`: ambil 2 frame per detik dan simpan sebagai JPEG berkualitas tinggi (`-q:v 2`, skala 2–31, makin kecil makin bagus).
3. Cetak jumlah frame yang dihasilkan.

Butuh `ffmpeg` terpasang (misalnya `brew install ffmpeg`). Skrip memakai `set -euo pipefail`, jadi langsung berhenti kalau ada error atau argumen yang kurang.

### 2.4 Status
**Tidak dipakai di rencana saat ini**, karena tidak ada pengumpulan data video baru. Skrip tetap disimpan kalau nanti dibutuhkan, misalnya untuk merekam video smoke test. Hasil dari skrip ini tetap **tidak boleh** dipakai sebagai data latih atau angka utama.

---

## 3. Status: sudah vs belum

| Pekerjaan | Status | Tiket |
|---|---|---|
| `split_by_group.py` berfungsi (uji cepat: 0 grup bocor) | ✅ Ada, untuk metadata gaya `alat_id`/`sesi_id` | — |
| `extract_frames.sh` | ✅ Ada, tidak dipakai saat ini | — |
| Split train/val/test SachaIZADI per grup dHash (±60/20/20, 779 foto layak) | ❌ Belum | **#1** |
| Laporan sebaran HQ/MQ/LQ per split | ❌ Belum | #1 |

---

## 4. Pertanyaan dosen yang mungkin muncul

**T: Kenapa tidak dibagi acak per foto saja?**
J: Foto dari sesi yang sama hampir identik. Kalau dibagi per foto, data uji "bocor" ke data latih dan akurasinya terlihat tinggi palsu. Pembagian per kelompok memastikan model diuji pada foto yang benar-benar belum pernah dilihatnya.

**T: SachaIZADI tidak punya info sesi. Bagaimana mencegah bocor?**
J: Foto yang nyaris identik dideteksi dengan dHash di `pipeline/unify.py`, lalu dikelompokkan. Hasilnya 856 grup untuk 858 foto. Pembaginya dikerjakan di tiket #1.
