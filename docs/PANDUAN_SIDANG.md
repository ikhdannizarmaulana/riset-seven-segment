# Panduan Memahami Riset & Bekal Tanya-Jawab Dosen

> Dokumen ini menjelaskan **cara kerja sistem**, **apa yang sudah dikerjakan**, **isi skripsi per bab**, dan **bank pertanyaan dosen beserta jawabannya**. Semua angka di sini berasal dari file di repo. Sumber setiap angka ditulis supaya bisa dicek ulang.
>
> Status per 5 November 2026. Yang **sudah** dikerjakan dan yang **belum** dipisahkan dengan tegas, supaya tidak ada klaim berlebihan saat sidang.

---

## 1. Ringkasan 1 menit (hafalkan ini)

> "Skripsi saya membandingkan tiga metode untuk membaca angka di layar **seven segment**, seperti layar pompa BBM, meteran, atau tensimeter: **Tesseract yang di-fine-tune**, **CNN per-digit**, dan **CRNN-CTC**. Ketiganya diuji pada kondisi foto yang berbeda-beda: bersih, silau, buram, miring, redup, dan terpotong. Tujuannya mengetahui metode mana yang paling akurat dan paling cepat untuk tiap kondisi.
>
> Datanya gabungan **data sintetis** buatan generator sendiri dan **foto nyata**, terutama 858 foto layar pompa BBM yang saya label ulang karena label aslinya ternyata dibulatkan. Hasilnya saya wujudkan dalam **aplikasi web**: foto dari HP, layar dideteksi otomatis, lalu ketiga metode membaca angkanya dan hasilnya tampil berdampingan beserta waktu prosesnya."

---

## 2. Konsep dasar (bahasa sederhana)

### 2.1 Seven segment itu apa?
Angka yang tersusun dari **7 garis (segmen)** berlabel a–g. Setiap angka 0–9 adalah kombinasi segmen yang menyala. Contoh: angka **1** hanya menyalakan 2 segmen kanan, angka **8** menyalakan semuanya. Titik desimal (DP) adalah titik kecil terpisah.

```
 aaa
f   b
 ggg
e   c
 ddd  .
```

- **LCD**: segmen gelap di atas latar terang (misalnya pompa BBM, kalkulator).
- **LED**: segmen terang di atas latar gelap (misalnya jam digital merah).

### 2.2 Kenapa susah dibaca komputer?
1. **Segmennya terputus.** OCR umum dilatih pada huruf cetak yang garisnya menyambung, jadi "8" seven segment sering dibaca sebagai karakter lain.
2. **Titik desimal sangat kecil.** Kalau hilang, `204.84` terbaca `20484`. Ini kesalahan fatal untuk angka liter atau rupiah.
3. **Kondisi foto nyata:** pantulan cahaya (glare), buram, miring, redup, digit terpotong bingkai, dan **ghosting** (segmen mati masih samar terlihat).
4. **Angka yang mirip:** 0↔8 (beda segmen tengah saja), 1↔7 (beda segmen atas), 5↔6, 6↔8.

### 2.3 OCR itu apa?
OCR (*Optical Character Recognition*) adalah proses mengubah gambar tulisan menjadi teks yang bisa diproses komputer.

---

## 3. Cara kerja sistem (pipeline)

```
Foto HP ─► [1] Deteksi layar ─► [2] Crop + prapemrosesan ─► [3] OCR (3 engine) ─► [4] Hasil + waktu
                 (bbox)               (gray, CLAHE, dll.)        Tesseract / CNN / CRNN
```

### [1] Deteksi layar
Foto 2048×1152 berisi banyak hal lain (badan pompa, latar). Langkah ini mencari **kotak (bounding box)** posisi layar. Tanpa langkah ini, OCR akan membaca "sampah".
- Status: **belum dibuat** (tiket #2, #3). Saat ini pengguna masih menggambar kotak layar secara manual di app.

### [2] Prapemrosesan
Membuat gambar lebih "bersih" sebelum dibaca:
- **Grayscale**: buang warna.
- **Deteksi polaritas**: LCD (gelap di atas terang) dan LED (terang di atas gelap) disamakan arahnya.
- **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*): menaikkan kontras per bagian gambar. Membantu kondisi redup.
- **Threshold (Otsu)**: mengubah gambar jadi hitam-putih.
- **Resize**: menyamakan tinggi gambar.

**Prinsip keadilan:** ketiga engine diberi crop dan prapemrosesan yang **sama**, supaya yang dibandingkan memang metodenya, bukan kualitas inputnya.

### [3] Tiga engine OCR yang dibandingkan

| | **Tesseract fine-tuned** | **CNN per-digit** | **CRNN-CTC** |
|---|---|---|---|
| Analogi | Orang yang sudah pandai membaca, lalu dilatih ulang khusus membaca angka seven segment | Membaca angka **satu per satu**: potong tiap digit, tebak masing-masing | Membaca **satu baris sekaligus** dari kiri ke kanan, seperti mata menyapu |
| Cara kerja | OCR open-source Google/HP. Intinya jaringan **LSTM**. Model dasar `eng` di-*fine-tune* (dilatih lanjut) dengan data seven segment memakai `tesstrain`. | (a) **Segmentasi**: crop layar dipotong per digit. (b) Tiap potongan masuk **CNN** kecil yang mengklasifikasi 0–9, ditambah satu output apakah ada titik desimal. | **CNN** mengekstrak fitur visual → **BiLSTM** membaca urutan fitur kiri-kanan → **CTC** mengubah urutan prediksi menjadi teks tanpa perlu tahu posisi tiap digit. |
| Kelebihan | Sudah matang, tidak perlu merancang arsitektur, banyak dipakai industri | Sederhana, cepat, mudah dijelaskan | Tidak perlu memotong per digit, jadi tahan terhadap digit yang berdempetan |
| Kelemahan | Arsitekturnya tidak didesain untuk seven segment, dan sensitif terhadap prapemrosesan | **Bergantung pada segmentasi.** Kalau pemotongan salah, hasilnya pasti salah. | Butuh data lebih banyak, dan training lebih rumit |
| Ukuran di pilot | — (memakai traineddata `7seg`/`ssd` pihak ketiga) | ±1,41 juta parameter | ±0,56 juta parameter |

**CTC itu apa?** *Connectionist Temporal Classification.* Model mengeluarkan prediksi untuk setiap "kolom" gambar, misalnya `1 1 _ 2 2 _ . 5`. Simbol `_` (blank) memisahkan. CTC menggabungkan prediksi berulang lalu membuang blank, sehingga hasilnya `12.5`. Karena itu kita **tidak perlu menandai posisi tiap digit** saat membuat data latih; cukup teks labelnya.

**Fine-tuning itu apa?** Melanjutkan training model yang sudah pintar (pre-trained) dengan data khusus kita. Lebih cepat dan butuh data lebih sedikit daripada melatih dari nol.

#### Detail: "fine-tune sendiri" pada Tesseract (tiket #5)

**Apa bedanya dengan pilot?**

| | Di pilot (sekarang) | Yang wajib di skripsi |
|---|---|---|
| Model | `7seg.traineddata` / `ssd.traineddata` | `sevenseg_ft.traineddata` (nama usulan) |
| Siapa yang melatih | **Orang lain** (Shreeshrii, GitHub), dengan font dan data milik dia | **Kita sendiri**, dengan data kita |
| Data latih | Tidak diketahui persis | Sintetis + potongan layar foto pompa BBM bagian **train** |
| Status | Hanya baseline pembanding | Belum dikerjakan |

**Analogi:** model `eng` itu seperti orang yang sudah lancar membaca koran. Fine-tuning adalah memberinya latihan tambahan khusus membaca angka di layar pompa bensin. Dia tidak belajar dari nol, cukup dilatih tambahan.

**Langkahnya:**
1. **Siapkan data latih.** Setiap contoh berupa satu gambar baris angka (`.png`) dan file teks berisi jawabannya (`.gt.txt`, misalnya `128.28`). Sumbernya gambar sintetis dari generator dan potongan layar foto bagian train, yang diproses dengan prapemrosesan seragam dari #4.
2. **Jalankan `tesstrain`**, tool resmi Tesseract (di dalamnya ada `lstmtraining`). Training mulai dari model `eng`, dan sebagai pembanding juga dari `ssd`. Jaringan LSTM di dalam Tesseract menyesuaikan bobotnya dengan data kita.
3. **Pantau error di data val** (CER val). Pilih hasil terbaik dan hentikan sebelum model mulai hafal (*overfitting*).
4. **Hasilnya file `sevenseg_ft.traineddata`**, yang dipakai di aplikasi (`TESS_LANG=sevenseg_ft`) dan diuji di data test.
5. **Batasi karakter keluaran** dengan whitelist `0123456789.`, supaya Tesseract tidak mengeluarkan huruf.

**Kenapa wajib:**
- Judul skripsi menyebut **"Tesseract Fine-Tuned"**.
- **Keadilan:** CNN dan CRNN dilatih dengan data kita. Kalau Tesseract tidak dilatih dengan data yang sama, perbandingannya berat sebelah.

**Urutan pengerjaan:** #13 (validasi label) → #1 (pembagian data) → #4 (prapemrosesan seragam) → **#5 (fine-tune Tesseract)**. Tiket #6 (CNN) dan #7 (CRNN) dikerjakan paralel dengan #5.

**Pertanyaan dosen yang mungkin muncul:** "Kenapa tidak melatih Tesseract dari nol?" Jawab: data kita tidak cukup untuk melatih dari nol. Fine-tuning memanfaatkan kemampuan dasar membaca yang sudah ada, sehingga cukup dengan data sedikit. Ini juga cara yang direkomendasikan dokumentasi Tesseract untuk font atau domain baru.

### [4] Output
Teks angka, confidence (tingkat keyakinan), dan waktu proses tiap tahap (deteksi, prapemrosesan, OCR) dalam milidetik.

---

## 4. Data

### 4.1 Tiga sumber data

| Sumber | Jumlah | Dipakai untuk | Catatan |
|---|---|---|---|
| **Sintetis** (generator sendiri, `pilot/generator.py`) | Tak terbatas (pilot: 12.000 latih, 2.400 uji) | Pra-latih dan uji terkontrol per kondisi | Segmen digambar sebagai poligon. Ada variasi LCD/LED, miring, tebal segmen, dan 8 kondisi degradasi. |
| **Foto pompa BBM SachaIZADI** (GitHub) | 858 foto, **779 layak** | Latih + uji utama (foto nyata) | Kualitas HQ 242, MQ 250, LQ 366. **Tidak punya lisensi**, jadi fotonya tidak di-commit. |
| **Dataset publik lain** | ±495 citra | **Uji eksternal** (generalisasi) | Tensimeter `chakibmed` (MIT), LED `cramatsu` (MIT, tanpa titik desimal), LCD meter `2005JIAZHI` (CC BY-NC) |

Sumber: `strategi_data.md`, `pipeline/unified/counts.txt`.

**Kenapa pakai data sintetis?** Foto nyata yang berlabel dan berlisensi jelas sangat sedikit. Data sintetis bisa dibuat dalam jumlah besar, labelnya otomatis 100% benar, dan kondisinya bisa diatur, misalnya "buat 300 gambar silau". Ini juga dilakukan di literatur (Finnegan dkk. 2019, Mitsubishi Electric).

### 4.2 Cerita pelabelan ulang (temuan penting, bisa dipakai di Bab 3/4)
1. Label asli SachaIZADI hanya berisi **angka bulat liter**, padahal layar menampilkan desimal (misalnya `66.02`).
2. Saat dicek, label asli ternyata **tidak konsisten**. Sebagian besar hanya membuang desimal (layar `66.02` → `66`). Tetapi **93 foto** dinaikkan +1: layar `128.28` dilabeli `129`, dan `63.96` dilabeli `64`. Yang dinaikkan pun tidak hanya desimal ≥ .50, karena `128.28` juga ikut naik. Rincian lengkap ada di `labeling/README.md`.
   **Akibatnya:** kalau label asli dipakai mentah-mentah, OCR yang membaca dengan **benar** justru dihitung **salah**.
3. Karena itu ke-858 foto dilabel ulang lengkap dengan desimalnya:
   - Pass 1: semua foto dibaca dengan model AI vision secara **buta**, tanpa melihat label asli. Karakter yang tidak terbaca ditandai `?`.
   - Pass 2: 102 foto sulit dibaca ulang dengan zoom.
4. Hasil dibandingkan dengan label asli (angka bulat):

| Status | Jumlah | Arti |
|---|---|---|
| match | 688 | Angka bulatnya sama |
| orig_rounded_up | 93 | Label asli = bacaan + 1 (label asli dinaikkan, tidak konsisten) |
| unreadable | 74 | Masih ada karakter yang tidak terbaca (silau, terpotong) |
| mismatch | 3 | Beda jauh. Perlu dicek manual, kemungkinan label asli yang salah. |

Sumber: `labeling/labels_final.csv`. Tingkat keyakinan: 516 tinggi, 326 sedang, 16 rendah.

⚠️ **Penting soal kejujuran akademik:** pelabelan ulang dibantu **model AI**. Ini harus **diungkapkan** di Bab 3 dan **divalidasi manusia**. Saran: periksa manual sampel acak (misalnya 100 foto), lalu laporkan persentase kesepakatan manusia vs AI. Validasi ini **belum dilakukan**. Fiturnya akan dibuat di aplikasi (tiket #13, lihat 4.4).

### 4.3 Pembagian data: mana untuk latih, mana untuk uji (rencana, tiket #1)

**Kenapa data harus dibagi?** Model yang diuji dengan foto yang sudah pernah dilihatnya saat latihan akan terlihat pintar padahal hanya hafal. Analoginya seperti ujian yang soalnya sama persis dengan latihan. Karena itu data dibagi tiga, dan bagian uji **tidak boleh disentuh** sampai evaluasi akhir.

| Data | Peran | Analogi |
|---|---|---|
| Sintetis (generator) | **Latih** | Buku latihan soal yang bisa dicetak sebanyak apa pun |
| Foto pompa BBM, bagian **train** (±60%) | **Latih** | Latihan soal dari ujian tahun lalu |
| Foto pompa BBM, bagian **val** (±20%) | **Memilih model terbaik** saat training (kapan berhenti, setelan mana yang dipakai) | Tryout |
| Foto pompa BBM, bagian **test** (±20%) | **Uji utama skripsi**. Angka di Bab 4 diambil dari sini. | Ujian sesungguhnya |
| Dataset publik lain (tensimeter, LED, meter) | **Uji eksternal**: apakah model tetap bisa membaca alat yang belum pernah dilihat | Ujian dari sekolah lain |
| Foto HP sendiri (±30–50) | **Uji coba aplikasi** end-to-end saja. Bukan untuk latih, bukan angka utama. | Gladi bersih sebelum demo |

Aturan pembagian:
- Pembagian dilakukan **per grup foto** (`split_group`), supaya foto yang hampir sama tidak terpisah ke latih dan uji. Kalau sampai terpisah, itu disebut **kebocoran data** (*data leakage*).
- Proporsi kualitas HQ/MQ/LQ dibuat seimbang di ketiga bagian.
- Foto `unreadable` dan yang belum layak dikeluarkan dari metrik utama, lalu dilaporkan terpisah.
- Hasil pembagian disimpan sebagai file dan di-commit, sehingga semua eksperimen memakai pembagian yang sama dan bisa diulang.

Jawaban singkat bila dosen bertanya: *"Data uji tidak pernah dipakai untuk training. Pembagiannya per grup foto, dikunci dalam file, dan foto HP saya hanya untuk uji coba aplikasi."*

### 4.4 Validasi manusia atas label AI (tiket #13)
- **Apa:** fitur di aplikasi (`/validate`). Foto ditampilkan bersama label dari AI, lalu kamu memilih **Setuju / Koreksi / Tidak terbaca**.
- **Kenapa:** label dibuat oleh AI, jadi harus ada bukti bahwa manusia sudah memeriksanya. Bukti itu berupa catatan per foto (siapa, kapan, keputusan) dan angka **persentase kesepakatan manusia vs AI**.
- **Bagaimana dipakai di skripsi:** validasi sampel acak (misalnya 100 foto) dengan seed tetap. Hasilnya ditulis di Bab 3, misalnya: "100 sampel divalidasi manual; kesepakatan X%; Y label dikoreksi." Foto berkeyakinan rendah dan 3 foto `mismatch` juga divalidasi.
- **Efek ke dataset:** kalau ada label dari manusia, label itu yang dipakai (`label_source=human`). Kalau tidak ada, label AI yang dipakai (`label_source=ai`).
- **Status:** belum dibuat (#13).

---

## 5. Metrik evaluasi

| Metrik | Arti | Contoh |
|---|---|---|
| **Exact Match (EM)** | Persentase gambar yang **seluruh** angkanya benar persis | Label `12.50`, prediksi `12.50` → benar. Prediksi `12.5` → salah. |
| **CER** (*Character Error Rate*) | Jumlah edit karakter (sisip, hapus, ganti; jarak Levenshtein) ÷ panjang label. **Makin kecil makin baik.** | Label `12.50`, prediksi `12.80`: 1 ganti dari 5 karakter → CER 0,2 |
| **Akurasi per digit** | Persentase digit yang benar | — |
| **Akurasi titik desimal** | Apakah titik desimal ada di posisi yang benar | Dipisah karena ini galat utama Tesseract |
| **Latency** | Waktu proses per gambar (ms), diukur di mesin dan thread yang sama | — |
| **Uji McNemar** | Uji statistik untuk mengetahui apakah perbedaan akurasi dua metode **signifikan**, bukan kebetulan. Cocok untuk data berpasangan (gambar yang sama dibaca dua metode). | p < 0,05 → berbeda signifikan |

---

## 6. Apa yang SUDAH dikerjakan (fakta)

| # | Pekerjaan | Hasil | Lokasi |
|---|---|---|---|
| 1 | Survei literatur | 22 paper (2018–2026), dataset publik, celah riset | `survei_literatur_seven_segment_AMR.md` |
| 2 | Strategi & verifikasi data | 10 dataset publik dicek aksesnya, lisensinya, dan isinya | `strategi_data.md` |
| 3 | Pilot (studi kelayakan) | Ketiga keluarga metode jalan di laptop. Hasil di bawah. | `pilot/` |
| 4 | Penyatuan dataset | 1.841 citra dari 6 sumber dalam satu manifest | `pipeline/unified/manifest.csv` |
| 5 | Pelabelan ulang 858 foto | 779 layak latih, temuan pembulatan label | `labeling/` |
| 6 | Aplikasi web MVP | FastAPI + HTML. Ambil foto, ROI manual, Tesseract, koreksi & simpan label, ringkasan. 6 uji otomatis lulus. | `app/` |
| 7 | Perencanaan | PRD dan 11 tiket per tahap + 1 roadmap | `docs/PRD.md`, Issues #1–#12 |

### Hasil pilot (data **sintetis**, n = 2.400; sumber `pilot/README_pilot.md`)

| Metode | Exact match | CER | Latency (ms/citra) |
|---|---|---|---|
| Tesseract `eng` (bawaan) | 0,139 | 0,527 | 53 |
| Tesseract `7seg` (traineddata publik) + prapemrosesan | 0,618 | 0,204 | 25 |
| CNN per-digit | 0,995 | 0,002 | 3,9 |
| CRNN-CTC | 0,988 | 0,004 | 8,4 |

- Kondisi paling berat untuk Tesseract adalah **glare** (EM 0,04–0,14).
- Dari 54 kesalahan Tesseract `7seg` pada gambar bersih, **41 hanya karena titik desimal hilang**.
- Uji **OOD** (gaya font berbeda, DSEG7): CNN 0,953, CRNN 0,885, Tesseract `7seg` 0,383.

**Cara menyampaikan angka ini dengan benar:** ini hasil **data sintetis**, latih dan uji dari generator yang sama, sehingga merupakan **batas atas yang optimistis**. Selain itu, CNN masih memakai posisi digit "kunci jawaban" (oracle) dan Tesseract belum di-fine-tune sendiri. **Jangan klaim CNN 99,5% sebagai hasil skripsi.** Hasil sesungguhnya adalah evaluasi pada foto nyata (tiket #10).

---

## 7. Yang BELUM dikerjakan (jangan diklaim)
- Detektor layar otomatis (#2, #3).
- Prapemrosesan seragam dan segmentasi digit nyata untuk CNN (#4).
- **Fine-tuning Tesseract sendiri** (#5). Sesuai judul, ini wajib ada.
- Training ulang CNN dan CRNN dengan foto nyata, lalu ekspor ke app (#6, #7, #8).
- UI tiga engine berdampingan (#9).
- Evaluasi pada foto nyata, metrik per kondisi, dan McNemar (#10). **Ini hasil utama skripsi.**
- Fitur dan proses validasi manusia atas label AI (#13, lihat 4.4).
- Tampilan aplikasi: design system (#15), alur unggah-saja yang otomatis (#14), dan layar proses langkah demi langkah (#16).

---

## 8. Isi skripsi per bab

**Bab 1 — Pendahuluan**
- Latar belakang: pembacaan alat ukur seven segment masih manual, dan OCR umum gagal membacanya.
- Rumusan masalah:
  1. Bagaimana akurasi dan waktu proses Tesseract fine-tuned, CNN per-digit, dan CRNN-CTC dalam membaca angka seven segment?
  2. Bagaimana pengaruh kondisi citra (silau, buram, miring, redup, terpotong) terhadap masing-masing metode?
  3. Metode mana yang paling sesuai diterapkan pada aplikasi?
- Tujuan, manfaat, dan batasan masalah: angka 1–6 digit + titik desimal, tanpa tanda minus, layar LCD/LED, bukan meter roda mekanik atau analog.

**Bab 2 — Tinjauan Pustaka**
- Teori: seven segment, OCR, Tesseract (LSTM) dan fine-tuning, CNN, RNN/LSTM/BiLSTM, CTC, prapemrosesan (CLAHE, Otsu), metrik, uji McNemar.
- Penelitian terkait: lihat bagian 9 (posisi riset).

**Bab 3 — Metodologi**
- Alur penelitian: studi literatur → pengumpulan data → pelabelan → prapemrosesan → training 3 model → integrasi aplikasi → evaluasi → analisis.
- Dataset: sumber, lisensi, pelabelan ulang (dengan pengungkapan penggunaan AI + validasi manusia), dan pembagian data.
- Perancangan tiap model (arsitektur, hiperparameter) dan perancangan aplikasi (diagram alir, use case, arsitektur).
- Skenario pengujian: metrik, kondisi, dan uji statistik.

**Bab 4 — Hasil & Pembahasan**
- Implementasi aplikasi (tangkapan layar).
- Tabel EM/CER/latency per metode × kondisi, pada uji nyata dan uji eksternal.
- Uji McNemar.
- Analisis kesalahan: digit tertukar (0↔8, 1↔7), titik desimal hilang, kegagalan segmentasi.
- Perbandingan dengan penelitian terkait.

**Bab 5 — Kesimpulan & Saran**
- Jawaban atas rumusan masalah.
- Saran: inference on-device, perluasan dataset multi-alat, dan IoT (ESP32-CAM).

---

## 9. Posisi riset terhadap penelitian lain (sering ditanya: "apa bedanya?")

| Penelitian | Yang dilakukan | Bedanya dengan skripsi ini |
|---|---|---|
| Effendi dkk. 2025 (JIIF Unpad) | YOLOv8 + CRNN-CTC, 218 citra uji, dataset tidak publik | Hanya satu metode OCR. Skripsi ini membandingkan **tiga** metode, per kondisi, dengan set uji lebih besar dan data/kode terbuka. *(Detail paper ini perlu dibaca ulang langsung sebelum dikutip.)* |
| ITM Web Conf 63 (2024) | Model STR pretrained **tanpa fine-tune**: PARSeq terbaik 56,97% | Menunjukkan model umum tanpa adaptasi kurang bagus. Skripsi ini justru menguji efek **fine-tuning / training khusus**. |
| YOLO-OCR meter air, ETASR 2026 (Telkom Univ. Purwokerto) | YOLOv8 + PP-OCR, meter air | Domainnya meter air (roda), metodenya berbeda |
| Laroca dkk. 2019/2021 | Pipeline deteksi + OCR untuk meter listrik | Inspirasi pipeline dua tahap (deteksi → baca) |
| Finnegan dkk. 2019 (Oxford) | Latih dengan digit sintetis, uji pada foto alat medis | Mendukung pendekatan data sintetis |
| CNN tensimeter (Frontiers) | CNN vs Tesseract per kualitas foto | Mendukung analisis per kondisi |

**Kontribusi skripsi (jawaban "apa kebaruannya?"):**
1. Perbandingan **tiga paradigma** OCR (OCR engine yang di-fine-tune, klasifikasi per karakter, pengenalan urutan) pada data dan pipeline yang **identik**.
2. Analisis **per kondisi citra** dan **titik desimal** sebagai metrik terpisah.
3. **Koreksi label** dataset publik (temuan pembulatan) dan rilis label yang lebih akurat.
4. **Aplikasi** yang bisa dipakai dan direproduksi (kode dan langkah terbuka).

---

## 10. Bank pertanyaan dosen + jawaban

### A. Latar belakang & motivasi
**Q1. Kenapa topik ini penting?**
Pembacaan alat ukur seven segment masih banyak yang manual, sehingga lambat dan rawan salah ketik. OCR umum seperti Tesseract bawaan gagal membacanya: di pilot, EM-nya hanya 0,139 pada data sintetis. Jadi dibutuhkan metode yang diadaptasi khusus.

**Q2. Kenapa tidak pakai Google Vision atau ChatGPT saja?**
Layanan cloud berbayar, butuh internet, datanya dikirim ke pihak ketiga, dan tidak bisa dikontrol atau direproduksi untuk penelitian. Model kecil (CNN 1,4 juta parameter) bisa jalan offline dalam hitungan milidetik. Fokus skripsi ini adalah memahami **metode**, bukan memakai layanan jadi.

**Q3. Apa kebaruan penelitianmu?**
Lihat bagian 9: perbandingan tiga paradigma pada pipeline yang identik, analisis per kondisi dan titik desimal, koreksi label dataset publik, dan aplikasi yang reproducible.

### B. Metode
**Q4. Kenapa memilih tiga metode ini?**
Ketiganya mewakili tiga pendekatan berbeda:
- Tesseract: OCR engine umum yang diadaptasi.
- CNN per-digit: klasifikasi per karakter, yang membutuhkan segmentasi.
- CRNN-CTC: pengenalan urutan tanpa segmentasi.

Ketiganya juga paling sering muncul di literatur AMR (*automatic meter reading*).

**Q5. Apa beda CNN per-digit dan CRNN?**
CNN harus **memotong** layar per digit dulu, lalu menebak tiap potongan. CRNN membaca **seluruh baris** sekaligus, dan CTC mengurus penyelarasannya. Jadi CRNN tidak bergantung pada segmentasi.

**Q6. Jelaskan CTC.**
Lihat bagian 3. Model memprediksi karakter per kolom gambar, termasuk simbol blank. CTC menggabungkan prediksi berulang dan membuang blank. Saat training, CTC menjumlahkan semua kemungkinan penyelarasan, sehingga cukup label teks tanpa posisi karakter.

**Q7. Fine-tuning Tesseract itu bagaimana?**
Mulai dari model LSTM `eng` (atau `ssd`), lalu dilatih lanjut dengan gambar baris angka seven segment beserta teksnya (`.png` + `.gt.txt`) memakai `lstmtraining`/`tesstrain`. Hasilnya file `.traineddata` baru. *(Status: belum dikerjakan, tiket #5.)*

**Q8. Kenapa perlu prapemrosesan, dan kenapa harus sama untuk ketiga metode?**
Prapemrosesan menormalkan kontras dan polaritas LCD/LED. Kalau setiap metode diberi prapemrosesan berbeda, perbedaan hasilnya tidak bisa dipastikan berasal dari metodenya. Itulah prinsip **perbandingan adil**.

**Q9. Kenapa CNN di pilot hampir 100%? Bukankah terlalu bagus?**
Benar, dan itu disadari. Ada tiga sebabnya: (1) latih dan uji dari generator yang sama, (2) posisi digit diambil dari kunci jawaban (oracle), bukan dari segmentasi nyata, (3) belum diuji di foto nyata. Karena itu angka pilot hanya **studi kelayakan**, bukan hasil skripsi.

### C. Data
**Q10. Datanya dari mana? Berapa banyak?**
Lihat bagian 4.1: data sintetis, 858 foto pompa BBM (779 layak), dan ±495 citra publik lain untuk uji eksternal.

**Q11. Kenapa tidak mengumpulkan foto sendiri?**
Foto nyata dengan beragam kondisi sudah tersedia. Tantangan utamanya justru **kualitas label**, dan itu sudah diperbaiki. Foto HP sendiri (30–50 foto) hanya dipakai sebagai uji coba aplikasi end-to-end, bukan untuk training dan bukan angka utama (lihat 4.3).

**Q12. Bagaimana labelnya dibuat? Bisa dipercaya?**
Label dibuat ulang secara buta oleh model AI vision. Karakter yang ragu diberi tanda `?` dan tidak ditebak. Foto sulit dibaca ulang dengan zoom. Konsistensinya terlihat dari perbandingan dengan label asli: 688 cocok, dan 93 sisanya hanya beda +1 karena label asli dinaikkan. Hanya 3 yang beda jauh. Ditambah validasi manual sampel acak lewat fitur `/validate` di aplikasi, yang mencatat siapa memvalidasi, kapan, dan persentase kesepakatannya *(tiket #13; lakukan sebelum sidang)*.

**Q13. Bolehkah memakai AI untuk melabel?**
Boleh, asal **diungkapkan** dan **divalidasi**. Ini juga praktik umum (pseudo-labeling, misalnya Sensors 2026 memakai pseudo-label). Ikuti kebijakan kampus soal penggunaan AI.

**Q14. Bagaimana mencegah kebocoran data?**
Pembagian train/val/test dilakukan per grup foto dan dikunci dalam file. Data uji tidak pernah dipakai untuk training atau tuning.

**Q15. Lisensinya?**
SachaIZADI tidak punya lisensi, jadi fotonya tidak didistribusikan ulang dan hanya dipakai untuk riset dengan sitasi. Dataset lain berlisensi MIT atau CC BY-NC (non-komersial, aman untuk skripsi).

### D. Eksperimen & evaluasi
**Q16. Metriknya apa saja dan kenapa?**
Lihat bagian 5. EM untuk kebenaran mutlak (yang penting bagi pengguna), CER untuk melihat seberapa dekat prediksi yang salah, akurasi titik desimal karena itu galat paling fatal, dan latency untuk kelayakan aplikasi.

**Q17. Bagaimana tahu satu metode benar-benar lebih baik?**
Dengan uji McNemar pada prediksi berpasangan di set uji yang sama. Kalau p < 0,05, perbedaannya signifikan.

**Q18. Kondisi citra ditentukan bagaimana?**
- Data sintetis: diatur langsung oleh generator.
- Foto nyata: setiap foto diberi tag kondisi (`glare`, `blur`, `tilt`, `dim`, `occlusion`, `cut_off`, `dirty`, `clean`) saat pelabelan, sehingga hasil bisa dirinci per kondisi.

**Q19. Hardware-nya apa?**
Laptop Apple M4. Training memakai GPU Apple (MPS). Latency diukur di CPU dengan 1 thread, supaya ketiga metode diukur setara.

### E. Aplikasi
**Q20. Arsitektur aplikasinya?**
FastAPI (Python) di laptop, antarmuka HTML yang dibuka dari browser HP lewat Wi-Fi yang sama. Ketiga engine berada di belakang satu interface `OCREngine`. Model deep learning dijalankan dengan ONNX Runtime.

**Q21. Kenapa web app, bukan aplikasi Android?**
Fokus skripsi adalah perbandingan metode. Web app membuat ketiga engine jalan di mesin dan prapemrosesan yang sama (adil), biayanya Rp0, dan pembuatannya cepat. Android/on-device masuk ke saran pengembangan.

**Q22. Kalau layar tidak terdeteksi?**
Aplikasi meminta pengguna menggambar kotak layar secara manual sebagai cadangan.

**Q21a. Aplikasinya bisa jalan di Windows?**
Bisa. Aplikasi dan pipeline didukung di Windows 10/11, macOS, dan Linux.
- **Setup Windows:** cukup satu skrip PowerShell (`scripts\windows\setup_app.ps1`). Skrip ini memasang Python dan Tesseract, menyiapkan lingkungan, lalu menjalankan uji otomatis.
- **Bukti:** setiap perubahan kode diuji otomatis di ketiga sistem operasi lewat GitHub Actions (CI). Hasilnya bisa ditunjukkan di tab *Actions* repo.
- **Hal khusus Windows yang ditangani:**
  - Tesseract tidak ada di PATH, jadi app mencarinya sendiri di `C:\Program Files\Tesseract-OCR`.
  - Encoding teks dipaksa UTF-8.
  - Path file disimpan dalam format yang sama untuk semua OS.
  - Firewall perlu diizinkan supaya HP bisa mengakses.
- **Training:** di laptop dengan GPU NVIDIA otomatis memakai CUDA. Tanpa GPU NVIDIA memakai CPU (lebih lambat). Di Mac memakai GPU Apple (MPS).
- **Pengecualian:** fine-tuning Tesseract (`tesstrain`) membutuhkan lingkungan Linux, jadi di Windows dijalankan lewat WSL2.

**Q22a. Bagaimana pengguna memakai aplikasinya?** *(rencana #14, #15)*
Cukup **unggah atau ambil foto**. Deteksi layar, prapemrosesan, dan ketiga metode berjalan otomatis, lalu hasilnya tampil berdampingan. Pengguna hanya ditanya "Apakah angkanya benar?". Semua parameter teknis disembunyikan di "Mode peneliti". Tampilannya memakai **design system**, yaitu kumpulan komponen dan aturan visual yang seragam, sehingga konsisten dan nyaman di HP.

**Q22b. Bisa tunjukkan bagaimana metodenya bekerja?** *(rencana #16)*
Bisa. Saat memproses, aplikasi menampilkan **langkah demi langkah dengan gambar asli dari foto itu**, bukan animasi buatan:
foto asli → kotak deteksi layar → potongan layar → grayscale → penyamaan terang-gelap LCD/LED → peningkatan kontras (CLAHE) → threshold → cara kerja tiap metode → hasil.

Visual cara kerja tiap metode:
- **Tesseract:** gambar input akhir dan teks yang dibaca.
- **CNN:** kotak potongan tiap digit, plus grafik batang keyakinan 0–9 untuk tiap digit.
- **CRNN:** tebakan per kolom (misalnya `1 1 _ 2 2 _ . 5`), lalu digabung dan blank dibuang menjadi `12.5`. Inilah cara kerja CTC.

Setiap langkah menampilkan waktu prosesnya. Saat presentasi, langkah bisa dimajukan dan dimundurkan satu per satu.

**Kenapa fitur ini berguna untuk sidang:** dosen bisa melihat langsung bahwa (1) ketiga metode menerima input yang sama, yaitu prinsip keadilan, (2) di mana metode gagal, misalnya potongan digit CNN yang salah pada foto silau, dan (3) bahwa angkanya berasal dari proses nyata, bukan hasil yang ditempel.

### F. Keterbatasan & lanjutan
**Q23. Apa keterbatasan penelitianmu?**
- Foto nyata didominasi satu domain (LCD pompa BBM).
- Sebagian foto tidak terbaca bahkan oleh manusia, sehingga dikeluarkan.
- Tidak mencakup tanda minus atau meter roda mekanik.
- Latency diukur di laptop, bukan HP.

**Q24. Kalau diberi waktu lebih, apa yang akan dikerjakan?**
Dataset multi-alat milik sendiri, inference langsung di HP (ONNX/TFLite), dan detektor plus OCR dalam satu model (end-to-end).

---

## 11. Glosarium cepat
- **Bounding box (bbox):** kotak koordinat (x1, y1, x2, y2) penanda objek.
- **ROI** (*Region of Interest*): area gambar yang diproses, di sini area layar.
- **Segmentasi digit:** memotong crop layar menjadi gambar per digit.
- **Epoch:** satu putaran penuh model melihat seluruh data latih.
- **Overfitting:** model hafal data latih tetapi buruk di data baru.
- **OOD** (*out-of-distribution*): data uji yang gayanya berbeda dari data latih.
- **ONNX:** format standar file model agar bisa dijalankan di luar PyTorch.
- **Seed:** angka acak awal, supaya hasil training bisa diulang sama.
- **Pseudo-label:** label yang dibuat oleh model, bukan manusia.
