# Strategi Pengadaan Data — Skripsi OCR Angka Seven Segment

**Judul kandidat:** Analisis Komparatif Tesseract Fine-Tuned, CNN Per-Digit, dan CRNN-CTC untuk Pengenalan Angka Seven Segment pada Variasi Kondisi Citra
**Tanggal cek:** 4 November 2026 (WIB)
**Pembanding utama:** Effendi dkk. 2025 (JIIF Unpad): YOLOv8 + CRNN-CTC, 218 citra uji, datasetnya tidak publik (dari konteks, tidak saya cek ulang).

**Arti tanda**
- **[V] Terverifikasi:** saya cek sendiri lewat API, unduhan, atau skrip Python pada 4-11-2026.
- **[S] Snippet:** diambil dari cuplikan hasil pencarian atau survei sebelumnya, belum saya buka sendiri. Cek dulu sebelum dikutip.
- **[E] Estimasi:** perkiraan saya untuk perencanaan, bukan data terukur.

---

## 1. Ringkasan untuk keputusan

1. **Data utama tetap perlu dikumpulkan sendiri.** Tidak ada dataset publik yang sekaligus (a) berupa foto nyata, (b) memuat banyak jenis alat LCD dan LED, (c) berlabel string, (d) punya anotasi kondisi citra (redup/silau/miring/blur), dan (e) bisa diunduh tanpa login dengan lisensi jelas. [V] Kesimpulan ini berdasarkan tabel di Bagian 2.
2. **Dataset publik paling berguna sebagai pelengkap.**
   - **Pra-latih sintetis:** `MiXaiLL76/7SEG_OCR`.
   - **Uji lintas-dataset (eksternal):**
     - `chakibmed/bp-monitor-seven-segment` (LCD tensimeter nyata)
     - Kaggle `cramatsu` (modul LED nyata)
     - GitHub `SachaIZADI` (LCD pompa bensin nyata, 858 foto, tetapi tanpa lisensi)
     - Sampel `2005JIAZHI` (LCD meter utilitas, CC BY-NC)
   Hasil uji lintas-dataset ini memperkuat klaim "variasi kondisi" di luar data milik sendiri.
3. **Oxford IBME:** zip datasetnya **putus** (redirect ke beranda lab). [V] Yang masih tersedia hanya kode MATLAB-nya.
4. **UFPR-AMR dan Copel-AMR tidak realistis untuk S1.** Keduanya perlu perjanjian lisensi. Copel-AMR bahkan **menolak tanda tangan mahasiswa atau dosen** dan harus ditandatangani pimpinan institusi. [V] Domainnya juga meter listrik Brasil, bukan berfokus pada seven segment.

---

## 2. Verifikasi dataset publik

Semua sampel ada di `/Users/ikhdan/riset-seven-segment/data/`, total **118 MB** (di bawah batas 200 MB). [V]

| # | Dataset | Bisa diunduh tanpa login? | Ukuran asli | Format label | Lisensi | Hasil hitung aktual (Python) |
|---|---|---|---|---|---|---|
| 1 | HF `MiXaiLL76/7SEG_OCR` | **Ya** [V]: `gated:false`, datasets-server API jalan | 1 parquet 201.880.890 B, 3.333 baris, split *train only* [V] | kolom `image` + `text` (string, termasuk `.` dan `-`, mis. `6.691`, `-7891`) [V] | MIT (tag HF) [V] | Diunduh 100 baris pertama → `hf_7seg_ocr/images/` + `labels.csv`. Panjang string: 5 karakter 53, 4 karakter 27, 6 karakter 16, 3 karakter 3, 2 karakter 1. Citra **sintetis**, 232×96 px, gaya LED sian. [V] |
| 2 | Roboflow `mba/seven-segments-display-digits` | **Tidak**: API unduh membalas `"This method requires your API key."` [V], jadi perlu akun | 473 citra, 3 versi; v1: train 377 / valid 94 / test 2, resize 640×640 [V] | Deteksi objek per karakter, 13 kelas (`- . + 0–9`) [V]; ekspor YOLO/COCO/VOC [V] | CC BY 4.0 **[S]** (tidak terlihat di HTML yang saya ambil). Halaman menyebut *cloned from* `shrinand-aggarwal/digits-zktws` [V] | Tidak diunduh (butuh akun) |
| 3 | Kaggle `cramatsu/7-segment-display-yolov8` | **Ya** [V]: `kaggle.com/api/v1/datasets/download/...` redirect ke URL GCS bertanda tangan, tanpa login | 22.468.312 B [V] | YOLOv8 `.txt` + `data.yaml`, `nc: 10` (0–9). **Tidak ada kelas titik desimal** [V] | MIT (metadata API) [V] | 670 jpg + 670 txt (train 560 / valid 110, tanpa test). **390 citra sumber unik** (augmentasi rotasi ×2). Tidak ada sumber yang muncul di 2 split. Jumlah box per citra: 4 box 641, 3 box 22, 2 box 4, 1 box 2, 0 box 1. Isinya **satu modul LED "Quad Display"**, grayscale 640×640. Ada label yang tidak lengkap: `00_66…jpg` hanya punya 3 box padahal tampil 4 digit. String hasil turunan (box diurutkan menurut x) ada di `kaggle_cramatsu/derived_strings.csv` [V] |
| 4 | GitHub `SachaIZADI/Seven-Segment-OCR` | **Ya** (`git clone --depth 1`) [V] | 85 MB setelah `.git` (81 MB) dihapus [V] | CSV `;`: `used_liter;image;cadran_1..4` (`X` = digit kosong) [V] | **Tidak ada file LICENSE** (GitHub API `license` → 404) [V]. Artinya semua hak dilindungi: pakai hanya untuk evaluasi dengan sitasi, atau minta izin | 858 foto berlabel (HQ 242, MQ 250, LQ 366), semua file cocok dengan CSV. 2 digit: 534, 3 digit: 324. Foto nyata **LCD pompa bensin**, contoh 2048×1152, banyak pantulan dan blur. **Label hanya bagian bulat liter** (foto tampil `66.02`, label `66`) [V]. Ada juga crop per-digit `Datasets_digits/0..10` (kelas `10` = 1.110 file, kemungkinan non-digit/kosong **[S] interpretasi**) dan 688 file di `Datasets_frames` [V] |
| 5 | Oxford IBME (Finnegan dkk. 2019, *J. Med. Eng. Technol.* 43(6):341–355) | **Tidak (link putus)** [V]: `cameralab.eng.ox.ac.uk/data/bp_bg_meters.zip` → 302 ke `eng.ox.ac.uk/lcmt` (HTML); Wayback CDX hanya menyimpan respons 302 [V] | — | — | — | Masih tersedia: repo Bitbucket publik `eoinf96/7-segmentdigitgenerator` (generator digit sintetis MATLAB, PNG 56×56) dan `eoinf96/automatic_detection_bp_bg_meters` (kode MATLAB) [V]. Kontak di halaman: eoin.finnegan@eng.ox.ac.uk [V]. Kalau mau, coba minta datanya lewat email |
| 6 | UFPR-AMR | **Tidak** (lewat permintaan) [V] | 2.000 citra, 3 kamera, 2.340×4.160 s.d. 3.120×4.160; split 800/400/800; 5 digit per meter [V] | teks: kamera, posisi counter, bacaan, posisi tiap digit [V] | Hanya akademik, non-komersial [V] | Tidak diunduh. Jenis counter (dugaan mayoritas elektromekanik/rol) **[S]**, cek di paper |
| 7 | Copel-AMR | **Tidak** [V] | 12.500 citra 480×640; split 5.000/2.500/5.000; 2.500 tidak terbaca [V] | bacaan, 4 sudut counter, bbox tiap digit [V] | Perjanjian lisensi harus ditandatangani pimpinan institusi; **"We cannot accept licenses signed by students or faculty members."** [V] | Tidak diunduh |

### Temuan tambahan (pencarian HF API, 4-11-2026)

| # | Dataset | Akses | Isi (hasil hitung) | Lisensi | Kegunaan |
|---|---|---|---|---|---|
| 8 | HF `chakibmed/bp-monitor-seven-segment` | **Ya**, diunduh penuh 449 file / 1,5 MB [V] | `readings`: 55 foto (dev 29, test 26; 53 crop 160 px dari aplikasi + 2 foto utuh), label sistolik/diastolik/nadi. `digits`: 388 crop 24×32 (dev 201, test 187). Split per foto. **Satu tensimeter, satu orang** [V] | MIT [V] | Uji eksternal LCD tensimeter. README mencantumkan baseline CNN 10k parameter: 99,5% per-digit di test [V] |
| 9 | HF `2005JIAZHI/Annotated-Digital-LCD-Utility-Meter-And-Serial-Numbers-Dataset` | **Ya**, 152 file / 3,3 MB [V] | **50** jpg + YOLO txt + VOC xml (sampel gratis dari dataset berbayar 2.290 "kits"), 25 kelas (digit layar 0–9, digit serial terpisah) [V] | CC BY-NC 4.0 [V] | Uji eksternal kecil LCD meter utilitas. Boleh untuk skripsi (non-komersial) |
| 10 | HF `tomasko1987/seven-segment-digits` | Tidak di-gate, tapi shard parquet 339–474 MB/file, total ±3,86 GB; datasets-server gagal (*scan size limit*) [V] | README hanya berisi lisensi; isi **belum diperiksa** | MIT [V] | Tidak diunduh (melebihi batas). Bisa dicek nanti satu shard saja |
| 11 | HF `HelloImMrGrey/7SEG_OCR` | ada di hasil pencarian [V] | belum diperiksa (namanya mirip #1, mungkin salinan) | MIT (tag) [V] | — |

Isi folder `data/`:

```
data/hf_7seg_ocr/ (100 img + labels.csv + rows_0_100.json)
data/kaggle_cramatsu/ (670 img+txt, derived_strings.csv)
data/github_sachaizadi/ (repo tanpa .git)
data/hf_chakibmed_bp-monitor-seven-segment/
data/hf_2005JIAZHI_.../
data/hf_tomasko1987_seven-segment-digits/ (kosong, hanya cek)
```

**Implikasi metodologis:**
- **Kaggle #3 dan HF #1 bukan uji independen.**
  - #3 hanya satu modul LED. Labelnya juga tidak bisa langsung dipakai sebagai string, karena ada box hilang dan tidak ada titik desimal.
  - #1 sintetis. Kartu datasetnya sendiri menulis "TRAIN ONLY".
- **Sacha #4 perlu label ulang** kalau ingin membaca desimal. Kalau tidak, evaluasi hanya digit bulatnya dan sebutkan di skripsi.

---

## 3. SOP Pengumpulan Data Sendiri

### 3.1 Alat target

Harga di bawah adalah cuplikan marketplace dari hasil pencarian **[S]**, bisa berubah. Cek lagi sebelum membeli.

| Alat | Display | Kemudahan | Harga contoh [S] | Catatan variasi label |
|---|---|---|---|---|
| Multimeter digital DT830B | LCD | Beli | Rp37.900–Rp62.000 (Shopee/Tokopedia) | Ukur baterai/resistor berbeda, putar potensiometer agar nilai bervariasi |
| Timbangan dapur SF-400 | LCD | Beli | Rp34.300–Rp39.000 | Timbang benda berbeda-beda |
| Kalkulator LCD (mis. Casio MX-12B) | LCD | Pinjam/beli | Rp75.000–Rp103.680 | **Label bisa direncanakan** (lihat 4.1) |
| Termometer/higrometer ruangan HTC-1 | LCD | Beli/pinjam | mulai Rp149.900 (monotaro) | Nilai berubah lambat, jadi variasi label rendah |
| Xiaomi LYWSD03MMC (BLE) | LCD | Beli | Rp96.400–Rp109.000 | Bisa memberi ground truth otomatis via BLE (4.1) |
| Jam digital LED meja/dinding | LED | Beli/pinjam | contoh Rp59.900 | Ground truth dari timestamp (4.1) |
| Tensimeter digital rumah | LCD | **Pinjam** (keluarga/puskesmas) | — | Data kesehatan, lihat etika 3.8 |
| kWh meter prabayar (rumah/kos) | LCD | Sudah ada | — | Ada nomor meter/ID, harus di-blur |
| Timbangan digital pedagang pasar | LED/LCD | Izin pedagang | — | Lihat etika |
| Modul LED 4-digit untuk Arduino (opsional) | LED | Beli | **belum dicek** | Generator data LED berlabel otomatis (4.1) |

**Rekomendasi minimal [E]:** 8 alat, yaitu 5 LCD + 3 LED. Contoh set:
- **LCD:** multimeter, timbangan dapur, kalkulator, termometer, tensimeter/kWh
- **LED:** jam LED, timbangan pasar LED, modul LED Arduino

Kalau bisa, sediakan dua unit berbeda merek untuk tiap jenis alat, supaya tersedia data untuk uji *unseen device*.

### 3.2 Kondisi citra dan target jumlah

Kondisi yang direkam:
- **normal:** cahaya ruangan merata, tegak lurus
- **redup:** lampu dimatikan, hanya cahaya jendela/senter jauh
- **silau:** lampu atau senter HP diarahkan ke layar sampai ada pantulan
- **miring:** sudut 30–45° horizontal/vertikal
- **blur:** sengaja goyang atau tidak fokus; bisa juga ambil dari frame video saat bergerak

Kalau ingin variabel kontinu, catat juga `lux` (aplikasi lux meter HP), `sudut`, dan `jarak_cm`.

**Dasar ukuran sampel.** Rumus selang kepercayaan 95% akurasi: ±1,96·√(p(1−p)/n) [V, dihitung dengan Python]:

| n uji per sel | p=0,80 | p=0,90 | p=0,95 |
|---|---|---|---|
| 100 | ±7,8 pp | ±5,9 pp | ±4,3 pp |
| 200 | ±5,5 pp | ±4,2 pp | ±3,0 pp |
| 300 | ±4,5 pp | ±3,4 pp | ±2,5 pp |

Agar perbedaan 5–6 pp antar-model per kondisi bisa dibaca, set uji sebaiknya punya **≥100 citra per kondisi per jenis display**. Pakai uji McNemar berpasangan antar-model pada citra yang sama; uji berpasangan ini lebih hemat sampel.

**Target [E]:**

| | normal | redup | silau | miring | blur | Total |
|---|---|---|---|---|---|---|
| LCD | 300 | 300 | 300 | 300 | 300 | 1.500 |
| LED | 200 | 200 | 200 | 200 | 200 | 1.000 |
| **Total foto nyata** | | | | | | **±2.500** |

Dengan rasio ±70/15/15, set uji sekitar 375 citra, jauh di atas 218 citra uji Effendi dkk. Kalau waktu terbatas, prioritaskan **set uji yang besar dan bersih**. Data latih bisa ditambah dari sintetis (HF #1, generator Oxford, atau generator buatan sendiri).

**Variasi label penting untuk CRNN.** Satu sesi tidak boleh berisi angka yang sama berulang-ulang.
- Targetkan ≥60% string unik di set uji.
- Seluruh digit 0–9 harus muncul di setiap posisi.
- Kalau ada, sertakan juga titik desimal dan tanda minus.

`tools/split_by_group.py` mencetak berapa string uji yang juga muncul di train.

### 3.3 Cara pengambilan

1. **HP:** pakai 2–3 HP berbeda kalau bisa, dan catat model HP di kolom `hp`. Mode foto biasa, resolusi bawaan, HDR off agar konsisten (catat kalau HDR dinyalakan). Jarak 15–50 cm, layar alat mengisi ±30–60% frame.
2. **Foto vs video:**
   - **Foto** untuk set uji: tiap foto independen.
   - **Video → frame** untuk data latih, dengan `tools/extract_frames.sh VIDEO ALAT03 S012 2` (fps=2). Sudah diuji dengan video sintetis dan menghasilkan 6 frame dari 3 detik [V].
   - Frame dari video yang sama sangat mirip, jadi **seluruh frame satu sesi harus masuk satu split**.
3. **Definisi sesi:** satu sesi = satu alat × satu waktu pengambilan (±10–15 menit) × satu HP. ID-nya `S###`. Kalau satu sesi mencakup beberapa kondisi, kondisi dicatat per foto.
   - Saran agar distribusi kondisi per split seimbang: rekam semua 5 kondisi di setiap sesi.
   - Saat uji coba skrip dengan metadata dummy (1 kondisi per sesi), pembagian acak per grup menghasilkan sel kosong, misalnya val/redup = 0 [V]. Ini alasan saran di atas.
4. **Urutan kerja per sesi [E]:** sekitar 10 menit untuk 40–60 foto.
   1. Foto "papan sesi": kertas bertuliskan ALAT/SESI/tanggal, berguna untuk sinkronisasi.
   2. Ambil 8–12 foto per kondisi, ubah nilai di layar di antara foto.
   3. Catat di form/CSV.

### 3.4 Estimasi waktu (jam kerja) [E]

| Aktivitas | Asumsi | Jam |
|---|---|---|
| Persiapan alat, form, izin | — | 4–6 |
| Pengambilan 2.500 foto | ±50 foto per 10–15 menit + bongkar pasang | 10–15 |
| Ekstraksi frame + rename + metadata | skrip | 2–3 |
| Label string | 4–8 detik/citra dengan pintasan keyboard | 3–6 |
| Label ROI layar (bbox/4 titik) | 5–10 detik/citra | 4–7 |
| Box per-digit, hanya bila CNN per-digit tidak memakai segmentasi otomatis | ±20–30 detik/citra; lakukan pada subset saja | 5–10 |
| Verifikasi ulang set uji 100% + audit 10% train | — | 3–5 |
| **Total** | | **±30–50 jam** |

Tips: label box per-digit bisa dihindari. Setelah ROI di-rektifikasi, potong dengan pitch tetap atau proyeksi vertikal, lalu hitung kegagalan segmentasi sebagai bagian dari error pipeline CNN per-digit. Cara ini juga adil sebagai bahan perbandingan.

### 3.5 Estimasi biaya [E]

Disusun dari harga [S] di 3.1:
- Beli 4 alat murah (multimeter, timbangan, jam LED, termometer Xiaomi): ±Rp230.000–Rp310.000
- Kertas/print form izin, senter, kain hitam: ±Rp50.000
- Sisanya pinjam: kalkulator, tensimeter, HP teman, kWh rumah
- **Total kira-kira Rp300.000–Rp600.000**

Software semuanya gratis.

### 3.6 Tools pelabelan gratis

| Tool | Status | Cocok untuk |
|---|---|---|
| Label Studio | open source, Apache 2.0 [S: README GitHub] | ROI rectangle/polygon + TextArea transkripsi dalam satu tugas. Self-host lokal (`pip install label-studio`) |
| CVAT Community | self-hosted, MIT [S: docs.cvat.ai] | bbox per-digit + atribut; ekspor YOLO/COCO |
| Roboflow (free tier) | ada tier $0 [S: roboflow.com/pricing] | Mudah dipakai, tapi data dan kebijakan lisensi di cloud mereka; cek apakah proyek wajib publik |
| Skrip CSV sendiri | — | Cukup untuk label string: tampilkan gambar, ketik label, simpan baris CSV. Paling cepat kalau ROI tidak perlu |

### 3.7 Struktur folder dan metadata

Template CSV dibuat: `metadata_template.csv`. Struktur folder:

```
riset-seven-segment/
├── raw/<alat_id>/<sesi_id>/<alat>_<sesi>_<nnnn>.jpg   # asli, read-only
├── processed/roi/...                                  # crop/rektifikasi layar
├── metadata.csv                                       # satu baris per citra
├── metadata_split.csv                                 # + kolom split (hasil skrip)
├── alat.csv                                           # alat_id, jenis, merek, LCD/LED, jumlah_digit, desimal?
├── sesi.csv                                           # sesi_id, alat_id, tanggal, lokasi, hp, operator
└── tools/ (split_by_group.py, extract_frames.sh)
```

Kolom `metadata.csv`:

```
file, label, alat_id, jenis_alat, jenis_display, kondisi, sesi_id, split, hp, tanggal, sumber_label, diverifikasi_oleh, catatan
```

- `label`: string persis seperti di layar, termasuk `.` dan `-`. Digit mati/kosong tidak ditulis.
- `kondisi`: salah satu dari `normal|redup|silau|miring|blur`, atau kombinasi dengan `+`.
- `sumber_label`: `manual|serial|ble|timestamp|rencana|prelabel_dikoreksi`.

### 3.8 Pembagian train/val/test

Jalankan `tools/split_by_group.py metadata.csv --out metadata_split.csv --group alat_id,sesi_id --test-alat ALAT08`. Diuji dengan metadata dummy 800 baris: 0 grup bocor [V].

1. **Split per sesi (grup), bukan per foto:** 70/15/15 berdasarkan jumlah gambar.
2. **Test "alat tak terlihat":** 1–2 unit alat ditahan penuh sebagai `test_unseen`, satu LCD dan satu LED. Ini menguji generalisasi.
3. Laporkan akurasi per kondisi × per jenis display × per split.
4. Uji eksternal memakai dataset publik #3, #4, #8, #9 tanpa fine-tuning.
5. Jangan melakukan augmentasi sebelum split. Augmentasi hanya pada train.

### 3.9 Etika dan izin

- **Wajah, tangan, dan latar:** framing hanya ke layar. Wajah orang yang tidak sengaja terekam di-blur atau fotonya dibuang. Hapus EXIF GPS sebelum dipublikasikan (`exiftool` belum terpasang di mesin ini [V]). Simpan timestamp di CSV kalau diperlukan.
- **Tensimeter (data kesehatan):**
  - Pakai alat milik sendiri atau keluarga, dengan persetujuan tertulis singkat.
  - Jangan merekam nama/identitas pasien.
  - Nilai tensi tidak dikaitkan ke orang (ID anonim).
  - Jangan mengambil di fasilitas kesehatan tanpa izin resmi.
  - Data kesehatan termasuk data pribadi yang bersifat spesifik menurut UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi **[S]**. Cek pasalnya sebelum dikutip.
- **Pedagang pasar:** minta izin lisan dan dokumentasikan, misalnya formulir persetujuan singkat. Jangan merekam wajah, nama kios, atau transaksi pembeli. Tawarkan untuk tidak menyebut lokasi.
- **kWh prabayar:** blur nomor meter/ID pelanggan sebelum dirilis.
- **Kampus:** tanyakan ke pembimbing apakah perlu surat izin atau *ethical clearance*. Kemungkinan besar tidak perlu, kecuali melibatkan data pasien.
- **Rilis dataset:** setelah anonimisasi, rilis di HF/Zenodo dengan lisensi CC BY 4.0 [E, saran]. Ini juga kontribusi skripsi, karena dataset Effendi dkk. tidak publik.
- **Lisensi data pihak ketiga:**
  - #4 tanpa lisensi: hanya evaluasi + sitasi; jangan didistribusikan ulang.
  - #9 CC BY-NC.
  - #1, #3, #8 MIT: cantumkan atribusi.

---

## 4. Trik mempercepat pelabelan

### 4.1 Ground truth otomatis atau terencana

| Trik | Cara | Status | Risiko yang harus dicek |
|---|---|---|---|
| **Kalkulator + daftar angka rencana** | Buat daftar 500 angka acak (variasi panjang, desimal, minus) dengan Python; ketik berurutan dan foto; label = indeks daftar | ide [E], tanpa alat tambahan | Salah ketik. Foto ulang "papan sesi" tiap 50 angka sebagai checkpoint; audit 10% |
| **Jam digital + EXIF timestamp** | Samakan jam dengan jam HP (NTP); label = HH:MM dari `DateTimeOriginal` | ide [E] | Buang foto ±2 detik dari pergantian menit; format 12/24 jam; titik dua bukan digit |
| **Modul LED 4-digit + Arduino** | Mikrokontroler menampilkan angka acak dan mengirim `timestamp,angka` lewat serial; video direkam, frame dicocokkan berdasarkan waktu atau lewat pola sinkron (mis. `8888` lalu `0000`) | ide [E], harga modul belum dicek | Sinkronisasi waktu video vs log |
| **Multimeter dengan PC link** (UNI-T UT61E, RS232/USB) | Software merekam bacaan berkala; cocokkan dengan waktu foto | fitur interface [S: halaman UNI-T & Batronix] | Laju update layar vs log; pakai hanya bacaan stabil (nilai sama ≥2 sampel berturut) |
| **Multimeter Bluetooth** (OWON B35T, dsb.) | Aplikasi mencatat data | fitur Bluetooth/datalogging [S: owon.com.hk, ulasan lygte-info] | Harga tidak saya cek; perlu izin aplikasi |
| **Termometer Xiaomi LYWSD03MMC + firmware pvvx/ATC** | Firmware kustom menyiarkan suhu/kelembapan lewat BLE advertising; log dengan HP/laptop | [S: README github.com/pvvx/ATC_MiThermometer] | Flash firmware berisiko; pembulatan di layar bisa berbeda dari nilai BLE, jadi verifikasi pada 50 sampel |
| **Run-length di video** | Untuk alat yang nilainya diam (timbangan dengan beban tetap), label sekali per segmen waktu | [E] | Semua frame segmen masuk satu split |

### 4.2 Pre-label dengan model, lalu koreksi manual

1. Latih CRNN-CTC kecil pada data sintetis (HF #1 + generator), atau pakai Tesseract (`--psm 7`, whitelist `0123456789.-`). Jalankan pada ROI.
2. Simpan prediksi + skor keyakinan ke kolom `label`, dengan `sumber_label=prelabel_dikoreksi`.
3. Di Label Studio, impor sebagai *predictions*. Pelabel cukup menekan Enter kalau prediksi benar dan mengetik ulang kalau salah. Urutkan dari keyakinan terendah.
4. Siklus: setelah ±500 citra benar, latih ulang model dan pre-label sisanya. Ini akan makin cepat [E].
5. **Wajib:** set uji **100% diverifikasi manusia**. Idealnya 2 orang pada 10–20% sampel; laporkan tingkat kesepakatan, misalnya persentase kecocokan string. Ini untuk mencegah bias, karena model yang dipakai untuk pre-label tidak boleh "menentukan" label uji.

Untuk deteksi ROI layar, YOLO yang dilatih pada Roboflow/Kaggle (#2/#3) bisa memberi bbox awal, lalu dikoreksi manual.

---

## 5. Langkah berikutnya

1. Konfirmasi set alat dengan pembimbing dan buat `alat.csv`.
2. Uji coba 1 sesi (±50 foto) untuk mengukur waktu label per citra. Ganti angka [E] di atas dengan hasil nyata.
3. Kalau perlu, kirim email ke kontak Oxford untuk meminta `bp_bg_meters.zip`.
4. Periksa satu shard `tomasko1987` (±339 MB) kalau ingin tambahan data latih.
5. Pilih label ulang desimal di Sacha #4, atau evaluasi bagian bulat saja.
