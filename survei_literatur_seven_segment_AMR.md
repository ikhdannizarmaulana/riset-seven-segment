# Survei Literatur: Pengenalan Digit Seven-Segment / AMR / LCD-LED (2018–2026)
Disusun: 4 Nov 2026. Legenda verifikasi: **[V]** = angka/klaim dibaca langsung dari halaman/PDF; **[S]** = hanya dari snippet mesin pencari / belum dibuka penuh; **[?]** = tidak ditemukan/tidak tercantum.

## 1. Tabel paper kunci

| # | Judul (venue, tahun) | Metode | Dataset | Hasil dilaporkan | URL |
|---|---|---|---|---|---|
| 1 | Convolutional Neural Networks for Automatic Meter Reading (J. Electronic Imaging, 2019) – Laroca dkk. | Fast-YOLO (deteksi counter) + bandingkan CR-NET (YOLO-based per-digit), CRNN, Multi-Task; augmentasi penyeimbang digit | UFPR-AMR (2.000 img, 5 digit) | CR-NET+DA: 98,30% digit, **94,13% counter**; CRNN+DA 92,30%; Multi-Task tanpa DA 0%. Meter-Integration: 97,30% **[V]** | https://arxiv.org/abs/1902.09600 |
| 2 | Towards Image-based AMR in Unconstrained Scenarios (IEEE Access, 2021) – Laroca dkk. | Fast-YOLOv4-SmallObj → CDCC-NET (deteksi sudut + klasifikasi meter rusak/illegible, rektifikasi) → Fast-OCR | UFPR-AMR + Copel-AMR (12.500 img, 2.500 ilegibel) | End-to-end 94,75% (UFPR-AMR), 96,98% (Copel-AMR); baseline TRBA 92,34% rata2; ≥99% bila menolak bacaan confidence rendah (mis. 99,33% rata2 di threshold tertinggi) **[V]** | https://arxiv.org/abs/2009.10181 |
| 3 | Fully Convolutional Sequence Recognition Network for Water Meter Number Reading (IEEE Access, 2019) – Yang dkk. | FCSRN + *Augmented Loss* untuk digit "mid-state" (setengah berputar) | SCUT-WMN | Angka akurasi **[?]** (IEEE diblokir); fokus mid-state **[V via repo dataset]** | https://ieeexplore.ieee.org/document/8606091/ |
| 4 | Automated method for detecting and reading seven-segment digits from images of blood glucose metres and BP monitors (J. Med. Eng. & Tech., 2019) – Finnegan dkk., Oxford | Retinex + MSER + HOG + NN, dilatih pada **digit sintetis** | Sintetis + foto glukometer/tensimeter (publik) | 93% akurasi digit; F1 lokasi 0,87 (glukometer) / 0,80 (tensimeter) **[V]** | https://eng.ox.ac.uk/lcmt/resources/resources-seven-segment |
| 5 | CNN-Based LCD Transcription of Blood Pressure From a Mobile Phone Camera (Frontiers, ~2021 [S]) | Preprocessing OpenCV + CNN; baseline Tesseract & Google Vision | 7.205 foto tensimeter dari bidan (label kualitas: blur, dark, reflection, far, cropped) | CNN: 90,7% (SBP)/91,1% (DBP) gambar baik; ~65–66% gambar buruk. **Tesseract (tanpa training): 20,2%/14,3%**; Google Vision ~42–43% **[V]** | https://pmc.ncbi.nlm.nih.gov/articles/PMC8177819/ |
| 6 | Image-Based Automatic Watermeter Reading under Challenging Environments (Sensors 21(2):434, 2021) | YOLOv3 + orientation alignment + spatial-layout guidance (menangani rolling digit) | 10.000 img (9.500 train/500 test, milik sendiri) | Digit error 24,32% (basic) → **3,79%** (OA+SG) **[V]** | https://www.mdpi.com/1424-8220/21/2/434 |
| 7 | Automated Medical Device Display Reading Using Deep Learning Object Detection (arXiv 2210.01325, 2022) – Moreira | EfficientDet / EfficientDet-Lite (deteksi digit sbg objek) | Foto alat medis; test 104 img/438 digit | EfficientDet-Lite1: 100% deteksi & klasifikasi digit (test kecil) **[V]** | https://arxiv.org/abs/2210.01325 |
| 8 | PACMAN: pulse oximeter digit detection and reading in a low-resource setting (arXiv 2212.04964, 2022) | YOLOv5 + auto-orientation + clustering | Foto oksimeter nyata | 81,0–89,5% **[V]** | https://arxiv.org/abs/2212.04964 |
| 9 | Research on water meter reading recognition based on deep learning (Sci. Reports, 2022) | YOLOv3 (crop kotak hitam) + deteksi digit | 4.000 img | Akurasi 90,61% (Plan B) **[V]** | https://www.nature.com/articles/s41598-022-17255-3 |
| 10 | Water Meter Reading for Smart Grid Monitoring (Sensors 23(1):75, 2022/2023) | YOLOv5 lokalisasi meter+digit | challengedata.ens.fr #30 (1.000 img) | mAP dilaporkan; angka tidak dikutip di sini | https://pmc.ncbi.nlm.nih.gov/articles/PMC9823547/ |
| 11 | Benchmarking YOLO Models for Automatic Reading in Smart Metering (ICISSP, 2023) – Hattak dkk. | Perbandingan beberapa YOLO | Water meter | mAP@0.5 AUC 0,862; AUC kelas "reading digit" hanya 0,327 **[V]** | https://iris.uniroma1.it/retrieve/4f099bd2-7ed5-421d-8222-e0eb407cf6f9/Hattak_pre-print_Benchmarking-YOLO_2023.pdf.pdf |
| 12 | Lightweight single pass numerical reading extraction for displays in the wild (Electronic Imaging, 2023) | Model ringan single-pass untuk 7-seg di perangkat portabel | [?] | Angka **[?]** (hanya abstrak) | https://library.imaging.org/ei/articles/35/7/IMAGE-282 |
| 13 | Detecting and recognizing seven segment digits using a deep learning approach (ITM Web Conf 63, 2024; UNITEN) | Bandingkan model STR **pretrained tanpa fine-tune**: DBNet, PARSeq, SVTR, ABINet, NRTR | 7Seg (Paddle AI Studio, 165 img) + custom | PARSeq terbaik 56,97% di 7Seg; per kondisi (tanpa/dgn karakter khusus): ideal 93,18/79,55, blur 91,43/71,43, rotated 86,71/69,04, **glare 81,82/68,18**; titik desimal "menyatu" karena LED bleeding **[V]** | https://www.itm-conferences.org/articles/itmconf/abs/2024/06/itmconf_amict2023_01007/itmconf_amict2023_01007.html |
| 14 | Development of seven segment display recognition using TensorFlow on Raspberry Pi (IEEE conf., ~2022 [S]) – IIUM | Tiny YOLO vs SSD-MobileNet di Raspberry Pi 4 | [?] | Kualitatif "akurat dan cepat" **[V abstrak]** | https://irep.iium.edu.my/102470/ |
| 15 | Automatic Generation of Seven-Segment Display Image for ML-Based Digital Meter Reading (PHM Society Asia-Pacific, tahun [?]) – Mitsubishi Electric | Generator citra 7-seg sintetis otomatis, tanpa anotasi manual | Sintetis → uji nyata | Akurasi baca 96,8% **[V]** | https://papers.phmsociety.org/index.php/phmap/article/download/3612/2095 |
| 16 | Towards Accurate Readings of Water Meters by Eliminating Transition Error: New Dataset and Effective Solution (IEEE TIM, 2025) – Zhang dkk. | Fokus "transition error" (roda digit berputar) + pointer | WMeter5K (5.000 img, kamera add-on) | Angka **[?]** (hanya repo dibuka) | https://github.com/ZZZHANG-jx/WMeter-Reader |
| 17 | Water meter reading recognition method based on character attention mechanism (PLOS ONE, 2025) | ResNet-FPN deteksi + koreksi kemiringan + CNN char-attention | DataFountain (1.000 train/500 test) | Deteksi 99,8%, karakter 97,6%, **baris teks 88,6%** **[V]** | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0332119 |
| 18 | Semi-Supervised Seven-Segment LED Display Recognition with an Integrated Data-Acquisition Framework (Sensors 26(1):265, 2026) | Self-training semi-supervised + SE-Attention + adversarial training AdvGAN++; K-means sampling | Lokal (500–3000 lx) + publik | 89,3% lokal, 98,1% publik (pseudo-label baseline 76,2/87,4) **[V]** | https://pmc.ncbi.nlm.nih.gov/articles/PMC12788288/ |
| 19 | Enhancing Water Meter Digit Recognition Using YOLO-OCR (ETASR 16(3), 2026) – **Telkom Univ. Purwokerto** | YOLOv8m + PP-OCRv4, CLAHE | 3.250 img (2.250 ideal/1.000 non-ideal) | RA 90,90% (train non-ideal) vs 81,81% (full); CER 0,0151 **[V]** — *catatan: sudah ada riset Indonesia serupa* | https://etasr.com/index.php/ETASR/article/download/17291/6576/89458 |
| 20 | A Comprehensive Dataset for Word-Wheel Water Meter Reading Under Challenging Conditions (Scientific Data 13:479, 2026) | Dataset + baseline UNet/DeepLabV3+/PSPNet/SegFormer, VGG/ResNet/DenseNet | >50.000 img Hangzhou | Baseline ada (angka tidak diekstrak) | https://www.nature.com/articles/s41597-026-06809-z |
| 21 | Do Vision-Language Models Measure Up? MeasureBench (arXiv 2510.26865, 2025/2026) | Benchmark VLM membaca alat ukur | Real + sintetis | VLM frontier "struggle"; RFT di data sintetis membantu **[V]** | https://arxiv.org/abs/2510.26865 |
| 22 | Generative Data Augmentation for AMR Using CNNs (IEEE Access, 2022) – Sripanuskul dkk. | Y-Autoencoder untuk generate citra digit | [S] | **[S]** | https://ieeexplore.ieee.org/document/9729827/ |

### Alat/baseline non-paper
| Alat | Fakta terverifikasi | URL |
|---|---|---|
| Tesseract `tessdata_ssd` (7seg/ssd traineddata) | Apache-2.0; fine-tune dari eng best; dilatih 3 font 7-seg **sintetis**; issue: traineddata lama lebih baik di foto nyata **[V]** | https://github.com/Shreeshrii/tessdata_ssd |
| Tesseract `letsgodigital` | dalam repo display_ocr, GPL-2.0 **[V]** | https://github.com/arturaugusto/display_ocr |
| ssocr | GPL-3, v2.26.0; deterministik, thresholding+segmentasi, tanpa ML **[V]** | https://www.unix-ag.uni-kl.de/~auerswal/ssocr/ |
| AI-on-the-edge-device (ESP32-CAM) | TFLite di ESP32, ROI per digit, ~8,9k stars **[V]** | https://github.com/jomjol/AI-on-the-edge-device |
| Benchmark hobi (anshin) | Pada 120 sampel sintetis: SVTR ringan 79,17% exact; PARSeq 19,17%; TrOCR 10,83% (tanpa adaptasi 7-seg) **[V, non-peer-review]** | https://github.com/Tinnci/anshin/blob/main/seven_segment_ocr/MODEL_EVALUATION.md |
| OICWS lcd-digit-recognition | YOLOv8+CRNN, klaim 98% real-world, AGPL-3.0 **[V, klaim sendiri]** | https://github.com/OICWS/lcd-digit-recognition |
| Daftar kurasi | Awesome Image-based Meter Recognition | https://github.com/ZZZHANG-jx/Awesome-Image-based-Meter-Recognition-Reading |

## 2. Dataset publik

| Dataset | Isi | Akses / Lisensi | URL |
|---|---|---|---|
| UFPR-AMR | 2.000 img meter listrik Copel (800/400/800), 5 digit, anotasi posisi digit | Academic, **non-komersial**, via request **[V]** | https://github.com/raysonlaroca/ufpr-amr-dataset |
| Copel-AMR | 12.500 img lapangan (2.500 ilegibel/rusak) | Registrasi form + terms of use **[V]** | https://web.inf.ufpr.br/vri/databases/copel-amr/ |
| UFPR-ADMR-v1 (dial, bukan 7-seg) | 2.000 img dial meter | upon request **[S]** | https://web.inf.ufpr.br/vri/databases/ufpr-admr/ |
| SCUT-WMN | 5.000 noisy + 1.000 clean water meter, ada digit "mid-state" | Lisensi tidak tercantum **[?]** | https://github.com/HCIILAB/Water-Meter-Number-DataSet |
| Word-wheel water meter (Sci Data 2026) | Deteksi: 32.714 train/13.398 test (512²); Rekognisi: 5-digit 11.867/2.000, 6-digit 2.555/400; label half-word, reversed, multi-hot kondisi (blur, reflektif, kotor) | Dryad doi:10.5061/dryad.7d7wm3860 (lisensi Dryad belum dicek; artikel CC BY-NC-ND) | https://doi.org/10.5061/dryad.7d7wm3860 |
| WMeter5K | 5.000 img water meter, anotasi per roda digit + pointer | OneDrive; lisensi **[?]** | https://github.com/ZZZHANG-jx/WMeter-Reader |
| Oxford seven-segment (Finnegan 2019) | Digit sintetis + foto glukometer/tensimeter | "made public"; lisensi **[?]** | https://eng.ox.ac.uk/lcmt/resources/resources-seven-segment |
| Sensors 2026 LED dataset | Data LED industri | Zenodo doi:10.5281/zenodo.17855380 (belum dibuka) | https://doi.org/10.5281/zenodo.17855380 |
| MiXaiLL76/7SEG_OCR (HF) | Sintetis 7-seg, label string termasuk "-" dan "." , TRAIN ONLY | lisensi tidak terbaca **[?]** | https://huggingface.co/datasets/MiXaiLL76/7SEG_OCR |
| Kaggle 7-Segment Display: YOLOV8 (cramatsu) | Per-digit kelas 0–9 format YOLO | MIT **[V]** | https://www.kaggle.com/datasets/cramatsu/7-segment-display-yolov8 |
| Kaggle seven segment display YOLOV5 (loclaurote) | Display harga | **[S]** | https://www.kaggle.com/datasets/loclaurote/seven-segment-display-dataset-yolov5 |
| Roboflow "Seven segments display digits" (MBA) | 473 img, deteksi | CC BY 4.0 **[V]** | https://universe.roboflow.com/mba/seven-segments-display-digits |
| Roboflow labmonitor / UK / Digital Meter | 698 img / 265 img / ? | **[S]** | https://universe.roboflow.com/labmonitor/7-segment-digit-number-display ; https://universe.roboflow.com/uk-jklsw/7-segment-display-8qb3b |
| CoccaGuo Seven-Segment-Display-Dataset | 7-seg terklasifikasi (release Des 2021) | tanpa lisensi tercantum | https://github.com/CoccaGuo/Seven-Segment-Display-Dataset |
| Water Meters (Toloka/UniDataPro, Kaggle) | 1.244 foto + mask + OCR | CC BY-NC-ND 4.0 **[S]** | https://www.kaggle.com/datasets/unidpro/water-meters |
| DataFountain water meter | 1.000 train/500 test (dipakai PLOS 2025) | **[?]** | https://www.datafountain.cn/datasets/5857 |
| ENS Challenge Data #30 | 1.000 img water meter | "freely available for research" **[V via PMC9823547]** | https://challengedata.ens.fr/challenges/30 |

## 3. Masalah terbuka (berbasis bukti)
- **Glare/pantulan**: kondisi terburuk PARSeq (68,18%) [ITM 2024]; label "reflection" menurunkan CNN BP ke ~65% [PMC8177819]; kategori "reflective or dark" di Sci Data 2026.
- **Titik desimal & tanda**: DP menyatu dengan digit karena LED bleeding; membuang karakter khusus menaikkan akurasi 79,55→93,18% (artinya DP/minus justru sumber error utama) [ITM 2024].
- **Rolling/half digit**: SCUT-WMN mid-state, Sci Data "half-word", TIM 2025 "transition error", Sensors 2021 spatial guidance.
- **Model STR umum gagal tanpa adaptasi**: Tesseract 14–20%, Google Vision ~42% (BP); PARSeq pretrained 56,97% (7Seg).
- **Kualitas buruk**: drop ~25 poin dari gambar baik ke buruk [PMC8177819]; Laroca: rektifikasi + reject option menaikkan ke ≥99%.
- **Segmen mati / LCD ghosting / LCD vs LED cross-domain / low light**: **tidak ditemukan** studi kuantitatif khusus dalam penelusuran ini → celah.
- **Edge**: ada sistem (ESP32-CAM, RPi SSD-MobileNet) tetapi jarang evaluasi akurasi–latensi–energi yang sistematis dan terpublikasi.

## 4. Celah riset untuk S1 (4–6 bulan, Colab)
1. **Klasifikasi tingkat-segmen (7 bit + DP) vs 10-kelas digit** — model multi-label memprediksi status tiap segmen lalu decode via tabel; uji robust terhadap **segmen mati** (augmentasi segment-dropout sintetis) dan bisa mendeteksi pola invalid → reject. Belum ditemukan paper yang mengevaluasi segmen rusak.
2. **Penanganan titik desimal & tanda minus secara eksplisit** — kelas/head DP terpisah atau CTC dengan vocabulary {0-9, ., -} + metrik terpisah (DP-accuracy). Bukti masalah: ITM 2024.
3. **Benchmark terkontrol per-kondisi pada dataset lokal Indonesia** (mis. kWh meter prabayar PLN [LCD], timbangan, tensimeter) dengan label kondisi (glare/blur/miring/low-light/LCD-vs-LED): ssocr vs Tesseract-ssd/letsgodigital vs CNN per-digit vs CRNN-CTC vs PARSeq fine-tuned vs YOLO-digit. Kontribusi = dataset terlabel kondisi + analisis kegagalan, bukan sekadar "Tesseract vs CNN". Catatan: water meter Indonesia sudah diteliti (ETASR 2026, Telkom Purwokerto) → pilih domain LCD/kWh atau alat medis.
4. **Sintetis-ke-nyata (sim2real) dengan anggaran label minim** — generator 7-seg (font DSEG + glare, ghost segments LCD, blur, perspektif) lalu kurva "berapa foto nyata yang dibutuhkan" (0/50/100/500). Dasar: Finnegan 2019, Mitsubishi 96,8%, tessdata_ssd yang gagal di foto nyata.
5. **Reject option / confidence calibration untuk billing** — reproduksi ide Laroca (≥99% saat menolak bacaan ragu) pada model ringan; metrik coverage-vs-accuracy. Murah komputasi, bernilai praktis.
6. **Model ringan terkuantisasi INT8 di ESP32-S3/Raspberry Pi** — bandingkan CNN per-digit (gaya AI-on-the-edge) vs tiny-CRNN vs YOLOv8n-digit: akurasi, latensi, RAM, (opsional) energi.
7. **Rolling digit water meter dengan label half-digit + konstrain temporal monoton** (bacaan tidak boleh turun antar-frame) memakai dataset Sci Data 2026/SCUT-WMN — publik, besar, CPU/Colab cukup untuk CRNN kecil.
8. **VLM open-source (Qwen-VL/PaliGemma kecil, LoRA) vs model spesialis kecil** untuk 7-seg — MeasureBench menunjukkan VLM lemah membaca alat ukur; S1 dapat mengukur zero-shot vs LoRA vs CRNN dengan biaya Colab. (Risiko: VRAM; pilih model ≤3B.)
