# Jalankan pilot (data sintetis, 3 metode) di Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\windows\run_pilot.ps1
#
# Sama dengan urutan perintah di pilot\README_pilot.md, versi Windows.
# GPU: kalau laptop punya NVIDIA + PyTorch versi CUDA, training otomatis memakai GPU ('cuda');
# kalau tidak, memakai CPU (lebih lambat, tetap jalan).
# Prasyarat: Python 3.12 (py launcher) dan Tesseract (lihat setup_app.ps1),
# serta traineddata eng/ssd/7seg/ssd_plus di pilot\tessdata_extra\ (lihat README).

$ErrorActionPreference = "Stop"
$Pilot = Resolve-Path (Join-Path $PSScriptRoot "..\..\pilot")
Set-Location $Pilot
$env:PYTHONUTF8 = "1"          # baca/tulis file teks sebagai UTF-8 (default Windows bukan UTF-8)
$env:OMP_THREAD_LIMIT = "1"    # Tesseract 1 thread agar latency sebanding antar-metode

$need = @("eng", "ssd", "7seg")
foreach ($n in $need) {
    if (-not (Test-Path "tessdata_extra\$n.traineddata")) {
        throw "tessdata_extra\$n.traineddata belum ada. Unduh sesuai README (bagian 'Menjalankan pilot')."
    }
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.12 -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install numpy pillow opencv-python-headless pytesseract torch
}
$py = ".\.venv\Scripts\python.exe"

& $py generator.py --out data/test  --per-condition 300 --seed 2026
& $py generator.py --out data/train --n 12000 --mixed --seed 1
& $py eval_tesseract.py
& $py train_cnn.py --epochs 6
& $py train_crnn.py --epochs 10
& $py eval_ood_dseg.py --per-condition 50
& $py make_report.py
Write-Host "Selesai. Hasil: pilot\results.csv dan pilot\results.json" -ForegroundColor Green
