# Setup aplikasi OCR Seven Segment di Windows 10/11 (PowerShell).
# Jalankan dari root repo:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup_app.ps1
#
# Yang dilakukan:
#   1. Cek Python 3.12 (py launcher) — kalau belum ada, pasang lewat winget.
#   2. Cek Tesseract OCR — kalau belum ada, pasang lewat winget (UB-Mannheim.TesseractOCR).
#   3. Buat virtual environment app\.venv dan pasang paket dari app\requirements.txt.
#   4. Jalankan uji otomatis (pytest) sebagai bukti instalasi berhasil.

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$App  = Join-Path $Root "app"

function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Write-Host "== 1/4 Cek Python 3.12" -ForegroundColor Cyan
$pyOk = $false
if (Have "py") {
    try { & py -3.12 --version | Out-Null; $pyOk = ($LASTEXITCODE -eq 0) } catch { $pyOk = $false }
}
if (-not $pyOk) {
    if (-not (Have "winget")) { throw "Python 3.12 belum ada dan winget tidak tersedia. Pasang manual dari https://www.python.org/downloads/ (centang 'Add python.exe to PATH'), lalu jalankan ulang skrip ini." }
    Write-Host "Memasang Python 3.12 lewat winget..."
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    Write-Host "Python terpasang. Tutup PowerShell ini, buka lagi, lalu jalankan ulang skrip." -ForegroundColor Yellow
    exit 0
}
& py -3.12 --version

Write-Host "== 2/4 Cek Tesseract OCR" -ForegroundColor Cyan
$tess = $env:TESSERACT_CMD
if (-not $tess) {
    $cands = @("$env:ProgramFiles\Tesseract-OCR\tesseract.exe", "${env:ProgramFiles(x86)}\Tesseract-OCR\tesseract.exe", "$env:LOCALAPPDATA\Programs\Tesseract-OCR\tesseract.exe")
    if (Have "tesseract") { $tess = (Get-Command tesseract).Source }
    else { $tess = $cands | Where-Object { Test-Path $_ } | Select-Object -First 1 }
}
if (-not $tess) {
    if (-not (Have "winget")) { throw "Tesseract belum ada. Unduh installer dari https://github.com/UB-Mannheim/tesseract/wiki lalu jalankan ulang skrip ini." }
    Write-Host "Memasang Tesseract lewat winget..."
    winget install -e --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements
    $tess = "$env:ProgramFiles\Tesseract-OCR\tesseract.exe"
}
& $tess --version | Select-Object -First 1
Write-Host "Tesseract: $tess  (app mencari lokasi ini otomatis; bisa juga set env TESSERACT_CMD)"

Write-Host "== 3/4 Virtual environment + paket" -ForegroundColor Cyan
Set-Location $App
if (-not (Test-Path ".venv\Scripts\python.exe")) { & py -3.12 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "== 4/4 Uji otomatis" -ForegroundColor Cyan
$env:PYTHONUTF8 = "1"
& .\.venv\Scripts\python.exe -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw "Uji otomatis gagal. Lihat pesan di atas." }

Write-Host ""
Write-Host "Selesai. Jalankan aplikasi dengan:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\windows\run_app.ps1"
