# Jalankan aplikasi OCR Seven Segment di Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\windows\run_app.ps1 [-Port 8000]
#
# Menampilkan alamat untuk laptop dan untuk HP di Wi-Fi yang sama.
# Pertama kali dijalankan, Windows Firewall akan bertanya: pilih "Allow" untuk jaringan Private
# supaya HP bisa membuka aplikasinya.

param([int]$Port = 8000)
$ErrorActionPreference = "Stop"
$App = Resolve-Path (Join-Path $PSScriptRoot "..\..\app")
Set-Location $App

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Belum di-setup. Jalankan dulu: powershell -ExecutionPolicy Bypass -File scripts\windows\setup_app.ps1"
}

$env:PYTHONUTF8 = "1"
$ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
       Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } |
       Select-Object -ExpandProperty IPAddress

Write-Host "Laptop : http://localhost:$Port" -ForegroundColor Green
foreach ($ip in $ips) { Write-Host "HP     : http://${ip}:$Port   (HP harus di Wi-Fi yang sama)" -ForegroundColor Green }
Write-Host "Tekan Ctrl+C untuk berhenti."

& .\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port $Port
