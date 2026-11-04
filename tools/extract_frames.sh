#!/usr/bin/env bash
# Ekstrak frame dari video satu SESI ke folder sesi tsb, nama file mengandung alat & sesi.
# Pemakaian: ./extract_frames.sh VIDEO.mp4 ALAT03 S012 [fps=2]
# Hasil: raw/ALAT03/S012/ALAT03_S012_f0001.jpg ...
# fps rendah (1-2) mengurangi frame hampir identik. Tetap: SATU sesi = SATU split.
set -euo pipefail
VID="$1"; ALAT="$2"; SESI="$3"; FPS="${4:-2}"
OUT="raw/${ALAT}/${SESI}"; mkdir -p "$OUT"
ffmpeg -hide_banner -loglevel error -i "$VID" -vf "fps=${FPS}" -q:v 2 "${OUT}/${ALAT}_${SESI}_f%04d.jpg"
echo "$(ls "$OUT" | wc -l) frame -> $OUT"
