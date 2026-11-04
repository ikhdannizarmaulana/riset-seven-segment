#!/usr/bin/env python3
"""Satukan dataset publik seven-segment ke satu manifest CSV.

Output:
  pipeline/unified/manifest.csv   satu baris per citra (citra asli TIDAK disalin)
  pipeline/unified/counts.csv     ringkasan jumlah per dataset/unit/display
  pipeline/unified/counts.txt     ringkasan yang sama dalam bentuk teks

Jalankan:  ../pilot/.venv/bin/python unify.py
"""
import csv
import glob
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(ROOT, 'pipeline', 'unified')

COLUMNS = [
    'id', 'source_dataset', 'unit', 'original_split', 'split_group', 'image_path',
    'label', 'label_source', 'label_flag', 'label_parts', 'label_alt',
    'display_type', 'is_synthetic', 'license',
    'width', 'height', 'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2', 'notes',
]


def img_size(path):
    with Image.open(path) as im:
        return im.size


def read_csv_any(path, delimiter=','):
    """Baca CSV yang mungkin ber-line-ending '\\r' saja (SachaIZADI)."""
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        txt = f.read()
    lines = [ln for ln in re.split(r'\r\n|\r|\n', txt) if ln.strip()]
    return list(csv.DictReader(lines, delimiter=delimiter))


def row(**kw):
    r = {c: '' for c in COLUMNS}
    r.update({k: ('' if v is None else v) for k, v in kw.items()})
    return r


# ---------------------------------------------------------------- cramatsu
def load_cramatsu():
    """Kaggle cramatsu/7-segment-display-yolov8 (LED Quad Display, MIT).

    - Nama file '{AB}_{CD}_jpg.rf.<hash>.jpg' menyandikan angka yang tampil (AB.CD).
      Dicek visual pada 6 sampel (termasuk 4 yang konflik dengan box YOLO): nama file benar.
      Maka label = 'ABCD' (label_source=filename). Titik desimal setelah digit-2 tampak menyala
      di sampel yang dicek, tetapi dataset tidak punya kelas titik -> disimpan di label_alt 'AB.CD'.
    - String turunan box YOLO (urut x) dibandingkan; beda -> label_flag.
    - Augmentasi Roboflow (2 salinan per sumber) di-dedupe: simpan 1 per sumber
      (file pertama menurut urutan nama), salinan lain dicatat di notes.
    - split_group = sumber (semua satu modul LED, jadi grup per sumber adalah yang terbaik yang ada).
    """
    base = os.path.join(DATA, 'kaggle_cramatsu', 'segment_indigator.v4i.yolov8')
    by_src = defaultdict(list)
    for split in ('train', 'valid', 'test'):
        for p in sorted(glob.glob(os.path.join(base, split, 'images', '*.jpg'))):
            src = os.path.basename(p).split('_jpg')[0]
            by_src[src].append((split, p))
    rows, stats = [], Counter()
    for src in sorted(by_src):
        copies = by_src[src]
        splits = {s for s, _ in copies}
        split, p = copies[0]
        lbl_path = os.path.join(base, split, 'labels', os.path.basename(p)[:-4] + '.txt')
        boxes = []
        if os.path.exists(lbl_path):
            for ln in open(lbl_path):
                parts = ln.split()
                if len(parts) == 5:
                    c, x, y, w, h = int(parts[0]), *map(float, parts[1:])
                    boxes.append((x, y, w, h, c))
        boxes.sort()
        yolo_str = ''.join(str(b[4]) for b in boxes)
        m = re.fullmatch(r'(\d\d)_(\d\d)', src)
        W, H = img_size(p)
        flags = []
        if m:
            label = m.group(1) + m.group(2)
            alt = f'{m.group(1)}.{m.group(2)}'
            lsrc = 'filename'
        else:
            label, alt, lsrc = yolo_str, '', 'yolo_boxes_sorted_x'
            flags.append('filename_unparsed')
        if len(boxes) < 4:
            flags.append(f'yolo_incomplete_{len(boxes)}boxes')
        if yolo_str != label:
            flags.append(f'yolo_mismatch:{yolo_str or "EMPTY"}')
        if len(splits) > 1:
            flags.append('source_in_multiple_splits')
        crop = ['', '', '', '']
        if boxes:
            x1 = min((b[0] - b[2] / 2) for b in boxes) * W
            y1 = min((b[1] - b[3] / 2) for b in boxes) * H
            x2 = max((b[0] + b[2] / 2) for b in boxes) * W
            y2 = max((b[1] + b[3] / 2) for b in boxes) * H
            crop = [max(0, round(x1)), max(0, round(y1)), min(W, round(x2)), min(H, round(y2))]
        stats['flagged'] += bool(flags)
        rows.append(row(
            id=f'cramatsu_{src}', source_dataset='kaggle_cramatsu', unit='string',
            original_split=split, split_group=f'cramatsu:{src}', image_path=p,
            label=label, label_source=lsrc, label_flag='|'.join(flags), label_alt=alt,
            display_type='LED', is_synthetic=0, license='MIT', width=W, height=H,
            crop_x1=crop[0], crop_y1=crop[1], crop_x2=crop[2], crop_y2=crop[3],
            notes=f'aug_copies={len(copies)}; crop=union of YOLO digit boxes (may miss digits if flagged); '
                  f'other_copies=' + ';'.join(os.path.basename(c[1]) for c in copies[1:]),
        ))
    print(f'[cramatsu] {sum(len(v) for v in by_src.values())} files -> {len(rows)} unique sources, '
          f'{stats["flagged"]} flagged (YOLO incomplete/mismatch)')
    return rows


# ---------------------------------------------------------------- SachaIZADI
def dhash(path, size=8):
    with Image.open(path) as im:
        im.draft('L', (size * 16, size * 16))  # decode JPEG murah
        im = im.convert('L').resize((size + 1, size))
        px = list(im.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def load_sacha(dup_thresh=4):
    """GitHub SachaIZADI/Seven-Segment-OCR (LCD pompa bensin, tanpa lisensi).

    Label = used_liter (integer saja; foto menampilkan desimal) -> label_source=original_integer.
    Dicek konsisten dengan cadran_1..4 ('X' = kosong).
    Tidak ada info perangkat/sesi. split_group dibuat dari klaster near-duplicate
    (dHash 64-bit, jarak Hamming <= dup_thresh, union-find) agar foto yang nyaris identik
    tidak terpecah ke train dan test.
    """
    base = os.path.join(DATA, 'github_sachaizadi', 'Datasets')
    items = []
    for q in ('HQ', 'MQ', 'LQ'):
        for r in read_csv_any(os.path.join(base, f'{q}_digital.csv'), delimiter=';'):
            p = os.path.join(base, f'{q}_digital', r['image'])
            if not os.path.exists(p):
                print(f'[sacha] MISSING {p}', file=sys.stderr)
                continue
            cad = ''.join(r[f'cadran_{i}'] for i in range(1, 5)).replace('X', '')
            items.append((q, r, p, cad))
    hashes = [dhash(p) for _, _, p, _ in items]
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if bin(hashes[i] ^ hashes[j]).count('1') <= dup_thresh:
                parent[find(i)] = find(j)
    clusters = defaultdict(list)
    for i in range(len(items)):
        clusters[find(i)].append(i)
    cid = {}
    for k, (root, members) in enumerate(sorted(clusters.items(), key=lambda kv: min(kv[1]))):
        for i in members:
            cid[i] = (k, len(members))
    rows, mism = [], 0
    for i, (q, r, p, cad) in enumerate(items):
        W, H = img_size(p)
        label = str(int(r['used_liter']))
        flags = []
        if cad != label:
            flags.append(f'cadran_mismatch:{cad}')
            mism += 1
        k, n = cid[i]
        rows.append(row(
            id=f'sacha_{q}_{r["image"][:12]}', source_dataset='github_sachaizadi', unit='string',
            original_split=q, split_group=f'sacha:dup{k:04d}', image_path=p,
            label=label, label_source='original_integer', label_flag='|'.join(flags),
            display_type='LCD', is_synthetic=0,
            license='NONE (no LICENSE file; all rights reserved - evaluation with citation only)',
            width=W, height=H,
            notes=f'quality={q}; label is integer part only (display shows decimals); '
                  f'dhash_cluster_size={n}',
        ))
    multi = sum(1 for m in clusters.values() if len(m) > 1)
    print(f'[sacha] {len(rows)} images, {len(clusters)} dHash groups '
          f'({multi} groups with >1 image, largest={max(len(m) for m in clusters.values())}), '
          f'cadran mismatches={mism}')
    return rows


# ---------------------------------------------------------------- bp-monitor
def load_bp():
    """HF chakibmed/bp-monitor-seven-segment (LCD tensimeter, MIT).

    readings: layar menampilkan 3 angka bertumpuk (sistolik atas, diastolik tengah, nadi bawah).
      label       = gabungan digit urut atas->bawah, mis. 137/88/85 -> '1378885'
                    (cocok dengan keluaran OCR yang dibaca berurutan lalu disaring ke digit saja)
      label_parts = '137;88;85' untuk penilaian per-field
      split_group = reading_id (1 foto = 1 grup); semua dari satu alat & satu orang.
    digits: crop 24x32 'ink map' per digit -> unit='digit' (untuk CNN per-digit, BUKAN untuk Vision API).
      split_group = reading_id asal, sehingga digit tidak bocor dari foto yang sama.
    """
    base = os.path.join(DATA, 'hf_chakibmed_bp-monitor-seven-segment')
    rows = []
    for split in ('dev', 'test'):
        for r in read_csv_any(os.path.join(base, 'readings', split, 'metadata.csv')):
            p = os.path.join(base, 'readings', split, r['file_name'])
            W, H = img_size(p)
            parts = [r['systolic'], r['diastolic'], r['pulse']]
            rows.append(row(
                id=f'bp_{r["reading_id"]}', source_dataset='hf_bp_monitor', unit='string',
                original_split=split, split_group=f'bp:{r["reading_id"]}', image_path=p,
                label=''.join(parts), label_source='metadata_sys_dia_pulse_concat',
                label_parts=';'.join(parts), label_alt='\n'.join(parts).replace('\n', '/'),
                display_type='LCD', is_synthetic=0, license='MIT', width=W, height=H,
                notes=f'image_kind={r["image_kind"]}; multi-line display (sys/dia/pulse)',
            ))
    nread = len(rows)
    for split in ('dev', 'test'):
        for r in read_csv_any(os.path.join(base, 'digits', split, 'metadata.csv')):
            p = os.path.join(base, 'digits', split, r['file_name'])
            rows.append(row(
                id=f'bpdigit_{split}_{r["file_name"][:-4]}', source_dataset='hf_bp_monitor_digits',
                unit='digit', original_split=split, split_group=f'bp:{r["reading_id"]}',
                image_path=p, label=r['label'], label_source='metadata',
                display_type='LCD', is_synthetic=0, license='MIT', width=24, height=32,
                notes=f'ink map; row={r["row"]}; position={r["position"]}',
            ))
    print(f'[bp] {nread} readings + {len(rows) - nread} digit crops')
    return rows


# ---------------------------------------------------------------- 2005JIAZHI
def load_jiazhi():
    """HF 2005JIAZHI utility meter LCD sample (50 citra). Lisensi tag HF: CC BY-NC 4.0,
    tetapi README juga melarang redistribusi file mentah -> jangan unggah ulang citranya.

    Label = box digit layar (kelas '0'-'9') + 'comma' yang pusatnya di dalam box 'scoreboard',
    urut x; 'comma' ditulis sebagai '.' (format desimal Eropa). Digit serial (s0..s9) diabaikan.
    crop = box 'scoreboard'. split_group = per citra (meter berbeda-beda, tidak ada info alat).
    """
    base = os.path.join(DATA, 'hf_2005JIAZHI_Annotated-Digital-LCD-Utility-Meter-And-Serial-Numbers-Dataset')
    rows = []
    for xp in sorted(glob.glob(os.path.join(base, '*.xml'))):
        t = ET.parse(xp).getroot()
        p = xp[:-4] + '.jpg'
        W, H = img_size(p)
        objs = []
        for o in t.iter('object'):
            bb = o.find('bndbox')
            objs.append((o.find('name').text, *[float(bb.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]))
        sb = [o for o in objs if o[0] == 'scoreboard']
        flags = []
        if len(sb) != 1:
            flags.append(f'scoreboard_count_{len(sb)}')
        if sb:
            _, X1, Y1, X2, Y2 = sb[0]
            inside = [o for o in objs if (o[0].isdigit() or o[0] == 'comma')
                      and X1 <= (o[1] + o[3]) / 2 <= X2 and Y1 <= (o[2] + o[4]) / 2 <= Y2]
        else:
            X1 = Y1 = X2 = Y2 = ''
            inside = [o for o in objs if o[0].isdigit() or o[0] == 'comma']
        inside.sort(key=lambda o: (o[1] + o[3]) / 2)
        label = ''.join('.' if o[0] == 'comma' else o[0] for o in inside)
        if label.count('.') > 1:
            flags.append('multiple_commas')
        dig = [o for o in inside if o[0].isdigit()]
        if dig:
            hs = [o[4] - o[2] for o in dig]
            cy = [(o[2] + o[4]) / 2 for o in dig]
            if max(cy) - min(cy) > 0.6 * (sum(hs) / len(hs)):
                flags.append('digits_not_single_row')
        stray = [o for o in objs if o[0].isdigit() and o not in inside]
        if stray:
            flags.append(f'{len(stray)}_screen_digits_outside_scoreboard')
        rows.append(row(
            id=f'jiazhi_{os.path.basename(p)[-9:-4]}', source_dataset='hf_jiazhi_utility_lcd',
            unit='string', original_split='sample', split_group=f'jiazhi:{os.path.basename(p)[-9:-4]}',
            image_path=p, label=label, label_source='voc_boxes_in_scoreboard_sorted_x',
            label_flag='|'.join(flags), display_type='LCD', is_synthetic=0,
            license='CC BY-NC 4.0 (HF tag); README forbids redistribution of raw files',
            width=W, height=H,
            crop_x1=round(X1) if sb else '', crop_y1=round(Y1) if sb else '',
            crop_x2=round(X2) if sb else '', crop_y2=round(Y2) if sb else '',
            notes='comma rendered as "."; serial-number characters ignored',
        ))
    print(f'[jiazhi] {len(rows)} images, {sum(1 for r in rows if r["label_flag"])} flagged')
    return rows


# ---------------------------------------------------------------- 7SEG_OCR (synthetic)
def load_7seg():
    """HF MiXaiLL76/7SEG_OCR, 100 baris pertama (sintetis, MIT). Label string asli (boleh '.' dan '-')."""
    base = os.path.join(DATA, 'hf_7seg_ocr')
    rows = []
    for r in read_csv_any(os.path.join(base, 'labels.csv')):
        p = os.path.join(base, r['file'])
        rows.append(row(
            id=f'7seg_{os.path.basename(p)[:-4]}', source_dataset='hf_7seg_ocr_synthetic', unit='string',
            original_split='train', split_group='7seg:synthetic', image_path=p, label=r['text'],
            label_source='dataset_text', display_type='LED', is_synthetic=1, license='MIT',
            width=r['w'], height=r['h'], notes='synthetic; dataset card says TRAIN ONLY',
        ))
    print(f'[7seg_ocr] {len(rows)} synthetic images')
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_cramatsu() + load_sacha() + load_bp() + load_jiazhi() + load_7seg()
    ids = Counter(r['id'] for r in rows)
    dup = [k for k, v in ids.items() if v > 1]
    assert not dup, f'duplicate ids: {dup[:5]}'
    for r in rows:
        assert os.path.isabs(r['image_path']) and os.path.exists(r['image_path']), r['image_path']
        # Simpan path RELATIF terhadap root repo dengan pemisah '/', supaya manifest
        # bisa dipakai di Windows/macOS/Linux dan di folder clone mana pun.
        r['image_path'] = Path(os.path.relpath(r['image_path'], ROOT)).as_posix()
    out = os.path.join(OUT_DIR, 'manifest.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    cnt = defaultdict(lambda: Counter())
    for r in rows:
        k = (r['source_dataset'], r['unit'], r['display_type'], r['is_synthetic'], r['license'][:40])
        cnt[k]['n'] += 1
        cnt[k]['flagged'] += bool(r['label_flag'])
    groups = defaultdict(set)
    for r in rows:
        groups[(r['source_dataset'], r['unit'])].add(r['split_group'])
    lines = ['source_dataset,unit,display_type,is_synthetic,n_images,n_split_groups,n_label_flagged,license']
    for k in sorted(cnt):
        lines.append(f'{k[0]},{k[1]},{k[2]},{k[3]},{cnt[k]["n"]},{len(groups[(k[0], k[1])])},'
                     f'{cnt[k]["flagged"]},"{k[4]}"')
    real_str = sum(1 for r in rows if r['unit'] == 'string' and str(r['is_synthetic']) == '0')
    lines.append(f'TOTAL,,,,{len(rows)},,,')
    lines.append(f'REAL_STRING_IMAGES,,,,{real_str},,,')
    with open(os.path.join(OUT_DIR, 'counts.csv'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    txt = '\n'.join(lines)
    with open(os.path.join(OUT_DIR, 'counts.txt'), 'w') as f:
        f.write(txt + '\n')
    print('\n' + txt)
    print(f'\nmanifest -> {out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
