#!/usr/bin/env python3
"""Degradasi citra terkontrol, deterministik (seeded), level 1-3.

Jenis degradasi (parameter diskalakan relatif terhadap sisi terpendek citra, s = min(H, W)):
  gaussian_blur   sigma = {1: 0.004, 2: 0.008, 3: 0.016} * s
  motion_blur     panjang kernel = {1: 0.02, 2: 0.04, 3: 0.07} * s, sudut acak (seeded) 0-180 derajat
  low_light       I' = brightness * (I/255)^gamma * 255; (gamma, brightness) = (1.5,.6) (2.2,.4) (3.0,.25)
  glare           sorotan spekular Gauss eliptis (putih), radius {.12,.20,.30}*s, intensitas {.5,.8,1.0},
                  pusat acak di dalam area display (crop box) bila ada
  perspective     4 sudut digeser acak hingga {0.05, 0.10, 0.18} * lebar/tinggi (tilt), border replicate
  gaussian_noise  sigma = {8, 18, 32} (skala 0-255), aditif per piksel/kanal
  jpeg            kualitas JPEG = {30, 12, 5}
  occlusion       persegi panjang abu-gelap menutup {6, 12, 20}% luas area display, posisi acak di dalamnya

Seed tiap (id, degradasi, level) = sha256(global_seed|id|degradasi|level) -> hasil sama tiap dijalankan.
Level 0 = 'clean' (hanya resize/crop yang sama) ikut ditulis agar perbandingan adil.

Contoh:
  python degrade.py --datasets kaggle_cramatsu hf_bp_monitor --n-per-dataset 5 --out degraded/demo
  python degrade.py --contact-sheet cramatsu_05_10 --out degraded/demo
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, 'unified', 'manifest.csv')

LEVELS = (1, 2, 3)
PARAMS = {
    'gaussian_blur': {1: dict(sigma_rel=0.004), 2: dict(sigma_rel=0.008), 3: dict(sigma_rel=0.016)},
    'motion_blur': {1: dict(len_rel=0.02), 2: dict(len_rel=0.04), 3: dict(len_rel=0.07)},
    'low_light': {1: dict(gamma=1.5, brightness=0.6), 2: dict(gamma=2.2, brightness=0.4),
                  3: dict(gamma=3.0, brightness=0.25)},
    'glare': {1: dict(radius_rel=0.12, intensity=0.5), 2: dict(radius_rel=0.20, intensity=0.8),
              3: dict(radius_rel=0.30, intensity=1.0)},
    'perspective': {1: dict(shift_rel=0.05), 2: dict(shift_rel=0.10), 3: dict(shift_rel=0.18)},
    'gaussian_noise': {1: dict(sigma=8), 2: dict(sigma=18), 3: dict(sigma=32)},
    'jpeg': {1: dict(quality=30), 2: dict(quality=12), 3: dict(quality=5)},
    'occlusion': {1: dict(area_frac=0.06), 2: dict(area_frac=0.12), 3: dict(area_frac=0.20)},
}
DEGRADATIONS = list(PARAMS)


def seed_for(global_seed, img_id, deg, level):
    h = hashlib.sha256(f'{global_seed}|{img_id}|{deg}|{level}'.encode()).hexdigest()
    return int(h[:16], 16)


def region(box, shape):
    """Area display (x1,y1,x2,y2) di koordinat citra kerja; seluruh citra bila tak ada box."""
    H, W = shape[:2]
    if box is None:
        return 0, 0, W, H
    return box


def apply(img, deg, level, rng, box=None):
    """Kembalikan (citra_terdegradasi, params_aktual dict). img = BGR uint8."""
    p = dict(PARAMS[deg][level])
    H, W = img.shape[:2]
    s = min(H, W)
    f = img.astype(np.float32)
    if deg == 'gaussian_blur':
        sigma = max(0.3, p['sigma_rel'] * s)
        out = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        p['sigma_px'] = round(sigma, 2)
    elif deg == 'motion_blur':
        L = max(3, int(round(p['len_rel'] * s)) | 1)
        ang = float(rng.uniform(0, 180))
        k = np.zeros((L, L), np.float32)
        k[L // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), ang, 1.0)
        k = cv2.warpAffine(k, M, (L, L))
        k /= k.sum()
        out = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)
        p.update(kernel_px=L, angle_deg=round(ang, 1))
    elif deg == 'low_light':
        out = p['brightness'] * np.power(f / 255.0, p['gamma']) * 255.0
    elif deg == 'glare':
        x1, y1, x2, y2 = region(box, img.shape)
        cx, cy = rng.uniform(x1, x2), rng.uniform(y1, y2)
        r = p['radius_rel'] * s
        ax = r * rng.uniform(1.0, 1.8)
        ay = r * rng.uniform(0.5, 1.0)
        th = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        m = np.exp(-0.5 * ((u / ax) ** 2 + (v / ay) ** 2)) * p['intensity']
        m = np.clip(m * 1.6, 0, 1)[..., None]  # inti jenuh -> efek pantulan
        out = f * (1 - m) + 255.0 * m
        p.update(center=[round(cx), round(cy)], axes_px=[round(ax), round(ay)])
    elif deg == 'perspective':
        d = p['shift_rel']
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        jit = rng.uniform(0, d, size=(4, 2)) * np.float32([W, H])
        sign = np.float32([[1, 1], [-1, 1], [-1, -1], [1, -1]])  # geser ke dalam
        dst = src + jit * sign
        M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        p['dst_corners'] = dst.round(1).tolist()
    elif deg == 'gaussian_noise':
        out = f + rng.normal(0, p['sigma'], size=img.shape)
    elif deg == 'jpeg':
        ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, p['quality']])
        out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    elif deg == 'occlusion':
        x1, y1, x2, y2 = region(box, img.shape)
        rw, rh = x2 - x1, y2 - y1
        area = p['area_frac'] * rw * rh
        aspect = rng.uniform(0.3, 1.0)  # lebar/tinggi -> balok vertikal (menutup sebagian digit)
        oh = min(rh, np.sqrt(area / aspect))
        ow = min(rw, area / oh)
        ox = rng.uniform(x1, x2 - ow)
        oy = rng.uniform(y1, y2 - oh)
        out = img.copy()
        color = int(rng.integers(20, 80))
        cv2.rectangle(out, (int(ox), int(oy)), (int(ox + ow), int(oy + oh)), (color,) * 3, -1)
        p['rect_xywh'] = [int(ox), int(oy), int(ow), int(oh)]
    else:
        raise ValueError(deg)
    return np.clip(out, 0, 255).astype(np.uint8), p


def load_working(r, use_crop, pad, max_side):
    """Baca citra -> (citra kerja BGR, box display di koordinat citra kerja atau None)."""
    img = cv2.imread(r['image_path'], cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(r['image_path'])
    box = None
    if r.get('crop_x1') not in (None, ''):
        box = [int(float(r[k])) for k in ('crop_x1', 'crop_y1', 'crop_x2', 'crop_y2')]
    if use_crop and box:
        H, W = img.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        X1, Y1, X2, Y2 = max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)
        img = img[Y1:Y2, X1:X2]
        box = [x1 - X1, y1 - Y1, x2 - X1, y2 - Y1]
    if max_side and max(img.shape[:2]) > max_side:
        sc = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        if box:
            box = [int(round(v * sc)) for v in box]
    return img, box


def read_manifest(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def select(rows, args):
    rows = [r for r in rows if r['unit'] == args.unit]
    if args.datasets:
        rows = [r for r in rows if r['source_dataset'] in args.datasets]
    if args.ids:
        want = set(args.ids)
        rows = [r for r in rows if r['id'] in want]
    if args.exclude_flagged:
        rows = [r for r in rows if not r['label_flag']]
    if args.n_per_dataset:
        rnd = random.Random(args.seed)
        by = {}
        for r in rows:
            by.setdefault(r['source_dataset'], []).append(r)
        rows = []
        for k in sorted(by):
            rs = sorted(by[k], key=lambda r: r['id'])
            rows += rnd.sample(rs, min(args.n_per_dataset, len(rs)))
    return rows


def run(args):
    rows = select(read_manifest(args.manifest), args)
    degs = args.degradations or DEGRADATIONS
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, 'degradation_manifest.csv')
    fields = ['id', 'source_dataset', 'label', 'degradation', 'level', 'params', 'seed', 'path',
              'orig_image_path']
    n = 0
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            img, box = load_working(r, args.use_crop, args.pad, args.max_side)
            d = os.path.join(args.out, 'images', r['id'])
            os.makedirs(d, exist_ok=True)
            jobs = [('clean', 0)] + [(g, l) for g in degs for l in args.levels]
            for g, l in jobs:
                sd = seed_for(args.seed, r['id'], g, l)
                if g == 'clean':
                    out, p = img, {}
                else:
                    out, p = apply(img, g, l, np.random.default_rng(sd), box)
                p.update(work_hw=list(img.shape[:2]), use_crop=bool(args.use_crop and box))
                path = os.path.join(d, f'{g}_L{l}.{args.format}')
                if args.format == 'jpg':  # hemat disk, tapi menambah artefak kompresi ringan (q=95)
                    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(path, out)
                w.writerow(dict(id=r['id'], source_dataset=r['source_dataset'], label=r['label'],
                                degradation=g, level=l, params=json.dumps(p), seed=sd,
                                path=os.path.abspath(path), orig_image_path=r['image_path']))
                n += 1
    print(f'{len(rows)} source images x {1 + len(degs) * len(args.levels)} conditions = {n} files')
    print(f'manifest -> {out_csv}')
    return out_csv


def contact_sheet(args):
    rows = [r for r in read_manifest(args.manifest) if r['id'] == args.contact_sheet]
    if not rows:
        sys.exit(f'id not found: {args.contact_sheet}')
    r = rows[0]
    img, box = load_working(r, args.use_crop, args.pad, args.max_side)
    tile_h = 200
    sc = tile_h / img.shape[0]
    tw = int(img.shape[1] * sc)
    if tw > 360:
        sc = 360 / img.shape[1]
        tw, tile_h = 360, int(img.shape[0] * sc)
    lab_h = 22
    degs = args.degradations or DEGRADATIONS
    rows_img = []
    for g in degs:
        tiles = []
        for l in (0,) + tuple(args.levels):
            if l == 0:
                out = img
            else:
                out, _ = apply(img, g, l, np.random.default_rng(seed_for(args.seed, r['id'], g, l)), box)
            t = cv2.resize(out, (tw, tile_h), interpolation=cv2.INTER_AREA)
            bar = np.full((lab_h, tw, 3), 255, np.uint8)
            cv2.putText(bar, f'{g} L{l}' if l else 'clean', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, t]))
            tiles.append(np.full((tile_h + lab_h, 4, 3), 255, np.uint8))
        rows_img.append(np.hstack(tiles))
        rows_img.append(np.full((4, rows_img[-1].shape[1], 3), 255, np.uint8))
    sheet = np.vstack(rows_img)
    title = np.full((30, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"{r['id']}  label={r['label']}  seed={args.seed}", (6, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 1, cv2.LINE_AA)
    sheet = np.vstack([title, sheet])
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'contact_sheet_{r["id"]}.png')
    cv2.imwrite(path, sheet)
    print(f'contact sheet -> {path} ({sheet.shape[1]}x{sheet.shape[0]})')
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--out', default=os.path.join(HERE, 'degraded', 'run'))
    ap.add_argument('--datasets', nargs='*', help='filter source_dataset')
    ap.add_argument('--ids', nargs='*', help='filter id')
    ap.add_argument('--unit', default='string')
    ap.add_argument('--n-per-dataset', type=int, default=0, help='sampling acak (seeded) per dataset; 0 = semua')
    ap.add_argument('--exclude-flagged', action='store_true', help='buang baris yang label_flag-nya terisi')
    ap.add_argument('--degradations', nargs='*', choices=DEGRADATIONS)
    ap.add_argument('--levels', nargs='*', type=int, default=list(LEVELS), choices=LEVELS)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--use-crop', action='store_true', help='potong ke crop box (+pad) bila tersedia')
    ap.add_argument('--pad', type=float, default=0.15)
    ap.add_argument('--max-side', type=int, default=1024, help='resize sisi terpanjang (0 = asli)')
    ap.add_argument('--format', choices=['png', 'jpg'], default='png',
                    help='png = lossless (default, ~0.4 MB/file pada 1024 px); jpg q95 = hemat disk')
    ap.add_argument('--contact-sheet', metavar='ID', help='buat contact sheet satu citra saja')
    args = ap.parse_args(argv)
    if args.contact_sheet:
        return contact_sheet(args)
    return run(args)


if __name__ == '__main__':
    main()
