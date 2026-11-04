"""Generator citra sintetis seven segment (pilot skripsi).

Segmen digambar sendiri sebagai poligon (7 segmen + titik desimal), sehingga
posisi tiap digit (bounding box) diketahui pasti -> bisa dipakai untuk CNN per-digit.

Contoh:
  python generator.py --out data/test --per-condition 300 --seed 2026
  python generator.py --out data/train --n 12000 --seed 1 --mixed
"""
import argparse, json, os, random, math
from multiprocessing import Pool
import numpy as np
import cv2

SEG = {'0': 'abcdef', '1': 'bc', '2': 'abged', '3': 'abgcd', '4': 'fgbc',
       '5': 'afgcd', '6': 'afgedc', '7': 'abc', '8': 'abcdefg', '9': 'abcdfg'}
CONDITIONS = ['clean', 'noise', 'blur', 'perspective', 'dim', 'glare', 'ghost', 'combined']
DEGRADATIONS = ['noise', 'blur', 'perspective', 'dim', 'glare', 'ghost']
SHIFT = 4  # sub-pixel untuk cv2.fillPoly
SF = 1 << SHIFT


def digit_polys(w, h, t, gap, slant):
    """Poligon 7 segmen pada frame lokal digit (garis tengah segmen di x=0..w, y=0..h)."""
    ht = t / 2.0
    ins = gap  # ujung segmen hampir bertemu di sudut

    def hseg(y, x1, x2):
        return [(x1 + ins, y), (x1 + ins + ht, y - ht), (x2 - ins - ht, y - ht),
                (x2 - ins, y), (x2 - ins - ht, y + ht), (x1 + ins + ht, y + ht)]

    def vseg(x, y1, y2):
        return [(x, y1 + ins), (x + ht, y1 + ins + ht), (x + ht, y2 - ins - ht),
                (x, y2 - ins), (x - ht, y2 - ins - ht), (x - ht, y1 + ins + ht)]

    m = h / 2.0
    P = {'a': hseg(0, 0, w), 'g': hseg(m, 0, w), 'd': hseg(h, 0, w),
         'f': vseg(0, 0, m), 'b': vseg(w, 0, m), 'e': vseg(0, m, h), 'c': vseg(w, m, h)}
    out = {}
    for k, pts in P.items():
        out[k] = np.array([(x + (h - y) * slant, y) for x, y in pts], np.float64)
    return out


def sample_text(rng):
    n = rng.randint(1, 6)
    digits = ''.join(rng.choice('0123456789') for _ in range(n))
    dp = rng.randint(0, n - 2) if (n >= 2 and rng.random() < 0.5) else -1
    label = ''.join(d + ('.' if i == dp else '') for i, d in enumerate(digits))
    return digits, dp, label


def render(rng, digits, dp, display, ghost_alpha):
    h = rng.uniform(36, 80)               # tinggi digit (px)
    w = h * rng.uniform(0.45, 0.6)
    t = h * rng.uniform(0.10, 0.16)       # tebal segmen
    gap = t * rng.uniform(0.05, 0.25)
    slant = rng.choice([0.0, rng.uniform(0.05, 0.15)])
    spacing = t * rng.uniform(1.6, 2.6)
    pitch = w + t + spacing
    mx = h * rng.uniform(0.25, 0.7)
    my = h * rng.uniform(0.2, 0.5)
    n = len(digits)
    W = int(math.ceil(2 * mx + n * pitch + h * slant))
    H = int(math.ceil(2 * my + h + t))
    on = np.zeros((H, W), np.float32)
    off = np.zeros((H, W), np.float32)
    boxes = []
    for i, d in enumerate(digits):
        ox = mx + i * pitch + t / 2
        oy = my + t / 2
        polys = digit_polys(w, h, t, gap, slant)
        for s, p in polys.items():
            pts = np.round((p + [ox, oy]) * SF).astype(np.int32)
            target = on if s in SEG[d] else off
            cv2.fillPoly(target, [pts], 1.0, lineType=cv2.LINE_AA, shift=SHIFT)
        # titik desimal (selalu digambar: on bila dp, else ghost)
        cx, cy = ox + w + t / 2 + spacing * 0.45, oy + h
        tgt = on if i == dp else off
        cv2.circle(tgt, (int(cx * SF), int(cy * SF)), int(t * 0.6 * SF), 1.0,
                   -1, lineType=cv2.LINE_AA, shift=SHIFT)
        # kotak sel digit (termasuk area DP), 4 sudut setelah slant
        x0, x1 = ox - t * 0.6, ox + w + t / 2 + spacing * 0.45 + t * 0.7
        y0, y1 = oy - t * 0.6, oy + h + t * 0.6
        quad = [(x0 + h * slant + t * 0.6 * slant, y0), (x1 + h * slant, y0), (x1, y1), (x0, y1)]
        boxes.append(quad)
    on = np.clip(on, 0, 1)
    off = np.clip(off, 0, 1) * (1 - on)

    if display == 'lcd':
        bg = np.array([rng.uniform(150, 195), rng.uniform(165, 210), rng.uniform(130, 175)], np.float32)
        fg = np.array([rng.uniform(15, 60)] * 3, np.float32) + [0, rng.uniform(0, 10), 0]
        grad = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.95, 1.1), H, dtype=np.float32)[:, None, None]
        img = np.ones((H, W, 3), np.float32) * bg * grad
        m = cv2.GaussianBlur(on, (0, 0), 0.6) + off * ghost_alpha
        img = img * (1 - m[..., None]) + fg * m[..., None]
        color = 'dark'
    else:
        color = rng.choice(['red', 'green'])
        c = np.array([rng.uniform(230, 255), rng.uniform(20, 70), rng.uniform(20, 60)] if color == 'red'
                     else [rng.uniform(40, 90), rng.uniform(220, 255), rng.uniform(60, 110)], np.float32)
        bg = np.array([rng.uniform(5, 35)] * 3, np.float32)
        img = np.ones((H, W, 3), np.float32) * bg
        glow = cv2.GaussianBlur(on, (0, 0), t * rng.uniform(0.6, 1.4)) * rng.uniform(0.3, 0.8)
        core = np.clip(on + glow, 0, 1.3)
        img = img + c * core[..., None] + c * (off * ghost_alpha)[..., None]
        # inti segmen sedikit memutih (overexposure LED)
        img = img + 255 * 0.25 * (on ** 2)[..., None] * rng.uniform(0, 1)
    return np.clip(img, 0, 255), boxes, dict(digit_h=round(h, 1), slant=round(slant, 3))


def apply_degr(rng, img, boxes, degs, params):
    H, W = img.shape[:2]
    if 'perspective' in degs:
        ang = rng.uniform(-12, 12)
        j = rng.uniform(0.04, 0.10)
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = src + np.float32([[rng.uniform(-j, j) * W, rng.uniform(-j, j) * H] for _ in range(4)])
        Mr = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        dst = cv2.transform(dst[None], Mr)[0]
        mnx, mny = dst.min(0)
        dst -= [mnx, mny]
        nW, nH = int(np.ceil(dst[:, 0].max())), int(np.ceil(dst[:, 1].max()))
        M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        img = cv2.warpPerspective(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        boxes = [cv2.perspectiveTransform(np.float32(b)[None], M)[0].tolist() for b in boxes]
        params['rot_deg'] = round(ang, 2); params['persp_jitter'] = round(j, 3)
        H, W = img.shape[:2]
    if 'glare' in degs:
        g = np.zeros((H, W), np.float32)
        cx, cy = rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H
        ax, ay = rng.uniform(0.15, 0.45) * W, rng.uniform(0.2, 0.6) * H
        cv2.ellipse(g, (int(cx), int(cy)), (int(ax), int(ay)), rng.uniform(0, 180), 0, 360, 1.0, -1)
        g = cv2.GaussianBlur(g, (0, 0), max(2, 0.15 * min(ax, ay)))
        a = rng.uniform(0.35, 0.75)
        img = img * (1 - a * g[..., None]) + 255 * a * g[..., None]
        params['glare_alpha'] = round(a, 2)
    if 'dim' in degs:
        gamma, sc = rng.uniform(1.6, 2.6), rng.uniform(0.3, 0.6)
        img = 255 * (img / 255) ** gamma * sc
        params['gamma'] = round(gamma, 2); params['bright_scale'] = round(sc, 2)
    if 'blur' in degs:
        if rng.random() < 0.5:
            s = rng.uniform(1.2, 2.5) * params['digit_h'] / 50
            img = cv2.GaussianBlur(img, (0, 0), s); params['blur'] = f'gauss{s:.2f}'
        else:
            k = int(rng.uniform(5, 11) * params['digit_h'] / 50) | 1
            ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1
            ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), rng.uniform(0, 180), 1), (k, k))
            img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6)); params['blur'] = f'motion{k}'
    if 'noise' in degs:
        s = rng.uniform(12, 30)
        img = img + np.random.default_rng(rng.randint(0, 2**31)).normal(0, s, img.shape)
        params['noise_sigma'] = round(s, 1)
        img = np.clip(img, 0, 255).astype(np.uint8)
        q = rng.randint(20, 60)
        _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED).astype(np.float32)
        params['jpeg_q'] = q
    return np.clip(img, 0, 255).astype(np.uint8), boxes


def make_one(args):
    idx, seed, condition, out_dir = args
    rng = random.Random(seed)
    digits, dp, label = sample_text(rng)
    display = rng.choice(['lcd', 'led'])
    if condition == 'combined':
        degs = rng.sample(DEGRADATIONS, rng.randint(2, 3))
    elif condition == 'clean':
        degs = []
    else:
        degs = [condition]
    ghost = rng.uniform(0.15, 0.35) if 'ghost' in degs else 0.0
    img, boxes, params = render(rng, digits, dp, display, ghost)
    params['ghost_alpha'] = round(ghost, 2)
    img, boxes = apply_degr(rng, img, boxes, degs, params)
    H, W = img.shape[:2]
    bboxes = []
    for q in boxes:
        q = np.array(q)
        x0, y0 = q.min(0); x1, y1 = q.max(0)
        bboxes.append([int(max(0, x0)), int(max(0, y0)), int(min(W, math.ceil(x1))), int(min(H, math.ceil(y1)))])
    fn = f'{condition}_{idx:05d}.png'
    cv2.imwrite(os.path.join(out_dir, 'images', fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return dict(file=fn, label=label, digits=digits, dp_index=dp, condition=condition,
                degradations=degs, display=display, color=('dark' if display == 'lcd' else None),
                params=params, width=W, height=H, digit_bboxes=bboxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-condition', type=int, default=0, help='jumlah per kondisi (set uji)')
    ap.add_argument('--n', type=int, default=0, help='jumlah total (mode --mixed)')
    ap.add_argument('--mixed', action='store_true', help='kondisi diacak seragam (set latih)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'images'), exist_ok=True)
    jobs = []
    if a.mixed:
        r = random.Random(a.seed)
        for i in range(a.n):
            jobs.append((i, a.seed * 10_000_000 + i, r.choice(CONDITIONS), a.out))
    else:
        for ci, c in enumerate(CONDITIONS):
            for i in range(a.per_condition):
                jobs.append((i, a.seed * 10_000_000 + ci * 100_000 + i, c, a.out))
    with Pool() as p:
        recs = p.map(make_one, jobs, chunksize=64)
    with open(os.path.join(a.out, 'labels.jsonl'), 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    print(f'{len(recs)} citra -> {a.out}')


if __name__ == '__main__':
    main()
