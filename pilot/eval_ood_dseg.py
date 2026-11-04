"""Uji generalisasi (out-of-distribution): font DSEG7 (keshikan/DSEG, OFL) yang TIDAK
dipakai saat pelatihan CNN/CRNN (latih = poligon buatan generator.py).

  python eval_ood_dseg.py --per-condition 50
Menghasilkan data/test_dseg/ dan results/ood_dseg.json
"""
import json, os, random, argparse, time
from multiprocessing import Pool
import numpy as np, cv2, torch
from PIL import Image, ImageDraw, ImageFont
from generator import CONDITIONS, DEGRADATIONS, sample_text, apply_degr
from metrics import score, aggregate

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = [os.path.join(HERE, 'fonts/fonts-DSEG_v046', p) for p in
         ['DSEG7-Classic/DSEG7Classic-Regular.ttf', 'DSEG7-Classic/DSEG7Classic-Bold.ttf',
          'DSEG7-Modern/DSEG7Modern-Regular.ttf', 'DSEG7-Classic/DSEG7Classic-Italic.ttf']]
OUT = os.path.join(HERE, 'data/test_dseg')


def render_dseg(rng, digits, dp, display, ghost):
    size = int(rng.uniform(40, 80))
    font = ImageFont.truetype(rng.choice([f for f in FONTS if os.path.exists(f)]), size)
    text = ''.join(d + ('.' if i == dp else '') for i, d in enumerate(digits))
    gtext = '8.' * len(digits)   # DSEG: '.' sempit, '8.' = semua segmen (ghost)
    tmp = ImageDraw.Draw(Image.new('L', (10, 10)))
    gb = tmp.textbbox((0, 0), gtext, font=font)
    mx, my = int(size * rng.uniform(0.3, 0.7)), int(size * rng.uniform(0.2, 0.5))
    W, H = gb[2] - gb[0] + 2 * mx, gb[3] - gb[1] + 2 * my
    org = (mx - gb[0], my - gb[1])
    on = Image.new('L', (W, H), 0); off = Image.new('L', (W, H), 0)
    ImageDraw.Draw(off).text(org, gtext, font=font, fill=255)
    # gambar karakter satu per satu agar posisi sel sama dgn ghost '8.'
    d_on = ImageDraw.Draw(on)
    boxes = []
    for i, ch in enumerate(digits):
        x = org[0] + tmp.textlength('8.' * i, font=font)
        s = ch + ('.' if i == dp else '')
        d_on.text((x, org[1]), s, font=font, fill=255)
        b = tmp.textbbox((x, org[1]), '8.', font=font)
        boxes.append([(b[0] - 2, b[1] - 2), (b[2] + 2, b[1] - 2), (b[2] + 2, b[3] + 2), (b[0] - 2, b[3] + 2)])
    on = np.asarray(on, np.float32) / 255; off = np.clip(np.asarray(off, np.float32) / 255 - on, 0, 1)
    t = size * 0.12
    if display == 'lcd':
        bg = np.array([rng.uniform(150, 195), rng.uniform(165, 210), rng.uniform(130, 175)], np.float32)
        fg = np.array([rng.uniform(15, 60)] * 3, np.float32)
        img = np.ones((H, W, 3), np.float32) * bg
        m = on + off * ghost
        img = img * (1 - m[..., None]) + fg * m[..., None]
    else:
        red = rng.random() < 0.5
        c = np.array([245, 45, 40] if red else [60, 240, 90], np.float32)
        img = np.ones((H, W, 3), np.float32) * rng.uniform(5, 35)
        glow = cv2.GaussianBlur(on, (0, 0), t * rng.uniform(0.6, 1.4)) * rng.uniform(0.3, 0.8)
        img = img + c * np.clip(on + glow, 0, 1.3)[..., None] + c * (off * ghost)[..., None]
    return np.clip(img, 0, 255), boxes, dict(digit_h=size * 0.8, font_size=size)


def make(a):
    idx, seed, cond = a
    rng = random.Random(seed)
    digits, dp, label = sample_text(rng)
    display = rng.choice(['lcd', 'led'])
    degs = rng.sample(DEGRADATIONS, rng.randint(2, 3)) if cond == 'combined' else ([] if cond == 'clean' else [cond])
    ghost = rng.uniform(0.15, 0.35) if 'ghost' in degs else 0.0
    img, boxes, params = render_dseg(rng, digits, dp, display, ghost)
    img, boxes = apply_degr(rng, img, boxes, degs, params)
    H, W = img.shape[:2]
    bb = []
    for q in boxes:
        q = np.array(q); x0, y0 = q.min(0); x1, y1 = q.max(0)
        bb.append([int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))])
    fn = f'dseg_{cond}_{idx:04d}.png'
    cv2.imwrite(os.path.join(OUT, 'images', fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return dict(file=fn, label=label, digits=digits, dp_index=dp, condition=cond, display=display,
                degradations=degs, digit_bboxes=bb)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--per-condition', type=int, default=50)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'images'), exist_ok=True)
    jobs = [(i, 777_000_000 + ci * 10_000 + i, c) for ci, c in enumerate(CONDITIONS) for i in range(a.per_condition)]
    with Pool() as p:
        recs = p.map(make, jobs)
    with open(os.path.join(OUT, 'labels.jsonl'), 'w') as f:
        for r in recs: f.write(json.dumps(r) + '\n')
    print(len(recs), 'citra DSEG')
    res = {}
    # Tesseract
    import eval_tesseract as et
    for v in ['eng_pre', 'ssd_pre', '7seg_pre']:
        with Pool(8) as p:
            rows = p.map(et.job, [(r, v, OUT) for r in recs])
        res[v] = aggregate(rows)
    # CNN per-digit
    from train_cnn import DigitCNN, load as cload, to_string
    m = DigitCNN(); m.load_state_dict(torch.load(os.path.join(HERE, 'results/cnn_perdigit.pt'), map_location='cpu')); m.eval()
    X, _, _, own = cload(OUT, recs)
    with torch.no_grad():
        ld, lp = m(X)
    pd, pp = ld.argmax(1).numpy(), lp.argmax(1).numpy()
    rows = []
    for k, r in enumerate(recs):
        pr = to_string(pd[own == k], pp[own == k]); s = score(r['label'], pr); s['condition'] = r['condition']; rows.append(s)
    res['cnn_perdigit'] = aggregate(rows)
    # CRNN
    from train_crnn import CRNN, load as rload, decode
    m = CRNN(); m.load_state_dict(torch.load(os.path.join(HERE, 'results/crnn_ctc.pt'), map_location='cpu')); m.eval()
    with torch.no_grad():
        preds = decode(m(rload(OUT, recs)))
    rows = []
    for r, pr in zip(recs, preds):
        s = score(r['label'], pr); s['condition'] = r['condition']; rows.append(s)
    res['crnn_ctc'] = aggregate(rows)
    for k, v in res.items():
        print(f"{k:13s} ALL EM={v['ALL']['exact_match']:.3f} CER={v['ALL']['cer']:.3f} digit={v['ALL']['digit_acc']:.3f}")
    json.dump(res, open(os.path.join(HERE, 'results/ood_dseg.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
