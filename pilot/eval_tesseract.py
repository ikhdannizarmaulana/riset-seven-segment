"""Baseline Tesseract pada set uji sintetis.

Varian:
  eng_raw  : eng (bawaan), citra mentah RGB
  eng_pre  : eng + prapemrosesan sederhana (gray, inversi bila latar gelap, resize, Otsu)
  ssd_pre  : ssd.traineddata (Shreeshrii/tessdata_ssd) + prapemrosesan
  7seg_pre : 7seg.traineddata (Shreeshrii/tessdata_ssd) + prapemrosesan
Semua: --psm 7, --oem 1 (LSTM), whitelist 0123456789.

Akurasi dihitung paralel (multiprocessing); latency diukur terpisah secara serial
pada subset agar tidak terdistorsi oleh kontensi CPU.
"""
import json, os, sys, time, argparse
from multiprocessing import Pool
import cv2, numpy as np, pytesseract
from metrics import score, aggregate

HERE = os.path.dirname(os.path.abspath(__file__))
TESSDATA = os.path.join(HERE, 'tessdata_extra')
VARIANTS = {'eng_raw': ('eng', False), 'eng_pre': ('eng', True),
            'ssd_pre': ('ssd', True), '7seg_pre': ('7seg', True)}


def preprocess(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([g[:3].ravel(), g[-3:].ravel(), g[:, :3].ravel(), g[:, -3:].ravel()])
    if border.mean() < 100:            # latar gelap (LED) -> balik jadi teks gelap di latar terang
        g = 255 - g
    s = 80.0 / g.shape[0]
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.copyMakeBorder(b, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)


def ocr(path, variant):
    lang, pre = VARIANTS[variant]
    img = cv2.imread(path)
    if pre:
        img = preprocess(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cfg = f'--tessdata-dir {TESSDATA} --oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.'
    t0 = time.perf_counter()
    txt = pytesseract.image_to_string(img, lang=lang, config=cfg)
    dt = (time.perf_counter() - t0) * 1000
    return ''.join(c for c in txt if c in '0123456789.'), dt


def job(a):
    rec, variant, root = a
    pred, dt = ocr(os.path.join(root, 'images', rec['file']), variant)
    r = score(rec['label'], pred)
    r.update(file=rec['file'], condition=rec['condition'], display=rec['display'],
             label=rec['label'], pred=pred, variant=variant)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'data/test'))
    ap.add_argument('--latency-n', type=int, default=10, help='citra per kondisi utk ukur latency serial')
    ap.add_argument('--out', default=os.path.join(HERE, 'results/tesseract.json'))
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(a.data, 'labels.jsonl'))]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print('tesseract', pytesseract.get_tesseract_version())
    results, preds = {}, []
    for v in VARIANTS:
        t0 = time.time()
        with Pool(8) as p:
            rows = p.map(job, [(r, v, a.data) for r in recs], chunksize=8)
        preds += rows
        # latency serial
        lat = []
        by = {}
        for r in recs:
            by.setdefault(r['condition'], []).append(r)
        for c, rs in by.items():
            for r in rs[:a.latency_n]:
                lat.append(ocr(os.path.join(a.data, 'images', r['file']), v)[1])
        agg = aggregate(rows)
        agg['ALL']['latency_ms_mean'] = round(float(np.mean(lat)), 1)
        agg['ALL']['latency_ms_median'] = round(float(np.median(lat)), 1)
        agg['ALL']['latency_n'] = len(lat)
        results[v] = agg
        print(f"{v}: EM={agg['ALL']['exact_match']} CER={agg['ALL']['cer']} digit={agg['ALL']['digit_acc']} "
              f"lat={agg['ALL']['latency_ms_mean']}ms  ({time.time()-t0:.0f}s)", flush=True)
        for c in agg:
            if c != 'ALL':
                print(f"   {c:12s} EM={agg[c]['exact_match']:.3f} CER={agg[c]['cer']:.3f} digit={agg[c]['digit_acc']:.3f}")
    json.dump(results, open(a.out, 'w'), indent=1)
    with open(a.out.replace('.json', '_predictions.jsonl'), 'w') as f:
        for r in preds:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
