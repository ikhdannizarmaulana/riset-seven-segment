"""CNN kecil per-digit (PyTorch).

Setiap citra dipotong per digit memakai bounding box dari generator (oracle segmentation),
lalu CNN memprediksi (a) kelas digit 0-9 dan (b) ada/tidaknya titik desimal di kanan digit.
String direkonstruksi dari urutan prediksi per digit.

  python train_cnn.py --epochs 6
"""
import json, os, time, argparse, random
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from metrics import score, aggregate

HERE = os.path.dirname(os.path.abspath(__file__))
CH, CW = 64, 40


def crops_of(img_gray, rec):
    out = []
    H, W = img_gray.shape
    for x0, y0, x1, y1 in rec['digit_bboxes']:
        px, py = int(0.08 * (x1 - x0)), int(0.06 * (y1 - y0))
        c = img_gray[max(0, y0 - py):min(H, y1 + py), max(0, x0 - px):min(W, x1 + px)]
        if c.size == 0:
            c = np.zeros((CH, CW), np.uint8)
        c = cv2.resize(c, (CW, CH), interpolation=cv2.INTER_AREA).astype(np.float32)
        c = (c - c.mean()) / (c.std() + 1e-5)   # standardisasi per crop (LCD/LED)
        out.append(c)
    return out


def load(root, recs):
    X, yd, yp, owner = [], [], [], []
    for k, r in enumerate(recs):
        g = cv2.imread(os.path.join(root, 'images', r['file']), cv2.IMREAD_GRAYSCALE)
        for i, c in enumerate(crops_of(g, r)):
            X.append(c); yd.append(int(r['digits'][i])); yp.append(int(i == r['dp_index'])); owner.append(k)
    return (torch.tensor(np.stack(X))[:, None], torch.tensor(yd), torch.tensor(yp), np.array(owner))


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128))   # 64x40 -> 8x5
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 8 * 5, 256), nn.ReLU())
        self.hd, self.hp = nn.Linear(256, 10), nn.Linear(256, 2)

    def forward(self, x):
        z = self.fc(self.f(x))
        return self.hd(z), self.hp(z)


def to_string(pd, pp):
    return ''.join(str(int(d)) + ('.' if p else '') for d, p in zip(pd, pp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--out', default=os.path.join(HERE, 'results/cnn_perdigit.json'))
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tr = [json.loads(l) for l in open(os.path.join(HERE, 'data/train/labels.jsonl'))]
    te = [json.loads(l) for l in open(os.path.join(HERE, 'data/test/labels.jsonl'))]
    nval = len(tr) // 10
    t0 = time.time()
    Xtr, Ydtr, Yptr, _ = load(os.path.join(HERE, 'data/train'), tr[nval:])
    Xva, Ydva, Ypva, _ = load(os.path.join(HERE, 'data/train'), tr[:nval])
    print(f'crop latih={len(Xtr)} val={len(Xva)} (load {time.time()-t0:.1f}s) device={dev}', flush=True)
    m = DigitCNN().to(dev)
    print('parameter:', sum(p.numel() for p in m.parameters()))
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * ((len(Xtr) + 255) // 256))
    t_train = time.time()
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(Xtr)); tl = 0
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            x = Xtr[idx].to(dev)
            x = x + 0.1 * torch.randn_like(x)  # augmentasi ringan
            ld, lp = m(x)
            loss = F.cross_entropy(ld, Ydtr[idx].to(dev)) + F.cross_entropy(lp, Yptr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
        m.eval()
        with torch.no_grad():
            ld, lp = m(Xva.to(dev))
            vd = (ld.argmax(1).cpu() == Ydva).float().mean().item()
            vp = (lp.argmax(1).cpu() == Ypva).float().mean().item()
        print(f'epoch {ep+1}: loss={tl/len(Xtr):.4f} val_digit_acc={vd:.4f} val_dp_acc={vp:.4f}', flush=True)
    train_s = time.time() - t_train
    torch.save(m.state_dict(), os.path.join(HERE, 'results/cnn_perdigit.pt'))

    # evaluasi uji
    Xte, _, _, own = load(os.path.join(HERE, 'data/test'), te)
    with torch.no_grad():
        ld, lp = m(Xte.to(dev))
    pd, pp = ld.argmax(1).cpu().numpy(), lp.argmax(1).cpu().numpy()
    rows = []
    for k, r in enumerate(te):
        sel = own == k
        pred = to_string(pd[sel], pp[sel])
        s = score(r['label'], pred)
        s.update(file=r['file'], condition=r['condition'], label=r['label'], pred=pred)
        rows.append(s)
    agg = aggregate(rows)

    # latency: CPU 1 thread, 1 citra per panggilan (baca file + crop + inferensi)
    torch.set_num_threads(1)
    mc = DigitCNN(); mc.load_state_dict({k: v.cpu() for k, v in m.state_dict().items()}); mc.eval()
    lat = []
    sub = [r for i, r in enumerate(te) if i % 30 == 0]  # 80 citra merata semua kondisi
    with torch.no_grad():
        for r in sub:
            t1 = time.perf_counter()
            g = cv2.imread(os.path.join(HERE, 'data/test/images', r['file']), cv2.IMREAD_GRAYSCALE)
            x = torch.tensor(np.stack(crops_of(g, r)))[:, None]
            ld, lp = mc(x)
            to_string(ld.argmax(1).numpy(), lp.argmax(1).numpy())
            lat.append((time.perf_counter() - t1) * 1000)
    agg['ALL'].update(latency_ms_mean=round(float(np.mean(lat)), 2), latency_ms_median=round(float(np.median(lat)), 2),
                      latency_n=len(lat), train_seconds=round(train_s, 1), epochs=a.epochs, device_train=dev,
                      n_train_images=len(tr) - nval)
    print('UJI ALL:', agg['ALL'])
    for c in agg:
        if c != 'ALL':
            print(f"   {c:12s} EM={agg[c]['exact_match']:.3f} CER={agg[c]['cer']:.3f} digit={agg[c]['digit_acc']:.3f}")
    json.dump({'cnn_perdigit': agg}, open(a.out, 'w'), indent=1)
    with open(a.out.replace('.json', '_predictions.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
