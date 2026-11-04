"""CRNN-CTC kecil (PyTorch) pada string penuh -- tanpa segmentasi digit.

  python train_crnn.py --epochs 15
"""
import json, os, time, argparse, random
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from metrics import score, aggregate

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = '0123456789.'
IH, IW = 32, 256


def prep(g):
    h, w = g.shape
    nw = min(IW, max(8, int(round(w * IH / h))))
    r = cv2.resize(g, (nw, IH), interpolation=cv2.INTER_AREA).astype(np.float32)
    r = (r - r.mean()) / (r.std() + 1e-5)
    out = np.full((IH, IW), r[:, -1:].mean(), np.float32)   # pad kanan
    out[:, :nw] = r
    return out


def load(root, recs):
    X = [prep(cv2.imread(os.path.join(root, 'images', r['file']), cv2.IMREAD_GRAYSCALE)) for r in recs]
    return torch.tensor(np.stack(X))[:, None]


class CRNN(nn.Module):
    def __init__(self, nc=len(VOCAB) + 1):
        super().__init__()
        def c(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.cnn = nn.Sequential(*c(1, 32), nn.MaxPool2d(2),            # 16x128
                                 *c(32, 64), nn.MaxPool2d(2),           # 8x64
                                 *c(64, 128), *c(128, 128), nn.MaxPool2d((2, 1)),  # 4x64
                                 *c(128, 128), nn.MaxPool2d((4, 1)))   # 1x64
        self.rnn = nn.LSTM(128, 96, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(192, nc)

    def forward(self, x):
        z = self.cnn(x).squeeze(2).permute(0, 2, 1)   # B,T=64,C
        z, _ = self.rnn(z)
        return self.fc(z)                              # B,T,nc


def decode(logits):
    out = []
    for seq in logits.argmax(-1).cpu().numpy():
        s, prev = [], -1
        for k in seq:
            if k != prev and k != 0:
                s.append(VOCAB[k - 1])
            prev = k
        out.append(''.join(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--out', default=os.path.join(HERE, 'results/crnn_ctc.json'))
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'  # CNN+LSTM di MPS, loss CTC dihitung di CPU
    torch.set_num_threads(8)
    tr = [json.loads(l) for l in open(os.path.join(HERE, 'data/train/labels.jsonl'))]
    te = [json.loads(l) for l in open(os.path.join(HERE, 'data/test/labels.jsonl'))]
    nval = len(tr) // 10
    Xtr, Xva = load(os.path.join(HERE, 'data/train'), tr[nval:]), load(os.path.join(HERE, 'data/train'), tr[:nval])
    Ltr = [r['label'] for r in tr[nval:]]; Lva = [r['label'] for r in tr[:nval]]
    m = CRNN().to(dev)
    print('parameter:', sum(p.numel() for p in m.parameters()), flush=True)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    bs = 128
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * ((len(Xtr) + bs - 1) // bs))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    t_train = time.time()
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(Xtr)); tl = 0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            x = (Xtr[idx] + 0.1 * torch.randn(len(idx), 1, IH, IW)).to(dev)
            labs = [Ltr[j] for j in idx.tolist()]
            tgt = torch.tensor([VOCAB.index(ch) + 1 for s in labs for ch in s])
            tl_ = torch.tensor([len(s) for s in labs])
            lp = m(x).log_softmax(-1).permute(1, 0, 2).cpu()   # T,B,C
            loss = ctc(lp, tgt, torch.full((len(idx),), lp.shape[0], dtype=torch.long), tl_)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step(); sched.step()
            tl += loss.item() * len(idx)
        m.eval()
        with torch.no_grad():
            pv = decode(m(Xva.to(dev)))
        em = np.mean([p == g for p, g in zip(pv, Lva)])
        print(f'epoch {ep+1}: loss={tl/len(Xtr):.4f} val_exact={em:.4f} ({time.time()-t_train:.0f}s)', flush=True)
    train_s = time.time() - t_train
    torch.save(m.state_dict(), os.path.join(HERE, 'results/crnn_ctc.pt'))

    Xte = load(os.path.join(HERE, 'data/test'), te)
    with torch.no_grad():
        preds = decode(m(Xte.to(dev)))
    rows = []
    for r, p in zip(te, preds):
        s = score(r['label'], p); s.update(file=r['file'], condition=r['condition'], label=r['label'], pred=p)
        rows.append(s)
    agg = aggregate(rows)
    torch.set_num_threads(1)
    m = m.cpu()
    lat = []
    with torch.no_grad():
        for r in [r for i, r in enumerate(te) if i % 30 == 0]:
            t1 = time.perf_counter()
            g = cv2.imread(os.path.join(HERE, 'data/test/images', r['file']), cv2.IMREAD_GRAYSCALE)
            decode(m(torch.tensor(prep(g))[None, None]))
            lat.append((time.perf_counter() - t1) * 1000)
    agg['ALL'].update(latency_ms_mean=round(float(np.mean(lat)), 2), latency_ms_median=round(float(np.median(lat)), 2),
                      latency_n=len(lat), train_seconds=round(train_s, 1), epochs=a.epochs, device_train=dev,
                      n_train_images=len(tr) - nval)
    print('UJI ALL:', agg['ALL'])
    for c in agg:
        if c != 'ALL':
            print(f"   {c:12s} EM={agg[c]['exact_match']:.3f} CER={agg[c]['cer']:.3f} digit={agg[c]['digit_acc']:.3f}")
    json.dump({'crnn_ctc': agg}, open(a.out, 'w'), indent=1)
    with open(a.out.replace('.json', '_predictions.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
