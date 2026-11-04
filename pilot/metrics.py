"""Metrik bersama: exact match, CER, akurasi per digit."""
from collections import defaultdict


def lev_align(a, b):
    """Levenshtein distance + jumlah karakter yang cocok (match) pada alignment optimal."""
    n, m = len(a), len(b)
    D = [[(0, 0)] * (m + 1) for _ in range(n + 1)]  # (dist, -matches)
    for i in range(1, n + 1):
        D[i][0] = (i, 0)
    for j in range(1, m + 1):
        D[0][j] = (j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = a[i - 1] == b[j - 1]
            d, mm = D[i - 1][j - 1]
            cand = [(d + (0 if same else 1), mm - (1 if same else 0)),
                    (D[i - 1][j][0] + 1, D[i - 1][j][1]),
                    (D[i][j - 1][0] + 1, D[i][j - 1][1])]
            D[i][j] = min(cand)
    return D[n][m][0], -D[n][m][1]


def score(gt, pred):
    dist, _ = lev_align(gt, pred)
    gd, pd = gt.replace('.', ''), pred.replace('.', '')
    _, dmatch = lev_align(gd, pd)
    return dict(exact=int(gt == pred), edits=dist, gt_len=len(gt),
                digit_correct=dmatch, digit_total=len(gd))


def aggregate(rows, key='condition'):
    """rows: list of dict berisi field score + condition + latency_ms (opsional)."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
        groups['ALL'].append(r)
    out = {}
    for g, rs in groups.items():
        out[g] = dict(
            n=len(rs),
            exact_match=round(sum(r['exact'] for r in rs) / len(rs), 4),
            cer=round(sum(r['edits'] for r in rs) / sum(r['gt_len'] for r in rs), 4),
            digit_acc=round(sum(r['digit_correct'] for r in rs) / sum(r['digit_total'] for r in rs), 4),
        )
    return out
