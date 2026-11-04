#!/usr/bin/env python3
"""Bagi metadata.csv menjadi train/val/test PER KELOMPOK (default: alat_id+sesi_id),
sehingga frame/foto dari sesi yang sama tidak pernah tersebar ke dua split (anti-leakage).

Pemakaian:
  python3 split_by_group.py metadata.csv --out metadata_split.csv \
      --group alat_id,sesi_id --ratio 0.7,0.15,0.15 --seed 42 [--test-alat ALAT07,ALAT08]

--test-alat: alat yang SELURUHNYA dipakai sebagai test "alat tak terlihat" (unseen device).
Hanya stdlib. Mencetak ringkasan jumlah per split x kondisi dan cek kebocoran.
"""
import argparse, csv, random, collections, sys

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--out", required=True)
ap.add_argument("--group", default="alat_id,sesi_id")
ap.add_argument("--ratio", default="0.7,0.15,0.15")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--test-alat", default="")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.csv, newline="", encoding="utf-8")))
if not rows: sys.exit("metadata kosong")
keys = a.group.split(",")
for k in keys + ["file", "label", "kondisi"]:
    if k not in rows[0]: sys.exit(f"kolom '{k}' tidak ada di CSV")
unseen = set(x for x in a.test_alat.split(",") if x)
r_tr, r_va, r_te = map(float, a.ratio.split(","))

groups = collections.defaultdict(list)
for r in rows: groups[tuple(r[k] for k in keys)].append(r)
gids = sorted(g for g in groups if groups[g][0]["alat_id"] not in unseen)
random.Random(a.seed).shuffle(gids)
n = sum(len(groups[g]) for g in gids); acc = 0; assign = {}
for g in gids:  # isi train dulu sampai kuota (berdasar jumlah gambar), lalu val, sisanya test
    frac = acc / n if n else 1
    assign[g] = "train" if frac < r_tr else ("val" if frac < r_tr + r_va else "test")
    acc += len(groups[g])
for g in groups:
    if groups[g][0]["alat_id"] in unseen: assign[g] = "test_unseen"

out = []
for g, rs in groups.items():
    for r in rs: r = dict(r); r["split"] = assign[g]; out.append(r)
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

# ringkasan + cek leakage
tab = collections.Counter((r["split"], r["kondisi"]) for r in out)
splits = sorted({s for s, _ in tab}); conds = sorted({c for _, c in tab})
print("split".ljust(12) + "".join(c[:10].rjust(11) for c in conds) + "  total")
for s in splits:
    print(s.ljust(12) + "".join(str(tab[(s, c)]).rjust(11) for c in conds) + "  " + str(sum(tab[(s, c)] for c in conds)))
per_group = collections.defaultdict(set)
for r in out: per_group[tuple(r[k] for k in keys)].add(r["split"])
leak = [g for g, s in per_group.items() if len(s) > 1]
lab = collections.defaultdict(set)
for r in out: lab[r["split"]].add(r["label"])
print("grup bocor ke >1 split:", len(leak))
if "test" in lab and "train" in lab:
    print("label string test yang juga ada di train: %d dari %d" % (len(lab["test"] & lab["train"]), len(lab["test"])))
