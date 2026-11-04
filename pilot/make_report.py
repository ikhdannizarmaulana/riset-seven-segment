"""Gabungkan hasil -> results.json, results.csv, results_ood_dseg.csv; salin contoh citra."""
import json, os, csv, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
res = {}
res.update(json.load(open(f'{R}/tesseract.json')))
res.update(json.load(open(f'{R}/cnn_perdigit.json')))
res.update(json.load(open(f'{R}/crnn_ctc.json')))
ood = json.load(open(f'{R}/ood_dseg.json'))
json.dump({'test_poligon_in_distribution': res, 'test_dseg_ood': ood}, open(f'{HERE}/results.json', 'w'), indent=1)
conds = ['clean', 'noise', 'blur', 'perspective', 'dim', 'glare', 'ghost', 'combined', 'ALL']


def write(path, data):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metode', 'kondisi', 'n', 'exact_match', 'cer', 'digit_acc', 'latency_ms_mean'])
        for meth, agg in data.items():
            for c in conds:
                a = agg[c]
                w.writerow([meth, c, a['n'], a['exact_match'], a['cer'], a['digit_acc'], a.get('latency_ms_mean', '')])


write(f'{HERE}/results.csv', res)
write(f'{HERE}/results_ood_dseg.csv', ood)
# markdown ringkas (exact match per kondisi)
for name, data in [('IN-DIST', res), ('OOD-DSEG', ood)]:
    print(f'\n### {name} exact match')
    print('| metode | ' + ' | '.join(conds) + ' |')
    print('|' + '---|' * (len(conds) + 1))
    for m, agg in data.items():
        print(f'| {m} | ' + ' | '.join(f"{agg[c]['exact_match']:.3f}" for c in conds) + ' |')
    print(f'\n### {name} CER / digit_acc (ALL) + latency')
    for m, agg in data.items():
        a = agg['ALL']
        print(f"| {m} | {a['cer']:.4f} | {a['digit_acc']:.4f} | {a.get('latency_ms_mean','-')} | {a.get('latency_ms_median','-')} |")
    print(f'\n### {name} CER per kondisi')
    for m, agg in data.items():
        print(f'| {m} | ' + ' | '.join(f"{agg[c]['cer']:.3f}" for c in conds) + ' |')
    print(f'\n### {name} digit_acc per kondisi')
    for m, agg in data.items():
        print(f'| {m} | ' + ' | '.join(f"{agg[c]['digit_acc']:.3f}" for c in conds) + ' |')

# contoh citra: 1 per kondisi dari set uji + 2 DSEG
ex = os.path.join(HERE, 'contoh_gambar'); os.makedirs(ex, exist_ok=True)
recs = [json.loads(l) for l in open(f'{HERE}/data/test/labels.jsonl')]
seen = {}
for r in recs:
    if r['condition'] not in seen:
        seen[r['condition']] = r
for c, r in seen.items():
    shutil.copy(f"{HERE}/data/test/images/{r['file']}", f"{ex}/{c}_{r['display']}_label-{r['label']}.png")
d = [json.loads(l) for l in open(f'{HERE}/data/test_dseg/labels.jsonl')]
for r in [d[0], d[300]]:
    shutil.copy(f"{HERE}/data/test_dseg/images/{r['file']}", f"{ex}/dseg_{r['condition']}_{r['display']}_label-{r['label']}.png")
print('\ncontoh:', sorted(os.listdir(ex)))
