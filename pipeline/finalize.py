#!/usr/bin/env python3
"""Tahap 0 (#1): Penyatuan label final dan penguncian split train/val/test.

Menggabungkan label desimal dari labeling/labels_final.csv ke manifest,
membagi dataset SachaIZADI secara deterministik (berdasarkan split_group & stratifikasi HQ/MQ/LQ),
memasukkan dataset publik ke external_test, dan mencatat foto yang tidak layak ke excluded.csv.

Output:
  pipeline/unified/manifest.csv      (manifest termutakhirkan dengan kolom usable, conditions, confidence, split)
  data/splits/train.csv              (60% SachaIZADI layak)
  data/splits/val.csv                (20% SachaIZADI layak)
  data/splits/test.csv               (20% SachaIZADI layak)
  data/splits/external_test.csv      (dataset publik eksternal)
  data/splits/excluded.csv           (foto tidak layak / ? / mismatch / low confidence beserta alasan)

Jalankan:
  python pipeline/finalize.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import os
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_MANIFEST = ROOT / "pipeline" / "unified" / "manifest.csv"
DEFAULT_LABELS = ROOT / "labeling" / "labels_final.csv"
DEFAULT_SPLITS_DIR = ROOT / "data" / "splits"

FINAL_COLUMNS = [
    "id", "source_dataset", "unit", "original_split", "split_group", "image_path",
    "label", "label_source", "label_flag", "label_parts", "label_alt",
    "display_type", "is_synthetic", "license",
    "width", "height", "crop_x1", "crop_y1", "crop_x2", "crop_y2", "notes",
    "usable", "conditions", "confidence", "split", "reason",
]

EXCLUDED_COLUMNS = list(FINAL_COLUMNS)


def read_csv_dict(path: Path) -> list[dict[str, str]]:
    with open(path, mode="r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dict(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def merge_and_finalize(
    manifest_path: Path = DEFAULT_MANIFEST,
    labels_path: Path = DEFAULT_LABELS,
    ratio: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 2026,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    """Gabungkan manifest dengan labels_final dan bentuk split train/val/test/external_test/excluded."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file tidak ditemukan: {labels_path}")

    labels_rows = read_csv_dict(labels_path)
    manifest_rows = read_csv_dict(manifest_path)

    # 1. Validasi integritas file labels_final vs manifest
    labels_by_file: dict[str, dict[str, str]] = {}
    for r in labels_rows:
        fname = r.get("file", "").strip()
        if not fname:
            raise ValueError(f"Baris pada labels_final.csv tidak memiliki 'file': {r}")
        if fname in labels_by_file:
            raise ValueError(f"Duplikat file di labels_final.csv: {fname}")
        labels_by_file[fname] = r

    sacha_manifest_files: set[str] = set()
    for r in manifest_rows:
        if r.get("source_dataset") == "github_sachaizadi":
            fname = Path(r["image_path"]).name
            sacha_manifest_files.add(fname)

    # Negative check (Acceptance Criteria):
    # Bila ada file di labels_final yang tidak ada di manifest, skrip harus berhenti dengan error jelas.
    extra_in_labels = set(labels_by_file.keys()) - sacha_manifest_files
    if extra_in_labels:
        raise ValueError(
            f"labels_final.csv memuat {len(extra_in_labels)} file yang tidak ada di manifest: "
            f"{sorted(extra_in_labels)[:5]}"
        )

    missing_in_labels = sacha_manifest_files - set(labels_by_file.keys())
    if missing_in_labels:
        raise ValueError(
            f"Manifest SachaIZADI memuat {len(missing_in_labels)} file yang tidak ada di labels_final.csv: "
            f"{sorted(missing_in_labels)[:5]}"
        )

    # 2. Penggabungan atribut ke setiap baris manifest
    updated_manifest: list[dict[str, str]] = []
    usable_sacha: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    external_test_rows: list[dict[str, str]] = []

    for r in manifest_rows:
        row_dict = {col: r.get(col, "") for col in FINAL_COLUMNS}
        source = r.get("source_dataset", "")

        if source == "github_sachaizadi":
            fname = Path(r["image_path"]).name
            lbl = labels_by_file[fname]

            # Label lama dari dataset asli disimpan di label_alt (dari original_label),
            # sedangkan label_ai (desimal) menjadi label utama.
            row_dict["label_alt"] = lbl.get("original_label", "")
            row_dict["label"] = lbl.get("label_ai", "")
            row_dict["usable"] = lbl.get("usable_for_training", "")
            row_dict["conditions"] = lbl.get("conditions", "")
            row_dict["confidence"] = lbl.get("confidence", "")
            row_dict["label_source"] = "ai_vision"

            # Klasifikasi excluded vs usable
            is_usable = row_dict["usable"] == "True" and ("?" not in row_dict["label"])

            if is_usable:
                usable_sacha.append(row_dict)
            else:
                row_dict["split"] = "excluded"
                status_vs_orig = lbl.get("status_vs_original", "")
                conf = lbl.get("confidence", "")

                if status_vs_orig == "mismatch":
                    reason = "mismatch_label_discrepancy"
                elif conf == "low":
                    reason = "low_confidence"
                elif status_vs_orig == "unreadable" or ("?" in row_dict["label"]):
                    reason = "unreadable_contains_question_mark"
                else:
                    reason = "not_usable_for_training"

                row_dict["reason"] = reason
                excluded_rows.append(row_dict)
        else:
            # Dataset publik eksternal
            row_dict["usable"] = "True"
            row_dict["conditions"] = "clean"
            row_dict["confidence"] = "high"
            row_dict["split"] = "external_test"
            external_test_rows.append(row_dict)

        updated_manifest.append(row_dict)

    # 3. Stratifikasi per Quality (HQ/MQ/LQ) & Pengelompokan split_group untuk Usable SachaIZADI
    r_tr, r_va, r_te = ratio
    if abs(r_tr + r_va + r_te - 1.0) > 1e-5:
        raise ValueError(f"Ratio harus berjumlah 1.0, didapat: {ratio}")

    by_quality: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for r in usable_sacha:
        q = r.get("original_split", "UNKNOWN")
        grp = r.get("split_group", "")
        by_quality[q][grp].append(r)

    rng = random.Random(seed)
    group_assignments: dict[str, str] = {}

    for q in sorted(by_quality.keys()):
        groups = sorted(by_quality[q].keys())
        rng.shuffle(groups)

        total_q_imgs = sum(len(by_quality[q][g]) for g in groups)
        target_tr = int(round(r_tr * total_q_imgs))
        target_va = int(round(r_va * total_q_imgs))

        acc = 0
        for g in groups:
            g_size = len(by_quality[q][g])
            if acc < target_tr:
                sp = "train"
            elif acc < target_tr + target_va:
                sp = "val"
            else:
                sp = "test"
            acc += g_size
            group_assignments[g] = sp

    # Berikan split ke setiap baris usable
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for r in usable_sacha:
        sp = group_assignments[r["split_group"]]
        r["split"] = sp
        if sp == "train":
            train_rows.append(r)
        elif sp == "val":
            val_rows.append(r)
        elif sp == "test":
            test_rows.append(r)
        else:
            raise ValueError(f"Split tidak dikenal: {sp}")

    splits_dict = {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
        "external_test": external_test_rows,
        "excluded": excluded_rows,
    }

    return updated_manifest, splits_dict


def verify_acceptance_criteria(
    manifest_rows: list[dict[str, str]],
    splits_dict: dict[str, list[dict[str, str]]],
) -> None:
    """Verifikasi semua syarat acceptance criteria Issue #1."""
    train_rows = splits_dict["train"]
    val_rows = splits_dict["val"]
    test_rows = splits_dict["test"]
    excluded_rows = splits_dict["excluded"]

    # 1. Cek kebocoran split_group
    groups_by_split: dict[str, set[str]] = {
        "train": {r["split_group"] for r in train_rows},
        "val": {r["split_group"] for r in val_rows},
        "test": {r["split_group"] for r in test_rows},
    }

    leak_tr_va = groups_by_split["train"] & groups_by_split["val"]
    leak_tr_te = groups_by_split["train"] & groups_by_split["test"]
    leak_va_te = groups_by_split["val"] & groups_by_split["test"]

    assert not leak_tr_va, f"Bocor split_group antara train & val: {leak_tr_va}"
    assert not leak_tr_te, f"Bocor split_group antara train & test: {leak_tr_te}"
    assert not leak_va_te, f"Bocor split_group antara val & test: {leak_va_te}"

    # 2. Cek bahwa foto ber-? atau usable=false tidak ada di train/val/test
    for sp_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        bad_usable = [r["id"] for r in rows if r.get("usable") != "True"]
        bad_qmark = [r["id"] for r in rows if "?" in r.get("label", "")]
        assert not bad_usable, f"Ada baris usable!=True di {sp_name}: {bad_usable}"
        assert not bad_qmark, f"Ada baris berkarakter '?' di {sp_name}: {bad_qmark}"

    # 3. Cek proporsi stratifikasi HQ/MQ/LQ berbeda <= 5 poin persen
    tot_usable = len(train_rows) + len(val_rows) + len(test_rows)
    all_usable = train_rows + val_rows + test_rows
    overall_q = Counter(r["original_split"] for r in all_usable)

    for sp_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        sp_q = Counter(r["original_split"] for r in rows)
        n_sp = len(rows)
        for q in ("HQ", "MQ", "LQ"):
            p_overall = overall_q[q] / tot_usable
            p_sp = sp_q[q] / n_sp
            diff_pp = abs(p_sp - p_overall) * 100.0
            assert diff_pp <= 5.0, (
                f"Proporsi kualitas {q} pada {sp_name} ({p_sp*100:.2f}%) "
                f"berbeda {diff_pp:.2f} pp dari populasi ({p_overall*100:.2f}%), melebihi batas 5.0 pp!"
            )

    # 4. Cek excluded rows memiliki kolom reason
    for r in excluded_rows:
        assert r.get("reason"), f"Baris excluded tanpa reason: {r.get('id')}"


def print_summary(
    manifest_rows: list[dict[str, str]],
    splits_dict: dict[str, list[dict[str, str]]],
) -> None:
    """Cetak ringkasan komprehensif dari hasil finalisasi."""
    train = splits_dict["train"]
    val = splits_dict["val"]
    test = splits_dict["test"]
    ext = splits_dict["external_test"]
    exc = splits_dict["excluded"]

    total = len(manifest_rows)
    print("=" * 72)
    print("HASIL FINALISASI MANIFEST & PEMBAGIAN SPLIT (TAHAP 0 / TIKET #1)")
    print("=" * 72)
    print(f"Total citra manifest       : {total}")
    print(f"  • SachaIZADI Layak       : {len(train) + len(val) + len(test)} (Train: {len(train)}, Val: {len(val)}, Test: {len(test)})")
    print(f"  • SachaIZADI Excluded    : {len(exc)}")
    print(f"  • Dataset Eksternal      : {len(ext)}")
    print("-" * 72)

    print("\n1. STRATIFIKASI KUALITAS SACHAIZADI:")
    print("Kualitas      Train (60%)           Val (20%)            Test (20%)           Total")
    print("-" * 72)
    for q in ("HQ", "MQ", "LQ"):
        tr = sum(1 for r in train if r["original_split"] == q)
        va = sum(1 for r in val if r["original_split"] == q)
        te = sum(1 for r in test if r["original_split"] == q)
        tot = tr + va + te
        print(f"{q:<12}{tr:>4} ({tr/len(train):>6.2%})     {va:>4} ({va/len(val):>6.2%})     {te:>4} ({te/len(test):>6.2%})     {tot:>4}")
    print("-" * 72)
    print(f"{'Total':<12}{len(train):>4} ({1.0:>6.2%})     {len(val):>4} ({1.0:>6.2%})     {len(test):>4} ({1.0:>6.2%})     {len(train)+len(val)+len(test):>4}")

    print("\n2. SEBARAN KONDISI CITRA PADA SPLIT UTAMA:")
    conds = sorted({c for r in (train + val + test) for c in r.get("conditions", "").split("|") if c})
    header = f"{'Kondisi':<14}{'Train':>10}{'Val':>10}{'Test':>10}{'Total':>10}"
    print(header)
    print("-" * len(header))
    for c in conds:
        tr_c = sum(1 for r in train if c in r.get("conditions", "").split("|"))
        va_c = sum(1 for r in val if c in r.get("conditions", "").split("|"))
        te_c = sum(1 for r in test if c in r.get("conditions", "").split("|"))
        tot_c = tr_c + va_c + te_c
        print(f"{c:<14}{tr_c:>10}{va_c:>10}{te_c:>10}{tot_c:>10}")

    print("\n3. RINCIAN FOTO EXCLUDED:")
    reasons = Counter(r.get("reason") for r in exc)
    for rsn, cnt in reasons.most_common():
        print(f"  • {rsn:<35}: {cnt:>3} foto")

    print("\n4. RINCIAN DATASET EKSTERNAL (external_test):")
    ext_sources = Counter(r.get("source_dataset") for r in ext)
    for src, cnt in ext_sources.most_common():
        print(f"  • {src:<30}: {cnt:>4} citra")

    print("=" * 72)
    print("VERIFIKASI ACCEPTANCE CRITERIA: SEMUA LULUS (TIDAK ADA KEBOCORAN)")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Finalisasi manifest dan kunci split dataset.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path ke manifest.csv input")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Path ke labels_final.csv")
    parser.add_argument("--out-manifest", type=Path, default=DEFAULT_MANIFEST, help="Path manifest.csv output")
    parser.add_argument("--out-splits-dir", type=Path, default=DEFAULT_SPLITS_DIR, help="Folder output data/splits/")
    parser.add_argument("--seed", type=int, default=2026, help="Seed deterministik")
    parser.add_argument("--ratio", type=str, default="0.6,0.2,0.2", help="Rasio train,val,test")
    parser.add_argument("--check-only", action="store_true", help="Hanya jalankan verifikasi tanpa menulis file")

    args = parser.parse_args()
    ratio_parts = tuple(map(float, args.ratio.split(",")))
    if len(ratio_parts) != 3:
        sys.exit(f"Rasio harus 3 angka dipisah koma (mis. 0.6,0.2,0.2), didapat: {args.ratio}")

    updated_manifest, splits = merge_and_finalize(
        manifest_path=args.manifest,
        labels_path=args.labels,
        ratio=ratio_parts,
        seed=args.seed,
    )

    verify_acceptance_criteria(updated_manifest, splits)

    if not args.check_only:
        # Tulis manifest termutakhirkan
        write_csv_dict(args.out_manifest, updated_manifest, FINAL_COLUMNS)

        # Tulis file split
        args.out_splits_dir.mkdir(parents=True, exist_ok=True)
        write_csv_dict(args.out_splits_dir / "train.csv", splits["train"], FINAL_COLUMNS)
        write_csv_dict(args.out_splits_dir / "val.csv", splits["val"], FINAL_COLUMNS)
        write_csv_dict(args.out_splits_dir / "test.csv", splits["test"], FINAL_COLUMNS)
        write_csv_dict(args.out_splits_dir / "external_test.csv", splits["external_test"], FINAL_COLUMNS)
        write_csv_dict(args.out_splits_dir / "excluded.csv", splits["excluded"], EXCLUDED_COLUMNS)

    print_summary(updated_manifest, splits)


if __name__ == "__main__":
    main()
