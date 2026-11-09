"""Unit test untuk pipeline/finalize.py (Tahap 0 / Tiket #1).

Bisa dijalankan dengan:
  python -m unittest pipeline/tests/test_finalize.py
atau:
  pytest pipeline/tests/test_finalize.py
"""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.finalize import (
    DEFAULT_LABELS,
    DEFAULT_MANIFEST,
    merge_and_finalize,
    read_csv_dict,
    verify_acceptance_criteria,
    write_csv_dict,
    FINAL_COLUMNS,
)


class TestFinalizeDataset(unittest.TestCase):
    def test_finalize_integrity_and_acceptance_criteria(self):
        """Memastikan eksekusi normal memenuhi seluruh kriteria penerimaan Tiket #1."""
        updated_manifest, splits = merge_and_finalize(
            manifest_path=DEFAULT_MANIFEST,
            labels_path=DEFAULT_LABELS,
            ratio=(0.6, 0.2, 0.2),
            seed=2026,
        )

        # 1. Total citra harus 1841
        self.assertEqual(len(updated_manifest), 1841)

        # 2. Distribusi kuota SachaIZADI layak: 779 = 468 + 156 + 155
        self.assertEqual(len(splits["train"]), 468)
        self.assertEqual(len(splits["val"]), 156)
        self.assertEqual(len(splits["test"]), 155)
        self.assertEqual(len(splits["excluded"]), 79)
        self.assertEqual(len(splits["external_test"]), 983)

        # 3. Acceptance criteria lulus verifikasi (kebocoran & stratifikasi)
        verify_acceptance_criteria(updated_manifest, splits)

        # 4. Pastikan tidak ada karakter '?' di split utama
        for sp in ("train", "val", "test"):
            for r in splits[sp]:
                self.assertNotIn("?", r["label"])
                self.assertEqual(r["usable"], "True")

        # 5. Pastikan semua excluded memiliki reason yang valid
        valid_reasons = {
            "mismatch_label_discrepancy",
            "low_confidence",
            "unreadable_contains_question_mark",
            "not_usable_for_training",
        }
        for r in splits["excluded"]:
            self.assertIn(r["reason"], valid_reasons)

    def test_negative_case_extra_file_in_labels(self):
        """Negative case: file di labels_final yang tidak ada di manifest harus memunculkan ValueError."""
        labels_rows = read_csv_dict(DEFAULT_LABELS)
        dummy_row = copy.deepcopy(labels_rows[0])
        dummy_row["file"] = "file_palsu_tidak_ada_di_manifest_123456.jpg"
        labels_rows.append(dummy_row)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_labels = Path(tmpdir) / "labels_corrupt.csv"
            write_csv_dict(tmp_labels, labels_rows, list(labels_rows[0].keys()))

            with self.assertRaises(ValueError) as ctx:
                merge_and_finalize(
                    manifest_path=DEFAULT_MANIFEST,
                    labels_path=tmp_labels,
                )
            self.assertIn("tidak ada di manifest", str(ctx.exception))

    def test_negative_case_missing_file_in_labels(self):
        """Negative case: file manifest yang hilang dari labels_final harus memunculkan ValueError."""
        labels_rows = read_csv_dict(DEFAULT_LABELS)
        labels_rows.pop(0)  # hapus satu baris

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_labels = Path(tmpdir) / "labels_incomplete.csv"
            write_csv_dict(tmp_labels, labels_rows, list(labels_rows[0].keys()))

            with self.assertRaises(ValueError) as ctx:
                merge_and_finalize(
                    manifest_path=DEFAULT_MANIFEST,
                    labels_path=tmp_labels,
                )
            self.assertIn("tidak ada di labels_final.csv", str(ctx.exception))

    def test_determinism_identical_runs(self):
        """Memastikan eksekusi dua kali dengan seed yang sama menghasilkan keluaran yang identik."""
        _, splits1 = merge_and_finalize(seed=2026)
        _, splits2 = merge_and_finalize(seed=2026)

        for sp in ("train", "val", "test", "external_test", "excluded"):
            self.assertEqual(len(splits1[sp]), len(splits2[sp]))
            ids1 = [r["id"] for r in splits1[sp]]
            ids2 = [r["id"] for r in splits2[sp]]
            self.assertEqual(ids1, ids2)


if __name__ == "__main__":
    unittest.main()
