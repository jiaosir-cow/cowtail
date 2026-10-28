"""Synthetic tests only. Does not open results, train, connect, or delete files."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_six_results as scorer


class ScoreTests(unittest.TestCase):
    def test_absent_class_predictions_remain_errors(self):
        y = np.array([0, 0, 2, 2])
        p = np.eye(4)[[0, 3, 2, 2]]
        v, classes, matrix = scorer.compute_metrics(y, p, ["a", "a", "b", "b"], "window")
        self.assertAlmostEqual(v["macro_f1_fixed4"], (2 / 3 + 1) / 4)
        self.assertAlmostEqual(v["macro_f1_supported"], (2 / 3 + 1) / 2)
        self.assertAlmostEqual(v["predictions_absent_class_fraction"], .25)
        self.assertEqual(v["accuracy"], .75)
        self.assertIsNone(v["mcc4_complete_only"])
        self.assertIsNone(classes[1]["recall"])
        self.assertIsNone(classes[3]["recall"])
        self.assertEqual(classes[3]["f1_fixed_convention"], 0)
        self.assertEqual(matrix[0, 3], 1)

    def test_interval_weight_is_invariant_to_identical_window_duplication(self):
        y = np.array([0, 0, 1, 2, 3])
        p = np.eye(4)[[0, 1, 1, 2, 3]]
        events = np.array(["a", "a", "b", "c", "d"])
        duplicate = np.array([0, 1, 0, 1, 0, 1, 2, 3, 4])
        a, _, _ = scorer.compute_metrics(y, p, events, "interval_balanced_window")
        b, _, _ = scorer.compute_metrics(y[duplicate], p[duplicate], events[duplicate], "interval_balanced_window")
        for key in ["macro_f1_fixed4", "macro_f1_supported", "mcc4_diagnostic", "accuracy", "walking_ap"]:
            self.assertAlmostEqual(a[key], b[key], places=13)
        self.assertAlmostEqual(a["weight_sum"], 4)
        window_a, _, _ = scorer.compute_metrics(y, p, events, "window")
        window_b, _, _ = scorer.compute_metrics(y[duplicate], p[duplicate], events[duplicate], "window")
        self.assertNotEqual(window_a["accuracy"], window_b["accuracy"])

    def test_probability_composition_detects_corruption(self):
        heads = np.array([[[.7, .3], [.8, .2], [.6, .4]]])
        states = np.array([[.8 * .7 * .6, .8 * .3, .2, .8 * .7 * .4]])
        scorer.validate_probabilities(states, heads)
        with self.assertRaisesRegex(ValueError, "composition"):
            scorer.validate_probabilities(states[:, [1, 0, 2, 3]], heads)
        with self.assertRaisesRegex(ValueError, "normalization"):
            scorer.validate_probabilities(states * .9, heads)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            scorer.validate_probabilities(states * np.nan, heads)

    @staticmethod
    def truth_fixture():
        counts = [6342, 19841, 1563, 328, 554, 283]
        intervals = [209, 486, 51, 17, 11, 7]
        aliases = np.concatenate([[alias] * n for alias, n in zip(scorer.IDS, counts)])
        cows = np.array([scorer.IDS[a] for a in aliases])
        events = np.concatenate([[f"{i}_{j % c}" for j in range(n)] for i, (n, c) in enumerate(zip(counts, intervals))])
        windows = pd.DataFrame({"row_index": np.arange(sum(counts)), "cow_alias": aliases,
                                "cow_id": cows, "event_id": events})
        labels = np.concatenate([np.arange(n, dtype=np.int64) % c % 4 for n, c in zip(counts, intervals)])
        labels[(aliases == "cowD") & (labels == 3)] = 0
        labels[(aliases == "COW_OTHER") & (labels == 1)] = 0
        arrays, folds = {}, {}
        for number, (alias, cow) in enumerate(scorer.IDS.items(), 1):
            fid = f"f{number:02d}"
            indices = np.flatnonzero(cows == cow)
            y = labels.copy()
            y[indices] = -1
            arrays[f"{fid}/train_labels.npy"] = y
            arrays[f"{fid}/test_indices.npy"] = indices
            folds[fid] = {"test_cow": cow, "test_window_count": len(indices)}
        def load(path, **kwargs):
            return arrays[Path(path).parent.name + "/" + Path(path).name]
        return windows, labels, arrays, folds, load

    def test_six_fold_truth_reconstruction_and_disagreement(self):
        windows, labels, arrays, folds, load = self.truth_fixture()
        with patch.object(scorer.pd, "read_csv", return_value=windows), patch.object(scorer.np, "load", side_effect=load):
            truth = scorer.reconstruct_truth(Path("unused"), folds)
            np.testing.assert_array_equal(truth.target4, labels)
            arrays["f02/train_labels.npy"][0] = (labels[0] + 1) % 4
            with self.assertRaisesRegex(ValueError, "disagree"):
                scorer.reconstruct_truth(Path("unused"), folds)

    def test_six_fold_truth_rejects_unmasked_heldout_and_reordered_test(self):
        windows, labels, arrays, folds, load = self.truth_fixture()
        with patch.object(scorer.pd, "read_csv", return_value=windows), patch.object(scorer.np, "load", side_effect=load):
            arrays["f01/train_labels.npy"][0] = labels[0]
            with self.assertRaisesRegex(ValueError, "mask"):
                scorer.reconstruct_truth(Path("unused"), folds)
            arrays["f01/train_labels.npy"][0] = -1
            arrays["f01/test_indices.npy"] = arrays["f01/test_indices.npy"][::-1]
            with self.assertRaisesRegex(ValueError, "indices"):
                scorer.reconstruct_truth(Path("unused"), folds)

    def test_reconstructed_original_interval_must_have_one_class(self):
        windows, labels, arrays, folds, load = self.truth_fixture()
        for number in range(2, 7):
            arrays[f"f{number:02d}/train_labels.npy"][0] = (labels[0] + 1) % 4
        with patch.object(scorer.pd, "read_csv", return_value=windows), patch.object(scorer.np, "load", side_effect=load):
            with self.assertRaisesRegex(ValueError, "original interval"):
                scorer.reconstruct_truth(Path("unused"), folds)

    def test_gate_failure_prevents_truth_read_and_output(self):
        with patch.object(scorer, "verify_gate", side_effect=ValueError("Missing formal COMPLETE")), \
             patch.object(scorer, "reconstruct_truth") as truth:
            with self.assertRaisesRegex(ValueError, "Missing formal"):
                scorer.score(Path("synthetic_nonexistent_bundle"), Path("synthetic_nonexistent_analysis"))
            truth.assert_not_called()

    def test_protocol_is_pinned_before_accepting_self_consistent_bundle(self):
        with patch.object(scorer, "read_json", return_value={}), patch.object(scorer, "sha", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "Pinned pre-run protocol"):
                scorer.verify_gate(Path("synthetic"))

    def test_three_aggregation_scopes_and_pairing_keep_all_directions(self):
        rows = []
        for number, (alias, cow) in enumerate(scorer.IDS.items(), 1):
            for method, budget, step, weighting in __import__("itertools").product(["scratch", "act", "ufro"], [10, 100], [1000, 3000], scorer.WEIGHTINGS):
                # A-D favour ACT; E/F reverse it. Both directions must survive.
                bonus = 0 if method == "scratch" else (.1 if number <= 4 else -.5)
                value = .5 + bonus
                rows.append({"unit_id": f"f{number}_{budget}_{method}", "fold_id": f"f{number:02d}",
                             "cow_alias": alias, "test_cow": cow, "replicate": 0, "method": method,
                             "budget_percent": budget, "step": step, "endpoint": "PRIMARY" if step == 1000 else "SENSITIVITY",
                             "weighting": weighting, "complete_four_class_truth": number <= 4,
                             "macro_f1_fixed4": value, "macro_f1_supported": value + .05,
                             "mcc4_diagnostic": value, "mcc4_complete_only": value if number <= 4 else None,
                             "accuracy": value, "walking_ap": value})
        summary, pairs = scorer.aggregate(pd.DataFrame(rows))
        self.assertEqual(len(summary), 72)
        self.assertEqual(len(pairs), 144)
        act = summary[(summary.method == "act") & (summary.budget_percent == 10) & (summary.step == 1000) & (summary.weighting == "window")]
        self.assertAlmostEqual(act[act.scope == "complete_four_class_A_to_D"].iloc[0].macro_f1, .6)
        self.assertLess(act[act.scope == "all_six_fixed4_diagnostic"].iloc[0].delta_f1_vs_scratch_pp, 0)
        ap = pairs[pairs.comparison == "act-scratch"]
        self.assertEqual(int((ap.macro_f1_fixed4_difference > 0).sum()), 32)
        self.assertEqual(int((ap.macro_f1_fixed4_difference < 0).sum()), 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
