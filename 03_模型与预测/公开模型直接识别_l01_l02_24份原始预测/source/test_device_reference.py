"""CPU helper tests; model tests run only where pinned PyTorch is installed."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import device_reference_worker as worker


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.random.default_rng(7).normal(size=(9, 3, 256)).astype(np.float32)
        self.rows = np.arange(21, 30)

    def test_partition_invariance_every_condition(self):
        for family, level in worker.CONDITIONS:
            a = worker.corrupt_raw(self.raw, self.rows, family, level, 91)
            b = np.concatenate([worker.corrupt_raw(self.raw[s], self.rows[s], family, level, 91)
                                for s in (slice(0, 2), slice(2, 7), slice(7, None))])
            np.testing.assert_array_equal(a, b)
            self.assertTrue(np.isfinite(a).all())

    def test_clean_identity_and_copy(self):
        result = worker.corrupt_raw(self.raw, self.rows, "clean", 0, 91)
        np.testing.assert_array_equal(result, self.raw)
        result[0, 0, 0] = 999
        self.assertNotEqual(result[0, 0, 0], self.raw[0, 0, 0])

    def test_physical_gaussian_scale_and_rotation_norm(self):
        a = worker.corrupt_raw(self.raw, self.rows, "gaussian", 1, 91)
        self.assertAlmostEqual(float((a - self.raw).std()), .01, delta=.001)
        b = worker.corrupt_raw(self.raw, self.rows, "rotation", 2, 91)
        np.testing.assert_allclose(np.linalg.norm(b, axis=1), np.linalg.norm(self.raw, axis=1), atol=1e-6)

    def test_median_fixed_reflect_and_loss_interpolation(self):
        x = np.zeros((1, 3, 256), dtype=np.float32)
        x[:, :, 7] = 10
        self.assertEqual(float(worker.median3(x).max()), 0)
        ramp = np.tile(np.arange(256, dtype=np.float32), (1, 3, 1))
        for family, level in (("packetloss", 2), ("burst", 2)):
            np.testing.assert_array_equal(worker.corrupt_raw(ramp, [3], family, level, 4), ramp)

    def test_no_rotation_in_training(self):
        self.assertNotIn("rotation", worker.TRAIN_FAMILIES)
        self.assertEqual(len(worker.CONDITIONS), 13)


class ContractTests(unittest.TestCase):
    def test_overlap_rejected_and_identity_disjoint_required(self):
        metadata = [{"cow_id": str(i // 2), "legacy_cow_id": str(i // 2),
                     "session_id": str(i), "window_id": str(i)} for i in range(4)]
        ok = worker.verify_indices(np.array([0, 1]), np.array([2, 3]), 4, metadata)
        self.assertEqual(ok["disjoint_identity_keys"], ["cow_id", "legacy_cow_id"])
        with self.assertRaises(ValueError):
            worker.verify_indices(np.array([0, 1]), np.array([1, 2]), 4, metadata)
        with self.assertRaises(ValueError):
            worker.verify_indices(np.array([0, 2]), np.array([1, 3]), 4, metadata)

    def test_metadata_rejects_target_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.csv"
            fields = ["row_index", "cow_id", "legacy_cow_id", "session_id", "scale_g", "label"]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, [0, "A", "oldA", "S", 1, "walking"])))
            with self.assertRaisesRegex(ValueError, "label-bearing"):
                worker.load_metadata(path, 1)

    def test_relative_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                worker.relative(folder, "../outside")


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Pinned PyTorch not installed in this CPU interpreter")
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(101)
        spec = importlib.util.spec_from_file_location("model_test", Path(__file__).with_name("model_paper_kernel8.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model_type = module.PaperKernel8CCNN
        self.model = worker.new_binary_model(self.model_type, torch).eval()

    def test_only_sixteen_affine_values_mutate_and_reload(self):
        teacher = copy.deepcopy(self.model)
        student = copy.deepcopy(self.model)
        parameters = worker.configure_affine(student)
        self.assertEqual(sum(p.numel() for p in parameters), 16)
        names = ("feature_bn.weight", "feature_bn.bias")
        before = worker.state_hash(student.state_dict(), names)
        refs = [p.detach().clone() for p in parameters]
        optimizer = torch.optim.Adam(parameters, lr=.001)
        batch = np.random.default_rng(4).normal(size=(4, 3, 256)).astype(np.float32)
        damaged = batch + np.random.default_rng(17).normal(0, .1, batch.shape).astype(np.float32)
        for _ in range(2):
            worker.train_step(student, teacher, optimizer, batch, damaged, refs, torch, torch.device("cpu"), {})
        self.assertEqual(worker.state_hash(student.state_dict(), names), before)
        self.assertTrue(any(not torch.equal(p, r) for p, r in zip(parameters, refs)))
        with tempfile.TemporaryDirectory() as folder:
            receipt = worker.checkpoint_roundtrip(student, Path(folder) / "model.pt", batch, self.model_type, torch, torch.device("cpu"))
            self.assertTrue(receipt["prediction_reload_exact"])

    def test_binary_source_strict_checkpoint_and_positive_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "inputs/checkpoints/act_seed0.pt"
            path.parent.mkdir(parents=True)
            metadata = {"source": "act", "seed": 0, "matched_optimizer_steps": 1600, "positive_class_index": 1}
            torch.save({"model_state": self.model.state_dict(), "metadata_json": json.dumps(metadata)}, path)
            loaded, info = worker.load_source(root, "act", 0, self.model_type, torch)
            self.assertEqual(info["positive_class_index"], 1)
            self.assertEqual(worker.state_hash(loaded.state_dict()), worker.state_hash(self.model.state_dict()))
            metadata["positive_class_index"] = 0
            torch.save({"model_state": self.model.state_dict(), "metadata_json": json.dumps(metadata)}, path)
            with self.assertRaisesRegex(ValueError, "positive class"):
                worker.load_source(root, "act", 0, self.model_type, torch)


if __name__ == "__main__":
    unittest.main()
