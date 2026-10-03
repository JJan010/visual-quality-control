"""Identity/acceptance gates only; these tests do not run GPU inference."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from visual_quality.inference.export_validation import (
    evidence_hashes, load_validated_export, sha256_file,
)


class ExportValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.export = self.run / "exports/new"
        self.export.mkdir(parents=True)
        (self.run / "calibration").mkdir()
        (self.run / "evaluation").mkdir()
        (self.run / "model.pt").write_bytes(b"synthetic checkpoint")
        (self.export / "feature_extractor.onnx").write_bytes(b"synthetic graph")
        (self.export / "export_check.json").write_text("{}")
        (self.run / "split.json").write_text("{}")
        (self.run / "evaluation/metrics.json").write_text("{}")
        (self.run / "evaluation/test_scores.csv").write_text("path,score\na,1\n")
        self.checkpoint = sha256_file(self.run / "model.pt")
        self.config = {"image_size": 256}
        self.write(self.run / "calibration/threshold.json", {
            "checkpoint_sha256": self.checkpoint, "threshold": 32.0, "config": self.config,
        })
        self.report = {
            "schema_version": 1, "kind": "patchcore_onnx_pipeline", "status": "passed",
            "checkpoint_sha256": self.checkpoint,
            "onnx_sha256": sha256_file(self.export / "feature_extractor.onnx"),
            "model_config": self.config, "threshold": 32.0,
            "torch_precision": "ieee", "ort_use_tf32": False,
            "blur_backend": "original_2d", "atol": 1e-4, "rtol": 1e-4,
            "evidence_hashes": evidence_hashes(self.run, self.export),
            "feature_parity_passed": False,
            "pipeline_checks": [{
                "batch_size": b, "checked_images": 83, "changed_decisions": 0,
                "changed_decisions_vs_saved": 0, "max_score_difference": 0.0001,
                "max_map_difference": 0.0004, "score_parity_passed": True,
                "map_parity_passed": True,
            } for b in (1, 8)],
            "binding_checks": [{"batch_size": b, "layer": layer,
                                "max_difference": 0.0, "parity_passed": True}
                               for b in (1, 3, 8) for layer in ("layer2", "layer3")],
        }
        self.publish()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def publish(self):
        path = self.export / "pipeline_validation.json"
        self.write(path, self.report)
        self.write(self.export / "runtime_manifest.json", {
            "schema_version": 1, "checkpoint_sha256": self.checkpoint,
            "onnx_sha256": self.report["onnx_sha256"], "model_config": self.config,
            "pipeline_validation_sha256": sha256_file(path),
        })

    def load(self):
        return load_validated_export(self.run, self.export)

    def test_valid_pipeline_approval_accepts_diagnostic_feature_difference(self):
        self.assertEqual(self.load()["status"], "passed")

    def test_replaced_graph_is_rejected(self):
        (self.export / "feature_extractor.onnx").write_bytes(b"different graph")
        with self.assertRaisesRegex(RuntimeError, "ONNX changed"):
            self.load()

    def test_replaced_checkpoint_is_rejected(self):
        (self.run / "model.pt").write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(RuntimeError, "inputs changed"):
            self.load()

    def test_changed_threshold_is_rejected(self):
        path = self.run / "calibration/threshold.json"
        calibration = json.loads(path.read_text())
        calibration["threshold"] = 33.0
        self.write(path, calibration)
        with self.assertRaisesRegex(RuntimeError, "inputs changed"):
            self.load()

    def test_modified_approval_report_is_rejected(self):
        path = self.export / "pipeline_validation.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.load()

    def test_diagnostic_status_cannot_authorize_build(self):
        self.report["status"] = "diagnostic_only"
        self.publish()
        with self.assertRaisesRegex(RuntimeError, "status"):
            self.load()

    def test_missing_batch_coverage_is_rejected(self):
        self.report["pipeline_checks"].pop()
        self.publish()
        with self.assertRaisesRegex(RuntimeError, "batch coverage"):
            self.load()

    def test_failed_numeric_or_decision_checks_are_rejected(self):
        for key, value in (("map_parity_passed", False), ("score_parity_passed", False),
                           ("changed_decisions", 1), ("changed_decisions_vs_saved", 1)):
            with self.subTest(key=key):
                result = self.report["pipeline_checks"][0]
                old = result[key]
                result[key] = value
                self.publish()
                with self.assertRaises(RuntimeError):
                    self.load()
                result[key] = old

    def test_incomplete_binding_coverage_is_rejected(self):
        self.report["binding_checks"].pop()
        self.publish()
        with self.assertRaisesRegex(RuntimeError, "binding coverage"):
            self.load()

    def test_runtime_identity_check_needs_no_training_files(self):
        (self.run / "evaluation/test_scores.csv").unlink()
        result = load_validated_export(export_dir=self.export,
                                      checkpoint_sha256=self.checkpoint,
                                      model_config=self.config)
        self.assertEqual(result["status"], "passed")
        with self.assertRaisesRegex(RuntimeError, "another checkpoint"):
            load_validated_export(export_dir=self.export, checkpoint_sha256="wrong",
                                  model_config=self.config)


if __name__ == "__main__":
    unittest.main()
