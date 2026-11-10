"""验证前景区域控制实际释放背景，并保持成功源、目标和监督标签不变。

@author pony
@date 2026-11-07
@version v1.1.0
@last_modified 2026-11-08
@changelog
    - v1.1.0 (2026-11-08): 覆盖仅降低vis标量的真实准备链，拒绝混入ROI干预或缺失比较基线。
    - v1.0.0 (2026-11-07): 覆盖原生空间归一化、bf16舍入、掩码失效及真实源准备链。
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_prepare_background_transfer import TARGET, background_spec, prepare
from p0_transfer_smoke import (MULTICONTROL_ORDER, MULTICONTROL_RECIPES, spatial_control_statistics,
                                validate_generation_spec, validate_multicontrol_trace,
                                expected_spatial_control_weights, validate_bound_source_video)
from p0_prepare_background_transfer import COMPOSITE_RECIPE


RECIPE = "multicontrol-roi-background"
WEIGHTS = MULTICONTROL_RECIPES[RECIPE]


def checksum(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def parent_spec(directory):
    return {"name": "original", "prompt": "previous warm light", "max_frames": 93,
            "video_path": str(directory / "source.mp4"),
            "num_video_frames_per_chunk": 93, "num_steps": 35, "seed": 42, "resolution": "480",
            "guided_generation_mask": str(directory / "mask.mp4"), "guided_generation_step_threshold": 34,
            **{name: {"control_path": str(directory / f"{name}.mp4"), "control_weight": value}
               for name, value in WEIGHTS.items()}}


class BackgroundRoiTests(unittest.TestCase):
    def spatial_fixture(self):
        common = np.zeros((1, 24, 4, 4, 1), dtype=np.float32)
        common[:, :, 1:3, 1:3] = 1
        return np.stack([common * value / sum(WEIGHTS.values()) for value in WEIGHTS.values()])

    def test_same_background_prompt_and_raw_weights_only_change_control_region(self):
        parent = parent_spec(pathlib.Path("/parent"))
        before = copy.deepcopy(parent)
        directory = pathlib.Path("/new")
        full = background_spec(parent, directory, "put the book in the caddy")
        roi = background_spec(parent, directory, "put the book in the caddy", "roi")
        self.assertEqual(parent, before)
        for name in MULTICONTROL_ORDER:
            self.assertEqual(roi[name], {**full[name], "mask_path": "/new/mask.mp4"})
        for key in set(full) - {*MULTICONTROL_ORDER, "name"}:
            self.assertEqual(roi[key], full[key])
        validate_generation_spec(roi, RECIPE, 35)
        with self.assertRaises(ValueError):
            validate_generation_spec(roi, "multicontrol-vis-edge", 35)
        for name in MULTICONTROL_ORDER:
            changed = copy.deepcopy(roi)
            changed[name].pop("mask_path")
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_generation_spec(changed, RECIPE, 35)

    def test_actual_space_weights_use_local_normalization_and_trace_is_bound_to_array(self):
        actual = self.spatial_fixture()
        mask = np.zeros((93, 4, 4), dtype=bool)
        mask[:, 1:3, 1:3] = True
        expected = expected_spatial_control_weights(mask, actual.shape, WEIGHTS)
        np.testing.assert_allclose(expected, actual)
        summary = spatial_control_statistics(actual, WEIGHTS, expected)
        self.assertAlmostEqual(summary["depth"]["weight_max"], 10 / 27)
        self.assertEqual(summary["edge"]["weight_zero_fraction"], .75)
        trace = [{"branch_order": MULTICONTROL_ORDER, "branches": {
            key: {"finite": True, "latent_nonzero": 11, **value} for key, value in summary.items()}}]
        validate_multicontrol_trace(trace, WEIGHTS, actual, expected)
        with self.assertRaises(RuntimeError):
            validate_multicontrol_trace(trace, WEIGHTS)
        changed = copy.deepcopy(trace)
        changed[0]["branches"]["depth"]["weight_mean"] += .01
        with self.assertRaises(RuntimeError):
            validate_multicontrol_trace(changed, WEIGHTS, actual, expected)

    def test_bfloat16_rounding_is_accepted_without_accepting_wrong_weights(self):
        actual = self.spatial_fixture()
        rounded = actual.copy()
        for index, value in enumerate([.07421875, .185546875, .37109375, .37109375]):
            rounded[index][actual[index] > 0] = value
        spatial_control_statistics(rounded, WEIGHTS, actual)
        rounded[1] *= .8
        with self.assertRaises(RuntimeError):
            spatial_control_statistics(rounded, WEIGHTS, actual)

    def test_silently_unmasked_misaligned_infinite_or_zero_controls_fail(self):
        actual = self.spatial_fixture()
        unmasked = np.ones_like(actual)
        for index, value in enumerate(WEIGHTS.values()):
            unmasked[index] *= value / sum(WEIGHTS.values())
        wrong_region = actual.copy()
        wrong_region[1] = np.roll(wrong_region[1], 1, axis=2)
        nonfinite = actual.copy(); nonfinite[0, 0, 0, 0, 0, 0] = np.nan
        for invalid in [unmasked, wrong_region, nonfinite, np.zeros_like(actual), actual * 2, actual[:, 0],
                        np.roll(actual, 1, axis=3), unmasked - actual]:
            with self.subTest(shape=invalid.shape), self.assertRaises(RuntimeError):
                spatial_control_statistics(invalid, WEIGHTS, actual)

    def source_fixture(self, root):
        source = root / "runs/p0/edge_anchor_input_fixture"
        destination = root / "runs/p0/new_background"
        source.mkdir(parents=True); destination.mkdir()
        cache = root / "data/manifest.json"
        cache.parent.mkdir()
        cache.write_text(json.dumps({"episodes": [{"episode_id": "train", "split": "demo_train", "env_success": True}]}))
        files = ["source.mp4", "edge.mp4", "vis.mp4", "depth.mp4", "seg.mp4", "mask.mp4",
                 "roi_masks.npy", "metric_depth.npy", "source_labels.npz"]
        for name in files:
            (source / name).write_bytes(("immutable " + name).encode())
        (source / "spec.json").write_text(json.dumps(parent_spec(source)))
        evidence = {"source_success": True, "replay_validation": {"passed": True},
                    "control_replay_validation": {"passed": True}, "source_episode_id": "train",
                    "instruction": "put the book in the caddy", "spec": str(source / "spec.json"),
                    "vis_control": {"control_weight": .5, "decoded_pixels_match": True},
                    "cache_manifest": str(cache), "cache_manifest_sha256": checksum(cache),
                    "files_sha256": {name: checksum(source / name) for name in files if name != "source_labels.npz"},
                    "labels_sha256": checksum(source / "source_labels.npz")}
        (source / "verified_input.json").write_text(json.dumps(evidence))
        (source / "probe.json").write_text(json.dumps({"stage": "edge_anchor_input", "status": "passed",
            "artifacts_sha256": {name: checksum(source / name) for name in ["verified_input.json", "spec.json"]}}))
        baseline = root / "runs/p0/background_transfer_input_baseline"
        baseline.mkdir()
        for name in files:
            shutil.copyfile(source / name, baseline / name)
        baseline_spec = background_spec(parent_spec(source), baseline, evidence["instruction"])
        (baseline / "spec.json").write_text(json.dumps(baseline_spec))
        baseline_evidence = {**evidence, "spec": str(baseline / "spec.json"), "target_condition": TARGET,
                             "composite_recipe": COMPOSITE_RECIPE, "control_recipe": "multicontrol-vis-edge",
                             "parent_verified_input": str(source / "verified_input.json"),
                             "parent_verified_input_sha256": checksum(source / "verified_input.json")}
        (baseline / "verified_input.json").write_text(json.dumps(baseline_evidence))
        (baseline / "probe.json").write_text(json.dumps({"stage": "background_transfer_input", "status": "passed",
            "artifacts_sha256": {name: checksum(baseline / name) for name in ["verified_input.json", "spec.json"]}}))
        generation = root / "runs/p0/native_baseline"
        generation.mkdir()
        output = generation / "native.mp4"
        output.write_bytes(b"actual baseline file for CPU provenance contract")
        baseline_probe = generation / "probe.json"
        baseline_probe.write_text(json.dumps({"stage": "transfer_smoke", "status": "passed", "result": {
            "recipe": "multicontrol-vis-edge", "num_steps": 35, "seed": 42, "batch": 1,
            "resolution_requested": "480", "guardrails_enabled": True, "outputs": [str(output)],
            "source_success_evidence": {"path": str(baseline / "verified_input.json"),
                                        "sha256": checksum(baseline / "verified_input.json")}},
            "artifacts_sha256": {output.name: checksum(output)}}))
        self.baseline_probe = baseline_probe
        return source, destination, cache

    def test_source_prepare_preserves_parent_and_records_future_intervention(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, destination, _ = self.source_fixture(root)
            before = {path.name: checksum(path) for path in source.iterdir()}
            with patch("p0_prepare_background_transfer.ROOT", root), \
                    patch("p0_prepare_background_transfer.ensure_gpu_idle"), \
                    patch("p0_prepare_background_transfer.sha256", checksum):
                prepare(destination, SimpleNamespace(source_input=source / "verified_input.json", control_scope="roi",
                                                       baseline_probe=self.baseline_probe))
            result = json.loads((destination / "verified_input.json").read_text())
            self.assertEqual(result["control_recipe"], RECIPE)
            self.assertEqual(result["target_condition"], TARGET)
            self.assertEqual(result["intervention"]["mask_sha256"], checksum(source / "mask.mp4"))
            self.assertFalse(result["eligible_for_training"])
            self.assertEqual(result["comparison_baseline"]["generation_probe_sha256"], checksum(self.baseline_probe))
            self.assertEqual(before, {path.name: checksum(path) for path in source.iterdir()})
            for name, expected in {**result["files_sha256"], "source_labels.npz": result["labels_sha256"]}.items():
                self.assertEqual(checksum(destination / name), expected)

    def test_changed_supervision_labels_rejected_before_generating(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, destination, _ = self.source_fixture(root)
            (source / "source_labels.npz").write_bytes(b"changed action labels")
            with patch("p0_prepare_background_transfer.ROOT", root), \
                    patch("p0_prepare_background_transfer.ensure_gpu_idle"), \
                    patch("p0_prepare_background_transfer.sha256", checksum), self.assertRaises(ValueError):
                prepare(destination, SimpleNamespace(source_input=source / "verified_input.json", control_scope="roi",
                                                       baseline_probe=self.baseline_probe))

    def test_low_vis_keeps_full_frame_geometry_controls_and_binds_actual_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, destination, _ = self.source_fixture(root)
            before = {path.name: checksum(path) for path in source.iterdir()}
            with patch("p0_prepare_background_transfer.ROOT", root), \
                    patch("p0_prepare_background_transfer.ensure_gpu_idle"), \
                    patch("p0_prepare_background_transfer.sha256", checksum):
                prepare(destination, SimpleNamespace(source_input=source / "verified_input.json", control_scope="full-frame",
                    appearance_weight=.1, baseline_probe=self.baseline_probe))
            result = json.loads((destination / "verified_input.json").read_text())
            spec = json.loads((destination / "spec.json").read_text())
            baseline = json.loads(pathlib.Path(result["comparison_baseline"]["source_spec"]).read_text())
            self.assertEqual(result["control_recipe"], "multicontrol-low-vis-background")
            self.assertEqual(result["vis_control"]["control_weight"], .1)
            self.assertEqual(result["vis_control"]["parent_control_weight"], .5)
            self.assertEqual(spec["prompt"], baseline["prompt"])
            self.assertEqual(result["target_condition"], TARGET)
            self.assertEqual(result["comparison_baseline"]["intervention_kind"], "vis_control_weight_reduction")
            self.assertFalse(result["eligible_for_training"])
            for name in MULTICONTROL_ORDER:
                self.assertNotIn("mask_path", spec[name])
                self.assertEqual(spec[name]["control_weight"], .1 if name == "vis" else baseline[name]["control_weight"])
            self.assertEqual(before, {path.name: checksum(path) for path in source.iterdir()})
            validate_generation_spec(spec, "multicontrol-low-vis-background", 35)
            with self.assertRaises(ValueError):
                validate_generation_spec(spec, "multicontrol-vis-edge", 35)

    def test_low_vis_rejects_two_interventions_and_requires_completed_baseline(self):
        for scope, baseline in [("roi", True), ("full-frame", False)]:
            with self.subTest(scope=scope, baseline=baseline), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                source, destination, _ = self.source_fixture(root)
                with patch("p0_prepare_background_transfer.ROOT", root), \
                        patch("p0_prepare_background_transfer.ensure_gpu_idle"), \
                        patch("p0_prepare_background_transfer.sha256", checksum), self.assertRaises(ValueError):
                    prepare(destination, SimpleNamespace(source_input=source / "verified_input.json", control_scope=scope,
                        appearance_weight=.1, baseline_probe=self.baseline_probe if baseline else None))

    def test_rgb_path_and_preparation_digest_cannot_be_rebound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, _, _ = self.source_fixture(root)
            spec = json.loads((source / "spec.json").read_text())
            evidence = json.loads((source / "verified_input.json").read_text())
            validate_bound_source_video(spec, evidence, source)
            other = source / "other.mp4"
            shutil.copyfile(source / "source.mp4", other)
            with self.assertRaisesRegex(RuntimeError, "RGB"):
                validate_bound_source_video({**spec, "video_path": str(other)}, evidence, source)
            (source / "spec.json").write_text(json.dumps({**spec, "prompt": "changed after preparation"}))
            with self.assertRaisesRegex(RuntimeError, "摘要"):
                validate_bound_source_video(spec, evidence, source)

    def test_roi_comparison_requires_real_baseline_and_rejects_a_changed_prompt(self):
        from p0_prepare_background_transfer import bind_comparison_baseline

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, destination, _ = self.source_fixture(root)
            evidence = json.loads((source / "verified_input.json").read_text())
            spec = background_spec(parent_spec(source), destination, evidence["instruction"], "roi")
            with patch("p0_prepare_background_transfer.ROOT", root), \
                    patch("p0_prepare_background_transfer.sha256", checksum):
                for baseline, changed in [(None, spec), (self.baseline_probe, {**spec, "prompt": "new target"})]:
                    with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                        bind_comparison_baseline(baseline, source / "verified_input.json", changed, evidence)


if __name__ == "__main__":
    unittest.main()
