"""检查独立场景生成素材不能混用源、伪造运行条件或遗漏检查帧。

@author pony
@date 2026-11-07
@version v1.2.0
@last_modified 2026-11-08
@changelog
    - v1.2.0 (2026-11-08): 低vis素材必须匹配实际0.1权重，防止配方声明与运行trace混用。
    - v1.1.0 (2026-11-07): 覆盖ROI实测数组校验、错误摘要、mask混用和配方降级。
    - v1.0.0 (2026-11-07): 覆盖来源摘要、原生控制、安全检查和全部有序帧的归档边界。
"""
from __future__ import annotations

import copy
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_generation_sample_materials import validate_sample_records
from p0_policy_runtime import sha256
from p0_transfer_smoke import MULTICONTROL_ORDER, MULTICONTROL_RECIPES, spatial_control_statistics


class IndependentSampleMaterialsTests(unittest.TestCase):
    def setUp(self):
        weights = MULTICONTROL_RECIPES["multicontrol-vis-edge"]
        branches = {key: {"finite": True, "latent_nonzero": 10,
                         **{metric: weights[key] for metric in ["weight_min", "weight_max", "weight_mean"]}}
                    for key in MULTICONTROL_ORDER}
        self.generation = {"stage": "transfer_smoke", "status": "passed", "result": {
            "source_success_evidence": {"sha256": "source"}, "variant": "base/edge+vis+depth+seg",
            "guardrails_enabled": True, "num_steps": 35, "seed": 42, "batch": 1, "sample_count": 1,
            "resolution_requested": "480", "multicontrol_trace": [{"branch_order": MULTICONTROL_ORDER, "branches": branches}]}}
        self.gate = {"source_input_sha256": "source", "generation_probe_sha256": "generation",
                     "segmented_frames": list(range(93)),
                     "per_frame": [{"frame": i, "rois": {"robot": {}, "bowl": {}}} for i in range(93)]}
        self.evidence = {"source_success": True, "replay_validation": {"passed": True},
                         "control_replay_validation": {"passed": True}, "roi_names": ["robot", "bowl"]}

    def test_valid_full_frame_sample_can_be_archived_without_claiming_training_eligibility(self):
        validate_sample_records(self.generation, self.gate, self.evidence, "generation", "source")

    def test_generation_and_gate_cannot_refer_to_different_sources(self):
        for field in ["source_input_sha256", "generation_probe_sha256"]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_sample_records(self.generation, {**self.gate, field: "wrong"}, self.evidence, "generation", "source")

    def test_partial_duplicate_or_unordered_frames_are_rejected(self):
        for gate in [{**self.gate, "segmented_frames": [0, 46, 92]},
                     {**self.gate, "per_frame": self.gate["per_frame"][:-1]},
                     {**self.gate, "per_frame": list(reversed(self.gate["per_frame"]))}]:
            with self.subTest(gate=gate["segmented_frames"][:2]), self.assertRaises(ValueError):
                validate_sample_records(self.generation, gate, self.evidence, "generation", "source")

    def test_failed_source_or_wrong_run_conditions_cannot_enter_the_video(self):
        for field, value in [("guardrails_enabled", False), ("num_steps", 4), ("seed", 7), ("batch", 2),
                             ("sample_count", 2), ("resolution_requested", "720")]:
            changed = copy.deepcopy(self.generation)
            changed["result"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_sample_records(changed, self.gate, self.evidence, "generation", "source")
        with self.assertRaises(ValueError):
            validate_sample_records(self.generation, self.gate, {**self.evidence, "source_success": False}, "generation", "source")

    def test_inactive_requested_branch_is_rejected(self):
        changed = copy.deepcopy(self.generation)
        changed["result"]["multicontrol_trace"][0]["branches"]["depth"]["latent_nonzero"] = 0
        with self.assertRaises(RuntimeError):
            validate_sample_records(changed, self.gate, self.evidence, "generation", "source")

    def spatial_sample(self, directory):
        recipe = "multicontrol-roi-background"
        weights = MULTICONTROL_RECIPES[recipe]
        common = np.zeros((1, 24, 4, 4, 1), dtype=np.float32)
        common[:, :, 1:3, 1:3] = 1
        actual = np.stack([common * weights[key] / sum(weights.values()) for key in MULTICONTROL_ORDER])
        generation, evidence = copy.deepcopy(self.generation), copy.deepcopy(self.evidence)
        files = {}
        for name in ["native_control_weights.npy", "expected_control_weights.npy"]:
            path = directory / name
            np.save(path, actual, allow_pickle=False)
            files[name] = sha256(path)
        statistics = spatial_control_statistics(actual, weights, actual)
        generation["artifacts_sha256"] = files
        generation["result"].update({"recipe": recipe, "control_scope": "robot_and_targets_roi_all_four_controls",
            "multicontrol_trace": [{"branch_order": MULTICONTROL_ORDER, "branches": {
                key: {"finite": True, "latent_nonzero": 10, **value} for key, value in statistics.items()}}],
            "native_control_weight_evidence": {"file": "native_control_weights.npy",
                "sha256": files["native_control_weights.npy"], "expected_file": "expected_control_weights.npy",
                "expected_sha256": files["expected_control_weights.npy"], "requested_raw_weights": weights,
                "source_roi_position_verified": True, "mask_sha256": "source-mask"}})
        evidence.update({"control_recipe": recipe, "files_sha256": {"mask.mp4": "source-mask"}})
        return generation, evidence

    def test_roi_sample_is_checked_against_saved_spatial_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            generation, evidence = self.spatial_sample(directory)
            validate_sample_records(generation, self.gate, evidence, "generation", "source",
                                    generation_directory=directory)
            with self.assertRaises(ValueError):
                validate_sample_records(generation, self.gate, evidence, "generation", "source")
            changed = copy.deepcopy(generation)
            changed["result"]["multicontrol_trace"][0]["branches"]["depth"]["weight_mean"] += .01
            with self.assertRaises(RuntimeError):
                validate_sample_records(changed, self.gate, evidence, "generation", "source",
                                        generation_directory=directory)

    def test_roi_array_digests_and_source_mask_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            generation, evidence = self.spatial_sample(directory)
            changed_source = copy.deepcopy(evidence)
            changed_source["files_sha256"]["mask.mp4"] = "different-mask"
            with self.assertRaises(ValueError):
                validate_sample_records(generation, self.gate, changed_source, "generation", "source",
                                        generation_directory=directory)
            for filename in ["native_control_weights.npy", "expected_control_weights.npy"]:
                original = (directory / filename).read_bytes()
                (directory / filename).write_bytes(b"changed")
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    validate_sample_records(generation, self.gate, evidence, "generation", "source",
                                            generation_directory=directory)
                (directory / filename).write_bytes(original)

    def test_roi_recipe_cannot_fall_back_to_scalar_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            generation, evidence = self.spatial_sample(directory)
            for recipe in ["multicontrol-vis-edge", "unregistered"]:
                changed = copy.deepcopy(generation)
                changed["result"]["recipe"] = recipe
                with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                    validate_sample_records(changed, self.gate, evidence, "generation", "source",
                                            generation_directory=directory)
            changed = copy.deepcopy(generation)
            changed["artifacts_sha256"]["expected_control_weights.npy"] = "different-run"
            with self.assertRaises(ValueError):
                validate_sample_records(changed, self.gate, evidence, "generation", "source",
                                        generation_directory=directory)

    def test_low_vis_materials_require_actual_reduced_control_weight(self):
        generation, evidence = copy.deepcopy(self.generation), copy.deepcopy(self.evidence)
        generation["result"]["recipe"] = "multicontrol-low-vis-background"
        evidence["control_recipe"] = "multicontrol-low-vis-background"
        branch = generation["result"]["multicontrol_trace"][0]["branches"]["vis"]
        branch.update({key: .1 for key in ["weight_min", "weight_max", "weight_mean"]})
        validate_sample_records(generation, self.gate, evidence, "generation", "source")
        branch["weight_mean"] = .5
        with self.assertRaises(RuntimeError):
            validate_sample_records(generation, self.gate, evidence, "generation", "source")


if __name__ == "__main__":
    unittest.main()
