"""Targeted checks for A/B/C/D. Written for review; not run during editing."""

from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data import Batch, Data
import yaml

from data.standardization import FixedVariableStandardizer, observation_fingerprint
from data.tep_dataset import RunRecord, TEPWindowDataset
from data.tep_hetero_graph import build_tep_hetero_graph, _shift_window_for_negative, window_statistics
from models import build_model
from models.common import extended_x_dict, heterodata_from_inputs
from models.input_representation import SharedVariableTCN
from models.qc_hgt import QCHGT
from scripts.run_input_ablation import input_plan, load_input_checkpoint, main
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.experiment_manifest import assert_matching_manifest, fingerprint


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return yaml.safe_load((ROOT / "configs" / "stage2.yaml").read_text(encoding="utf-8"))


def windows(split="train"):
    records = []
    for i, offset in enumerate((0.0, 10000.0)):
        values = torch.arange(8, dtype=torch.float32)[:, None].expand(-1, 52).clone() + offset
        records.append(RunRecord(
            run_uid=f"train|fault={i}|run=0", run_id="0", source_split="train", fault_id=i,
            length=8, origin="synthetic", values=values, labels=torch.full((8,), i),
        ))
    return TEPWindowDataset(records, split, 3, 1, 0.0, None, split == "train", 4)


def sample(training=False):
    raw = torch.arange(52 * 30, dtype=torch.float32).reshape(52, 30) / 100
    mask = torch.ones_like(raw, dtype=torch.bool)
    mask[0, 4:7] = False
    raw[~mask] = 0
    value = Data(x=raw, observed_mask=mask, y=torch.tensor([1]),
                 window_start=torch.tensor([0]), window_end=torch.tensor([29]),
                 fault_scenario=torch.tensor([1]))
    if training:
        value.q_star = torch.ones(52)
    return value


def scaler():
    return FixedVariableStandardizer(torch.arange(52), torch.full((52,), 2.0),
                                    torch.full((52,), 10), 1e-6, "synthetic_selection")


def enhanced_config(temporal=False):
    config = configuration()
    norm = scaler()
    flags = {"fixed_standardization": True, "identity": True, "temporal": temporal}
    config["graph"]["representation"] = flags
    model = config["models"]["qc_hgt"]
    model["input_normalization"] = "none"
    model["representation"] = {**flags, "scaler_fingerprint": norm.fingerprint}
    # Tiny deterministic fixtures test plumbing, not classification performance.
    model["hidden_dim"] = 8
    model["num_layers"] = 1
    model["consistency"]["memory"]["gradient_checkpointing"] = False
    for key in ("random_wrong_connection", "different_window_or_run", "temporal_misalignment"):
        model["consistency"]["negative_sampling"][key] = False
    return config, norm


class StandardizationTests(unittest.TestCase):
    def test_selected_overlaps_count_each_observation_once(self):
        dataset = windows()
        norm = FixedVariableStandardizer.fit(Subset(dataset, [0, 1]))
        # Windows [0,3), [1,4) cover four unique rows, not six weighted rows.
        self.assertTrue(torch.equal(norm.count, torch.full((52,), 4)))
        self.assertTrue(torch.allclose(norm.mean, torch.full((52,), 1.5, dtype=torch.float64)))
        expected = torch.arange(4, dtype=torch.float64).std(correction=0)
        self.assertTrue(torch.allclose(norm.scale, expected.expand(52)))

    def test_unselected_data_cannot_change_scaler_or_content_fingerprint(self):
        dataset = windows()
        chosen = Subset(dataset, [0, 1])
        before = FixedVariableStandardizer.fit(chosen).to_dict()
        digest = observation_fingerprint(chosen)
        dataset.runs[0].values[4:] = -100000
        dataset.runs[1].values[:] = 100000
        self.assertEqual(before, FixedVariableStandardizer.fit(chosen).to_dict())
        self.assertEqual(digest, observation_fingerprint(chosen))
        dataset.runs[0].values[0, 0] += 1
        self.assertNotEqual(digest, observation_fingerprint(chosen))

    def test_validation_and_test_cannot_fit(self):
        for split in ("val", "test"):
            with self.assertRaises(ValueError):
                FixedVariableStandardizer.fit(Subset(windows(split), [0, 1]))

    def test_missing_observations_excluded_and_constants_are_finite(self):
        dataset = windows()
        dataset.runs[0].values[0, 0] = float("nan")
        dataset.runs[0].values[:, 1] = 5
        norm = FixedVariableStandardizer.fit(Subset(dataset, [0, 1]))
        self.assertEqual(int(norm.count[0]), 3)
        self.assertAlmostEqual(float(norm.mean[0]), 2)
        self.assertAlmostEqual(float(norm.scale[1]), norm.epsilon)
        x = torch.ones(52, 4)
        x[0, 0] = float("nan")
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[1, :] = False
        output = norm.transform(x, mask)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(float(output[0, 0]), 0)
        self.assertTrue((output[1] == 0).all())

    def test_missing_variable_rejected_and_json_roundtrip(self):
        dataset = windows()
        dataset.runs[0].values[:, 0] = float("nan")
        with self.assertRaises(ValueError):
            FixedVariableStandardizer.fit(Subset(dataset, [0, 1]))
        norm = scaler()
        restored = FixedVariableStandardizer.from_dict(json.loads(json.dumps(norm.to_dict())))
        self.assertEqual(norm.fingerprint, restored.fingerprint)
        self.assertFalse(torch.allclose(norm.transform(torch.full((52, 30), 10.0), torch.ones(52, 30).bool()),
                                        norm.transform(torch.full((52, 30), 100.0), torch.ones(52, 30).bool())))


class RepresentationTests(unittest.TestCase):
    def test_nonfinite_values_remain_missing_after_scaling_and_shifting(self):
        config, norm = enhanced_config(temporal=True)
        original = sample(training=True)
        original.observed_mask[:] = True
        original.x[0, 4] = float("nan")
        original.x[41, 9] = float("inf")
        original.x[2, :] = float("nan")
        valid = original.observed_mask & torch.isfinite(original.x)
        graph = build_tep_hetero_graph(original, config["graph"], standardizer=norm)
        expected = window_statistics(norm.transform(original.x, valid), valid, config["graph"]["statistics"])
        shifted, shifted_mask = _shift_window_for_negative(original.x, valid, 3)
        negative = window_statistics(norm.transform(shifted, shifted_mask), shifted_mask,
                                     config["graph"]["statistics"])
        for node_type, positions in (("sensor", slice(0, 41)), ("actuator", slice(41, 52))):
            self.assertTrue(torch.equal(graph[node_type].input_mask, valid[positions]))
            self.assertTrue(torch.equal(graph[node_type].observed_mask, original.observed_mask[positions]))
            self.assertTrue(torch.allclose(graph[node_type].x, expected[positions]))
            self.assertTrue(torch.allclose(graph[node_type].temporal_negative_x, negative[positions]))
            self.assertTrue(torch.equal(graph[node_type].temporal_negative_mask, shifted_mask[positions]))
        encoded = SharedVariableTCN()(graph["sensor"].input_window, graph["sensor"].input_mask)
        self.assertTrue(torch.isfinite(encoded).all())
        self.assertTrue((encoded[2] == 0).all())

    def test_fixed_statistics_retain_absolute_level(self):
        config, norm = enhanced_config()
        graphs = []
        for level in (10.0, 100.0):
            value = sample()
            value.x[:] = level
            value.observed_mask[:] = True
            graphs.append(build_tep_hetero_graph(value, config["graph"], standardizer=norm))
        for node_type in ("sensor", "actuator"):
            # The retained mean changes by (100 - 10) / scale=2.
            delta = graphs[1][node_type].x[:, 0] - graphs[0][node_type].x[:, 0]
            self.assertTrue(torch.allclose(delta, torch.full_like(delta, 45.0)))

    def test_raw_quality_inputs_preserved_and_negative_windows_shifted(self):
        config, norm = enhanced_config(temporal=True)
        original = sample(training=True)
        graph = build_tep_hetero_graph(original, config["graph"], standardizer=norm)
        self.assertTrue(torch.equal(graph["sensor"].raw_window, original.x[:41]))
        self.assertTrue(torch.equal(graph["actuator"].raw_window, original.x[41:]))
        shifted, mask = _shift_window_for_negative(original.x, original.observed_mask, 3)
        normalized = norm.transform(shifted, mask)
        self.assertTrue(torch.equal(graph["sensor"].temporal_negative_window, normalized[:41]))
        self.assertTrue(torch.equal(graph["actuator"].temporal_negative_window, normalized[41:]))
        self.assertTrue(torch.equal(graph["sensor"].temporal_negative_mask, mask[:41]))

    def test_identity_and_temporal_inputs_survive_batch_adapter(self):
        config, norm = enhanced_config(temporal=True)
        graphs = [build_tep_hetero_graph(sample(True), config["graph"], standardizer=norm) for _ in range(2)]
        batch = Batch.from_data_list(graphs)
        restored = heterodata_from_inputs(extended_x_dict(batch), batch.edge_index_dict)
        self.assertTrue(torch.equal(restored["sensor"].variable_id, torch.arange(41).repeat(2)))
        self.assertTrue(torch.equal(restored["actuator"].variable_id, torch.arange(41, 52).repeat(2)))
        self.assertEqual(restored.scaler_fingerprint, [norm.fingerprint, norm.fingerprint])
        for node_type in ("sensor", "actuator"):
            for key in ("raw_window", "observed_mask", "input_window", "input_mask", "unit_id", "temporal_negative_window", "temporal_negative_mask"):
                self.assertTrue(torch.equal(restored[node_type][key], batch[node_type][key]))

    def test_all_switches_off_preserve_initial_state(self):
        config = configuration()["models"]["qc_hgt"]
        torch.manual_seed(11)
        default = QCHGT(21, config)
        explicit = deepcopy(config)
        explicit["representation"] = {"fixed_standardization": False, "identity": False, "temporal": False}
        torch.manual_seed(11)
        disabled = QCHGT(21, explicit)
        self.assertEqual(list(default.state_dict()), list(disabled.state_dict()))
        for key, value in default.state_dict().items():
            self.assertTrue(torch.equal(value, disabled.state_dict()[key]))

    def test_wrong_scaler_rejected_before_forward(self):
        config, norm = enhanced_config()
        graph = build_tep_hetero_graph(sample(), config["graph"], standardizer=norm)
        graph.scaler_fingerprint = "wrong"
        model = QCHGT(21, config["models"]["qc_hgt"])
        with self.assertRaises(ValueError):
            model(graph)

    def test_d_forward_backward_and_state_reload(self):
        config, norm = enhanced_config(temporal=True)
        graph = build_tep_hetero_graph(sample(True), config["graph"], standardizer=norm)
        model = QCHGT(21, config["models"]["qc_hgt"])
        output = model(graph)
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        gradients = [parameter.grad for parameter in model.temporal_encoder.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        model.eval()
        restored = QCHGT(21, config["models"]["qc_hgt"]).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(graph)["logits"], restored(graph)["logits"]))

    def test_physical_node_permutation_preserves_prediction(self):
        config, norm = enhanced_config(temporal=True)
        graph = build_tep_hetero_graph(sample(), config["graph"], standardizer=norm)
        permuted = graph.clone()
        order = torch.arange(40, -1, -1)
        inverse = torch.argsort(order)
        for key, value in graph["sensor"].items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 41:
                permuted["sensor"][key] = value[order]
        for edge_type in graph.edge_types:
            edges = graph[edge_type].edge_index.clone()
            if edge_type[0] == "sensor":
                edges[0] = inverse[edges[0]]
            if edge_type[2] == "sensor":
                edges[1] = inverse[edges[1]]
            permuted[edge_type].edge_index = edges
        model = QCHGT(21, config["models"]["qc_hgt"]).eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(graph)["logits"], model(permuted)["logits"], atol=1e-5, rtol=1e-5))

    def test_same_observations_keep_distinct_physical_embeddings(self):
        config, norm = enhanced_config()
        value = sample()
        value.x[:] = 1
        value.observed_mask[:] = True
        graph = build_tep_hetero_graph(value, config["graph"], standardizer=norm)
        graph["sensor"].x[1] = graph["sensor"].x[0]
        torch.manual_seed(10)
        model = QCHGT(21, config["models"]["qc_hgt"]).eval()
        features = model._encode_inputs(graph)["sensor"]
        self.assertFalse(torch.allclose(features[0], features[1]))


class TemporalAndProvenanceTests(unittest.TestCase):
    def test_enhanced_checkpoint_restores_preprocessing_and_prediction(self):
        config, norm = enhanced_config(temporal=True)
        model = build_model("qc_hgt", 0, 21, config).eval()
        optimizer = torch.optim.AdamW(model.parameters())
        metadata = {"schema_version": "qc_hgt_input_ablation.v1",
                    "experiment_fingerprint": "synthetic", "group": "D", "num_classes": 21,
                    "experiment": config, "scaler": norm.to_dict()}
        graph = build_tep_hetero_graph(sample(), config["graph"], standardizer=norm)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, model, optimizer, 1, {}, metadata=metadata)
            restored, restored_scaler, restored_metadata = load_input_checkpoint(path)
            restored_graph = build_tep_hetero_graph(sample(), restored_metadata["experiment"]["graph"],
                                                   standardizer=restored_scaler)
            self.assertEqual(norm.fingerprint, restored_scaler.fingerprint)
            with torch.no_grad():
                before = model(extended_x_dict(graph), graph.edge_index_dict)["logits"]
                after = restored(extended_x_dict(restored_graph), restored_graph.edge_index_dict)["logits"]
            self.assertTrue(torch.allclose(before, after))

    def test_checkpoint_rejects_inconsistent_graph_configuration_before_building_model(self):
        config, norm = enhanced_config(temporal=True)
        metadata = {"schema_version": "qc_hgt_input_ablation.v1",
                    "experiment_fingerprint": "synthetic", "group": "D", "num_classes": 21,
                    "experiment": config, "scaler": norm.to_dict()}
        metadata["experiment"]["graph"]["representation"]["temporal"] = False
        with patch("scripts.run_input_ablation.torch.load", return_value={"metadata": metadata}), \
             patch("scripts.run_input_ablation.build_model") as build:
            with self.assertRaisesRegex(ValueError, "switches disagree"):
                load_input_checkpoint(Path("unused.pt"))
            build.assert_not_called()

    def test_missing_gap_is_not_a_derivative_spike(self):
        inputs, _ = SharedVariableTCN.inputs(torch.tensor([[1., 100., 3., 4.]]),
                                            torch.tensor([[True, False, True, True]]))
        self.assertTrue(torch.equal(inputs[0, 2], torch.tensor([0., 0., 0., 1.])))
        model = SharedVariableTCN()
        self.assertTrue((model(torch.ones(2, 30), torch.zeros(2, 30).bool()) == 0).all())

    def test_temporal_convolutions_do_not_read_future_positions(self):
        encoder = SharedVariableTCN()
        original = torch.arange(30, dtype=torch.float32)[None, :]
        changed = original.clone()
        changed[:, 20:] = 1000
        mask = torch.ones_like(original, dtype=torch.bool)
        left, _ = encoder.inputs(original, mask)
        right, _ = encoder.inputs(changed, mask)
        for layer, padding in zip(encoder.convolutions, encoder.left_padding):
            left = F.relu(layer(F.pad(left, (padding, 0))))
            right = F.relu(layer(F.pad(right, (padding, 0))))
        self.assertTrue(torch.allclose(left[..., :20], right[..., :20]))

    def test_checkpoint_fingerprint_guard_and_metadata_roundtrip(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        metadata = {"experiment_fingerprint": "correct", "scaler": scaler().to_dict()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, 1, {"macro_f1": 0.1}, metadata=metadata)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, torch.device("cpu"), expected_fingerprint="wrong")
            payload = load_checkpoint(path, model, torch.device("cpu"), expected_fingerprint="correct")
            self.assertEqual(payload["metadata"], metadata)

    def test_source_or_config_changes_invalidate_manifest(self):
        left = {"experiment_fingerprint": fingerprint({"source": "a", "batch": 128})}
        right = {"experiment_fingerprint": fingerprint({"source": "b", "batch": 128})}
        with self.assertRaises(ValueError):
            assert_matching_manifest(left, right)

    def test_four_groups_share_data_and_training_budget(self):
        plans = [input_plan(ROOT / "configs" / "input_ablation.yaml", group) for group in "ABCD"]
        for _, experiment, data_config, plan in plans:
            self.assertEqual(plan["scope"], "validation_only")
            self.assertFalse(plan["test_evaluation"])
            self.assertEqual(experiment["training"], plans[0][1]["training"])
            self.assertEqual(experiment["sampling"], plans[0][1]["sampling"])
            self.assertEqual(data_config, plans[0][2])

    def test_cli_default_never_loads_data_or_trains(self):
        with patch("sys.argv", ["run_input_ablation.py", "--group", "D"]), \
             patch("scripts.run_input_ablation.run_input_experiment") as run, \
             patch("scripts.run_input_ablation.build_tep_datasets_from_config") as load, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            main()
            run.assert_not_called()
            load.assert_not_called()
            self.assertIn("disabled", output.getvalue())


if __name__ == "__main__":
    unittest.main()
