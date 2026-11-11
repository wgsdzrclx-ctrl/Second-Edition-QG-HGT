"""Isolated A/B/C/D validation-only experiments, requiring explicit --execute.

This entry point deliberately does not call the Stage-5 test evaluation loop.
Existing stage2/stage5 configurations continue to describe the original model.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from data.feature_dataset import deterministic_subset  # noqa: E402
from data.standardization import (  # noqa: E402
    FixedVariableStandardizer, observation_fingerprint, selection_manifest,
)
from data.tep_dataset import build_tep_datasets_from_config  # noqa: E402
from data.tep_hetero_graph import TEPGraphDataset  # noqa: E402
from models import build_model  # noqa: E402
from scripts.run_protocol import (  # noqa: E402
    _device, _experiment_and_data, _load_yaml, _loader, _resolve, _write_json,
)
from train import fit  # noqa: E402
from utils import create_logger, set_seed  # noqa: E402
from utils.experiment_manifest import (  # noqa: E402
    assert_matching_manifest, fingerprint, source_manifest,
)


GROUP_FLAGS = {
    "A": (False, False, False), "B": (True, False, False),
    "C": (True, True, False), "D": (True, True, True),
}
FLAG_NAMES = ("fixed_standardization", "identity", "temporal")


def input_plan(config_path: Path, group: str, protocol: str | None = None,
               seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read configuration only; safe to inspect before approving execution."""
    path = config_path.expanduser().resolve()
    direction = _load_yaml(path)
    selected_protocol = str(protocol or direction["default_protocol"])
    if selected_protocol not in ("protocol_1", "protocol_2"):
        raise ValueError("Only P1 and P2 are supported.")
    selected_seed = int(direction["default_seed"] if seed is None else seed)
    stage5_path = _resolve(path.parent, str(direction["stage5_config"]))
    _, experiment, data_config = _experiment_and_data(stage5_path, selected_protocol)
    if data_config["dataset"]["variant"] != "extended500":
        raise ValueError("The reviewed first-stage experiments use the 21-class extended500 dataset.")
    if selected_seed not in data_config["experiment"]["seeds"]:
        raise ValueError("Seed is not in the configured five-seed set.")
    flags = direction["groups"][group]
    if any(type(flags.get(name)) is not bool for name in FLAG_NAMES):
        raise ValueError("Representation switches must be YAML booleans.")
    if tuple(flags[name] for name in FLAG_NAMES) != GROUP_FLAGS[group]:
        raise ValueError("Groups must follow A -> B (scaling) -> C (+ID) -> D (+TCN).")
    representation = {**deepcopy(direction["representation"]), **flags}
    experiment["models"]["qc_hgt"]["representation"] = representation
    if flags["fixed_standardization"]:
        experiment["models"]["qc_hgt"]["input_normalization"] = "none"
    experiment["graph"]["representation"] = dict(flags)
    experiment["training"]["record_diagnostics"] = True
    if experiment["graph"].get("homogeneous_ablation", False):
        raise ValueError("Input experiments require the heterogeneous process graph.")
    # These changes would silently change the run-grouped corruption protocol.
    if int(experiment["training"]["num_workers"]) != 0:
        raise ValueError("First-stage runs retain num_workers=0 for deterministic epoch corruption.")
    root = _resolve(path.parent, str(direction["output_directory"]))
    original_results = (PROJECT_ROOT / "results" / "stage5").resolve()
    if root == original_results or original_results in root.parents:
        raise ValueError("Use a separate results directory; the Stage-5 baseline is protected.")
    plan = {
        "group": group, "protocol": selected_protocol, "seed": selected_seed,
        "scope": "validation_only", "test_evaluation": False,
        "run_directory": str(root / selected_protocol / group / f"seed_{selected_seed}"),
        "representation": representation,
        "train_windows": experiment["sampling"]["max_samples_per_split"]["train"],
        "validation_windows": experiment["sampling"]["max_samples_per_split"]["val"],
        "epochs": experiment["training"]["epochs"],
        "batch_size": experiment["training"]["batch_size"],
        "window_length": data_config["window"]["length"],
        "device": experiment["runtime"]["device"],
    }
    return direction, experiment, data_config, plan


def _selected(dataset: Any, split: str, experiment: dict[str, Any]) -> Any:
    sampling = experiment["sampling"]
    return deterministic_subset(
        dataset, sampling["max_samples_per_split"][split],
        int(sampling["base_seed"]) + int(sampling["seed_offsets"][split]),
        str(sampling.get("strategy", "uniform_windows")), sampling.get("windows_per_run"),
    )


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_input_experiment(config_path: Path, group: str, protocol: str | None = None,
                         seed: int | None = None, *, resume: bool = False) -> Path:
    direction, experiment, data_config, plan = input_plan(config_path, group, protocol, seed)
    run_directory = Path(plan["run_directory"])
    if run_directory.exists() and not resume:
        raise FileExistsError("Output directory already exists. Use --resume to verify a completed run, or select a new root.")
    selected_seed = plan["seed"]
    set_seed(selected_seed, data_config["experiment"]["seeds"],
             bool(experiment["runtime"]["deterministic"]),
             bool(experiment["runtime"]["deterministic_warn_only"]))
    torch.set_num_threads(int(experiment["runtime"]["cpu_threads"]))
    device = _device(str(experiment["runtime"]["device"]))
    # The existing builder checks run separation, including held-out run metadata.
    # Only train/val observations are selected/read here; no test loader is built.
    datasets = build_tep_datasets_from_config(data_config, selected_seed, print_leakage_report=True)
    selected = {split: _selected(datasets[split], split, experiment) for split in ("train", "val")}
    selections = {split: selection_manifest(dataset) for split, dataset in selected.items()}
    selected_content = {split: observation_fingerprint(dataset) for split, dataset in selected.items()}
    use_scaler = plan["representation"]["fixed_standardization"]
    scaler = FixedVariableStandardizer.fit(selected["train"], float(direction["standardization"]["epsilon"])) if use_scaler else None
    if scaler is not None:
        experiment["models"]["qc_hgt"]["representation"]["scaler_fingerprint"] = scaler.fingerprint
    graph_datasets = {
        split: TEPGraphDataset(dataset, experiment["graph"], homogeneous=False, standardizer=scaler)
        for split, dataset in selected.items()
    }
    sources = source_manifest(PROJECT_ROOT)
    environment = {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "torch-geometric", "numpy", "pyarrow", "PyYAML")},
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    scaler_state = scaler.to_dict() if scaler else None
    identity = {
        "schema_version": "qc_hgt_input_ablation.v1", "scope": "validation_only",
        "group": group, "protocol": plan["protocol"], "seed": selected_seed,
        "experiment": experiment, "data_config": data_config,
        "selection_sha256": {split: item["sha256"] for split, item in selections.items()},
        "selected_observations_sha256": selected_content,
        "scaler": scaler_state, "source_sha256": sources["sha256"], "environment": environment,
    }
    manifest = {**identity, "experiment_fingerprint": fingerprint(identity),
                "sources": sources, "selections": selections}
    result_path = run_directory / "result.json"
    checkpoint = run_directory / "best.pt"
    if run_directory.exists():
        manifest_path = run_directory / "manifest.json"
        if not manifest_path.exists() or not result_path.exists() or not checkpoint.exists():
            raise RuntimeError("Incomplete run: automatic best-checkpoint continuation is disabled. Use a new output root.")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert_matching_manifest(previous, manifest)
        assert_matching_manifest(result, manifest)
        if not result.get("complete"):
            raise RuntimeError("Incomplete run cannot be silently resumed as a completed experiment.")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert_matching_manifest(payload.get("metadata", {}), manifest)
        if _checkpoint_hash(checkpoint) != result.get("checkpoint_sha256"):
            raise ValueError("Checkpoint contents changed since the completed result was saved.")
        if payload["metadata"].get("scaler") != scaler_state:
            raise ValueError("Checkpoint scaler does not match the selected training data.")
        print(f"[verified completed run] {result_path}")
        return result_path

    # Atomic directory creation prevents concurrent processes overwriting a run.
    run_directory.mkdir(parents=True, exist_ok=False)
    _write_json(run_directory / "manifest.json", manifest, 2)
    _write_json(run_directory / "effective_config.json", {"experiment": experiment, "data": data_config}, 2)
    if scaler_state is not None:
        _write_json(run_directory / "scaler.json", scaler_state, 2)
    logger = create_logger(f"input_ablation.{plan['protocol']}.{group}.{selected_seed}",
                           run_directory / "training.log", str(experiment["logging"]["level"]))
    variant = str(data_config["dataset"]["variant"])
    num_classes = int(data_config["dataset"]["variants"][variant]["num_classes"])
    model = build_model("qc_hgt", 0, num_classes, experiment)
    metadata = {
        "schema_version": identity["schema_version"],
        "experiment_fingerprint": manifest["experiment_fingerprint"],
        "group": group, "protocol": plan["protocol"], "seed": selected_seed,
        "experiment": experiment, "data_config": data_config, "scaler": scaler_state,
        "source_sha256": sources["sha256"], "num_classes": num_classes,
        "selection_sha256": identity["selection_sha256"],
        "selected_observations_sha256": selected_content, "environment": environment,
    }
    result = {
        "schema_version": identity["schema_version"], "scope": "validation_only",
        "experiment_fingerprint": manifest["experiment_fingerprint"],
        "group": group, "protocol": plan["protocol"], "seed": selected_seed,
        "complete": False, "test_evaluated": False,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint),
    }
    _write_json(result_path, result, 2)
    started = time.perf_counter()
    trained = fit(model, _loader(graph_datasets["train"], "train", selected_seed, experiment),
                  _loader(graph_datasets["val"], "val", selected_seed, experiment),
                  device, num_classes, experiment, checkpoint, logger,
                  checkpoint_metadata=metadata)
    result.update({"complete": True, "best_epoch": trained["best_epoch"],
                   "validation": trained["best_validation"], "actual_epochs": len(trained["history"]),
                   "training_seconds_including_validation": time.perf_counter() - started,
                   "checkpoint_sha256": _checkpoint_hash(checkpoint)})
    _write_json(run_directory / "history.json", {"epochs": trained["history"]}, 2)
    _write_json(result_path, result, 2)
    logger.info("completed validation-only group=%s result=%s", group, result_path)
    return result_path


def load_input_checkpoint(path: Path, device: torch.device | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """Restore model AND preprocessing from one enhanced checkpoint.

    The caller must build TEPGraphDataset with the returned scaler and
    metadata['experiment']['graph']; forward rejects a mismatched scaler.
    No data fitting, training, or test evaluation is performed by this helper.
    """
    device = torch.device("cpu") if device is None else device
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    if not metadata.get("experiment_fingerprint") or "experiment" not in metadata:
        raise ValueError("This checkpoint lacks input-experiment preprocessing metadata.")
    if metadata.get("schema_version") != "qc_hgt_input_ablation.v1":
        raise ValueError("Unsupported input-experiment checkpoint format.")
    state = metadata.get("scaler")
    scaler = FixedVariableStandardizer.from_dict(state) if state is not None else None
    representation = metadata["experiment"]["models"]["qc_hgt"].get("representation", {})
    graph_representation = metadata["experiment"]["graph"].get("representation", {})
    group = metadata.get("group")
    if group not in GROUP_FLAGS:
        raise ValueError("Checkpoint must identify one reviewed A/B/C/D group.")
    for name, expected in zip(FLAG_NAMES, GROUP_FLAGS[group]):
        if representation.get(name) is not expected or graph_representation.get(name) is not expected:
            raise ValueError("Checkpoint graph/model switches disagree with its A/B/C/D group.")
    if bool(representation.get("fixed_standardization", False)) != (scaler is not None):
        raise ValueError("Checkpoint preprocessing configuration and saved scaler disagree.")
    if scaler is not None and representation.get("scaler_fingerprint") != scaler.fingerprint:
        raise ValueError("Checkpoint contains an inconsistent scaler fingerprint.")
    model = build_model("qc_hgt", 0, int(metadata["num_classes"]), metadata["experiment"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, scaler, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "input_ablation.yaml")
    parser.add_argument("--group", required=True, choices=tuple(GROUP_FLAGS))
    parser.add_argument("--protocol", choices=("protocol_1", "protocol_2"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true", help="Verify and reuse a completed matching run only.")
    parser.add_argument("--execute", action="store_true", help="Explicitly permit data fitting and training.")
    args = parser.parse_args()
    if not args.execute:
        _, _, _, plan = input_plan(args.config, args.group, args.protocol, args.seed)
        print(json.dumps({"execution": "disabled; --execute is required", **plan}, ensure_ascii=False, indent=2))
        return
    run_input_experiment(args.config, args.group, args.protocol, args.seed, resume=args.resume)


if __name__ == "__main__":
    main()
