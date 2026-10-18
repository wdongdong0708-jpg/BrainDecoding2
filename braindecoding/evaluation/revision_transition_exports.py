"""Export the completed Pallier2025 context and revision studies for paper plots.

The canonical evaluation JSONs remain the result source; this module checks
their shared evaluation contract before writing a flat CSV to reports/exports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from braindecoding.config import PROJECT_ROOT


DATASET = "Pallier2025"
VOCABULARY_SIZE = 50
METRIC = "macro_top10"
SEEDS = (0, 1, 2)
SUBJECT_SCOPE = "sub01-10"
QUERY_COUNT = 16960
RUN_ROOT = PROJECT_ROOT / "outputs" / "word_decoding" / "pallier2025"
EXPORT_PATH = PROJECT_ROOT / "data/results" / "exports" / "context_revision_curves.csv"
SOURCE_PATH = EXPORT_PATH.with_suffix(".sources.json")
FIELDS = (
    "dataset", "split", "analysis", "metric", "vocabulary_size", "target_set",
    "method", "x", "mean", "sd", "n", "seeds", "seed_values", "event_table_sha256",
    "vocabulary_manifest_sha256", "query_count", "effective_query_count",
)


def _run_dir(group: str, experiment_id: str, seed: int) -> Path:
    if group == "full":
        return RUN_ROOT / SUBJECT_SCOPE / experiment_id / f"seed-{seed:03d}"
    return RUN_ROOT / group / SUBJECT_SCOPE / experiment_id / f"seed-{seed:03d}"


def _load_run(group: str, experiment_id: str, seed: int, reference: dict | None):
    run_dir = _run_dir(group, experiment_id, seed)
    evaluation_path = run_dir / "evaluation" / "val.json"
    summary_path = run_dir / "training_summary.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = {
        "task": "word_decoding", "dataset": "pallier2025",
        "subject_scope": SUBJECT_SCOPE, "experiment_id": experiment_id, "seed": seed,
    }
    if evaluation["status"] != "evaluation_completed" or summary["status"] != "training_completed":
        raise ValueError(f"Run is incomplete: {run_dir}")
    if evaluation["experiment"] != identity or summary["experiment"] != identity:
        raise ValueError(f"Run identity mismatch: {run_dir}")
    if evaluation["data"]["split"] != "val" or evaluation["data"]["query_count"] != QUERY_COUNT:
        raise ValueError(f"Evaluation split or query count mismatch: {run_dir}")
    if summary["checkpoint"]["best"]["sha256"] != evaluation["checkpoint"]["sha256"]:
        raise ValueError(f"Checkpoint mismatch: {run_dir}")
    if summary["selection"]["split"] != "val" or summary["training"]["budget"] != {
        "type": "epochs", "value": 30
    }:
        raise ValueError(f"Training contract mismatch: {run_dir}")
    expected_metric = (
        "revision_common_delay2_retrieval_acc10_vocab=pallier2025_50_macro"
        if group == "rolling_revision"
        else "retrieval_acc10_vocab=pallier2025_50_macro"
    )
    if summary["selection"]["metric"] != expected_metric:
        raise ValueError(f"Checkpoint selection metric mismatch: {run_dir}")
    if summary["details"]["validation_rows"] != QUERY_COUNT:
        raise ValueError(f"Training validation set mismatch: {run_dir}")
    initialization = summary["details"]["initialization_audit"]
    if initialization["method"] not in {
        "cnn_only_best_checkpoint_brain_encoder_only",
        "main_word_best_checkpoint_brain_encoder_only",
    } or initialization["loaded_loss_state"] or initialization["loaded_transformer_state"]:
        raise ValueError(f"Warm-start method mismatch: {run_dir}")
    block = evaluation["vocabularies"][str(VOCABULARY_SIZE)]
    if block["candidate_count"] != VOCABULARY_SIZE or block["retrieval"][METRIC] is None:
        raise ValueError(f"Missing frozen vocabulary result: {run_dir}")
    if block["support"]["supported_candidate_count"] != VOCABULARY_SIZE:
        raise ValueError(f"50-word result has unsupported candidates: {run_dir}")
    fingerprint = {
        "event_table_sha256": evaluation["data"]["event_table_sha256"],
        "subjects": evaluation["data"]["subjects"],
        "vocabulary_manifest_sha256": block["manifest_sha256"],
        "warm_start_sha256": initialization["source_checkpoint_sha256"],
        "effective_query_count": summary["details"]["best_validation"]["query_count"],
    }
    if reference is not None and fingerprint != reference:
        raise ValueError(f"Source runs have incompatible evaluation contracts: {run_dir}")
    return evaluation, fingerprint, {
        "experiment": experiment_id,
        "seed": seed,
        "evaluation": str(evaluation_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": evaluation["checkpoint"]["sha256"],
    }


def _score(evaluation: dict) -> float:
    value = float(evaluation["vocabularies"][str(VOCABULARY_SIZE)]["retrieval"][METRIC])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid Macro Top-10 value in canonical evaluation.")
    return value


def _add(rows: list[dict], reference: dict, analysis: str, method: str, x: int | str,
         scores: list[tuple[int, float]]) -> None:
    values = [value for _, value in scores]
    if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError(f"Invalid scores for {analysis}/{method}/{x}.")
    rows.append({
        "dataset": DATASET, "split": "val", "analysis": analysis, "metric": METRIC,
        "vocabulary_size": VOCABULARY_SIZE, "target_set": "all_target",
        "method": method, "x": x, "mean": mean(values),
        "sd": stdev(values) if len(values) > 1 else "", "n": len(values),
        "seeds": ";".join(str(seed) for seed, _ in scores),
        "seed_values": ";".join(str(value) for _, value in scores),
        "event_table_sha256": reference["event_table_sha256"],
        "vocabulary_manifest_sha256": reference["vocabulary_manifest_sha256"],
        "query_count": QUERY_COUNT,
        "effective_query_count": reference["effective_query_count"],
    })


def collect_rows() -> tuple[list[dict], dict]:
    """Validate all runs before returning any rows or writing outputs."""
    reference = None
    sources = []
    cache = {}

    def get(group: str, experiment: str, seed: int) -> dict:
        nonlocal reference
        key = group, experiment, seed
        if key not in cache:
            evaluation, fingerprint, source = _load_run(group, experiment, seed, reference)
            reference = fingerprint
            cache[key] = evaluation
            sources.append(source)
        return cache[key]

    rows = []
    for radius in (0, 1, 2, 4, 8):
        scores = [(seed, _score(get("local_window", f"local_window_r{radius}", seed)))
                  for seed in SEEDS]
        _add(rows, reference, "context_window", "local", radius, scores)
    full_scores = [(seed, _score(get("full", "main_context_warmstart", seed)))
                   for seed in SEEDS]
    _add(rows, reference, "context_window", "full", "", full_scores)

    # The directed experiments currently exist only for seed 0. The ±2
    # condition is the independently trained local-window r2 run for seed 0.
    directed = [
        (0, _score(get("directed_local_window", "local_window_m2_p0", 0))),
        (1, _score(get("directed_local_window", "local_window_m2_p1", 0))),
        (2, _score(get("local_window", "local_window_r2", 0))),
    ]
    for delay, value in directed:
        _add(rows, reference, "future_increment", "independent", delay, [(0, value)])

    rolling_by_seed = {}
    for seed in SEEDS:
        evaluation = get("rolling_revision", "local_window_rolling_revision_k2_d2", seed)
        revision = evaluation["rolling_revision"]
        if (revision["protocol"] != "local_window_physical_truncation"
                or revision["history_radius"] != 2
                or revision["maximum_delay"] != 2
                or revision["shared_transformer"] is not True
                or revision["common_target_policy"] != "target_has_all_future_events_through_maximum_delay"):
            raise ValueError(f"Rolling protocol mismatch for seed {seed}.")
        rolling_by_seed[seed] = revision
    for delay in (0, 1, 2):
        scores = []
        for seed in SEEDS:
            block = rolling_by_seed[seed]["horizons"][str(delay)]
            if block["relative_window"] != [-2, delay]:
                raise ValueError(f"Rolling horizon mismatch for seed {seed}, delay {delay}.")
            target = block["all_targets"]
            if target["event_count"] != QUERY_COUNT:
                raise ValueError(f"Rolling all-target count mismatch for seed {seed}, delay {delay}.")
            vocabulary = target["vocabularies"][str(VOCABULARY_SIZE)]
            if vocabulary["manifest_sha256"] != reference["vocabulary_manifest_sha256"]:
                raise ValueError(f"Rolling vocabulary mismatch for seed {seed}, delay {delay}.")
            scores.append((seed, float(vocabulary["retrieval"][METRIC])))
        _add(rows, reference, "rolling_revision", "rolling", delay, scores)
    _add(rows, reference, "rolling_revision", "offline_local", "",
         [(seed, _score(get("local_window", "local_window_r2", seed))) for seed in SEEDS])
    _add(rows, reference, "rolling_revision", "full", "", full_scores)
    return rows, {"schema_version": 1, "dataset": DATASET, "split": "val",
                  "metric": METRIC, "vocabulary_size": VOCABULARY_SIZE,
                  "target_set": "all_target", "fingerprint": reference, "sources": sources}


def main() -> tuple[Path, Path]:
    rows, manifest = collect_rows()
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXPORT_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    SOURCE_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return EXPORT_PATH, SOURCE_PATH


if __name__ == "__main__":
    for path in main():
        print(path)

# --- Merged from pallier_revision_event_exports.py ---
"""Export paired Top-10 word-event transitions for Pallier rolling revision.

The canonical JSONs contain summary metrics but no per-event decisions. This
module reruns inference from their exact checkpoints and checks every derived
metric against the saved evaluation before exporting transition counts.
"""


import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from braindecoding.config import PROJECT_ROOT
from braindecoding.evaluation.retrieval import fixed_vocabulary_retrieval
from braindecoding.models import build_brain_embedding_model
from braindecoding.tasks.word_decoding.pallier2025 import train as training
from braindecoding.training.runtime import load_checkpoint
from braindecoding.training.word import encode_loader


EXPORT_PATH = PROJECT_ROOT / "data/results" / "exports" / "rolling_revision_transitions.csv"
SOURCE_PATH = EXPORT_PATH.with_suffix(".sources.json")
EXPERIMENT = "local_window_rolling_revision_k2_d2"
FIELDS = (
    "dataset", "split", "vocabulary_size", "top_k", "target_set", "seed",
    "from_delay", "to_delay", "eligible_word_events", "wrong_to_right",
    "right_to_wrong", "right_to_right", "wrong_to_wrong", "macro_before",
    "macro_after", "micro_before", "micro_after", "evaluation_sha256",
    "checkpoint_sha256",
)


def count_transitions(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    """Partition one paired set of Boolean Top-10 decisions."""
    before = np.asarray(before, dtype=bool)
    after = np.asarray(after, dtype=bool)
    if before.ndim != 1 or after.shape != before.shape or len(before) == 0:
        raise ValueError("Paired decisions must be nonempty, aligned 1D arrays.")
    return {
        "eligible_word_events": len(before),
        "wrong_to_right": int((~before & after).sum()),
        "right_to_wrong": int((before & ~after).sum()),
        "right_to_right": int((before & after).sum()),
        "wrong_to_wrong": int((~before & ~after).sum()),
    }


def _assert_metric(actual: float, expected: float, name: str) -> None:
    # Re-evaluating CUDA checkpoints can move a few borderline Top-10 queries
    # across the cutoff; reject protocol-scale drift, not harmless FP jitter.
    if not np.isclose(actual, expected, rtol=0, atol=1e-3):
        raise ValueError(f"Recomputed {name} differs from canonical evaluation: {actual} vs {expected}.")


def _compute_seed(seed: int, evaluation: dict, source: dict) -> list[dict]:
    run_dir = _run_dir("rolling_revision", EXPERIMENT, seed)
    config = training.load_config(run_dir / "resolved_config.yaml")
    if config["experiment"]["id"] != EXPERIMENT or config["training"]["seed"] != seed:
        raise ValueError(f"Resolved configuration has wrong experiment identity: {run_dir}")
    training.set_seed(seed)
    device = training.choose_device(config["training"].get("device", "auto"))
    event_path = training.ensure_canonical_inputs(config)
    table = training.dataset_module.load_event_table(
        event_path, split="val", subjects=config["dataset"]["subjects"], trainable_only=True,
    )
    dataset = training.build_dataset(config, table)
    loader, _ = training.make_loader(
        dataset, config["training"], shuffle=False,
        group_column=training.loader_group_column(config),
    )
    checkpoint_path = run_dir / "best.pt"
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != evaluation["checkpoint"]["sha256"]:
        raise ValueError(f"Checkpoint SHA differs from canonical evaluation: {run_dir}")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("task") != "word_decoding/Pallier2025":
        raise ValueError(f"Wrong checkpoint task: {run_dir}")
    if checkpoint.get("text_embedding_config") != config["text_embedding"]:
        raise ValueError(f"Text embedding contract mismatch: {run_dir}")
    if tuple(checkpoint.get("channel_names", ())) != tuple(dataset.channel_names):
        raise ValueError(f"Channel contract mismatch: {run_dir}")
    model = build_brain_embedding_model(
        dataset.channel_count, dataset.channel_positions, dataset.subject_count,
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    encoded = encode_loader(model, loader, device, amp=config["training"].get("amp", True))
    del model
    if len(encoded["words"]) != QUERY_COUNT or set(encoded["revision_predictions"]) != {0, 1, 2}:
        raise ValueError(f"Missing evaluation rows or revision delays: {run_dir}")
    common = encoded["revision_availability_masks"][2].numpy().astype(bool)
    if int(common.sum()) != evaluation["rolling_revision"]["horizons"]["0"]["common_targets"]["event_count"]:
        raise ValueError(f"Common target count differs from canonical evaluation: {run_dir}")

    words = np.asarray(encoded["words"])[common].tolist()
    targets = encoded["targets"][common]
    vocabulary = training.frozen_vocabulary(50)
    hits = {}
    metrics = {}
    canonical_macro = {}
    selected_reference = None
    for delay in (0, 1, 2):
        block = evaluation["rolling_revision"]["horizons"][str(delay)]["common_targets"]["vocabularies"]["50"]
        summary, ranks, selected = fixed_vocabulary_retrieval(
            encoded["revision_predictions"][delay][common], targets, words,
            vocabulary, top_ks=(1, 10), vocabulary_name="pallier2025_50",
        )
        if selected_reference is None:
            selected_reference = selected
        elif not np.array_equal(selected, selected_reference):
            raise ValueError(f"Vocabulary query alignment changed across delays: {run_dir}")
        _assert_metric(summary["macro_recall_at_10"], block["retrieval"]["macro_top10"], f"seed {seed} K{delay} macro")
        _assert_metric(summary["micro_recall_at_10"], block["retrieval"]["top10"], f"seed {seed} K{delay} micro")
        hits[delay] = ranks <= 10
        metrics[delay] = summary
        canonical_macro[delay] = block["retrieval"]["macro_top10"]

    rows = []
    for before, after in ((0, 1), (1, 2)):
        counts = count_transitions(hits[before], hits[after])
        if counts["wrong_to_right"] - counts["right_to_wrong"] != int(hits[after].sum() - hits[before].sum()):
            raise ValueError("Transition net gain is inconsistent with paired hits.")
        rows.append({
            "dataset": "Pallier2025", "split": "val", "vocabulary_size": 50,
            "top_k": 10, "target_set": "common_delay2", "seed": seed,
            "from_delay": before, "to_delay": after, **counts,
            "macro_before": canonical_macro[before],
            "macro_after": canonical_macro[after],
            "micro_before": metrics[before]["micro_recall_at_10"],
            "micro_after": metrics[after]["micro_recall_at_10"],
            "evaluation_sha256": source["evaluation_sha256"],
            "checkpoint_sha256": evaluation["checkpoint"]["sha256"],
        })
    return rows


def main() -> tuple[Path, Path]:
    rows = []
    sources = []
    reference = None
    for seed in SEEDS:
        evaluation, fingerprint, source = _load_run("rolling_revision", EXPERIMENT, seed, reference)
        reference = fingerprint
        print(f"Recomputing paired word events: seed {seed}", flush=True)
        rows.extend(_compute_seed(seed, evaluation, source))
        sources.append(source)
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXPORT_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    SOURCE_PATH.write_text(json.dumps({
        "schema_version": 1, "event_definition": "paired Top-10 retrieval hit",
        "target_set": "common_delay2", "vocabulary_size": 50,
        "curve_metric_source": "canonical_evaluation",
        "transition_source": "checkpoint_reinference",
        "maximum_allowed_recomputed_metric_drift": 0.001,
        "fingerprint": reference, "sources": sources,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return EXPORT_PATH, SOURCE_PATH


if __name__ == "__main__":
    for path in main():
        print(path)
