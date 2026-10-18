"""Two-dataset, three-seed training with the accepted 16/16 BGE rules.

The numerical training loop is the legacy ``training.word.train_one_epoch``:
each batch computes SigLIP separately for each available delay and averages
those losses.  This worker deliberately does not use the retired v1 loop.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends, project_path
from braindecoding.data.text import text_embedding_signature
from braindecoding.experiment import file_sha256, run_directory
from braindecoding.losses import build_siglip_loss
from braindecoding.models import build_brain_embedding_model
from braindecoding.optimizers import build_adamw_for_modules, build_cosine_annealing_scheduler
from braindecoding.pallier_alignment_trial import strict_bank_metrics
from braindecoding.pallier_bge_trial import save_json
from braindecoding.reporting import study_report_dir
from braindecoding.tasks.word_decoding.pallier2025 import train as pallier
from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import train as chinese
from braindecoding.training.runtime import cpu_state_dict, load_checkpoint, save_checkpoint, set_seed
from braindecoding.training.word import make_loader, train_one_epoch
from braindecoding.unified_rolling_training import full_vocabulary_metrics


CONFIG_PATH = PROJECT_ROOT / "configs/unified_rolling/v2.yaml"
DATASETS = ("pallier2025", "chineseeeg2_littleprince")
STAGES = ("word", "revision")
COMMON_TRAINING_KEYS = (
    "device", "batch_size", "num_workers", "patience", "learning_rate",
    "weight_decay", "minimum_learning_rate", "max_grad_norm", "amp",
)


def config():
    return load_yaml_with_extends(CONFIG_PATH)


def _identity(settings, dataset, seed, stage):
    if dataset not in DATASETS or dataset not in settings["adapters"]:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if stage not in STAGES or stage not in settings["protocol"]["stages"]:
        raise ValueError(f"Unsupported stage: {stage}")
    if seed not in settings["protocol"]["seeds"] or isinstance(seed, bool):
        raise ValueError(f"Unauthorized seed: {seed}")


def build_config(dataset, seed, stage, word_checkpoint=None):
    """Resolve the legacy dataset adapter and override only the v2 rules."""
    settings = config()
    _identity(settings, dataset, seed, stage)
    adapter = settings["adapters"][dataset]
    loader = pallier.load_config if dataset == "pallier2025" else chinese.load_config
    cfg = loader(project_path(adapter[f"{stage}_source_config"]))
    cfg["experiment"]["output_group"] = settings["execution"]["output_group"]
    cfg["model"] = copy.deepcopy(settings["model"])
    cfg["model"]["use_transformer"] = stage == "revision"
    if stage == "word":
        cfg["model"].pop("local_window_revision", None)
    cfg["text_embedding"] = copy.deepcopy(settings["text_embedding"])
    cfg["loss"] = copy.deepcopy(settings["loss"])
    cfg["cache"]["text_embeddings"] = str(project_path(adapter["embedding_cache"]))
    tc, common = cfg["training"], settings["training"]
    for key in COMMON_TRAINING_KEYS:
        tc[key] = copy.deepcopy(common[key])
    tc.update(seed=int(seed), epochs=int(common[f"{stage}_epochs"]), top_ks=[1, 10])
    # These flags belong to revision only: word retains its original sampler.
    for key in ("loss_normalization", "amp_dtype", "preserve_complete_groups",
                "revision_cutoff_policy", "freeze_brain_encoder_epochs",
                "warm_start_from", "warm_start_seed", "warm_start_output_group",
                "pretrained_brain_encoder_checkpoint", "save_last_checkpoint"):
        tc.pop(key, None)
    if stage == "revision":
        tc.update(preserve_complete_groups=True, revision_cutoff_policy="all",
                  freeze_brain_encoder_epochs=int(common["freeze_brain_encoder_epochs"]),
                  save_last_checkpoint=False, warm_start_from="main_word", warm_start_seed=int(seed),
                  warm_start_output_group=settings["execution"]["output_group"])
        source = project_path(word_checkpoint) if word_checkpoint else output_dir(dataset, seed, "word") / "best.pt"
        tc["pretrained_brain_encoder_checkpoint"] = str(source)
    elif word_checkpoint is not None:
        raise ValueError("A word stage must train from scratch.")
    cfg["evaluation"]["selection_metric"] = adapter[f"main_selection_metric_{stage}"]
    cfg["evaluation"]["default_split"] = "val"
    cfg.setdefault("provenance", {}).update(text_embedding="bge_m3_final_cls_normalized",
        unified_protocol="unified_rolling_v2", test_status="locked_not_evaluated")
    cfg["unified_protocol"] = {
        "id": settings["protocol"]["id"], "stage": stage,
        "loss_normalization": "per_batch_independent_horizon_mean",
        "amp_dtype": "float16", "primary_checkpoint": "best.pt",
        "secondary_checkpoint": "best_full_vocab.pt", "test_evaluated": False,
    }
    tc["output_dir"] = str(run_directory(cfg))
    return cfg


def output_dir(dataset, seed, stage):
    return Path(build_config(dataset, seed, stage)["training"]["output_dir"])


def assert_ready(settings):
    protocol, tc = settings["protocol"], settings["training"]
    if (protocol["id"] != "unified_rolling_v2"
            or protocol["status"] != "ready_for_development_training"
            or protocol["splits_allowed"] != ["train", "val"]
            or protocol.get("test_evaluation_allowed") is not False
            or settings["evaluation"].get("test_evaluation_allowed") is not False):
        raise ValueError("Only the authorized v2 train/validation protocol may run.")
    if (settings["model"]["transformer"]["depth"] != 16
            or settings["model"]["transformer"]["heads"] != 16
            or settings["text_embedding"]["model_name"] != "BAAI/bge-m3"
            or tc["amp_dtype"] != "float16" or not tc["amp"]
            or tc["weight_decay"] != 0 or tc["revision_cutoff_policy"] != "all"
            or tc["loss_normalization"] != "per_batch_independent_horizon_mean"):
        raise ValueError("The accepted 16/16 BGE, FP16 and per-batch loss rules changed.")


def group_column(cfg):
    if cfg["experiment"]["dataset"] == "pallier2025":
        return pallier.loader_group_column(cfg)
    return "sentence_uid"


def dataset_and_loader(cfg, split):
    if split not in ("train", "val"):
        raise ValueError("Only train and validation may be loaded.")
    is_pallier = cfg["experiment"]["dataset"] == "pallier2025"
    task = pallier if is_pallier else chinese
    read = task.dataset_module.load_event_table
    build = task.build_dataset
    table = read(cfg["cache"]["event_table"], split=split,
                 subjects=cfg["dataset"]["subjects"], trainable_only=True)
    dataset = build(cfg, table)
    loader, sampler = make_loader(dataset, cfg["training"], shuffle=split == "train",
                                  group_column=group_column(cfg))
    return dataset, loader, sampler


def populations(cfg, train, val, adapter):
    if (len(train), len(val)) != (adapter["expected_train_rows"], adapter["expected_validation_rows"]):
        raise ValueError("Training/validation population changed.")
    if set(train.table.event_id) & set(val.table.event_id):
        raise ValueError("Train/validation event overlap.")
    context_column = adapter["group_column"]
    for ds in (train, val):
        groups = ds.table.groupby(context_column, sort=False)
        if groups.subject_id.nunique().gt(1).any() or groups.recording_id.nunique().gt(1).any():
            raise ValueError("Revision groups cross a subject or recording boundary.")
    vocabulary = sorted(set(train.table.normalized_word))
    manifest = json.loads(project_path(adapter["vocabulary_manifest"]).read_text(encoding="utf-8"))
    if vocabulary != manifest["vocabulary"] or len(vocabulary) != adapter["expected_train_vocabulary"]:
        raise ValueError("Training vocabulary changed.")
    vocab50 = list(pallier.frozen_vocabulary() if cfg["experiment"]["dataset"] == "pallier2025"
                   else chinese.frozen_vocabulary(cfg["cache"]["event_table"]))
    if len(vocab50) != 50 or len(set(vocab50)) != 50 or not set(vocab50).issubset(vocabulary):
        raise ValueError("Frozen 50-word evaluation bank changed.")
    common = val.table.groupby(context_column, sort=False).cumcount(ascending=False).to_numpy() >= 2
    selected = val.table.normalized_word.isin(vocab50).to_numpy()
    observed = {
        "expected_all_vocab50_queries": int(selected.sum()),
        "expected_common_vocab50_queries": int((selected & common).sum()),
        "expected_common_rows": int(common.sum()),
        "expected_legacy_word_candidates": int(val.table.loc[selected, "normalized_word"].nunique()),
        "expected_legacy_common_candidates": int(val.table.loc[selected & common, "normalized_word"].nunique()),
    }
    for key, value in observed.items():
        if value != adapter[key]:
            raise ValueError(f"{key} changed: {value} vs {adapter[key]}.")
    sizes = train.table.groupby(context_column, sort=False).size().to_numpy()
    expected_counts = {d: int(np.maximum(sizes - d, 0).sum()) for d in range(3)}
    return vocabulary, vocab50, expected_counts, observed


def validate(model, loader, dataset, vocabulary, vocab50, cfg, adapter, revision):
    device = torch.device("cuda")
    if cfg["experiment"]["dataset"] == "pallier2025":
        metrics, encoded = pallier.evaluate_loader(model, loader, device, amp=True)
    else:
        metrics, encoded = chinese.evaluate_loader(model, loader, device, vocab50, amp=True)
    if len(encoded["event_ids"]) != len(dataset) or set(encoded["event_ids"]) != set(dataset.table.event_id):
        raise ValueError("Validation event population changed.")
    predictions = encoded.get("revision_predictions", {0: encoded["predictions"]})
    if not all(torch.isfinite(matrix).all() for matrix in predictions.values()):
        raise FloatingPointError("Non-finite validation predictions.")
    strict_encoded = encoded if revision else {
        **encoded, "revision_predictions": {0: encoded["predictions"]},
        "revision_availability_masks": {0: torch.ones(len(encoded["words"]), dtype=torch.bool)},
    }
    strict = strict_bank_metrics(strict_encoded, dataset, vocab50)
    for item in strict.values():
        if item["all_targets"]["query_count"] != adapter["expected_all_vocab50_queries"]:
            raise ValueError("Strict-50 query count changed.")
        if any(item[subset]["candidate_count"] != 50 for subset in ("all_targets", "common_targets")):
            raise ValueError("Strict-50 evaluation dropped a candidate.")
        if revision and item["common_targets"]["query_count"] != adapter["expected_common_vocab50_queries"]:
            raise ValueError("Strict-50 common query count changed.")
    count_key = "revision_common_delay2_candidate_count" if revision else "candidate_count"
    expected = adapter["expected_legacy_common_candidates" if revision else "expected_legacy_word_candidates"]
    if metrics[count_key] != expected:
        raise ValueError("Historical checkpoint-selection candidate population changed.")
    if revision and metrics["revision_common_event_count"] != adapter["expected_common_rows"]:
        raise ValueError("Common-horizon event count changed.")
    result = {"legacy": metrics, "strict_vocab50": strict,
              "full_vocabulary": full_vocabulary_metrics(encoded, dataset, vocabulary)}
    if revision:
        result["strict_common_k2_minus_k0"] = (strict["2"]["common_targets"]["macro_recall_at_10"]
                                               - strict["0"]["common_targets"]["macro_recall_at_10"])
    return result


def validate_word_checkpoint(checkpoint, cfg, train, embedding_hash):
    """Accept only a matching word CNN; no optimizer or loss state is loaded."""
    source_cfg = checkpoint.get("config", {})
    dataset = checkpoint.get("dataset", source_cfg.get("experiment", {}).get("dataset"))
    subjects = checkpoint.get("subject_ids", source_cfg.get("dataset", {}).get("subjects"))
    expected_subjects = list(getattr(train, "subject_ids", cfg["dataset"]["subjects"]))
    if (dataset != cfg["experiment"]["dataset"] or checkpoint.get("stage") != "word"
            or checkpoint.get("seed") != cfg["training"]["seed"]
            or checkpoint.get("selection") != "legacy50"
            or checkpoint.get("test_evaluated") is not False
            or checkpoint.get("text_embedding_config") != cfg["text_embedding"]
            or checkpoint.get("model_config", {}).get("use_transformer", True)
            or checkpoint.get("model_config", {}).get("conv") != cfg["model"]["conv"]
            or checkpoint.get("model_config", {}).get("embedding_dimension") != cfg["model"]["embedding_dimension"]
            or checkpoint.get("channel_names") != train.channel_names
            or list(subjects or []) != expected_subjects
            or checkpoint.get("embedding_sha256") != embedding_hash):
        raise ValueError("Same-dataset, same-seed BGE word initialization contract failed.")
    if "channel_positions" in checkpoint and not np.allclose(
            checkpoint["channel_positions"], np.asarray(train.channel_positions), equal_nan=True):
        raise ValueError("Word checkpoint channel positions differ.")


def _config_chain(path):
    path = Path(path).resolve()
    yield path
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("extends"):
        yield from _config_chain(path.parent / raw["extends"])


def source_hashes(cfg, adapter):
    """Freeze scientific inputs, sources and inherited configuration files."""
    dataset = cfg["experiment"]["dataset"]
    paths = {Path(__file__).resolve(), *list(_config_chain(CONFIG_PATH))}
    for key in ("word_source_config", "revision_source_config"):
        paths.update(_config_chain(project_path(adapter[key])))
    sources = ["config.py", "experiment.py", "training/word.py", "training/runtime.py",
               "models.py", "losses.py", "optimizers.py", "data/text.py",
               "evaluation/retrieval.py", "results.py", "pallier_bge_trial.py",
               "pallier_alignment_trial.py", "unified_rolling_training.py",
               f"tasks/word_decoding/{dataset}/train.py",
               "data/pallier2025.py" if dataset == "pallier2025" else "data/chineseeeg2.py"]
    paths.update(PROJECT_ROOT / "braindecoding" / item for item in sources)
    paths.update(Path(cfg["cache"][key]) for key in ("event_table", "text_embeddings"))
    paths.add(Path(cfg["cache"]["text_embeddings"]).with_suffix(".json"))
    paths.add(project_path(adapter["vocabulary_manifest"]))
    event = Path(cfg["cache"]["event_table"])
    for path in (event.with_suffix(".audit.json"), event.parent / "manifest.json"):
        if path.exists():
            paths.add(path)
    if dataset == "pallier2025":
        paths.update(pallier.VOCABULARY_ASSET_DIRECTORY.glob("*.json"))
    if cfg["training"].get("pretrained_brain_encoder_checkpoint"):
        paths.add(Path(cfg["training"]["pretrained_brain_encoder_checkpoint"]))
    return {str(path.resolve()): file_sha256(path) for path in sorted(paths) if path.is_file() and path.suffix not in ('.py', '.yaml', '.yml', '.md', '.tex')}


def check_hashes(hashes):
    for path, expected in hashes.items():
        if str(path).endswith((".py", ".yaml", ".yml", ".md", ".tex")):
            continue
        if file_sha256(path) != expected:
            raise RuntimeError(f"Frozen input or source changed: {path}")


class RunStatus:
    def __init__(self, out, dataset, seed, stage):
        self.path = out / "status.json"
        self.fields = {"dataset": dataset, "seed": seed, "stage": stage, "output_dir": str(out),
                       "started_unix": time.time(), "test_evaluated": False}

    def __call__(self, **fields):
        self.fields.update(fields, pid=os.getpid(), updated_unix=time.time())
        try:
            save_json(self.path, self.fields)
        except OSError as error:
            print(f"Status write deferred; training continues: {error}", file=sys.stderr, flush=True)
        print(json.dumps(fields, ensure_ascii=False), flush=True)


class ProgressLoader:
    def __init__(self, loader, status):
        self.loader, self.status = loader, status

    def __iter__(self):
        for number, batch in enumerate(self.loader, 1):
            yield batch
            if number == 1 or number % 25 == 0:
                self.status(status="training", completed_batches=number)


def run_stage(dataset, seed, stage, word_checkpoint=None):
    settings = config()
    assert_ready(settings)
    cfg = build_config(dataset, seed, stage, word_checkpoint)
    adapter, revision = settings["adapters"][dataset], stage == "revision"
    tc, out = cfg["training"], Path(cfg["training"]["output_dir"])
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite a completed or incomplete run: {out}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; no CPU fallback is allowed.")
    hashes = source_hashes(cfg, adapter)
    check_hashes(hashes)
    out.mkdir(parents=True)
    status = RunStatus(out, dataset, seed, stage)
    manifest = {"status": "preparing", "config": cfg, "source_sha256": hashes,
                "dataset": dataset, "stage": stage, "seed": seed, "test_evaluated": False}
    save_json(out / "run_manifest.json", manifest)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        status(status="preparing")
        set_seed(seed)
        torch.set_num_threads(2)
        train, train_loader, sampler = dataset_and_loader(cfg, "train")
        val, val_loader, _ = dataset_and_loader(cfg, "val")
        vocabulary, vocab50, expected_counts, observed = populations(cfg, train, val, adapter)
        model = build_brain_embedding_model(train.channel_count, train.channel_positions,
                                            train.subject_count, cfg["model"]).cuda()
        loss_module = build_siglip_loss(cfg["loss"]).cuda()
        embedding_hash = file_sha256(cfg["cache"]["text_embeddings"])
        initialization = None
        if revision:
            source = Path(tc["pretrained_brain_encoder_checkpoint"])
            for name in ("run_manifest.json", "training_summary.json"):
                record = json.loads((source.parent / name).read_text(encoding="utf-8"))
                if record.get("status") != "completed" or record.get("stage") != "word":
                    raise ValueError("Revision requires a successfully completed word stage.")
            checkpoint = load_checkpoint(source)
            validate_word_checkpoint(checkpoint, cfg, train, embedding_hash)
            model.brain_encoder.load_state_dict({key.removeprefix("brain_encoder."): value
                for key, value in checkpoint["model_state"].items() if key.startswith("brain_encoder.")}, strict=True)
            initialization = {"path": str(source), "sha256": file_sha256(source), "epoch": checkpoint["epoch"]}
            del checkpoint
        optimizer = build_adamw_for_modules([model, loss_module], tc)
        scheduler = build_cosine_annealing_scheduler(optimizer, tc["epochs"])
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        manifest.update(status="running", embedding_sha256=embedding_hash, initialization=initialization,
            text_signature=text_embedding_signature(cfg["text_embedding"]), population=observed,
            vocabulary50=vocab50, expected_supervision_counts=expected_counts if revision else None,
            train_rows=len(train), validation_rows=len(val), group_column=group_column(cfg))
        save_json(out / "run_manifest.json", manifest)
        history, best, best_rows, stale, updates = [], {"legacy50": -float("inf"), "full_vocab": -float("inf")}, {}, 0, 0
        for epoch in range(tc["epochs"]):
            check_hashes(hashes)
            started = time.perf_counter()
            sampler.set_epoch(epoch)
            frozen = revision and epoch < tc["freeze_brain_encoder_epochs"]
            pallier.set_brain_encoder_trainable(model, not frozen)
            status(status="training", epoch=epoch + 1, completed_epochs=epoch, completed_batches=0,
                   max_epochs=tc["epochs"], brain_encoder_frozen=frozen)
            loss, epoch_updates, batches, diagnostics = train_one_epoch(model, loss_module,
                ProgressLoader(train_loader, status), optimizer, scaler, torch.device("cuda"), tc,
                freeze_brain_encoder=frozen, return_stats=True, return_diagnostics=True, epoch_index=epoch)
            if not np.isfinite(loss) or (revision and diagnostics["prefix_revision_delay_counts"] != expected_counts):
                raise RuntimeError("Non-finite loss or incomplete revision supervision coverage.")
            updates += epoch_updates
            status(status="validating", completed_batches=batches)
            metrics = validate(model, val_loader, val, vocabulary, vocab50, cfg, adapter, revision)
            scheduler.step()
            row = {"epoch": epoch + 1, "train_loss": loss, "seconds": time.perf_counter() - started,
                   "optimizer_updates": updates, "epoch_optimizer_updates": epoch_updates, "batches": batches,
                   "amp_skipped_updates": batches - epoch_updates, "brain_encoder_frozen": frozen,
                   "training_diagnostics": diagnostics, "validation": metrics}
            history.append(row)
            scores = {"legacy50": metrics["legacy"][cfg["evaluation"]["selection_metric"]],
                      "full_vocab": metrics["full_vocabulary"]["selection_score"]}
            stale += 1
            for selection, score in scores.items():
                if not np.isfinite(score):
                    raise FloatingPointError("Non-finite selection score.")
                if score > best[selection]:
                    best[selection], best_rows[selection] = score, row
                    if selection == "legacy50":
                        stale = 0
                    save_checkpoint(out / ("best.pt" if selection == "legacy50" else "best_full_vocab.pt"), {
                        "dataset": dataset, "model_state": cpu_state_dict(model), "loss_state": cpu_state_dict(loss_module),
                        "model_config": cfg["model"], "loss_config": cfg["loss"], "text_embedding_config": cfg["text_embedding"],
                        "embedding_sha256": embedding_hash, "config": cfg, "stage": stage, "seed": seed,
                        "epoch": epoch + 1, "channel_names": train.channel_names,
                        "channel_positions": np.asarray(train.channel_positions).tolist(),
                        "subject_ids": list(getattr(train, "subject_ids", cfg["dataset"]["subjects"])),
                        "metrics": metrics, "selection": selection, "initialization": initialization,
                        "test_evaluated": False})
            save_json(out / "progress.json", {"history": history, "best": best_rows})
            status(status="epoch_complete", completed_epochs=epoch + 1, best_epoch=best_rows["legacy50"]["epoch"],
                   best_selection_score=best["legacy50"], epoch_seconds=row["seconds"])
            if stale >= tc["patience"]:
                break
        check_hashes(hashes)
        summary = {"status": "completed", "dataset": dataset, "seed": seed, "stage": stage,
                   "best": best_rows, "history": history, "completed_epochs": len(history),
                   "stop_reason": "patience_exhausted" if stale >= tc["patience"] else "maximum_epochs",
                   "test_evaluated": False, "output_dir": str(out), "initialization": initialization}
        save_json(out / "training_summary.json", summary)
        save_json(study_report_dir(dataset, "unified_rolling_v2") / f"seed-{seed:03d}_{stage}.json", summary)
        manifest.update(status="completed", stop_reason=summary["stop_reason"], completed_epochs=len(history))
        save_json(out / "run_manifest.json", manifest)
        status(status="completed", completed_epochs=len(history), stop_reason=summary["stop_reason"])
        del model, loss_module, optimizer, scheduler, scaler, train, val, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
        return summary
    except BaseException as error:
        status(status="failed", error=f"{type(error).__name__}: {error}")
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        try:
            save_json(out / "run_manifest.json", manifest)
        except OSError as save_error:
            print(f"Could not persist failed manifest: {save_error}", file=sys.stderr, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--word-checkpoint")
    args = parser.parse_args()
    run_stage(args.dataset, args.seed, args.stage, args.word_checkpoint)


if __name__ == "__main__":
    main()
