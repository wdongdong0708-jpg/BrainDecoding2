"""Low-cost staged timing-shortcut control, using the frozen v2 training loop.

The adapter is installed only inside a dedicated worker process. Existing real
training, held-out evaluation and running LLM jobs are never edited or stopped.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from braindecoding import unified_rolling_v2_training as neural
from braindecoding.config import PROJECT_ROOT as ROOT
from braindecoding.data.synthetic_timing import (
    CONDITIONS, GENERATOR_VERSION, SyntheticTimingDataset, overlap_statistics,
)
from braindecoding.experiment import file_sha256
from braindecoding.pallier_alignment_trial import strict_bank_metrics
from braindecoding.training.runtime import load_checkpoint, set_seed
from braindecoding.training.word import make_loader, train_one_epoch, encode_loader

CONFIG = ROOT / "configs/unified_rolling/synthetic_timing_v1.yaml"
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = PROTOCOL.with_suffix(".json")
REPORT = ROOT / "data/results/studies/synthetic_timing_v1"
RUNTIME = ROOT / ".cache/runs/synthetic_timing_v1"
_original_build_config = neural.build_config
_original_sources = neural.source_hashes
_original_word_validation = neural.validate_word_checkpoint


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def settings():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p, s = cfg["protocol"], cfg["synthetic"]
    if (p["dataset"] != "pallier2025" or p["conditions"] != list(CONDITIONS)
            or p["seeds"] != [0, 1, 2] or p["history_radius"] != 2 or p["horizons"] != [0, 1, 2]
            or p["test_selection_allowed"] or p["language_model_allowed"]
            or s["generator"] != GENERATOR_VERSION or s["sample_rate_hz"] != 50
            or s["window_seconds"] != 1 or s["gaussian_sigma_ms"] != [60, 120, 240]
            or s["pulse_rate_per_process_hz"] != 1 or s["padding_seconds"] != 1
            or s["training_refresh"] != "epoch"):
        raise ValueError("Synthetic timing-control scientific contract changed")
    return cfg


def configure_environment():
    """Recover the local dataset root from an existing frozen real run."""
    source = neural.config()["execution"]["reuse_runs"][0]["path"]
    resolved_path = ROOT / source / "resolved_config.yaml"
    if resolved_path.is_file():
        cfg = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        os.environ.setdefault("BRAINDATA_ROOT", str(Path(cfg["dataset"]["root"]).parent))
    else:
        from braindecoding.config import DEFAULT_BRAINDATA_ROOT
        os.environ.setdefault("BRAINDATA_ROOT", DEFAULT_BRAINDATA_ROOT)
    # Only needed to resolve the inherited Chinese text config for metadata stats.
    from braindecoding.config import DEFAULT_MODEL_ROOT
    os.environ.setdefault("BRAINDECODING_MODEL_ROOT", DEFAULT_MODEL_ROOT)


def build_config(condition, seed, stage, word_checkpoint=None):
    if condition not in CONDITIONS or seed not in (0, 1, 2) or stage not in ("word", "revision"):
        raise ValueError("Unauthorized condition, seed or stage")
    spec = settings()
    cfg = _original_build_config("pallier2025", seed, stage)
    group = f"{spec['execution']['output_group_prefix']}_{condition}"
    cfg["experiment"]["output_group"] = group
    out = ROOT / "outputs/word_decoding/pallier2025" / group / cfg["experiment"]["subject_scope"]
    cfg["training"]["output_dir"] = str(out / cfg["experiment"]["id"] / f"seed-{seed:03d}")
    if stage == "revision":
        cfg["training"]["warm_start_output_group"] = group
        cfg["training"]["pretrained_brain_encoder_checkpoint"] = str(
            Path(word_checkpoint) if word_checkpoint else out / "main_word" / f"seed-{seed:03d}" / "best.pt")
    elif word_checkpoint is not None:
        raise ValueError("Synthetic word CNN must start from scratch")
    cfg["synthetic_control"] = {**spec["synthetic"], "condition": condition,
                               "stream_grouping": "subject_recording_language_segment"}
    cfg["provenance"]["neural_input"] = f"synthetic_{condition}_no_neural_samples"
    return cfg


class EpochSampler:
    def __init__(self, sampler, dataset):
        self.sampler, self.dataset = sampler, dataset

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)


def dataset_for(cfg, split, table=None):
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    if table is None:
        table = neural.pallier.dataset_module.load_event_table(cfg["cache"]["event_table"],
            split=split, subjects=cfg["dataset"]["subjects"], trainable_only=True)
    resolved = copy.deepcopy(cfg)
    if split == "test":
        resolved["cache"]["text_embeddings"] = str(ROOT / "data/cache/derived/pallier2025/text/unified_heldout_test_v2/embeddings.npz")
    # Base initialization validates metadata and embeddings but zero_meg avoids
    # neural sample access. The synthetic wrapper constructs every batch field.
    base = neural.pallier.build_dataset(resolved, table, zero_meg=True)
    spec = cfg["synthetic_control"]
    return SyntheticTimingDataset(base, spec["condition"], split=split,
        training_seed=cfg["training"]["seed"],
        evaluation_seed=spec["validation_seed" if split == "val" else "test_seed"])


def dataset_and_loader(cfg, split):
    if split not in ("train", "val"):
        raise ValueError("Training can load only train and validation")
    dataset = dataset_for(cfg, split)
    loader, sampler = make_loader(dataset, cfg["training"], shuffle=split == "train",
                                 group_column=neural.group_column(cfg))
    epoch_sampler = EpochSampler(sampler, dataset)
    # DataLoader keeps the original sampler; the returned epoch hook updates
    # that same sampler and the dataset together, without rebinding its fields.
    return dataset, loader, epoch_sampler


def validate_word_checkpoint(checkpoint, cfg, train, embedding_hash):
    _original_word_validation(checkpoint, cfg, train, embedding_hash)
    if checkpoint.get("config", {}).get("synthetic_control") != cfg["synthetic_control"]:
        raise ValueError("Synthetic revision requires the matching synthetic word CNN; real checkpoints are forbidden")


def source_hashes(cfg, adapter):
    hashes = _original_sources(cfg, adapter)
    for path in (CONFIG, PROTOCOL, MANIFEST, Path(__file__),
                 ROOT / "braindecoding/data/synthetic_timing.py"):
        hashes[str(path.resolve())] = file_sha256(path)
    return hashes


def run_stage(condition, seed, stage):
    verify_manifest()
    # These names are process-local adapters. No frozen module is edited.
    neural.build_config = lambda dataset, seed, stage, word_checkpoint=None: build_config(
        condition, seed, stage, word_checkpoint)
    neural.dataset_and_loader = dataset_and_loader
    neural.validate_word_checkpoint = validate_word_checkpoint
    neural.source_hashes = source_hashes
    neural.study_report_dir = lambda dataset, name: REPORT / condition
    return neural.run_stage("pallier2025", seed, stage)


def verify_manifest():
    m = read(MANIFEST)
    if digest({k: v for k, v in m.items() if k != "manifest_sha256"}) != m["manifest_sha256"]:
        raise ValueError("Control manifest digest changed")
    neural.check_hashes(m["source_sha256"])
    return m


def audit_dataset(dataset):
    checks = []
    groups = dataset.table.groupby("runtime_context_uid", sort=False).indices
    for indices in groups.values():
        for left, right in zip(indices[:-1], indices[1:]):
            offset = int(dataset.starts[right] - dataset.starts[left])
            if 0 < offset < dataset.window_samples:
                a, b = dataset.raw_window(left), dataset.raw_window(right)
                equal = bool(np.array_equal(a[:, offset:], b[:, :-offset]))
                if equal != (dataset.condition == "shared"):
                    raise AssertionError("Shared/independent window identity control failed")
                checks.append({"left_event": dataset.event_ids[left], "sample_offset": offset,
                               "overlap_samples": dataset.window_samples - offset, "raw_overlap_equal": equal})
                break
        if len(checks) == 20:
            break
    if len(checks) < 20:
        raise ValueError("Not enough real-timestamp overlap checks")
    first = dataset.raw_window(0)
    dataset.set_epoch(1)
    refreshed = not np.array_equal(first, dataset.raw_window(0))
    if refreshed != (dataset.split == "train"):
        raise AssertionError("Epoch refresh/fixed-evaluation control failed")
    return {"condition": dataset.condition, "split": dataset.split,
            "passed": True, "epoch_changed_signal": refreshed, "checks": checks}


def prepare():
    if MANIFEST.exists():
        return verify_manifest()
    spec = settings()
    REPORT.mkdir(parents=True, exist_ok=True)
    overlap = {}
    for dataset in neural.DATASETS:
        path = ROOT / "data/cache/derived" / dataset / "events/events.csv"
        table = pd.read_csv(path, keep_default_na=False)
        overlap[dataset] = {"event_table_sha256": file_sha256(path),
            "one_second": overlap_statistics(table),
            "three_seconds_reference": overlap_statistics(table, window_seconds=3.0)}
    save(REPORT / "overlap.json", overlap)
    audits, hashes, population = [], {}, None
    for condition in CONDITIONS:
        cfg = build_config(condition, 0, "revision")
        train, val = dataset_for(cfg, "train"), dataset_for(cfg, "val")
        _, _, _, observed = neural.populations(cfg, train, val, neural.config()["adapters"]["pallier2025"])
        if population is not None and population != observed:
            raise ValueError("Synthetic conditions changed the population")
        population = observed
        audits.extend([audit_dataset(train), audit_dataset(val)])
        # Revision dependency is frozen by its worker when its CNN exists.
        source_cfg = copy.deepcopy(cfg)
        source_cfg["training"].pop("pretrained_brain_encoder_checkpoint", None)
        hashes.update(_original_sources(source_cfg, neural.config()["adapters"]["pallier2025"]))
        del train, val
        gc.collect()
    save(REPORT / "window_audit.json", {"passed": True, "audits": audits,
        "identity_checked_before_baseline": True,
        "baseline_note": "Window-specific baseline shifts shared samples by a channel-wise constant; clipping can alter this further."})
    for path in (CONFIG, PROTOCOL, Path(__file__), ROOT / "braindecoding/data/synthetic_timing.py",
                 ROOT / spec["execution"]["real_results"],
                 ROOT / "data/cache/derived/chineseeeg2_littleprince/events/events.csv",
                 ROOT / "data/cache/derived/pallier2025/text/unified_heldout_test_v2/embeddings.npz",
                 ROOT / "data/cache/derived/pallier2025/text/unified_heldout_test_v2/embeddings.json",
                 ROOT / "data/manifests/rolling_sentence/heldout_test_v2.json"):
        hashes[str(path.resolve())] = file_sha256(path)
    real = read(ROOT / spec["execution"]["existing_training_results"])
    if real["status"] != "completed":
        raise ValueError("Real training reference is incomplete")
    references = [r for r in real["completed"] if r["dataset"] == "pallier2025" and r["stage"] == "revision"]
    if sorted(r["seed"] for r in references) != [0, 1, 2]:
        raise ValueError("Three real rolling references are required")
    for row in references:
        if file_sha256(row["checkpoint"]) != row["checkpoint_sha256"]:
            raise ValueError("Real reference checkpoint changed")
        hashes[str(Path(row["checkpoint"]).resolve())] = row["checkpoint_sha256"]
    jobs = [{"condition": c, "seed": s, "stage": stage,
             "output_dir": build_config(c, s, stage)["training"]["output_dir"]}
            for s in (0, 1, 2) for c in CONDITIONS for stage in ("word", "revision")]
    m = {"protocol": spec["protocol"]["id"], "frozen_unix": time.time(), "settings": spec,
         "jobs": jobs, "pipeline_count": 6, "training_stage_count": 12,
         "population": population, "real_references": references, "source_sha256": hashes,
         "test_selection_allowed": False, "test_evaluated": False}
    m["manifest_sha256"] = digest(m)
    save(MANIFEST, m)
    export_report()
    return m


def smoke():
    """One CPU update per stage/condition with the full frozen model capacity."""
    verify_manifest()
    torch.set_num_threads(2)
    records = []
    for condition in CONDITIONS:
        cfg = build_config(condition, 0, "revision")
        table = neural.pallier.dataset_module.load_event_table(cfg["cache"]["event_table"],
            split="train", subjects=cfg["dataset"]["subjects"], trainable_only=True)
        # Diagnostic only: three consecutive events, with the K2 common target
        # in the frozen candidate set, so retrieval checks are not empty.
        candidates = set(neural.pallier.frozen_vocabulary())
        for start in np.flatnonzero(table["标准词"].isin(candidates)):
            part = table.iloc[start:start + 3]
            if len(part) == 3 and all(part[k].nunique() == 1 for k in ("受试者", "记录编号", "上下文编号")):
                table = part.copy()
                break
        else:
            raise ValueError("No suitable three-word smoke segment")
        dataset = dataset_for(cfg, "train", table)
        loader, _ = make_loader(dataset, cfg["training"], shuffle=False,
                               group_column=neural.group_column(cfg))
        cnn_state = None
        for stage in ("word", "revision"):
            set_seed(0)
            stage_cfg = build_config(condition, 0, stage)
            model = neural.build_brain_embedding_model(dataset.channel_count, dataset.channel_positions,
                                                       dataset.subject_count, stage_cfg["model"])
            if cnn_state is not None:
                model.brain_encoder.load_state_dict(cnn_state, strict=True)
            frozen = stage == "revision"
            neural.pallier.set_brain_encoder_trainable(model, not frozen)
            loss_module = neural.build_siglip_loss(stage_cfg["loss"])
            optimizer = neural.build_adamw_for_modules([model, loss_module], stage_cfg["training"])
            started = time.perf_counter()
            tc = {**stage_cfg["training"], "amp": False}
            result = train_one_epoch(model, loss_module, loader, optimizer,
                torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"), tc,
                freeze_brain_encoder=frozen, return_stats=True, return_diagnostics=True)
            loss, updates, batches, diagnostics = result
            if not np.isfinite(loss) or updates != 1:
                raise AssertionError("Synthetic smoke training update failed")
            encoded = encode_loader(model, loader, torch.device("cpu"), amp=False)
            if stage == "revision":
                if diagnostics["prefix_revision_delay_counts"] != {0: 3, 1: 2, 2: 1}:
                    raise AssertionError("K0/K1/K2 supervision coverage changed")
                strict = strict_bank_metrics(encoded, dataset, neural.pallier.frozen_vocabulary())
                if set(strict) != {"0", "1", "2"} or any(
                    r["common_targets"]["candidate_count"] != 50 for r in strict.values()):
                    raise AssertionError("Smoke evaluation lost a horizon or a candidate")
            else:
                cnn_state = {k: v.detach().clone() for k, v in model.brain_encoder.state_dict().items()}
            records.append({"condition": condition, "stage": stage, "train_loss": loss,
                "updates": updates, "seconds": time.perf_counter() - started,
                "parameters": sum(p.numel() for p in model.parameters()),
                "delay_counts": diagnostics["prefix_revision_delay_counts"], "passed": True})
            del model, loss_module, optimizer, encoded
            gc.collect()
        del cnn_state, dataset, loader
        gc.collect()
    result = {"passed": True, "device": "cpu", "formal_performance_estimate": False,
              "full_model_capacity": True, "records": records}
    save(REPORT / "smoke.json", result)
    return result


def evaluate_test(condition, seed):
    m = verify_manifest()
    cfg = build_config(condition, seed, "revision")
    out = Path(cfg["training"]["output_dir"])
    result_path = out / "evaluation/synthetic_test.json"
    if result_path.exists():
        result = read(result_path)
        if result["manifest_sha256"] != m["manifest_sha256"] or result["checkpoint_sha256"] != file_sha256(out / "best.pt"):
            raise ValueError("Existing test evaluation identity changed")
        return result
    summary = read(out / "training_summary.json")
    if summary["status"] != "completed" or summary["test_evaluated"] is not False:
        raise ValueError("Only completed validation-selected runs may be evaluated on test")
    cp = load_checkpoint(out / "best.pt")
    if cp["config"] != cfg or cp["epoch"] != summary["best"]["legacy50"]["epoch"]:
        raise ValueError("Checkpoint is not the frozen validation-selected primary model")
    dataset = dataset_for(cfg, "test")
    loader, _ = make_loader(dataset, cfg["training"], shuffle=False, group_column=neural.group_column(cfg))
    model = neural.build_brain_embedding_model(dataset.channel_count, dataset.channel_positions,
                                               dataset.subject_count, cfg["model"]).cuda()
    model.load_state_dict(cp["model_state"], strict=True)
    encoded = encode_loader(model, loader, torch.device("cuda"), amp=True)
    strict = strict_bank_metrics(encoded, dataset, neural.pallier.frozen_vocabulary())
    if (len(encoded["event_ids"]) != 18450 or any(r["common_targets"]["query_count"] != 7440
            or r["all_targets"]["query_count"] != 8930 or r["common_targets"]["candidate_count"] != 50
            for r in strict.values())):
        raise ValueError("Synthetic test population differs from the real held-out population")
    result = {"condition": condition, "seed": seed, "split": "test", "test_evaluated": True,
        "manifest_sha256": m["manifest_sha256"], "checkpoint_sha256": file_sha256(out / "best.pt"),
        "test_embedding_sha256": file_sha256(dataset.base.meg_cache_dir.parent.parent / "text/unified_heldout_test_v2/embeddings.npz"),
        "event_ids_sha256": digest(encoded["event_ids"]), "strict_vocab50": strict,
        "best_epoch": cp["epoch"]}
    save(result_path, result)
    del model, cp, dataset, loader, encoded
    gc.collect()
    torch.cuda.empty_cache()
    return result


def export_report():
    REPORT.mkdir(parents=True, exist_ok=True)
    real = pd.read_csv(ROOT / settings()["execution"]["real_results"])
    rows = [{"condition": "real", "seed": int(r.seed), "horizon": int(r.horizon),
             "population": r.population, "macro_top10": float(r.macro_recall_at_10),
             "query_count": int(r.query_count), "candidate_count": int(r.candidate_count)}
            for r in real[real.dataset.eq("pallier2025")].itertuples()]
    for seed in (0, 1, 2):
        for condition in CONDITIONS:
            out = Path(build_config(condition, seed, "revision")["training"]["output_dir"])
            path = out / "evaluation/synthetic_test.json"
            if path.exists():
                data = read(path)
                for h, value in data["strict_vocab50"].items():
                    for population, metric in value.items():
                        rows.append({"condition": condition, "seed": seed, "horizon": int(h),
                            "population": population, "macro_top10": metric["macro_recall_at_10"],
                            "query_count": metric["query_count"], "candidate_count": metric["candidate_count"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT / "test_by_seed.csv", index=False)
    common = frame[frame.population.eq("common_targets")]
    summaries, lines = [], ["# 词级时间捷径控制：当前结果", "", "固定 50 词候选集，词型宏平均 Top-10；表格使用 K0/K1/K2 共同目标词（test：7,440 个）。",
        "随机参考为 20%；标准差为三个训练种子的样本标准差。缺失项表示尚未完成，不能据此解释时间捷径。", "",
        "| 输入 | 已完成种子 | K0 | K1 | K2 | K2−K0 |", "|---|---:|---:|---:|---:|---:|"]
    labels = {"real": "真实 MEG（复用）", "shared": "合成重叠", "independent": "合成独立"}
    gains = {}
    for condition in ("real", *CONDITIONS):
        part = common[common.condition.eq(condition)]
        n = part.seed.nunique()
        scores = []
        for h in (0, 1, 2):
            values = part.loc[part.horizon.eq(h), "macro_top10"].to_numpy()
            mean = float(values.mean()) if len(values) else None
            sd = float(values.std(ddof=1)) if len(values) > 1 else None
            scores.append("—" if mean is None else f"{100 * mean:.2f}%" + (f" ± {100 * sd:.2f}" if sd is not None else "（单种子）"))
            summaries.append({"condition": condition, "horizon": h, "n_seeds": len(values), "mean": mean, "sample_sd": sd})
        pivot = part.pivot(index="seed", columns="horizon", values="macro_top10")
        delta = (pivot[2] - pivot[0]) if n else pd.Series(dtype=float)
        gains[condition] = {"mean": float(delta.mean()) if n else None,
                            "sample_sd": float(delta.std(ddof=1)) if n > 1 else None, "n_seeds": n}
        gain_text = "—" if not n else f"{100 * delta.mean():.2f} pp" + (f" ± {100 * delta.std(ddof=1):.2f}" if n > 1 else "")
        lines.append(f"| {labels[condition]} | {n}/3 | {' | '.join(scores)} | {gain_text} |")
    done = len(common[common.condition.isin(CONDITIONS)]) == 18
    result = {"status": "completed" if done else "pending_synthetic_training", "split": "test",
              "population": "common_targets", "summaries": summaries, "delay_gains": gains,
              "context_experiment_B_started": False}
    if done:
        a = common[common.condition.eq("shared")].pivot(index="seed", columns="horizon", values="macro_top10")
        b = common[common.condition.eq("independent")].pivot(index="seed", columns="horizon", values="macro_top10")
        differences = a - b
        result["shared_minus_independent_by_seed"] = differences.reset_index().to_dict("records")
        lines += ["", "Shared−Independent（同种子配对均值）：" + "，".join(
            f"K{h}：{100 * differences[h].mean():.2f} ± {100 * differences[h].std(ddof=1):.2f} pp" for h in (0, 1, 2)) + "。",
            "", "这项差异衡量本控制中可利用的共享窗口信息。Real−Shared 不定义为纯神经贡献。",
            "实验 B 尚未启动；先依据验证集结果判断是否值得增加局部半径与完整上下文训练，不能用测试集选择设置。"]
    if (REPORT / "overlap.json").exists():
        lines += ["", "1 秒窗口的实际时间重叠（同片段、实际相邻且可训练词）："]
        for dataset, record in read(REPORT / "overlap.json").items():
            o = record["one_second"]
            lines.append(f"- {dataset}：{o['pairs']:,} 对，重叠发生率 {100 * o['overlap_probability']:.2f}%，平均共享比例 {100 * o['mean_shared_fraction']:.2f}%。")
    lines += ["", "训练按 epoch 重生成；验证和测试的合成种子固定且不随模型种子变化。Shared 与 Independent 使用同样长度的填充信号、归一化、CNN、Transformer、损失和训练预算。",
              "Pallier 的逐窗基线校正会使重叠样本相差通道常量，因此样本一致性在基线校正前检验。",
              "", "实验设置来源：[SimpleB2T 附录 A.2](https://arxiv.org/html/2609.40359v1#A1.SS2)。"]
    save(REPORT / "results.json", result)
    (REPORT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def gpu_ready():
    cfg = settings()["execution"]
    path = ROOT / cfg["existing_gpu_task"]
    if path.exists():
        other = read(path)
        released = other.get("state") == "paused" and other.get("worker_exited") is True
        if not released and other.get("state") not in ("completed", "failed") and psutil.pid_exists(other.get("pid", -1)):
            return False, "existing_gpu_job_active"
    command = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    free = int(subprocess.check_output(command, text=True).splitlines()[0].strip())
    return free >= cfg["minimum_free_gpu_mib"], f"free_gpu_mib={free}"


def run_queue():
    m = verify_manifest()
    if not read(REPORT / "smoke.json")["passed"]:
        raise RuntimeError("CPU smoke must pass before formal training")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = RUNTIME / "queue.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, stream)
    state = {"state": "running", "pid": os.getpid(), "started_unix": time.time(),
             "manifest_sha256": m["manifest_sha256"], "completed_stages": 0, "total_stages": 12}
    try:
        for job in m["jobs"]:
            verify_manifest()
            out = Path(job["output_dir"])
            completed = out / "training_summary.json"
            if out.exists():
                if not completed.exists() or read(completed)["status"] != "completed":
                    raise RuntimeError(f"Partial run requires inspection; refusing automatic restart: {out}")
                manifest = read(out / "run_manifest.json")
                if manifest["config"] != build_config(job["condition"], job["seed"], job["stage"]):
                    raise ValueError("Completed synthetic run configuration differs")
                neural.check_hashes(manifest["source_sha256"])
            else:
                while True:
                    ready, reason = gpu_ready()
                    if ready:
                        break
                    state.update(state="waiting_for_gpu", reason=reason, current=job, updated_unix=time.time())
                    save(RUNTIME / "status.json", state)
                    time.sleep(settings()["execution"]["poll_seconds"])
                verify_manifest()
                state.update(state="training", current=job, updated_unix=time.time())
                save(RUNTIME / "status.json", state)
                args = [sys.executable, "-u", "-m", "braindecoding.synthetic_timing_control", "--worker",
                        "--condition", job["condition"], "--seed", str(job["seed"]), "--stage", job["stage"]]
                with (RUNTIME / f"{job['condition']}_seed{job['seed']}_{job['stage']}.log").open("a", encoding="utf-8") as log:
                    subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            state["completed_stages"] += 1
            if job["stage"] == "revision":
                # Keep inference in a short-lived process to return CPU/GPU RAM.
                subprocess.run([sys.executable, "-u", "-m", "braindecoding.synthetic_timing_control", "--test",
                    "--condition", job["condition"], "--seed", str(job["seed"])], cwd=ROOT, check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                export_report()
            save(RUNTIME / "status.json", {**state, "updated_unix": time.time()})
        export_report()
        state.update(state="completed", updated_unix=time.time())
        save(RUNTIME / "status.json", state)
    except BaseException as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}", updated_unix=time.time())
        save(RUNTIME / "status.json", state)
        raise
    finally:
        lock.unlink()


def launch():
    verify_manifest()
    if not read(REPORT / "smoke.json")["passed"]:
        raise RuntimeError("CPU smoke must pass before queue launch")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    destination = RUNTIME / "launch.json"
    # Exclusive intent prevents double launches, including an uncertain spawn.
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"state": "launch_requested", "requested_unix": time.time()}, stream)
    with (RUNTIME / "queue.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "braindecoding.synthetic_timing_control", "--queue"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    result = {"state": "launched", "pid": process.pid, "created": psutil.Process(process.pid).create_time()}
    save(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ("prepare", "smoke", "worker", "test", "queue", "launch", "report", "status"):
        actions.add_argument("--" + action, action="store_true")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--stage", choices=("word", "revision"))
    args = parser.parse_args()
    configure_environment()
    if args.prepare:
        result = prepare()
        result = {"prepared": True, "pipeline_count": result["pipeline_count"], "manifest_sha256": result["manifest_sha256"]}
    elif args.smoke:
        result = smoke()
    elif args.worker:
        if not args.condition or not args.stage:
            parser.error("--worker requires --condition and --stage")
        result = run_stage(args.condition, args.seed, args.stage)
        result = {"status": result["status"], "condition": args.condition, "seed": args.seed, "stage": args.stage}
    elif args.test:
        if not args.condition:
            parser.error("--test requires --condition")
        result = evaluate_test(args.condition, args.seed)
        result = {"test_evaluated": True, "condition": args.condition, "seed": args.seed}
    elif args.queue:
        result = run_queue()
    elif args.launch:
        result = launch()
    elif args.report:
        verify_manifest()
        result = export_report()
    else:
        result = read(RUNTIME / "status.json") if (RUNTIME / "status.json").exists() else {"state": "not_launched"}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
