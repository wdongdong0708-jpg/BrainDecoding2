"""CPU-only input-availability statistics for the completed held-out C1 replay.

No decoding or model loading occurs here. Window completion uses the actual
50 Hz, half-open extraction bounds; nominal onset + 1 s is a sensitivity check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from braindecoding.config import PROJECT_ROOT
from braindecoding.reporting import study_report_dir


ROOT = PROJECT_ROOT
STUDY = "single_revision_c1_input_wait_test_v2"
REPORT = study_report_dir(None, STUDY)
PARENT = ROOT / "data/manifests/rolling_sentence/heldout_test_v2.json"
STABLE = ROOT / "data/manifests/rolling_sentence/single_revision_c1_test_v2_stable.json"
COMPLETED = study_report_dir(None, "single_revision_c1_test_v2_stable")
LATENCY = ROOT / "data/results/exports/single_revision_c1_test_v2_stable_latency_by_word.csv"
BUDGETS = {"A": 0, "C1": 1, "C": 2}
NAMES = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2"}
SEEDS = [0, 1, 2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(value):
    payload = {k: v for k, v in value.items() if k != "manifest_sha256"}
    observed = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    require(observed == value["manifest_sha256"], "Frozen manifest identity changed")


def timing_for_group(onsets, window_ends, budget):
    """An overlapping next window adds its endpoint gap, not a second 1 s."""
    onsets, window_ends = (np.asarray(v, dtype=float) for v in (onsets, window_ends))
    require(onsets.ndim == 1 and onsets.size > 0 and onsets.shape == window_ends.shape,
            "Timing population is empty or misaligned")
    require(isinstance(budget, int) and budget >= 0, "Invalid future-window budget")
    require(np.isfinite(onsets).all() and np.isfinite(window_ends).all(), "Nonfinite timing")
    require((np.diff(onsets) >= 0).all() and (np.diff(window_ends) >= 0).all(),
            "Group timing is not monotonic; do not sort a frozen replay")
    require((window_ends >= onsets).all(), "Input completed before target onset")
    positions = np.arange(len(onsets))
    freeze = np.minimum(positions + budget, len(onsets) - 1)
    return dict(freeze=freeze, windows=freeze - positions,
                first=window_ends - onsets,
                extra=window_ends[freeze] - window_ends,
                final=window_ends[freeze] - onsets)


def pallier_bounds(onsets, clock, dataset_config):
    """Mirror recording_onset_to_sample/extract_word_window without signal I/O."""
    source_rate = float(clock["source_sampling_rate_hz"])
    target_rate = float(clock["output_sampling_rate_hz"])
    require(source_rate == dataset_config["source_sampling_rate_hz"]
            and target_rate == dataset_config["target_sampling_rate_hz"], "Pallier clock rate changed")
    origin = float(clock["source_first_samp"]) / source_rate
    # MNE preserves first_samp on its recording clock; the stored NumPy array
    # still starts at index zero. Extraction subtracts the source-clock origin.
    require(abs(clock["source_recording_start_seconds"] - origin) < 1e-9
            and abs(clock["output_recording_start_seconds"] - clock["output_first_samp"] / target_rate) < 1e-9
            and abs(clock["output_recording_start_seconds"] - origin) <= .5 / target_rate + 1e-9,
            "Pallier recording-clock convention changed")
    starts = np.rint((np.asarray(onsets) + dataset_config["window_start_offset_seconds"] - origin)
                     * target_rate).astype(np.int64)
    stops = starts + round(dataset_config["window_seconds"] * target_rate)
    require((starts >= 0).all() and (stops <= clock["output_sample_count"]).all(),
            "Pallier word window exceeds the cached recording")
    return starts, stops, origin + stops / target_rate


def describe(frame, column):
    values = frame[column].to_numpy(dtype=float)
    require(len(values) > 0 and np.isfinite(values).all(), "Empty or nonfinite descriptive statistics")
    q = np.quantile(values, [0, .5, .9, .95, .99, 1], method="linear")
    return dict(n=len(values), mean_seconds=float(values.mean()),
                subject_equal_mean_seconds=float(frame.groupby("subject")[column].mean().mean()),
                **{k: float(v) for k, v in zip(
                    ["min_seconds", "median_seconds", "p90_seconds", "p95_seconds", "p99_seconds", "max_seconds"], q)},
                zero_seconds_count=int(np.sum(np.abs(values) < 1e-8)),
                over_1_second_count=int(np.sum(values > 1 + 1e-8)))


def verify_schedule(latency, groups):
    """Verify every seed/condition/event; elapsed-time rows are NOT seed copies."""
    require(set(latency["condition"]) == {"C1", "C"} and sorted(latency["seed"].unique()) == SEEDS,
            "Completed latency export has an unexpected condition/seed population")
    identities = ["dataset", "condition", "seed", "event_id"]
    require(not latency.duplicated(identities).any(), "Duplicate latency decision")
    expected = []
    for dataset, info in groups.items():
        for group in info["groups"]:
            for position, event in enumerate(group["event_ids"]):
                for condition in ("C1", "C"):
                    freeze = min(position + BUDGETS[condition], len(group["event_ids"]) - 1)
                    for seed in SEEDS:
                        expected.append(dict(dataset=dataset, condition=condition, seed=seed, event_id=event,
                                             group=group["group"], subject=group["subject"], position=position,
                                             first_arrival=position, freeze_arrival=freeze,
                                             wait_subsequent_word_windows=freeze-position,
                                             allowed_revision_opportunities=freeze-position))
    expected = pd.DataFrame(expected).set_index(identities).sort_index()
    actual = latency.set_index(identities).sort_index()
    require(expected.index.equals(actual.index), "Latency export does not cover all frozen word positions")
    for column in expected:
        require(expected[column].equals(actual[column]), f"Latency schedule changed: {column}")
    changes = actual["actual_text_changes"]
    require((changes >= 0).all() and (changes <= actual["allowed_revision_opportunities"]).all(),
            "Actual text changes exceed the scheduled revision budget")
    return dict(word_position_seed_condition_decisions=len(latency),
                seeds_verified=SEEDS, schedule_identical_across_seeds=True,
                C1_max_wait_windows=int(latency.loc[latency.condition.eq("C1"), "wait_subsequent_word_windows"].max()),
                C1_max_text_changes=int(latency.loc[latency.condition.eq("C1"), "actual_text_changes"].max()))


def build_rows(parent, bind):
    rows, contracts = [], {}
    for dataset, info in parent["datasets"].items():
        jobs = [j for j in parent["checkpoints"] if j["dataset"] == dataset]
        require(sorted(j["seed"] for j in jobs) == SEEDS, "Missing frozen timing configuration")
        configs = []
        for job in jobs:
            path = Path(job["path"]) / "resolved_config.yaml"
            bind(path, parent["source_sha256"][str(path)])
            configs.append(yaml.safe_load(path.read_text(encoding="utf-8")))
        cfg = configs[0]
        fields = ["window_seconds", "window_start_offset_seconds", "source_sampling_rate_hz", "target_sampling_rate_hz"]
        contract = {k: cfg["dataset"][k] for k in fields}
        for other in configs:
            require({k: other["dataset"][k] for k in fields} == contract
                    and other["cache"] == cfg["cache"], "Timing contract/cache differs between seeds")
        require(contract["window_seconds"] == 1.0 and contract["window_start_offset_seconds"] == 0.0
                and contract["target_sampling_rate_hz"] == 50.0, "Frozen input-window contract changed")
        contracts[dataset] = contract
        event_path = Path(cfg["cache"]["event_table"])
        bind(event_path, parent["source_sha256"][str(event_path)])
        table = pd.read_csv(event_path, keep_default_na=False)
        require(not table["事件编号"].duplicated().any(), "Duplicate canonical event identity")
        table = table.set_index("事件编号")
        clocks, seen = {}, set()
        for group in info["groups"]:
            events = group["event_ids"]
            require(events and not seen.intersection(events), "Repeated/empty frozen test group")
            seen.update(events)
            records = table.loc[events]
            require(records["受试者"].eq(group["subject"]).all()
                    and records["记录编号"].eq(group["recording"]).all()
                    and records["数据划分"].eq("test").all()
                    and records["是否可训练"].astype(str).str.lower().eq("true").all(),
                    "Timing identity differs from frozen test population")
            onsets = records["开始时间"].to_numpy(dtype=float)
            if dataset == "pallier2025":
                if group["recording"] not in clocks:
                    clock_path = Path(cfg["cache"]["meg_dir"]) / group["subject"] / (
                        f"{group['subject']}_{cfg['dataset']['session']}_task-listen_{group['scope']}_meg.json")
                    bind(clock_path)
                    clocks[group["recording"]] = read(clock_path)
                clock = clocks[group["recording"]]
                require(clock["recording_id"] == group["recording"], "Pallier sidecar identity changed")
                starts, stops, ends = pallier_bounds(onsets, clock, cfg["dataset"])
                provenance = "BIDS_recording_clock_onset"
                voice, flags = [""] * len(events), [""] * len(events)
            else:
                starts = records["窗口开始目标采样点"].to_numpy(dtype=np.int64)
                stops = records["窗口结束目标采样点"].to_numpy(dtype=np.int64)
                require((stops-starts == 50).all()
                        and records["目标采样率Hz"].eq(50.).all()
                        and records["源采样率Hz"].eq(contract["source_sampling_rate_hz"]).all()
                        and records["窗口完整"].astype(str).str.lower().eq("true").all(),
                        "Chinese extraction bounds changed")
                expected_starts = np.floor(records["窗口开始源采样点"].to_numpy(dtype=float)
                                          * 50. / contract["source_sampling_rate_hz"] + .5).astype(np.int64)
                require(np.array_equal(starts, expected_starts), "Chinese sample-grid conversion changed")
                require(records["对齐状态"].eq("actual_reading_text_qwen_estimated_timing").all()
                        and records["同步状态"].eq("audio_seconds_used_as_eeg_seconds_by_user_request").all()
                        and records["已修正硬件延迟"].astype(str).str.lower().eq("false").all(),
                        "Chinese timing provenance changed")
                ends = stops / 50.
                provenance = "estimated_audio_clock_used_as_EEG_clock_no_hardware_delay_correction"
                voice = records["声音版本"].tolist()
                flags = records["时间戳质检提示"].tolist()
            nominal = onsets + contract["window_seconds"] + contract["window_start_offset_seconds"]
            require(np.max(np.abs(ends-nominal)) <= .012000001, "Unexpected input-window quantization")
            for condition, budget in BUDGETS.items():
                timing = timing_for_group(onsets, ends, budget)
                nominal_timing = timing_for_group(onsets, nominal, budget)
                for i, event in enumerate(events):
                    j = int(timing["freeze"][i])
                    rows.append(dict(dataset=dataset, subject=group["subject"], condition=condition,
                                     group=group["group"], recording=group["recording"], material=group["material"],
                                     scope=group["scope"], voice=voice[i], event_id=event, position=i,
                                     group_words=len(events), has_next_word=i+1 < len(events),
                                     freeze_position=j, freeze_event_id=events[j], wait_windows=int(timing["windows"][i]),
                                     target_onset_seconds=float(onsets[i]), freeze_word_onset_seconds=float(onsets[j]),
                                     input_start_target_sample=int(starts[i]), input_stop_target_sample=int(stops[i]),
                                     first_input_available_seconds=float(ends[i]), final_input_available_seconds=float(ends[j]),
                                     first_input_wait_seconds=float(timing["first"][i]),
                                     extra_input_wait_seconds=float(timing["extra"][i]),
                                     final_input_wait_seconds=float(timing["final"][i]),
                                     nominal_extra_input_wait_seconds=float(nominal_timing["extra"][i]),
                                     nominal_final_input_wait_seconds=float(nominal_timing["final"][i]),
                                     timing_provenance=provenance, timing_qc_flag=flags[i], freeze_timing_qc_flag=flags[j]))
        require(len(seen) == info["counts"]["events"]
                and len(info["groups"]) == info["counts"]["groups"], "Frozen dataset count changed")
    return pd.DataFrame(rows), contracts


def summarize(frame, parent):
    summaries, subject_rows = {}, []
    columns = {"first_input": "first_input_wait_seconds", "extra_input": "extra_input_wait_seconds",
               "final_input": "final_input_wait_seconds", "nominal_extra_input": "nominal_extra_input_wait_seconds",
               "nominal_final_input": "nominal_final_input_wait_seconds"}
    for dataset, data in frame.groupby("dataset", sort=False):
        cohorts = {}
        for name, subset in (("all_positions", data), ("has_next_word", data[data.has_next_word]),
                             ("segment_final", data[~data.has_next_word])):
            cohorts[name] = {}
            for condition, cell in subset.groupby("condition", sort=False):
                cohorts[name][condition] = {k: describe(cell, v) for k, v in columns.items()}
                for subject, subject_cell in cell.groupby("subject"):
                    for metric, column in columns.items():
                        subject_rows.append(dict(dataset=dataset, cohort=name, condition=condition, subject=subject,
                                                 metric=metric, **describe(subject_cell, column)))
        c1 = data[data.condition.eq("C1")]
        c = data[data.condition.eq("C")]
        require((c1[~c1.has_next_word].extra_input_wait_seconds == 0).all(), "Segment final waited for a future input")
        all_c1, all_c = cohorts["all_positions"]["C1"], cohorts["all_positions"]["C"]
        summaries[dataset] = dict(
            counts=dict(parent["datasets"][dataset]["counts"], eligible_for_one_window=int(c1.has_next_word.sum()),
                        segment_final_positions=int((~c1.has_next_word).sum()),
                        zero_seconds_nonfinal_positions=int((c1.has_next_word & c1.extra_input_wait_seconds.abs().lt(1e-8)).sum()),
                        C1_wait_window_counts={str(int(k)): int(v) for k, v in c1.wait_windows.value_counts().sort_index().items()},
                        C_wait_window_counts={str(int(k)): int(v) for k, v in c.wait_windows.value_counts().sort_index().items()},
                        timing_qc_flagged_positions=int(c1.timing_qc_flag.ne("").sum())),
            cohorts=cohorts,
            C_to_C1_mean_extra_wait_reduction_percent=(1-all_c1["extra_input"]["mean_seconds"] / all_c["extra_input"]["mean_seconds"]) * 100,
            quantization=dict(max_abs_extra_vs_nominal_seconds=float(
                (c1.extra_input_wait_seconds-c1.nominal_extra_input_wait_seconds).abs().max()),
                max_abs_final_vs_nominal_seconds=float(
                    (c1.final_input_wait_seconds-c1.nominal_final_input_wait_seconds).abs().max())),
            longest_C1_waits=c1.sort_values("extra_input_wait_seconds", ascending=False).head(10).to_dict(orient="records"))
        if dataset == "chineseeeg2_littleprince":
            clean = c1[c1.has_next_word & c1.timing_qc_flag.eq("") & c1.freeze_timing_qc_flag.eq("")]
            summaries[dataset]["unflagged_nonfinal_pairs_sensitivity"] = {
                k: describe(clean, v) for k, v in columns.items()}
    return summaries, pd.DataFrame(subject_rows)


def markdown(result):
    datasets = result["datasets"]
    lines = ["# C1 输入等待统计：完整冻结测试集", "",
             "状态：完成。逐词位核对两个数据集的三个种子；时间统计中每个受试者的测试词位只计一次，种子不重复增加分母。", "",
             "输入等待按实际 50 Hz 切窗的右开端点计算。设目标词起始为 tᵢ、其一秒输入窗终点为 eᵢ，"
             "则 A 首次输入齐备为 eᵢ；C1 在段内下一词输入齐备 eᵢ₊₁ 时冻结。额外等待为 eᵢ₊₁−eᵢ，"
             "总输入等待为 eᵢ₊₁−tᵢ。段末词沿用首次 K0，额外等待为零。C 最多等两个后续词窗；所有条件在冻结段末截断。", "",
             "两个数据集输入窗均为 1.0 秒、起始偏移 0 秒。Pallier 的 3 秒资格支持窗不作为模型输入等待。"
             "相邻一秒窗可以重叠，因此等一词窗不等于再加一秒，也不构成固定秒数上限。", "",
             "以下均值、分位数以全部测试词位汇总，包含段末词；分位数使用线性插值。受试者等权均值、"
             "逐受试者统计另存 JSON/CSV。重复刺激不视为独立统计样本，未对描述性等待时间做显著性检验。", "",
             "## 全部测试词位：额外输入等待（秒）", "",
             "| 数据集 | 词位数 | 策略 | 均值 | 中位数 | P90 | P95 | P99 | 最大值 |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for dataset, info in datasets.items():
        for condition in BUDGETS:
            s = info["cohorts"]["all_positions"][condition]["extra_input"]
            values = " | ".join(f"{s[k]:.3f}" for k in
                                ["mean_seconds", "median_seconds", "p90_seconds", "p95_seconds", "p99_seconds", "max_seconds"])
            lines.append(f"| {NAMES[dataset]} | {s['n']} | {condition} | {values} |")
    lines += ["", "## 从目标词开始到最终输入齐备（秒）", "",
              "| 数据集 | 策略 | 均值 | 中位数 | P95 | 最大值 |", "|---|---|---:|---:|---:|---:|"]
    for dataset, info in datasets.items():
        for condition in BUDGETS:
            s = info["cohorts"]["all_positions"][condition]["final_input"]
            values = " | ".join(f"{s[k]:.3f}" for k in ["mean_seconds", "median_seconds", "p95_seconds", "max_seconds"])
            lines.append(f"| {NAMES[dataset]} | {condition} | {values} |")
    lines += ["", "## C1 有后续词的词位（排除段末）", "",
              "| 数据集 | 可回修词位 | 占全部词位 | 额外等待均值 | 中位数 | P95 | 最大值 | 总等待均值 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    explanations = []
    for dataset, info in datasets.items():
        s = info["cohorts"]["has_next_word"]["C1"]["extra_input"]
        total = info["cohorts"]["has_next_word"]["C1"]["final_input"]
        counts = info["counts"]
        lines.append(f"| {NAMES[dataset]} | {s['n']} | {s['n']/counts['events']*100:.2f}% | "
                     f"{s['mean_seconds']:.3f} | {s['median_seconds']:.3f} | {s['p95_seconds']:.3f} | "
                     f"{s['max_seconds']:.3f} | {total['mean_seconds']:.3f} |")
        explanations += ["", f"{NAMES[dataset]}：段末 {counts['segment_final_positions']} 个词位额外等 0 窗；"
                         f"其余 {counts['eligible_for_one_window']} 个等 1 窗。非段末中 {counts['zero_seconds_nonfinal_positions']} 个"
                         f"词位因时间戳或采样端点重合而额外等待 0 秒，仍属于一次后续词窗机会。"
                         f"C1 相比 C 的平均额外等待减少 {info['C_to_C1_mean_extra_wait_reduction_percent']:.2f}%。"]
    lines += explanations
    chinese = datasets["chineseeeg2_littleprince"]
    clean = chinese["unflagged_nonfinal_pairs_sensitivity"]["extra_input"]
    lines += ["", "## 时间戳来源与边界", "",
              "Pallier2025 使用冻结 BIDS 记录时钟 onset，按每条连续 MEG 缓存的 source_first_samp 校正零点，"
              "再应用实际 50 Hz 最近采样点取整。ChineseEEG-2 使用冻结的实际朗读文本估计时间戳，"
              "沿用已授权的音频秒数作为 EEG 秒数；未修正硬件延迟。", "",
              f"中文测试词位中 {chinese['counts']['timing_qc_flagged_positions']} 个已有时间戳质控提示，主统计全部保留。"
              "最长额外等待 6.16 秒发生在 f1 第 9 章首词起点 0 秒到下一词 6.16 秒，"
              "首词时长为零且带“非正时长”标记，四名受试者共享这一刺激时间线。"
              "此最大值是冻结估计时间线上的数值，不能视为实测部署延迟上界。", "",
              f"仅作为敏感性检查，中文非段末且目标/下一词均无质控提示的 {clean['n']} 个词位，"
              f"额外等待均值 {clean['mean_seconds']:.3f} 秒、P95 {clean['p95_seconds']:.3f} 秒、"
              f"最大值 {clean['max_seconds']:.3f} 秒；这仍属于估计时间戳分析，不替代完整主统计。", "",
              "本报告衡量已知词位、已知段边界下的输入可用时间；不包含事件边界检测、预处理、"
              "神经前向、LM 重搜或输出传输耗时，也不把离线预处理视为经过验证的在线因果流程。", "",
              "## 取整敏感性与复现", "",
              "名义时间按 onset+1.0 秒另行计算，保存在 JSON/逐词 CSV。实际时间依据输入样本范围的右开端点，"
              "与最后一个样本的时间点区分开来。"]
    for dataset, info in datasets.items():
        lines.append(f"{NAMES[dataset]}：C1 额外等待相对名义值的逐词最大绝对差 "
                     f"{info['quantization']['max_abs_extra_vs_nominal_seconds']*1000:.3f} ms，"
                     f"总输入等待最大绝对差 {info['quantization']['max_abs_final_vs_nominal_seconds']*1000:.3f} ms。")
    lines += ["", "运行：`python -m braindecoding.c1_input_wait_test_v2`，仅使用 CPU。", "",
              "- [完整摘要](results.json)", "- [逐词位/策略时间明细](by_word.csv)",
              "- [逐受试者与队列统计](by_subject.csv)", "- [输入来源、输出哈希与核对记录](audit.json)", "",
              f"父测试 manifest：`{result['parent_manifest_sha256']}`。", "",
              f"完成的 C1 manifest：`{result['C1_manifest_sha256']}`。", ""]
    return "\n".join(lines)


def main():
    sources = {}

    def bind(path, expected=None):
        if Path(path).suffix in (".py", ".yaml", ".yml", ".md", ".tex"):
            return
        path = Path(path).resolve()
        key = path.relative_to(ROOT).as_posix()
        observed = sha256(path)
        require(expected is None or observed == expected, f"Frozen source hash changed: {key}")
        require(key not in sources or sources[key] == observed, f"Source changed during analysis: {key}")
        sources[key] = observed

    for path in (PARENT, STABLE, COMPLETED / "audit.json", COMPLETED / "results.json"):
        bind(path)
    parent, stable, audit, completed = read(PARENT), read(STABLE), read(COMPLETED / "audit.json"), read(COMPLETED / "results.json")
    verify_manifest(parent)
    verify_manifest(stable)
    require(stable["parent_manifest_sha256"] == parent["manifest_sha256"]
            and audit["manifest_sha256"] == stable["manifest_sha256"] == completed["manifest_sha256"],
            "Completed C1 provenance differs from frozen test")
    require(audit["status"] == "verified_complete" and completed["status"] == "completed"
            and completed["seeds"] == SEEDS, "C1 is not complete for all three seeds")
    bind(COMPLETED / "results.json", audit["results_sha256"])
    bind(COMPLETED / "results.md", audit["markdown_sha256"])
    latency_key = LATENCY.relative_to(ROOT).as_posix()
    bind(LATENCY, audit["exports"][latency_key])
    for code in ("braindecoding/data/pallier2025.py", "braindecoding/data/chineseeeg2.py"):
        bind(ROOT / code, audit["source_sha256"].get(code))
    bind(Path(__file__))
    bind(ROOT / "tests/test_c1_input_wait_test_v2.py")
    require(len(audit["sources"]) == 6, "Missing completed C1 job sources")
    chunk_count = 0
    for source in audit["sources"]:
        bind(ROOT / source["path"], source["sha256"])
        for chunk in source["chunks"]:
            bind(ROOT / chunk["path"], chunk["sha256"])
            chunk_count += 1
    latency = pd.read_csv(LATENCY, keep_default_na=False)
    checks = verify_schedule(latency, parent["datasets"])
    frame, contracts = build_rows(parent, bind)
    datasets, by_subject = summarize(frame, parent)
    # Sources, including completed outputs, must still be identical after analysis.
    for key, expected in sources.items():
        require(sha256(ROOT / key) == expected, f"Source changed during analysis: {key}")
    checks.update(source_hashes_verified_before_and_after=True,
                  complete_job_sources=6, complete_chunk_sources=chunk_count,
                  elapsed_time_unique_word_positions=int(frame.condition.eq("C1").sum()),
                  elapsed_time_strategy_rows=len(frame), duplicate_seed_timing_rows=False,
                  group_timing_monotonic=True, all_test_events_retained=True,
                  segment_final_extra_wait_zero=True, GPU_used=False)
    result = dict(status="completed", study=STUDY, parent_manifest_sha256=parent["manifest_sha256"],
                  C1_manifest_sha256=stable["manifest_sha256"], seeds_verified=SEEDS,
                  timing_unit="seconds_at_half_open_input_window_end", input_window_contract=contracts,
                  aggregation="Pooled unique subject-word positions; subject-equal means also exported; no seed replication",
                  quantile_method="numpy linear", conditions=BUDGETS,
                  runtime_latency_measured=False, datasets=datasets, checks=checks)
    REPORT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORT / "by_word.csv", index=False, encoding="utf-8-sig")
    by_subject.to_csv(REPORT / "by_subject.csv", index=False, encoding="utf-8-sig")
    (REPORT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (REPORT / "results.md").write_text(markdown(result), encoding="utf-8")
    output_hashes = {name: sha256(REPORT / name) for name in ("by_word.csv", "by_subject.csv", "results.json", "results.md")}
    audit_result = dict(status="verified_complete", parent_manifest_sha256=parent["manifest_sha256"],
                        C1_manifest_sha256=stable["manifest_sha256"], source_sha256=sources,
                        output_sha256=output_hashes, checks=checks)
    (REPORT / "audit.json").write_text(json.dumps(audit_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="completed", report=REPORT.relative_to(ROOT).as_posix(), checks=checks), ensure_ascii=True))


if __name__ == "__main__":
    main()
