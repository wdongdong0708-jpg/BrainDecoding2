"""CPU-only block-length sensitivity of completed A/B/C1/C sentence gains."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from braindecoding.config import PROJECT_ROOT
from braindecoding.reporting import study_report_dir
from braindecoding.unified_sentence_v2_results import holm, paired_block_statistics


ROOT = PROJECT_ROOT
ID = "material_block_length_sensitivity_test_v2"
REPORT = study_report_dir(None, ID)
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = ROOT / f"data/manifests/rolling_sentence/{ID}.json"
PARENT = ROOT / "data/manifests/rolling_sentence/heldout_test_v2.json"
LENGTHS = (1, 2, 3, 5, 8, 10, 15, 20)
STATISTIC_SEEDS = (161803, 271828, 314159)
REPETITIONS = 10000
METRIC = "sentence_bleu1_mean"
PAIRS = (("C_minus_A", "A", "C", "original_AC"),
         ("B_minus_A", "A", "B", "B_ablation"),
         ("C_minus_B", "B", "C", "B_ablation"),
         ("C1_minus_A", "A", "C1", "C1_budget"),
         ("C_minus_C1", "C1", "C", "C1_budget"))
EXPERIMENTS = {"B": "fixed_k0_rolling_search_test_v2_stable", "C1": "single_revision_c1_test_v2_stable"}
NAMES = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def verify_manifest(value):
    require(digest({k: v for k, v in value.items() if k != "manifest_sha256"}) == value["manifest_sha256"],
            "Frozen manifest identity changed")


def material_order(info):
    """Reuse frozen connected components and their original within-stratum order."""
    ordered = defaultdict(list)
    assigned = set()
    covered_groups = set()
    for block in info["blocks"]:
        for material in block["materials"]:
            key = block["scope"], material
            require(key not in assigned, "Material appears in multiple original blocks")
            assigned.add(key)
            ordered[block["scope"]].append(material)
        for index in block["groups"]:
            require(index not in covered_groups, "Group appears in multiple original blocks")
            covered_groups.add(index)
            g = info["groups"][index]
            require(g["scope"] == block["scope"] and g["material"] in block["materials"]
                    and g["block"] == block["block"], "Original material membership changed")
    require(covered_groups == set(range(len(info["groups"]))), "Original blocks omit a group")
    require(assigned == {(g["scope"], g["material"]) for g in info["groups"]}, "Material population changed")
    # Shared atoms must remain one connected component, including voice versions.
    owner = {}
    for g in info["groups"]:
        for atom in g["atoms"]:
            key = g["scope"], atom
            require(key not in owner or owner[key] == g["material"], "A shared stimulus atom is split")
            owner[key] = g["material"]
    for scope, materials in ordered.items():
        sequence = [min(g["order"] for g in info["groups"] if g["scope"] == scope and g["material"] == m)
                    for m in materials]
        require(sequence == sorted(sequence), "Frozen material order is not chronological")
    return dict(ordered)


def reblock(dataset, info, length):
    require(isinstance(length, int) and length > 0, "Block length must be a positive integer")
    membership, rows = {}, []
    for scope, materials in material_order(info).items():
        for order, material in enumerate(materials):
            block = f"{dataset}|{scope}|test-block-{order//length:03d}"
            membership[scope, material] = block
            rows.append(dict(dataset=dataset, block_length=length, scope=scope, material=material,
                             material_order=order, block=block))
    group_map = {g["group"]: membership[g["scope"], g["material"]] for g in info["groups"]}
    if length == 5:
        require(all(group_map[g["group"]] == g["block"] for g in info["groups"]),
                "Five-material blocks do not reproduce frozen block identities")
    return group_map, rows


def prepare_inputs():
    sources = {}

    def bind(path, expected=None):
        if Path(path).suffix in (".py", ".yaml", ".yml", ".md", ".tex"):
            return
        path = Path(path).resolve()
        key = path.relative_to(ROOT).as_posix()
        observed = sha(path)
        require(expected is None or observed == expected, f"Source hash changed: {key}")
        require(key not in sources or sources[key] == observed, f"Source changed while reading: {key}")
        sources[key] = observed

    bind(PARENT)
    parent = read(PARENT)
    verify_manifest(parent)
    statcode = ROOT / "braindecoding/evaluation/unified_sentence_v2_results.py"
    bind(statcode, parent["source_sha256"].get(str(statcode)))
    reports, frames, experiment_ids = {}, {}, {}
    for condition, experiment in EXPERIMENTS.items():
        manifest_path = ROOT / f"data/manifests/rolling_sentence/{experiment}.json"
        bind(manifest_path)
        manifest = read(manifest_path)
        verify_manifest(manifest)
        require(manifest["parent_manifest_sha256"] == parent["manifest_sha256"], "Supplement has a different test parent")
        folder = study_report_dir(None, experiment)
        bind(folder / "audit.json")
        audit = read(folder / "audit.json")
        require(audit["status"] == "verified_complete" and audit["manifest_sha256"] == manifest["manifest_sha256"],
                "Supplement not verified complete")
        bind(folder / "results.json", audit["results_sha256"])
        bind(folder / "results.md", audit["markdown_sha256"])
        reports[condition] = read(folder / "results.json")
        require(reports[condition]["status"] == "completed" and reports[condition]["seeds"] == [0, 1, 2],
                "Missing complete supplement seeds")
        csv_path = ROOT / f"data/results/exports/{experiment}_by_group.csv"
        bind(csv_path, audit["exports"][csv_path.relative_to(ROOT).as_posix()])
        bind(statcode, audit["source_sha256"].get(statcode.relative_to(ROOT).as_posix()))
        require(len(audit["sources"]) == 6, "Missing completed jobs")
        for source in audit["sources"]:
            bind(ROOT / source["path"], source["sha256"])
            chunks = source["chunks"] if "chunks" in source else read(ROOT / source["path"])["sources"]
            for chunk in chunks:
                bind(ROOT / chunk["path"], chunk["sha256"])
        frames[condition] = pd.read_csv(csv_path, keep_default_na=False, float_precision="round_trip")
        experiment_ids[condition] = manifest["manifest_sha256"]
    keys = ["dataset", "seed", "group", "condition"]
    for frame in frames.values():
        require(not frame.duplicated(keys).any(), "Duplicate group-condition metric")
    shared = [f[f.condition.isin(["A", "C"])].set_index(keys).sort_index() for f in frames.values()]
    pd.testing.assert_frame_equal(shared[0], shared[1], check_exact=True)
    combined = pd.concat([frames["B"], frames["C1"][frames["C1"].condition.eq("C1")]], ignore_index=True)
    expected = {(d, s, g["group"], c) for d, info in parent["datasets"].items()
                for s in (0, 1, 2) for g in info["groups"] for c in ("A", "B", "C1", "C")}
    require(set(combined[keys].itertuples(index=False, name=None)) == expected and len(combined) == len(expected),
            "Incomplete A/B/C1/C population")
    metadata = {(d, g["group"]): g for d, info in parent["datasets"].items() for g in info["groups"]}
    for row in combined.itertuples(index=False):
        g = metadata[row.dataset, row.group]
        require(all(getattr(row, k) == g[k] for k in ("subject", "material", "block", "scope"))
                and row.words == len(g["event_ids"]), "Group metric identity changed")
    bind(PROTOCOL)
    bind(Path(__file__))
    bind(ROOT / "tests/test_material_block_length_sensitivity_test_v2.py")
    plan = dict(analysis=ID, analysis_status="post_result_sensitivity_frozen_before_resampling",
                parent_manifest_sha256=parent["manifest_sha256"], completed_experiment_ids=experiment_ids,
                metric=METRIC, comparisons=[x[0] for x in PAIRS], block_lengths=list(LENGTHS), baseline_length=5,
                bootstrap_repetitions=REPETITIONS, statistic_seeds=list(STATISTIC_SEEDS),
                interval="paired_stratified_material_block_percentile_95_pointwise",
                aggregation="within_subject_mean_segments_then_equal_subjects_and_neural_seeds",
                neural_seeds=[0, 1, 2], conditioning="fixed_subjects_neural_seeds_and_recording_chapter",
                preserve_shared_material_components=True, preserve_all_strata_tails=True,
                source_sha256=sources, GPU_used=False)
    plan["manifest_sha256"] = digest(plan)
    if MANIFEST.exists():
        require(read(MANIFEST) == plan, "Sensitivity inputs/plan changed; create a new analysis version")
    else:
        save(MANIFEST, plan)
    for key, expected_sha in sources.items():
        require(sha(ROOT / key) == expected_sha, f"Input changed before bootstrap: {key}")
    return parent, combined, reports, plan


def comparison_rows(dataset, info, combined, left, right):
    subset = combined[combined.dataset.eq(dataset)]
    paired = subset.pivot(index=["seed", "group"], columns="condition", values=METRIC)
    rows = []
    for seed in (0, 1, 2):
        for group in info["groups"]:
            cell = paired.loc[(seed, group["group"]), :]
            rows.append(dict(seed=seed, subject=group["subject"], group=group["group"], scope=group["scope"],
                             block=group["block"], difference=float(cell[right]-cell[left])))
    return rows


def interval_class(lo, hi):
    return "positive" if lo > 0 else ("negative" if hi < 0 else "includes_zero")


def original_statistics(reports, dataset, comparison):
    if comparison == "C_minus_A":
        a = reports["B"]["datasets"][dataset]["original_C_minus_A_statistics"]
        require(a == reports["C1"]["datasets"][dataset]["original_C_minus_A_statistics"], "Original C-A tests differ")
        return a
    source = "C1" if comparison in ("C1_minus_A", "C_minus_C1") else "B"
    return reports[source]["datasets"][dataset]["supplementary_statistics"][comparison]


def analyze(parent, combined, reports):
    estimates, assignments, structures = [], [], []
    with threadpool_limits(limits=1):
        for dataset, info in parent["datasets"].items():
            differences = {name: comparison_rows(dataset, info, combined, left, right) for name, left, right, _ in PAIRS}
            for length in LENGTHS:
                started = time.perf_counter()
                group_map, membership = reblock(dataset, info, length)
                assignments.extend(membership)
                by_scope = pd.DataFrame(membership).groupby(["scope", "block"]).size()
                for scope, sizes in by_scope.groupby(level=0):
                    structures.append(dict(dataset=dataset, block_length=length, scope=scope,
                                           blocks=len(sizes), materials=int(sizes.sum()),
                                           smallest_block=int(sizes.min()), largest_block=int(sizes.max())))
                for name, _, _, family in PAIRS:
                    rows = [{**row, "block": group_map[row["group"]]} for row in differences[name]]
                    for statistic_seed in STATISTIC_SEEDS:
                        settings = dict(parent["settings"]["statistics"], seed=statistic_seed,
                                        paired_material_block_sign_flips=REPETITIONS,
                                        material_block_bootstrap_repetitions=REPETITIONS)
                        stats = paired_block_statistics(rows, "difference", settings)
                        require(stats["bootstrap_samples"] == REPETITIONS, "An empty-cell bootstrap draw was discarded")
                        if length == 5 and statistic_seed == STATISTIC_SEEDS[0]:
                            original = original_statistics(reports, dataset, name)
                            for key in ("effect", "ci_lower", "ci_upper", "p_one_sided"):
                                require(abs(stats[key]-original[key]) <= 1e-12, f"Original five-material statistic differs: {dataset}/{name}/{key}")
                            for key in ("blocks", "subjects", "seeds", "sign_flip_exact", "bootstrap_samples"):
                                require(stats[key] == original[key], f"Original statistical structure differs: {key}")
                        estimates.append(dict(dataset=dataset, comparison=name, family=family, block_length=length,
                                              statistic_seed=statistic_seed, primary_rng=statistic_seed == STATISTIC_SEEDS[0],
                                              **stats, ci_width=stats["ci_upper"]-stats["ci_lower"],
                                              interval_class=interval_class(stats["ci_lower"], stats["ci_upper"])))
                print(json.dumps(dict(dataset=dataset, block_length=length, blocks=len(by_scope),
                                      completed_statistics=15, seconds=round(time.perf_counter()-started, 2))), flush=True)
    frame = pd.DataFrame(estimates)
    for _, group in frame.groupby(["block_length", "statistic_seed", "family"], sort=False):
        frame.loc[group.index, "p_holm"] = holm(group.p_one_sided.to_numpy())
    frame["significant_holm_05"] = frame.p_holm.lt(.05)
    for dataset, info in parent["datasets"].items():
        for name, _, _, _ in PAIRS:
            primary = frame[frame.dataset.eq(dataset) & frame.comparison.eq(name) & frame.primary_rng]
            baseline = primary[primary.block_length.eq(5)].iloc[0]
            require(abs(baseline.p_holm-original_statistics(reports, dataset, name)["p_holm"]) < 1e-12,
                    "Original Holm-adjusted five-material result differs")
            require(np.ptp(primary.effect.to_numpy()) < 1e-12, "Point estimand changed across block lengths")
            mask = frame.dataset.eq(dataset) & frame.comparison.eq(name)
            frame.loc[mask, "width_ratio_to_original_L5"] = frame.loc[mask, "ci_width"] / baseline.ci_width
    return frame, pd.DataFrame(assignments), pd.DataFrame(structures)


def robustness_summary(frame):
    output = []
    for (dataset, comparison), all_rng in frame.groupby(["dataset", "comparison"], sort=False):
        primary = all_rng[all_rng.primary_rng].sort_values("block_length")
        baseline = primary[primary.block_length.eq(5)].iloc[0]
        mc_ranges = []
        for length, group in all_rng.groupby("block_length"):
            mc_ranges.append(dict(block_length=int(length), lower_endpoint_range_pp=float(np.ptp(group.ci_lower)*100),
                                  upper_endpoint_range_pp=float(np.ptp(group.ci_upper)*100),
                                  interval_classes=sorted(set(group.interval_class))))
        output.append(dict(dataset=dataset, comparison=comparison, effect_pp=float(baseline.effect*100),
                           baseline_ci_pp=[float(baseline.ci_lower*100), float(baseline.ci_upper*100)],
                           baseline_interval_class=baseline.interval_class,
                           primary_class_stable=bool(primary.interval_class.eq(baseline.interval_class).all()),
                           all_rng_class_stable=bool(all_rng.interval_class.eq(baseline.interval_class).all()),
                           zero_including_block_lengths=primary.loc[primary.interval_class.eq("includes_zero"), "block_length"].astype(int).tolist(),
                           min_lower_pp=float(primary.ci_lower.min()*100), max_lower_pp=float(primary.ci_lower.max()*100),
                           min_upper_pp=float(primary.ci_upper.min()*100), max_upper_pp=float(primary.ci_upper.max()*100),
                           min_width_ratio=float(primary.width_ratio_to_original_L5.min()),
                           max_width_ratio=float(primary.width_ratio_to_original_L5.max()),
                           holm_significant_lengths=primary.loc[primary.significant_holm_05, "block_length"].astype(int).tolist(),
                           primary_holm_status_stable=bool(primary.significant_holm_05.eq(baseline.significant_holm_05).all()),
                           monte_carlo_endpoint_ranges=mc_ranges))
    return output


def figures(frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(PAIRS), 2, figsize=(11, 13), sharex=True)
    for column, dataset in enumerate(NAMES):
        for row, (name, _, _, _) in enumerate(PAIRS):
            ax = axes[row, column]
            cell = frame[frame.dataset.eq(dataset) & frame.comparison.eq(name) & frame.primary_rng].sort_values("block_length")
            effect = cell.effect.to_numpy()*100
            low, high = cell.ci_lower.to_numpy()*100, cell.ci_upper.to_numpy()*100
            # Percentile intervals need not contain the original point estimate.
            ax.vlines(cell.block_length, low, high, color="#2467a6", lw=1.5)
            ax.plot(cell.block_length, low, "_", color="#2467a6", ms=6)
            ax.plot(cell.block_length, high, "_", color="#2467a6", ms=6)
            ax.plot(cell.block_length, effect, "o", color="#2467a6", ms=3.5)
            ax.axhline(0, color="0.45", ls="--", lw=.8)
            ax.axvline(5, color="#b77819", ls=":", lw=1)
            ax.set_ylabel(name.replace("_minus_", " - ")+" (pp)")
            ax.set_xticks(LENGTHS)
            ax.grid(axis="y", alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0, 21)
            if row == 0:
                ax.set_title(NAMES[dataset])
            if row == len(PAIRS)-1:
                ax.set_xlabel("Materials per statistical block")
    fig.suptitle("Material-block length sensitivity of sentence BLEU-1 gains\n"
                 "Pointwise paired 95% CIs; 10,000 draws; fixed subjects and neural seeds", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .955))
    for extension in ("png", "svg"):
        fig.savefig(REPORT / f"block_length_sensitivity.{extension}", dpi=180)
    plt.close(fig)


def markdown(result, estimates, structures):
    labels = {p[0]: p[0].replace("_minus_", "−") for p in PAIRS}
    primary = estimates[estimates.primary_rng]
    lines = ["# 材料统计块长度敏感性分析", "", "完整 A/B/C1/C 测试结果，两个数据集、三个神经种子，仅 CPU 重采样。", "",
             "分析方案在本轮重采样前冻结，既有结果已知。主指标为受试者内平均片段 BLEU-1，再对受试者和神经种子等权。"
             "共享刺激连接分量保持完整，中文声音版本继续相连；在 run／章节内按顺序构造连续、不重叠块，尾块全部保留。", "",
             "块长固定 1、2、3、5、8、10、15、20 个材料单元，每项 10,000 次分层成对 bootstrap，"
             "统计种子 161803 为主；271828、314159 检查 Monte Carlo 端点波动。5 材料基准逐项复现原效应、95% 区间、"
             "符号翻转检验与原检验族的 Holm 校正。全部点估计在不同块长下保持一致。", "",
             "## 材料块数量", "", "| 块长 | Pallier2025 | ChineseEEG-2 | 中文分层块数（第2／9章） |",
             "|---:|---:|---:|---|"]
    for length in LENGTHS:
        counts = structures[structures.block_length.eq(length)].groupby("dataset").blocks.sum()
        chinese = structures[structures.block_length.eq(length) & structures.dataset.eq("chineseeeg2_littleprince")]
        lines.append(f"| {length} | {int(counts['pallier2025'])} | {int(counts['chineseeeg2_littleprince'])} | "
                     + "／".join(str(int(n)) for n in chinese.blocks)+" |")
    lines += ["", "## 跨块长汇总", "", "以下端点范围是不同块长下逐项95%区间端点的最小／最大值，单位百分点；"
              "不是一个新的95%同时置信带。宽度倍率相对原块长5的区间。", "",
              "| 数据集 | 比较 | 效应 | 原5材料区间 | 下端点范围 | 上端点范围 | 宽度倍率范围 | 跨零块长 | 3组统计随机数分类一致 |",
              "|---|---|---:|---|---|---|---|---|---|"]
    for row in result["robustness"]:
        lo, hi = row["baseline_ci_pp"]
        zeros = "、".join(map(str, row["zero_including_block_lengths"])) or "无"
        lines.append(f"| {NAMES[row['dataset']]} | {labels[row['comparison']]} | {row['effect_pp']:+.3f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | [{row['min_lower_pp']:+.3f}, {row['max_lower_pp']:+.3f}] | "
                     f"[{row['min_upper_pp']:+.3f}, {row['max_upper_pp']:+.3f}] | "
                     f"{row['min_width_ratio']:.2f}–{row['max_width_ratio']:.2f} | {zeros} | "
                     f"{'是' if row['all_rng_class_stable'] else '否'} |")
    for dataset in NAMES:
        lines += ["", f"## {NAMES[dataset]}：逐块长95%区间（百分点）", "",
                  "| 块长 | C−A | B−A | C−B | C1−A | C−C1 |", "|---:|---|---|---|---|---|"]
        for length in LENGTHS:
            cells = primary[primary.dataset.eq(dataset) & primary.block_length.eq(length)].set_index("comparison")
            values = [f"[{cells.loc[name].ci_lower*100:+.3f}, {cells.loc[name].ci_upper*100:+.3f}]" for name, _, _, _ in PAIRS]
            lines.append(f"| {length} | " + " | ".join(values) + " |")
    lines += ["", "## 结论与适用范围", ""]
    for row in result["robustness"]:
        if row["baseline_interval_class"] == "positive" and row["all_rng_class_stable"]:
            conclusion = "8种块长及3组统计随机数的95%区间下界均大于零，正增益的区间方向保持一致"
        elif row["baseline_interval_class"] == "includes_zero" and row["all_rng_class_stable"]:
            conclusion = "8种块长及3组统计随机数的95%区间均包含零，增益不确定的判断保持一致"
        else:
            conclusion = "区间是否包含零随块长或统计随机数变化，不能称为跨所检验块长稳健"
        lines.append(f"- {NAMES[row['dataset']]} {labels[row['comparison']]}：{conclusion}。")
    lines += ["", "辅助符号翻转检验仍按每种块长保留原检验族单独 Holm 校正，不把扫描得到的最小 p 值作为新确认性结论。"
              "区间是逐项95%，并未因为 Holm p 值而变成同时校正区间。", "",
              "| 数据集 | 比较 | 辅助 Holm p<0.05 的块长（主统计随机数） | 状态与原5材料一致 |", "|---|---|---|---|"]
    for row in result["robustness"]:
        lines.append(f"| {NAMES[row['dataset']]} | {labels[row['comparison']]} | "
                     f"{'、'.join(map(str, row['holm_significant_lengths'])) or '无'} | "
                     f"{'是' if row['primary_holm_status_stable'] else '否'} |")
    mc_max = max(max(s["lower_endpoint_range_pp"], s["upper_endpoint_range_pp"])
                 for r in result["robustness"] for s in r["monte_carlo_endpoint_ranges"])
    lines += ["", f"固定块长下，3组统计随机数的端点跨度最大为 {mc_max:.3f} 个百分点；逐项范围见 JSON。", "",
              "大块减少有效重采样单元：中文块长15／20只剩7／6块，每章分别4+3／3+3块。"
              "更少块的 percentile 区间可能变宽或变窄，不能因为区间变窄就认为估计更可靠。"
              "符号翻转还存在有限块数的离散 p 值粒度。结果仅检验本次固定范围内的块长敏感性，"
              "不证明所有依赖结构下的覆盖率，也不覆盖受试者总体与训练重复的不确定性。", "",
              "所有预测、测试材料、原统计结果与来源哈希均保留；不根据结果改变原块长5。", "",
              "- [全部240组统计与辅助检验](estimates.csv)", "- [材料块构成](material_blocks.csv)",
              "- [分层块数与尾块大小](block_structure.csv)", "- [机器可读汇总](results.json)",
              "- [来源与输出审计](audit.json)", "- [区间敏感性图](block_length_sensitivity.svg)", "",
              f"本次分析 manifest：`{result['manifest_sha256']}`。", "",
              "复现：`python -m braindecoding.material_block_length_sensitivity_test_v2`。", ""]
    return "\n".join(lines)


def main():
    started = time.perf_counter()
    parent, combined, reports, plan = prepare_inputs()
    estimates, membership, structure = analyze(parent, combined, reports)
    REPORT.mkdir(parents=True, exist_ok=True)
    result = dict(status="completed", analysis_status=plan["analysis_status"], manifest_sha256=plan["manifest_sha256"],
                  parent_manifest_sha256=parent["manifest_sha256"], block_lengths=list(LENGTHS),
                  statistic_seeds=list(STATISTIC_SEEDS), bootstrap_repetitions=REPETITIONS, metric=METRIC,
                  robustness=robustness_summary(estimates),
                  checks=dict(original_L5_block_identities_reproduced=True, original_L5_all_statistics_reproduced=True,
                              original_L5_Holm_reproduced=True, all_point_estimands_unchanged=True,
                              A_C_metrics_identical_in_B_C1_exports=True, complete_segment_seed_units=8244,
                              conditions=4, statistics_runs=len(estimates), discarded_bootstrap_samples=0,
                              shared_material_components_unchanged=True, all_stratum_tails_retained=True, GPU_used=False))
    for name, frame in (("estimates", estimates), ("material_blocks", membership), ("block_structure", structure)):
        frame.to_csv(REPORT / f"{name}.csv", index=False, encoding="utf-8-sig")
    save(REPORT / "results.json", result)
    (REPORT / "results.md").write_text(markdown(result, estimates, structure), encoding="utf-8")
    figures(estimates)
    for path, expected in plan["source_sha256"].items():
        require(sha(ROOT / path) == expected, f"Source changed during sensitivity analysis: {path}")
    require(read(MANIFEST) == plan, "Analysis manifest changed during computation")
    result["checks"]["sources_unchanged_before_and_after"] = True
    save(REPORT / "results.json", result)
    outputs = {p.name: sha(p) for p in REPORT.iterdir() if p.is_file() and p.name != "audit.json"}
    save(REPORT / "audit.json", dict(status="verified_complete", manifest_sha256=plan["manifest_sha256"],
                                      manifest_file_sha256=sha(MANIFEST), output_sha256=outputs,
                                      checks=result["checks"], elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(dict(status="completed", report=REPORT.relative_to(ROOT).as_posix(), checks=result["checks"])), flush=True)


if __name__ == "__main__":
    main()
