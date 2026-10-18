import copy
import pickle
from collections import OrderedDict

import numpy as np
import pytest

from braindecoding import pure_language_prompt_sensitivity_test_v2 as study
from braindecoding import pure_language_prompt_sensitivity_test_v2_results as results
from braindecoding import pure_language_baseline_test_v2 as original


def test_templates_are_predeclared_and_model_budgets_are_preserved():
    cfg, old = study.settings(), original.settings()
    assert cfg["prompt_diagnostic"]["candidate_ids"] == ["native_prose", "generic_prose", "story_opening"]
    assert cfg["prompt_diagnostic"]["use_test_references"] is False
    for key in ("beam_width", "proposal_count", "proposal_max_new_tokens", "proposal_batch_size",
        "score_batch_size", "score_cache_size", "score", "candidates", "revision_words"):
        assert cfg["decoding"][key] == old["decoding"][key]
    for pair in cfg["prompt_diagnostic"]["templates"].values():
        assert set(pair) == {"fr", "zh"}
        assert all("小王子" not in p and "Petit Prince" not in p and p.endswith("\n\n") for p in pair.values())


@pytest.mark.parametrize("words", [["1", "2"] * 6, ["le"] * 12, ["a", "b", "c", "d"] + ["a"] * 8])
def test_generation_diagnostic_rejects_numeric_or_repetition_collapse(words):
    assert not study.generation_diagnostics(words, study.settings()["prompt_diagnostic"])["passed"]


def test_prompt_selection_uses_predeclared_order_not_metric_or_probe_order():
    policy = study.settings()["prompt_diagnostic"]
    def probe(name, passed, fake_bleu):
        return dict(prompt_id=name, diagnostics={c: dict(passed=passed) for c in study.CONDITIONS},
            arbitrary_test_bleu= fake_bleu)
    later = probe("generic_prose", True, 1.)
    first = probe("native_prose", True, 0.)
    assert study.select_prompt([later, first], policy) is first
    first["diagnostics"]["LM_R"]["passed"] = False
    assert study.select_prompt([first, later], policy) is later
    with pytest.raises(ValueError, match="failed generation"):
        study.select_prompt([first], policy)


def scorer(prompt=(1, 2), language="fr", revision="revision"):
    value = object.__new__(study.FiniteWordLM)
    value.language, value.prompt_ids, value.context_tokens = language, prompt, 32768
    value.settings = dict(revision=revision, score="sum_all_word_subtokens", score_batch_size=32,
        proposal_count=5, proposal_max_new_tokens=16)
    value.cache, value.proposal_cache = OrderedDict({((), "le"): -2.}), OrderedDict({(): ("le",)})
    return value


@pytest.mark.parametrize("change", ["prompt", "language", "revision"])
def test_private_cache_rejects_cross_prompt_language_or_model(tmp_path, change):
    a, b = scorer(), scorer()
    file = tmp_path / "cache.pkl"
    a.save_cache(file)
    if change == "prompt":
        b.prompt_ids = (1, 3)
    elif change == "language":
        b.language = "zh"
    else:
        b.settings["revision"] = "other"
    with pytest.raises(ValueError, match="cache mismatch"):
        b.load_cache(file)


def test_private_cache_roundtrip_and_legacy_cache_rejection(tmp_path):
    a, b = scorer(), scorer()
    file = tmp_path / "cache.pkl"
    a.save_cache(file)
    b.cache.clear()
    b.proposal_cache.clear()
    b.load_cache(file)
    assert a.cache == b.cache and a.proposal_cache == b.proposal_cache
    with file.open("wb") as handle:
        pickle.dump(("fr", dict(a.cache), dict(a.proposal_cache)), handle)
    with pytest.raises(ValueError, match="cache mismatch"):
        b.load_cache(file)
    a.cache[((), "bad")] = np.nan
    a.save_cache(file)
    with pytest.raises(ValueError, match="Non-finite"):
        b.load_cache(file)


def rows_and_info():
    groups = [dict(group=f"g{b}-{u}", subject=u, material=f"m{b}", block=f"b{b}", scope="scope",
        event_ids=list(range(b + 1)), reference=["a"] * (b + 1)) for b in range(4) for u in ("u1", "u2")]
    info = dict(groups=groups, blocks=[dict(block=f"b{b}", scope="scope") for b in range(4)], counts=dict(test_groups=8))
    rows = []
    for version in results.VERSIONS:
        improved = version == "selected_prompt"
        for condition in study.CONDITIONS:
            for group in groups:
                rows.append(dict(prompt_version=version, condition=condition,
                    **{k: group[k] for k in ("subject", "material", "group", "block", "scope")},
                    words=len(group["event_ids"]), sentence_bleu1_mean=.1 + .2 * improved,
                    wer=.95 - .1 * improved, bleu1=.2 + .15 * improved,
                    aligned_word_accuracy=.01 + .05 * improved))
    return rows, info


def test_subject_equal_metrics_and_block_weights_match_without_fake_seeds():
    rows, info = rows_and_info()
    for row in rows:
        if row["prompt_version"] == "selected_prompt" and row["condition"] == "LM_A" and row["subject"] == "u1":
            row["sentence_bleu1_mean"] = 1. if row["words"] == 4 else 0.
            row["wer"] = 1. if row["words"] == 4 else 0.
    out = results.summarize(rows)
    subject = next(r for r in out["by_subject"] if r["prompt_version"] == "selected_prompt"
        and r["condition"] == "LM_A" and r["subject"] == "u1")
    assert subject["sentence_bleu1_mean"] == .25 and subject["wer"] == .4
    counts, values = results.block_arrays(rows, info)
    observed = results.weighted_scores(np.ones((1, 4)), counts, values)[..., 0]
    assert observed.shape == (2, 3, len(results.METRICS))
    for v, version in enumerate(results.VERSIONS):
        for c, condition in enumerate(study.CONDITIONS):
            row = next(r for r in out["summary"] if r["prompt_version"] == version and r["condition"] == condition)
            for k, name in enumerate(results.METRICS):
                assert observed[v, c, k] == pytest.approx(row[name])


def test_shared_material_pairing_and_wer_sign_and_search_comparison():
    rows, info = rows_and_info()
    policy = dict(seed=161803, material_block_bootstrap_repetitions=128, paired_material_block_sign_flips=128)
    paired, within = results.paired_statistics(rows, info, policy, 0)
    assert len(paired) == 3 and len(within) == 8
    for row in paired:
        assert row["independent_language_replicates_per_prompt"] == 1
        assert row["metrics"]["sentence_bleu1_mean"]["improvement"] == pytest.approx(.2)
        assert row["metrics"]["sentence_bleu1_mean"]["ci_lower"] == pytest.approx(.2)
        assert row["metrics"]["wer"]["improvement"] == pytest.approx(.1)
    assert all(r["improvement"] == pytest.approx(0.) for r in within)
    with pytest.raises(ValueError, match="Duplicate"):
        results.block_arrays(rows + [rows[0]], info)
    with pytest.raises(ValueError, match="Incomplete"):
        results.block_arrays(rows[:-1], info)


def test_language_prompt_settings_leave_frozen_model_configuration_untouched():
    m = dict(language_model=dict(language_prompt_template="old", score="sum_all_word_subtokens"),
        datasets=dict(d=dict(prompt_template="new")))
    before = copy.deepcopy(m)
    assert study.language_settings(m, "d")["language_prompt_template"] == "new"
    assert m == before


def test_full_export_has_six_prompt_comparisons_and_explicit_cross_prompt_neural_reference(tmp_path, monkeypatch):
    rows, info = rows_and_info()
    parent = dict(datasets={d: info for d in study.DATASETS})
    aggregate = results.summarize(rows)
    summary = [r for r in aggregate["summary"] if r["prompt_version"] == "original_prompt"]
    summary += [dict(condition=c, sentence_bleu1_mean=.4, wer=.8, bleu1=.3, aligned_word_accuracy=.1) for c in study.MATCH]
    previous = dict(datasets={d: dict(summary=summary) for d in study.DATASETS})
    monkeypatch.setattr(study, "OUT", tmp_path / "predictions")
    monkeypatch.setattr(study, "REPORT", tmp_path / "report")
    monkeypatch.setattr(study, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(study, "verify_sources", lambda *a, **k: None)
    monkeypatch.setattr(original, "verify_sources", lambda *a, **k: None)
    for dataset in study.DATASETS:
        file = study.OUT / dataset / "main.json"
        file.parent.mkdir(parents=True)
        file.write_text("{}", encoding="utf-8")
    study.write_json(study.RUNTIME / "pre_diagnostic_policy.json", dict(policy={}, code_sha256="code",
        config_sha256="config", protocol_sha256=study.sha256(study.PROTOCOL)))
    study.write_json(study.RUNTIME / "gpu_smoke.json", dict(policy={}, code_sha256="code", config_sha256="config"))
    original_read = study.read
    monkeypatch.setattr(study, "read", lambda p: parent if p == study.PARENT else previous
        if p == original.REPORT / "results.json" else {} if p == original.MANIFEST else original_read(p))
    monkeypatch.setattr(results, "load_rows", lambda *a: (rows, {v: {c: dict(total_words=20, unique_words=4,
        digit_words=0, most_common=[("a", 4)]) for c in study.CONDITIONS} for v in results.VERSIONS}))
    m = dict(manifest_sha256="test", parent_manifest_sha256="parent", original_language_manifest_sha256="old",
        settings=dict(statistics=dict(seed=161803, material_block_bootstrap_repetitions=128, paired_material_block_sign_flips=128)),
        datasets={d: dict(prompt_id="native_prose", prompt_template="example\n\n", prompt_template_sha256="prompt")
            for d in study.DATASETS})
    result = results.export(m)
    assert len(result["prompt_comparisons"]) == 6
    assert all("p_holm" in r["metrics"]["sentence_bleu1_mean"] for r in result["prompt_comparisons"])
    assert result["original_pure_language_metrics_reproduced"] is True
    assert result["neural_reference_has_same_prompt"] is False
    assert len(result["cross_prompt_neural_descriptive"]) == 32
    assert all(r["causal_attribution_allowed"] is False for r in result["cross_prompt_neural_descriptive"])
    assert original_read(study.REPORT / "audit.json")["status"] == "verified_complete"
    assert (study.REPORT / "results.md").exists()
    assert (study.REPORT / "summary.csv").exists()
