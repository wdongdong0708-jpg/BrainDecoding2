import copy

import numpy as np
import pytest

from braindecoding import neural_prompt_sensitivity_test_v2 as study
from braindecoding import neural_prompt_sensitivity_test_v2_results as results
from braindecoding.unified_sentence_v2_results import METRICS


def neural_segment(n=4):
    group = dict(group="g", event_ids=[f"e{i}" for i in range(n)], reference=["secret"] * n)
    predictions, masks = {}, {}
    for h in range(3):
        masks[h] = np.arange(n) + h < n
        predictions[h] = np.full((n, 1024), np.nan)
        predictions[h][masks[h]] = 0.
        predictions[h][masks[h], h] = 1.
    return group, predictions, masks, {e: i for i, e in enumerate(group["event_ids"])}


def test_rolling_replay_uses_only_available_horizons_and_generated_prefix(monkeypatch):
    from braindecoding import unified_sentence_batch as batch
    seen = []
    def fake_search(requests, *args, **kwargs):
        answers = []
        for r in requests:
            assert set(r) == {"queries", "prefix", "weight", "open_vocabulary", "pure_lm"}
            assert r["weight"] == .1 and r["open_vocabulary"] and not r["pure_lm"]
            assert np.isfinite(r["queries"]).all()
            horizons = r["queries"].argmax(axis=1).tolist()
            seen.append((list(r["prefix"]), horizons))
            answers.append(tuple(r["prefix"]) + tuple("abc"[h] for h in horizons))
        return answers
    monkeypatch.setattr(batch, "search_many", fake_search)
    group, predictions, masks, lookup = neural_segment()
    cfg = study.config()["decoding"]
    record = study.decode([group], predictions, masks, lookup, dict(vocabulary=["a", "b", "c"]),
        None, None, np.zeros((3, 1024)), cfg)[0]
    assert record["result"]["final"] == {"A": ["a"] * 4, "C": ["c", "c", "b", "a"]}
    assert seen == [([], [0]), ([], [0]), (["a"], [0]), ([], [1, 0]),
        (["a", "a"], [0]), ([], [2, 1, 0]), (["a", "a", "a"], [0]), (["c"], [2, 1, 0])]
    changed = dict(group, reference=["different_reference"] * 4)
    assert study.decode([changed], predictions, masks, lookup, dict(vocabulary=["a", "b", "c"]),
        None, None, np.zeros((3, 1024)), cfg) == [record]


def test_case_construction_rejects_cross_segment_horizon_and_weight_changes():
    group, predictions, masks, lookup = neural_segment()
    masks[2][-1] = True
    with pytest.raises(ValueError, match="segment boundary"):
        study.cases([group], predictions, masks, lookup)
    with pytest.raises(ValueError, match="weight changed"):
        study.cases([group], predictions, masks, lookup, .5)


def test_loader_contract_contains_no_references_or_material_identity():
    parent = dict(neural_settings=dict(language_model=dict(language_prompt_template="old")),
        settings=dict(decoding=dict(beam_width=5)), datasets={"x": dict(language="fr", vocabulary=["le"],
            embedding_cache="cache", groups=[dict(reference=["secret"])], title="secret title")})
    value = study.decoder_input(parent, "x", dict(prompt_template="new\n\n"))
    assert set(value["datasets"]["x"]) == {"language", "vocabulary", "embedding_cache"}
    assert value["neural_settings"]["language_model"]["language_prompt_template"] == "new\n\n"
    assert parent["neural_settings"]["language_model"]["language_prompt_template"] == "old"


def test_trace_audit_rejects_revising_a_frozen_word():
    group = dict(event_ids=["e0", "e1", "e2"])
    valid = dict(final={"A": ["a", "b", "c"], "C": ["c", "b", "a"]}, trace=[
        dict(arrival=0, A=["a"], C=["a"]), dict(arrival=1, A=["a", "b"], C=["b", "a"]),
        dict(arrival=2, A=["a", "b", "c"], C=["c", "b", "a"])])
    study.validate_result(group, valid)
    changed = copy.deepcopy(valid)
    changed["trace"][2]["A"][0] = "wrong"
    with pytest.raises(ValueError, match="frozen prefix"):
        study.validate_result(group, changed)


def rows_and_info(dataset="pallier2025"):
    groups = [dict(group=f"g{b}-{u}", subject=u, block=f"b{b}", material=f"m{b}", scope="scope",
        event_ids=[f"e{b}-{u}-{i}" for i in range(b + 1)]) for b in range(4) for u in ("u1", "u2")]
    info = dict(groups=groups, blocks=[dict(block=f"b{b}", scope="scope") for b in range(4)],
        counts=dict(events=20, groups=8, subjects=2, blocks=4))
    rows = []
    levels = dict(A=.10, C=.18, LM_A=.06, LM_R=.06, A_original=.08, C_original=.14)
    for group in groups:
        for c in results.ALL_CONDITIONS:
            for seed in results.condition_seeds(c):
                value = levels[c] + .001 * int(group["block"][1:]) + (0. if seed is None else .01 * seed)
                rows.append(dict(dataset=dataset, condition=c, seed=seed,
                    **{k: group[k] for k in ("group", "subject", "material", "block", "scope")},
                    words=len(group["event_ids"]), sentence_bleu1_mean=value, wer=1-value,
                    bleu1=value, aligned_word_accuracy=value))
    return rows, info


def test_language_has_one_trajectory_and_material_weights_match_direct_summary():
    rows, info = rows_and_info()
    summary = results.summarize(rows)
    lm = [r for r in summary["by_seed"] if r["condition"] == "LM_A"]
    assert len(lm) == 1 and lm[0]["seed"] is None
    counts, values = results.block_arrays(rows, info)
    direct = results.weighted_scores(np.ones((1, 4)), counts, values)[..., 0]
    for c, record in enumerate(summary["summary"]):
        for k, name in enumerate(METRICS):
            assert direct[c, k] == pytest.approx(record[name], abs=1e-12)
    duplicate = copy.deepcopy(next(r for r in rows if r["condition"] == "LM_A"))
    duplicate["seed"] = 0
    with pytest.raises(ValueError, match="undeclared"):
        results.block_arrays(rows + [duplicate], info)


def test_paired_effects_error_direction_and_prompt_gain_change():
    rows, info = rows_and_info()
    policy = dict(study.config()["statistics"], material_block_bootstrap_repetitions=100)
    _, pairs, changes = results.paired_statistics(rows, info, policy, 0)
    assert len(pairs) == 3
    contrast = next(r for r in pairs if r["comparison"] == "C_minus_A")
    for name in ("sentence_bleu1_mean", "wer", "aligned_word_accuracy"):
        assert contrast["metrics"][name]["improvement"] == pytest.approx(.08)
    assert "p_two_sided" not in contrast["metrics"]["wer"]
    change = next(r for r in changes if r["comparison"] == "rolling_gain_new_minus_original")
    assert change["metrics"]["sentence_bleu1_mean"]["improvement"] == pytest.approx(.02)


def test_export_reports_six_bleu_tests_and_reproduces_read_only_references(tmp_path, monkeypatch):
    policy = dict(study.config()["statistics"], material_block_bootstrap_repetitions=100)
    fixtures = {d: rows_and_info(d) for d in study.DATASETS}
    parent = dict(datasets={d: info for d, (_, info) in fixtures.items()})
    old, pure = dict(datasets={}), dict(datasets={})
    for d, (rows, info) in fixtures.items():
        summary = results.summarize(rows)["summary"]
        old["datasets"][d] = dict(summary=[dict(r, condition=r["condition"].removesuffix("_original"))
            for r in summary if r["condition"].endswith("_original")])
        pure["datasets"][d] = dict(summary=[dict(r, prompt_version="selected_prompt")
            for r in summary if r["condition"] in results.LANGUAGE_CONDITIONS])
    report = tmp_path / "report"
    monkeypatch.setattr(study, "REPORT", report)
    responses = {str(study.RUNTIME / "status.json"): dict(GPU_workers_exited=True, worker_pid=None),
        str(study.PARENT): parent, str(study.LM_MANIFEST): dict(manifest_sha256="lm"),
        str(study.OLD_SUMMARY): old, str(study.LM_REPORT / "results.json"): pure}
    for d in study.DATASETS:
        responses[str(study.OUT / d / "completion.json")] = dict(status="completed", manifest_sha256="new",
            completed_segment_seed_units=24)
    monkeypatch.setattr(study, "read", lambda p: responses[str(p)])
    monkeypatch.setattr(study, "verify_sources", lambda *a, **k: None)
    monkeypatch.setattr(study, "committed_prefix", lambda *a, **k: None)
    monkeypatch.setattr(study, "plan", lambda *a, **k: [])
    monkeypatch.setattr(results, "load_rows", lambda m, p, l, d: fixtures[d][0])
    m = dict(manifest_sha256="new", parent_manifest_sha256="parent", language_manifest_sha256="lm", jobs=[],
        settings=dict(statistics=policy), prompts={d: dict(prompt_id="native_prose") for d in study.DATASETS},
        total_segment_seed_units=48, source_sha256={})
    value = results.export(m)
    assert len(value["comparisons"]) == 6
    assert value["language_baselines_not_duplicated_as_seeds"] is True
    assert value["neural_and_language_prompts_identical"] is True
    for r in value["comparisons"]:
        assert "p_holm" in r["metrics"]["sentence_bleu1_mean"]
        assert "p_holm" not in r["metrics"]["wer"]
    assert (report / "audit.json").exists()
    assert "Acc" in (report / "results.md").read_text(encoding="utf-8")
