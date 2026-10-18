import copy

import numpy as np
import pytest

from braindecoding import pure_language_baseline_test_v2 as study
from braindecoding import pure_language_baseline_test_v2_results as results


DECODER = dict(candidates=1, beam_width=5, neural_scale=10.)


class LM:
    def __init__(self):
        self.words_seen = set()

    def next_words_many(self, prefixes):
        return {tuple(p): () for p in prefixes}

    def score_extensions(self, prefixes, words):
        self.words_seen.update(words)
        values = []
        for prefix, word in zip(prefixes, words):
            if not prefix:
                value = {"a": -1., "b": -2., "c": -5.}[word]
            else:
                value = (0. if word == "b" else -4.) if prefix[-1] == "b" else (-10. if word == "a" else -11.)
            values.append(value)
        return np.asarray(values)


def test_full_lexicon_is_scored_without_neural_vectors_or_embeddings(monkeypatch):
    scorer = LM()
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Neural arrays were read"))
    actual = study.replay(3, ("a", "b", "c"), scorer, DECODER)
    assert scorer.words_seen == {"a", "b", "c"}
    assert actual["trace"][0]["LM_A"] == ["a"]
    assert actual["trace"][0]["LM_A"] == actual["trace"][0]["LM_C1"] == actual["trace"][0]["LM_R"]
    assert actual["trace"][1]["LM_C1"] == actual["trace"][1]["LM_R"] == ["b", "b"]


@pytest.mark.parametrize("count", range(1, 8))
def test_actual_boundary_state_equals_independent_short_segment(count):
    longest = study.replay(7, ("a", "b", "c"), LM(), DECODER)
    independent = study.replay(count, ("a", "b", "c"), LM(), DECODER)
    assert study.crop_result(longest["trace"], count) == independent
    # This catches using the long segment's final prefix beyond a short boundary.
    if count == 1:
        assert independent["final"]["LM_R"] != longest["final"]["LM_R"][:count]


def test_policies_freeze_after_their_declared_revision_opportunities():
    actual = study.replay(7, ("a", "b", "c"), LM(), DECODER)
    study.validate_trace(actual["trace"])
    broken = copy.deepcopy(actual["trace"])
    broken[-1]["LM_A"][0] = "changed"
    with pytest.raises(ValueError, match="frozen"):
        study.validate_trace(broken)
    with pytest.raises(ValueError, match="length"):
        study.crop_result(actual["trace"], 8)


def test_open_oov_language_proposals_never_use_text_embeddings():
    class Outside:
        def next_words_many(self, prefixes):
            return {tuple(p): ("outside",) for p in prefixes}

        def score_extensions(self, prefixes, words):
            return np.asarray([1. if w == "outside" else 0. for w in words])
    result = study.replay(3, ("inside",), Outside(), DECODER)
    assert all(words == ["outside"] * 3 for words in result["final"].values())


def rows_and_info():
    groups = [dict(group=f"g{b}-{u}", subject=u, material=f"m{b}", block=f"b{b}", scope="scope", event_ids=list(range(b + 1)))
              for b in range(4) for u in ("u1", "u2")]
    info = dict(groups=groups, blocks=[dict(block=f"b{b}", scope="scope") for b in range(4)])
    rows = []
    for seed in range(3):
        for g in groups:
            for condition in results.ALL_CONDITIONS:
                neural = condition in study.MATCH
                rows.append(dict(seed=seed, condition=condition, **{k: g[k] for k in ("subject", "material", "group", "block", "scope")},
                    words=len(g["event_ids"]), sentence_bleu1_mean=.1 + .2 * neural, wer=.95 - .1 * neural,
                    bleu1=.2 + .15 * neural, aligned_word_accuracy=.01 + .05 * neural))
    return rows, info


def test_subject_segment_and_word_pooling_follow_original_protocol():
    rows, _ = rows_and_info()
    # Only one subject/condition/seed cell is changed: BLEU counts segments,
    # WER counts words, then both count subjects and seeds equally.
    for r in rows:
        if r["seed"] == 0 and r["subject"] == "u1" and r["condition"] == "A":
            r["sentence_bleu1_mean"] = 1. if r["words"] == 4 else 0.
            r["wer"] = 1. if r["words"] == 4 else 0.
    out = results.summarize(rows)
    row = next(r for r in out["by_subject_seed"] if r["seed"] == 0 and r["subject"] == "u1" and r["condition"] == "A")
    assert row["sentence_bleu1_mean"] == .25
    assert row["wer"] == .4


def test_shared_block_draws_and_signed_wer_reductions():
    rows, info = rows_and_info()
    policy = dict(seed=161803, material_block_bootstrap_repetitions=128, paired_material_block_sign_flips=128)
    comparisons, decomposition = results.paired_statistics(rows, info, policy, 0)
    assert len(comparisons) == 4
    for comparison in comparisons:
        assert comparison["unique_language_trajectories"] == 1
        assert comparison["neural_seeds"] == 3
        assert comparison["metrics"]["sentence_bleu1_mean"]["improvement"] == pytest.approx(.2)
        assert comparison["metrics"]["wer"]["improvement"] == pytest.approx(.1)
        assert comparison["metrics"]["sentence_bleu1_mean"]["ci_lower"] == pytest.approx(.2)
    assert len(decomposition) == 12
    assert all(r["difference_in_differences"] == pytest.approx(0.) for r in decomposition)
    counts, values = results.block_arrays(rows, info)
    observed = results.weighted_scores(np.ones((1, 4)), counts, values)
    summary = results.summarize(rows)["summary"]
    for c, item in enumerate(summary):
        for k, metric in enumerate(results.METRICS):
            assert observed[c, k, 0] == pytest.approx(item[metric])


def test_duplicate_material_observation_is_rejected():
    rows, info = rows_and_info()
    with pytest.raises(ValueError, match="duplicate"):
        results.block_arrays(rows + [rows[0]], info)


def test_config_enforces_neural_free_search_matching():
    cfg = study.settings()
    assert cfg["decoding"]["matched_controls"] == dict(A="LM_A", B="LM_R", C1="LM_C1", C="LM_R")
    assert cfg["protocol"]["neural_input_allowed"] is False
    assert cfg["execution"]["deduplicate_by_language_and_length"] is True


def test_nonfinite_language_score_stops_instead_of_selecting_a_word(monkeypatch):
    monkeypatch.setattr(study.BatchedWordLM, "score_extensions", lambda *a: np.asarray([np.nan]))
    scorer = object.__new__(study.FiniteWordLM)
    with pytest.raises(ValueError, match="Non-finite"):
        scorer.score_extensions([()], ["a"])


def test_full_export_preserves_all_conditions_and_eight_holm_comparisons(tmp_path, monkeypatch):
    rows, info = rows_and_info()
    info["counts"] = dict(test_groups=8)
    parent = dict(datasets={d: info for d in study.DATASETS})
    aggregate = results.summarize(rows)
    old = dict(datasets={d: dict(summary=aggregate["summary"]) for d in study.DATASETS})
    original_read = study.read
    monkeypatch.setattr(study, "verify_sources", lambda *a, **k: None)
    monkeypatch.setattr(study, "OUT", tmp_path / "predictions")
    monkeypatch.setattr(study, "REPORT", tmp_path / "report")
    for dataset in study.DATASETS:
        file = study.OUT / dataset / "main.json"
        file.parent.mkdir(parents=True)
        file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(study, "read", lambda p: parent if p == study.PARENT else old if p in study.OLD_REPORTS.values() else original_read(p))
    monkeypatch.setattr(results, "load_rows", lambda *a: rows)
    m = dict(manifest_sha256="test", parent_manifest_sha256="parent", settings=dict(statistics=dict(
        seed=161803, material_block_bootstrap_repetitions=128, paired_material_block_sign_flips=128)))
    out = results.export(m)
    assert len(out["comparisons"]) == 8
    assert all("p_holm" in r["metrics"]["sentence_bleu1_mean"] for r in out["comparisons"])
    assert all(len(info["summary"]) == 7 for info in out["datasets"].values())
    assert out["original_A_B_C1_C_metrics_reproduced"] is True
    audit = original_read(study.REPORT / "audit.json")
    assert audit["baseline_neural_array_reads"] == 0
    assert audit["deterministic_language_predictions_not_counted_as_independent_seeds"] is True
    assert (study.REPORT / "results.md").exists()
    assert (study.REPORT / "decomposition.csv").exists()
