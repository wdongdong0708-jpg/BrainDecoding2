from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import pytest

from braindecoding import f_local_sentence_test_v2 as study
from braindecoding import f_local_sentence_test_v2_results as report
from braindecoding.full_context_sentence_test_v2 import summarize
from braindecoding.unified_sentence_v2_results import METRICS


def neural(lengths):
    groups, positions = [], []
    for j, n in enumerate(lengths):
        ids = [f"group{j}|event{i}" for i in range(n)]
        groups.append(dict(group=f"g{j}", event_ids=ids, reference=["REFERENCE"] * n))
        positions.extend((event, i, n) for i, event in enumerate(ids))
    # The archive's row order deliberately differs from the segment order.
    positions.reverse()
    lookup = {event: i for i, (event, _, _) in enumerate(positions)}
    count = len(positions)
    predictions = {h: np.full((count, 1024), np.nan, dtype=np.float32) for h in range(3)}
    masks = {h: np.zeros(count, dtype=bool) for h in range(3)}
    for row, (_, i, n) in enumerate(positions):
        for h in range(3):
            if i + h < n:
                masks[h][row] = True
                predictions[h][row] = 0
                predictions[h][row, h * count + row] = 1
    return groups, predictions, masks, lookup


@pytest.mark.parametrize("n", [1, 2, 3, 4, 9])
def test_final_queries_select_last_available_horizon_without_crossing_boundary(n):
    groups, predictions, masks, lookup = neural([n, 4])
    before = {h: p.copy() for h, p in predictions.items()}
    q, horizons = study.final_queries(groups[0], predictions, masks, lookup)
    expected = [min(2, n - 1 - i) for i in range(n)]
    assert horizons == expected
    assert q.argmax(axis=1).tolist() == [h * len(lookup) + lookup[e] for h, e in zip(expected, groups[0]["event_ids"])]
    assert np.isfinite(q).all()
    assert all(np.array_equal(before[h], predictions[h], equal_nan=True) for h in range(3))


def test_boundary_mask_and_selected_invalid_values_stop():
    groups, predictions, masks, lookup = neural([3, 4])
    last = lookup[groups[0]["event_ids"][-1]]
    masks[2][last] = True
    with pytest.raises(ValueError, match="boundary"):
        study.final_queries(groups[0], predictions, masks, lookup)
    masks[2][last] = False
    predictions[0][last] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        study.final_queries(groups[0], predictions, masks, lookup)


def test_normalization_failure_stops():
    groups, predictions, masks, lookup = neural([3])
    predictions[2][lookup[groups[0]["event_ids"][0]]] *= 2
    with pytest.raises(ValueError, match="Unnormalized"):
        study.final_queries(groups[0], predictions, masks, lookup)


def test_loader_input_omits_reference_words_and_does_not_mutate_parent():
    parent = dict(neural_settings={"model": {"x": 1}}, settings={"decoding": {"x": 1}}, datasets={"d": dict(
        language="fr", vocabulary=["candidate"], embedding_cache="cache", groups=[{"reference": ["SECRET"]}])})
    loaded = study.decoder_input(parent, "d")
    assert set(loaded["datasets"]["d"]) == {"language", "vocabulary", "embedding_cache"}
    assert "SECRET" not in repr(loaded)
    loaded["neural_settings"]["model"]["x"] = 9
    assert parent["neural_settings"]["model"]["x"] == 1


def test_exactly_one_search_per_batch_with_empty_prefix_and_no_reference(monkeypatch):
    groups, predictions, masks, lookup = neural([4, 2])
    calls = []

    def search(requests, vocabulary, bank, scorer, embed, **settings):
        calls.append(requests)
        assert settings == dict(candidate_count=20, beam_width=5, neural_scale=10.)
        assert all(set(r) == {"queries", "prefix", "weight", "open_vocabulary"} for r in requests)
        assert all(r["prefix"] == () and r["weight"] == .1 and r["open_vocabulary"] for r in requests)
        return [tuple("candidate" for _ in r["queries"]) for r in requests]

    monkeypatch.setattr(study, "search_many", search)
    settings = dict(language_weight=.1, candidates=20, beam_width=5, neural_scale=10.)
    answers = study.decode(groups, predictions, masks, lookup, dict(vocabulary=["candidate"]),
                           None, None, np.zeros((1, 1024)), settings)
    assert len(calls) == 1
    assert answers[0]["horizons"] == [2, 2, 1, 0]
    assert answers[1]["horizons"] == [1, 0]
    assert "REFERENCE" not in repr(answers)


def test_whole_segment_beam_can_change_the_initially_best_first_word():
    groups, predictions, masks, lookup = neural([2])
    first_vector = predictions[1][lookup[groups[0]["event_ids"][0]]]
    bank = np.stack((first_vector, np.zeros(1024)))

    class Scorer:
        def next_words_many(self, prefixes):
            return {p: () for p in prefixes}

        def score_extensions(self, prefixes, words):
            return np.array([0. if not p or p[0] == "y" else -200. for p in prefixes])

    def embed(_):
        raise AssertionError("No outside candidates in this example")

    answers = study.decode(groups, predictions, masks, lookup, dict(vocabulary=["x", "y"]), Scorer(), embed,
                           bank, dict(language_weight=.1, candidates=2, beam_width=2, neural_scale=10.))
    assert answers[0]["prediction"][0] == "y"


def test_nonfinite_language_scores_stop_before_ranking():
    class Bad:
        def score_extensions(self, prefixes, words):
            return np.array([np.nan])

    with pytest.raises(ValueError, match="Non-finite LM"):
        study.CheckedScorer(Bad()).score_extensions([()], ["x"])


def metric_population():
    groups = [dict(group=f"{u}|{i}", event_ids=[f"{u}|{i}|{j}" for j in range(n)], subject=u,
                   block=f"b{i}", scope="scope") for u in ("u1", "u2") for i, n in enumerate((1, 3))]
    info = dict(groups=groups, blocks=[dict(block="b0", scope="scope"), dict(block="b1", scope="scope")])
    rows = []
    for seed in range(3):
        for g in groups:
            long = len(g["event_ids"]) == 3
            for condition in study.CONDITIONS:
                first = (0.2 if long else 0.1) if condition == "F-local" else (0.05 if condition == "F" else 0.)
                wer_reduction = (0.1 if long else 0.3) if condition == "F-local" else (0.05 if condition == "F" else 0.)
                rows.append(dict(seed=seed, condition=condition, **{k: g[k] for k in ("group", "subject", "block", "scope")},
                    words=len(g["event_ids"]), sentence_bleu1_mean=.2 + seed * .01 + first, wer=.9 - wer_reduction,
                    bleu1=.2 + first, aligned_word_accuracy=.1 + first))
    return info, rows


def test_metric_weighting_and_shared_bootstrap_decomposition():
    info, rows = metric_population()
    policy = dict(seed=123, material_block_bootstrap_repetitions=200)
    observed, pairs, decomposition = report.paired_statistics(rows, info, policy, 0)
    summary, _, _ = summarize(rows)
    lookup = {r["condition"]: r for r in summary}
    for i, condition in enumerate(study.CONDITIONS):
        for k, metric in enumerate(METRICS):
            assert observed[i, k] == pytest.approx(lookup[condition][metric], abs=1e-12)
    first = pairs[0]["metrics"]
    assert first["sentence_bleu1_mean"]["improvement"] == pytest.approx(.15)
    assert first["wer"]["improvement"] == pytest.approx(.15)  # (1*.3+3*.1)/4, not .20.
    assert first["sentence_bleu1_mean"]["ci_lower"] == pytest.approx(.1)
    assert first["sentence_bleu1_mean"]["ci_upper"] == pytest.approx(.2)
    for d in decomposition:
        assert d["waiting_mature_shared_input_and_search_increment"] + d["full_context_representation_increment"] == pytest.approx(d["total_F_vs_C_improvement"])
    assert decomposition[0]["full_context_representation_increment"] < 0


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_block", "wrong_words"])
def test_statistics_rejects_incomplete_or_unpaired_population(fault):
    info, rows = metric_population()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0]["block" if fault == "wrong_block" else "words"] = "other" if fault == "wrong_block" else 5
    with pytest.raises(ValueError):
        report.block_arrays(rows, info)


def test_material_sampling_is_shared_and_stratified():
    info = dict(blocks=[dict(block=f"b{i}", scope="a" if i < 2 else "b") for i in range(5)])
    cfg = dict(seed=123, material_block_bootstrap_repetitions=100)
    weights = report.bootstrap_weights(info, cfg, 0)
    assert np.all(weights[:, :2].sum(axis=1) == 2)
    assert np.all(weights[:, 2:].sum(axis=1) == 3)
    assert np.array_equal(weights, report.bootstrap_weights(info, cfg, 0))


def fake_chunk(m, job, groups, unit, previous, before, after):
    return dict(manifest_sha256=m["manifest_sha256"], dataset=job["dataset"], seed=job["seed"], condition="F-local",
                split="test", checkpoint_sha256=job["checkpoint_sha256"], prediction_sha256=job["prediction_sha256"],
                input_availability="wait_for_last_word_1s_window", search="single_full_segment_beam_from_empty_generated_prefix",
                selected_weight=.1, unit=unit, previous_chunk_sha256=previous, cache_before_sha256=before,
                cache_after=after, records=[dict(group=g["group"], prediction=["x"] * len(g["event_ids"]),
                    horizons=[min(2, len(g["event_ids"])-1-i) for i in range(len(g["event_ids"]))]) for g in groups])


def transactions(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "OUT", tmp_path / "out")
    ds = "d"
    initial = tmp_path / "initial.pkl"
    initial.write_bytes(b"initial")
    cache = dict(path=str(initial), sha256=study.sha256(initial))
    m = dict(manifest_sha256="manifest", initial_language_cache={ds: cache})
    job = dict(dataset=ds, seed=0, checkpoint_sha256="checkpoint", prediction_sha256="prediction")
    group = dict(group="g", event_ids=["e0", "e1", "e2"])
    tasks = [(job, i, [dict(group, group=f"g{i}")], study.OUT / ds / "seed-000/chunks" / f"chunk-{i:04d}.json") for i in range(3)]
    return m, job, tasks, cache


def test_resume_loads_only_last_committed_cache_and_ignores_orphan(tmp_path, monkeypatch):
    m, job, tasks, cache = transactions(tmp_path, monkeypatch)
    previous = None
    for unit, (_, _, groups, path) in enumerate(tasks[:2]):
        snapshot = tmp_path / f"unit-{unit}.pkl"
        snapshot.write_bytes(f"cache{unit}".encode())
        after = dict(path=str(snapshot), sha256=study.sha256(snapshot))
        value = fake_chunk(m, job, groups, unit, previous, cache["sha256"], after)
        study.write_json(path, value)
        previous, cache = study.sha256(path), after
    (tmp_path / "unit-2.pkl").write_bytes(b"uncommitted")
    chunks, restored, last_hash = study.committed_prefix(tasks, m, "d")
    assert len(chunks) == 2 and restored == cache and last_hash == previous
    Path(restored["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Last committed"):
        study.committed_prefix(tasks, m, "d")


def test_resume_rejects_chunk_gap(tmp_path, monkeypatch):
    m, job, tasks, cache = transactions(tmp_path, monkeypatch)
    _, _, groups, path = tasks[1]
    study.write_json(path, fake_chunk(m, job, groups, 1, None, cache["sha256"], cache))
    with pytest.raises(ValueError, match="gap"):
        study.committed_prefix(tasks, m, "d")


def test_chunk_rejects_other_seed_checkpoint_and_tail(tmp_path, monkeypatch):
    m, job, tasks, cache = transactions(tmp_path, monkeypatch)
    groups = tasks[0][2]
    valid = fake_chunk(m, job, groups, 0, None, cache["sha256"], cache)
    study.validate_chunk(valid, m, job, groups)
    for key, wrong in (("seed", 2), ("checkpoint_sha256", "other")):
        value = copy.deepcopy(valid)
        value[key] = wrong
        with pytest.raises(ValueError, match="provenance"):
            study.validate_chunk(value, m, job, groups)
    value = copy.deepcopy(valid)
    value["records"][0]["horizons"] = [2, 2, 2]
    with pytest.raises(ValueError, match="tail"):
        study.validate_chunk(value, m, job, groups)


def test_waiting_matches_F_for_each_event_without_seed_repetition(tmp_path, monkeypatch):
    group = dict(group="g", subject="u", block="b", scope="s", event_ids=["e0", "e1", "e2", "e3"])
    parent = dict(datasets={"d": dict(groups=[group], counts=dict(events=4, strict50_common=2))})
    path = tmp_path / "wait.csv"
    rows = [dict(dataset="d", event_id=f"e{i}", **{k: group[k] for k in ("group", "subject", "block", "scope")},
                 c_input_wait_s=min(3, 4-i), f_input_wait_s=4-i, common50=i < 2) for i in range(4)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    monkeypatch.setattr(study, "F_WAIT", path)
    events, summary = report.timing(parent)
    assert len(events) == 4
    assert all(r["f_local_input_wait_s"] == r["f_input_wait_s"] for r in events)
    local = next(r for r in summary if r["condition"] == "F-local" and r["population"] == "all_events")
    assert local["queries"] == 4 and local["mean_input_wait_s"] == 2.5
    assert local["mean_extra_wait_vs_C_s"] == .25


def test_complete_export_reproduces_original_metrics_and_emits_all_comparisons(tmp_path, monkeypatch):
    info, _ = metric_population()
    groups = info["groups"]
    for g in groups:
        g["reference"] = ["target"] * len(g["event_ids"])
        g["material"] = g["block"]
    info.update(language="fr", counts=dict(events=8, groups=4, subjects=2, materials=2, blocks=2, strict50_common=2))
    parent = dict(datasets={"d": info})
    for attribute, name in (("PARENT", "parent.json"), ("FULL", "full.json"), ("F_SUMMARY", "original.json"),
                            ("F_WAIT", "wait.csv"), ("OUT", "out"), ("REPORT", "report"), ("EXPORT", "exports")):
        monkeypatch.setattr(study, attribute, tmp_path / name)
    study.write_json(study.PARENT, parent)
    study.write_json(study.FULL, {})
    monkeypatch.setattr(study, "verify_sources", lambda *a, **k: None)
    monkeypatch.setattr(study, "committed_prefix", lambda *a, **k: ([], {}, None))
    policy = dict(seed=123, material_block_bootstrap_repetitions=100)
    m = dict(manifest_sha256="m", jobs=[], total_segment_seed_units=12, source_sha256={},
             settings=dict(statistics=policy, execution=dict(segment_batch_size=16)))
    original_rows, ac_payload, f_payload = [], {}, {}
    for seed in range(3):
        job = dict(dataset="d", seed=seed, checkpoint_sha256=f"cp{seed}", prediction_sha256=f"pred{seed}",
                   original_ac=f"ac{seed}", original_f=f"f{seed}")
        m["jobs"].append(job)
        chunk = fake_chunk(m, job, groups, seed, None, "cache", dict(path="none", sha256="cache"))
        for g, r in zip(groups, chunk["records"]):
            r["prediction"][0] = "target"
        path = study.OUT / "d" / f"seed-{seed:03d}" / "chunks/chunk-0000.json"
        study.write_json(path, chunk)
        main = dict(chunk, status="completed", sources=[dict(path=str(path), sha256=study.sha256(path))])
        study.write_json(path.parent.parent / "main.json", main)
        ac_payload[seed] = dict(records=[dict(group=g["group"], result=dict(final=dict(C=["wrong"] * len(g["event_ids"])))) for g in groups])
        f_payload[seed] = dict(records=[dict(group=g["group"], prediction=g["reference"]) for g in groups])
        for g in groups:
            for c in ("C", "F"):
                p = ["wrong"] * len(g["event_ids"]) if c == "C" else g["reference"]
                original_rows.append(dict(seed=seed, subject=g["subject"], condition=c, words=len(p),
                    **report.metric_row(g["reference"], p, False)))
    means, _, _ = summarize(original_rows)
    study.write_json(study.F_SUMMARY, dict(datasets={"d": dict(summary=means)}))
    study.write_json(study.OUT / "d/completion.json", dict(status="completed", manifest_sha256="m", completed_segment_seed_units=12))
    monkeypatch.setattr(study, "original_ac", lambda parent, ds, seed, path: ac_payload[seed])
    monkeypatch.setattr(study, "original_f", lambda parent, full, ds, seed, path: f_payload[seed])
    wait_rows = [dict(dataset="d", event_id=e, **{k: g[k] for k in ("group", "subject", "block", "scope")},
        c_input_wait_s=min(3, len(g["event_ids"])-i), f_input_wait_s=len(g["event_ids"])-i,
        common50=i+2 < len(g["event_ids"])) for g in groups for i, e in enumerate(g["event_ids"])]
    with study.F_WAIT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(wait_rows[0]))
        writer.writeheader(); writer.writerows(wait_rows)
    result = report.export(m)
    assert result["status"] == "completed" and result["original_C_F_metrics_reproduced"]
    assert result["new_significance_tests"] is False
    assert len(result["datasets"]["d"]["comparisons"]) == 3
    assert len(result["exports"]) == 6
    assert (study.REPORT / "results.md").exists()
    assert study.read(study.REPORT / "audit.json")["status"] == "verified_complete"
