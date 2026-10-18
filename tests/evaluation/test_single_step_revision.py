"""C1 causality, one-revision timing, provenance and complete paired export."""
import copy

import numpy as np
import pytest

from braindecoding import single_revision_c1_test_v2 as worker
from braindecoding import single_revision_c1_test_v2_results as results
from braindecoding import fixed_k0_rolling_search_test_v2 as shared
from braindecoding.unified_sentence_batch import replay_many
from braindecoding.unified_sentence_v2 import save, read, sha256
from braindecoding.unified_sentence_v2_results import METRICS, metric_row


pytestmark = pytest.mark.current_protocol


class NoLM:
    def score_extensions(self, prefixes, words):
        return np.zeros(len(words))


def test_c1_uses_k1_once_then_freezes_even_if_k2_would_reverse_it():
    k0 = np.array([[1., 0.]]*4)
    k1 = np.array([[0., 1.]]*4)
    k2 = k0.copy()
    c1 = worker.replay_c1_many([dict(predictions={0: k0, 1: k1}, weight=0., open_vocabulary=False)],
                              ["a", "b"], np.eye(2), NoLM(), None, candidate_count=2)[0]
    c = replay_many([dict(predictions={0: k0, 1: k1, 2: k2}, weight=0., open_vocabulary=False)],
                    ["a", "b"], np.eye(2), NoLM(), None, conditions="C", candidate_count=2)[0]
    assert c1["trace"][0]["C1"] == ["a"]
    assert c1["trace"][1]["C1"] == c["trace"][1]["C"] == ["b", "a"]
    assert c1["final"]["C1"] == ["b", "b", "b", "a"]
    assert c["final"]["C"] == ["a", "a", "b", "a"]
    worker.validate_c1_result(dict(event_ids=list(range(4))), c1)


def test_c1_cache_loader_never_reads_k2_and_reference_never_enters_cases(monkeypatch):
    class Archive:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __getitem__(self, key):
            assert key in {"k0", "k1", "event_ids", "mask1"}, "K2 must never be read"
            if key == "event_ids":
                return np.array(["a", "b", "c"])
            if key == "mask1":
                return np.array([True, True, False])
            return np.eye(3)
    monkeypatch.setattr(worker.np, "load", lambda *args, **kwargs: Archive())
    predictions, ids, mask = worker.load_k01("neural.npz")
    group = dict(event_ids=["b", "a", "c"], reference=["secret"]*3)
    case = worker.c1_cases([group], predictions, ids, mask)[0]
    assert set(case) == {"predictions", "weight", "open_vocabulary"}
    assert set(case["predictions"]) == {0, 1}
    np.testing.assert_array_equal(case["predictions"][1], np.eye(3)[[1, 0, 2]])
    with pytest.raises(ValueError, match="K1 unavailable"):
        worker.c1_cases([dict(event_ids=["c", "a"])], predictions, ids, mask)


def test_exact_two_word_window_per_arrival_and_no_extra_terminal_flush(monkeypatch):
    seen = []
    def search(requests, *args, **kwargs):
        arrival = len(seen)
        for r in requests:
            start = max(0, arrival-1)
            expected = np.eye(5)[[0]] if arrival == 0 else np.stack([np.eye(5)[arrival-1]*10, np.eye(5)[arrival]])
            np.testing.assert_array_equal(r["queries"], expected)
            assert len(r["prefix"]) == start
        seen.append(arrival)
        return [r["prefix"]+tuple(f"w-{arrival}-{i}" for i in range(len(r["queries"]))) for r in requests]
    monkeypatch.setattr(worker, "search_many", search)
    cases = [dict(predictions={0: np.eye(5)[:n], 1: np.eye(5)[:n]*10}, weight=.1, open_vocabulary=True)
             for n in (5, 1)]
    output = worker.replay_c1_many(cases, [], None, None, None)
    assert seen == list(range(5))
    assert len(output[1]["trace"]) == 1
    assert output[1]["final"]["C1"] == output[1]["trace"][0]["C1"]
    for n, result in zip((5, 1), output):
        worker.validate_c1_result(dict(event_ids=list(range(n))), result)


def fixture_result():
    return dict(final=dict(C1=["b", "b", "b", "a"]), trace=[
        dict(arrival=0, C1=["a"]), dict(arrival=1, C1=["b", "a"]),
        dict(arrival=2, C1=["b", "b", "a"]), dict(arrival=3, C1=["b", "b", "b", "a"])])


def test_resume_rejects_second_revision_overlong_wait_wrong_provenance_and_extra_arrival():
    group = dict(group="g", event_ids=list(range(4)))
    m, job = dict(manifest_sha256="fixed"), dict(dataset="dataset", seed=0)
    chunk = dict(**worker.provenance(m, job), records=[dict(group="g", result=fixture_result())])
    worker.validate_c1_chunk(chunk, m, job, [group])
    changed = copy.deepcopy(fixture_result())
    changed["trace"][2]["C1"][0] = "second-revision"
    with pytest.raises(ValueError, match="frozen prefix"):
        worker.validate_c1_result(group, changed)
    extra = copy.deepcopy(fixture_result())
    extra["trace"].append(dict(arrival=4, C1=["b", "b", "b", "a"]))
    with pytest.raises(ValueError, match="extra end flush"):
        worker.validate_c1_result(group, extra)
    for key, value in (("revision_words", 3), ("maximum_wait_subsequent_word_windows", 2),
                       ("neural_representation", "K0_K1_K2"), ("selected_weight", .5), ("seed", 1)):
        with pytest.raises(ValueError, match="provenance"):
            worker.validate_c1_chunk({**chunk, key: value}, m, job, [group])


def test_latency_counts_opportunities_even_without_actual_changes_and_excludes_last_word():
    group = dict(event_ids=["e0", "e1", "e2"], reference=["a", "a", "a"])
    ac = dict(final=dict(A=["a"]*3, C=["a"]*3),
              trace=[dict(arrival=i, A=["a"]*(i+1), C=["a"]*(i+1)) for i in range(3)])
    c1 = dict(final=dict(C1=["a"]*3), trace=[dict(arrival=i, C1=["a"]*(i+1)) for i in range(3)])
    rows = results.latency_rows(group, ac, c1, {})
    one = [r for r in rows if r["condition"] == "C1"]
    two = [r for r in rows if r["condition"] == "C"]
    assert [r["allowed_revision_opportunities"] for r in one] == [1, 1, 0]
    assert [r["freeze_arrival"] for r in one] == [1, 2, 2]
    assert [r["allowed_revision_opportunities"] for r in two] == [2, 1, 0]
    assert all(r["actual_text_changes"] == 0 for r in rows)
    changes = {r["comparison"]: r for r in results.c1_change_rows(group, ac, c1, {})}
    assert changes["C1_first_to_final"]["eligible"] == 2
    assert changes["C1_first_to_final"]["repaired"] == 0


def test_summary_retained_shares_preserve_negative_increment_and_wer_direction():
    rows = [dict(seed=seed, subject="s", condition=c, words=3,
                 **{metric: 1-v if metric == "wer" else v for metric in METRICS})
            for seed in (0, 1, 2) for c, v in (("A", .1), ("C1", .4), ("C", .3))]
    aggregate, gains, _ = results.summarize_c1(rows)
    assert {r["condition"] for r in aggregate["summary"]} == {"A", "C1", "C"}
    for gain in gains.values():
        assert gain["one_window_retained_share"] == pytest.approx(1.5)
        assert gain["second_window_increment_share"] == pytest.approx(-.5)


def test_full_export_requires_all_seeds_and_reproduces_ac(tmp_path, monkeypatch):
    for module in (worker, results, shared):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    for module in (worker, results):
        monkeypatch.setattr(module, "OUT", tmp_path / "output")
        monkeypatch.setattr(module, "REPORT", tmp_path / "report")
        monkeypatch.setattr(module, "ORIGINAL_SUMMARY", tmp_path / "original.json")
    parent = dict(manifest_sha256="parent", datasets={})
    m = dict(manifest_sha256="new", parent_manifest_sha256="parent", total_segment_seed_units=12,
             evidence="supplementary", source_sha256={}, jobs=[], settings=dict(
                 protocol=dict(datasets=["d1", "d2"], seeds=[0, 1, 2]), statistics=dict(
                     seed=1, paired_material_block_sign_flips=128, material_block_bootstrap_repetitions=128)))
    original = dict(datasets={})
    for dataset in ("d1", "d2"):
        groups = [dict(group=f"g{i}", subject="s", event_ids=[f"e{i}-{j}" for j in range(3)],
                       reference=["x", "y", "z"], material=f"m{i}", block=f"b{i}", scope="r") for i in range(2)]
        parent["datasets"][dataset] = dict(groups=groups, language="fr", counts=dict(groups=2, events=6))
        a, one, two = ["bad", "bad", "z"], ["x", "bad", "z"], ["x", "y", "z"]
        summary = [dict(condition=c, **metric_row(groups[0]["reference"], pred, False)) for c, pred in (("A", a), ("C", two))]
        original["datasets"][dataset] = dict(summary=summary,
            by_seed=[dict(seed=seed, **row) for seed in (0, 1, 2) for row in summary],
            primary_statistics=dict(effect=2/3, p_holm=.5))
        for seed in (0, 1, 2):
            old = dict(final=dict(A=a, C=two), trace=[dict(arrival=0, A=["bad"], C=["bad"]),
                dict(arrival=1, A=a[:2], C=["x", "bad"]), dict(arrival=2, A=a, C=two)])
            new = dict(final=dict(C1=one), trace=[dict(arrival=0, C1=["bad"]),
                dict(arrival=1, C1=["x", "bad"]), dict(arrival=2, C1=one)])
            ac_path, ac_chunk = tmp_path / f"{dataset}/{seed}/ac.json", tmp_path / f"{dataset}/{seed}/ac_chunk.json"
            fields = dict(manifest_sha256="parent", dataset=dataset, seed=seed, conditions=["A", "C"],
                          split="test", test_evaluated=True, selected_weight=.1)
            ac_records = [dict(group=g["group"], result=old) for g in groups]
            save(ac_chunk, dict(**fields, records=ac_records))
            save(ac_path, dict(**fields, status="completed", records=ac_records,
                               sources=[dict(path=shared.source_key(ac_chunk), sha256=sha256(ac_chunk))]))
            job = dict(dataset=dataset, seed=seed, original_ac=shared.source_key(ac_path))
            m["jobs"].append(job)
            path = results.OUT / dataset / f"seed-{seed:03d}" / "main.json"
            chunk_path = path.parent / "chunk.json"
            records = [dict(group=g["group"], result=new) for g in groups]
            save(chunk_path, dict(**worker.provenance(m, job), records=records))
            save(path, dict(**worker.provenance(m, job), status="completed", records=records,
                            sources=[dict(path=shared.source_key(chunk_path), sha256=sha256(chunk_path))]))
    save(results.ORIGINAL_SUMMARY, original)
    for key in ("d1", "d2"):
        monkeypatch.setitem(results.NAMES, key, key)
    report = results.export(m, parent)
    assert report["status"] == "completed" and report["original_AC_reproduced"]
    assert report["computed_conditions"] == ["C1"] and report["segment_seed_units"] == 12
    assert all(s["p_holm"] is not None for d in report["datasets"].values() for s in d["supplementary_statistics"].values())
    assert all(next(r for r in d["latency"] if r["condition"] == "C1")["max_wait_subsequent_word_windows"] == 1 for d in report["datasets"].values())
    path = results.OUT / "d1/seed-000/main.json"
    incomplete = read(path)
    incomplete["records"] = incomplete["records"][:-1]
    save(path, incomplete)
    with pytest.raises(ValueError, match="population"):
        results.export(m, parent)
