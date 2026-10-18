"""The B recovery shares only arrival zero and preserves every completed record."""
import copy

import numpy as np
import pytest

from braindecoding import fixed_k0_rolling_search_test_v2 as base
from braindecoding import fixed_k0_rolling_search_test_v2_stable as stable
from braindecoding.unified_sentence_batch import replay_many

pytestmark = pytest.mark.current_protocol


class StableLM:
    def score_extensions(self, prefixes, words):
        return np.array([.5 if prefix and prefix[-1] == "a" and word == "a" else 0.
                         for prefix, word in zip(prefixes, words)])


def test_shared_first_stage_matches_original_b_and_still_revises_with_k0_only():
    vectors = np.array([[.4, .6], [1., 0.], [.2, .8], [.8, .2], [.1, .9]])
    cases = [dict(predictions={0: vectors[:n].copy()}, weight=1., open_vocabulary=False) for n in (1, 2, 5)]
    settings = dict(candidate_count=2, beam_width=2, neural_scale=1.)
    original = replay_many(cases, ["a", "b"], np.eye(2), StableLM(), None, conditions="B", **settings)
    initial = [dict(arrival=0, words=r["trace"][0]["B"]) for r in original]
    recovered = stable.replay_shared_initial(cases, initial, ["a", "b"], np.eye(2), StableLM(), None, **settings)
    assert recovered == original
    assert recovered[1]["trace"][0]["B"] == ["b"]
    assert recovered[1]["final"]["B"] == ["a", "a"]
    for n, result in zip((1, 2, 5), recovered):
        base.validate_b_result(dict(event_ids=list(range(n))), result)
        np.testing.assert_array_equal(cases[(1, 2, 5).index(n)]["predictions"][0], vectors[:n])


def test_shared_extraction_cannot_access_reference_c_final_or_future_a():
    class Trace:
        def __getitem__(self, index):
            assert index == 0, "Future arrivals must never enter B"
            class Step(dict):
                def __getitem__(self, key):
                    assert key in ("arrival", "A"), "C must not enter B"
                    return super().__getitem__(key)
            return Step(arrival=0, A=["generated-first"], C=["forbidden"])
    class Result(dict):
        def __getitem__(self, key):
            assert key == "trace", "Final A/C output must never enter B"
            return Trace()
    class Group(dict):
        def __getitem__(self, key):
            assert key == "group", "Reference and future group text must never enter B"
            return super().__getitem__(key)
    assert stable.shared_initial([Group(group="g", reference=["secret"])],
                                 [dict(group="g", result=Result())]) == [dict(arrival=0, words=["generated-first"])]


def test_independent_search_starts_at_one_and_uses_exact_three_word_k0_window(monkeypatch):
    k0 = np.eye(6)
    seen = []
    def search(requests, *args, **kwargs):
        arrival = len(seen)+1
        start = max(0, arrival-2)
        request = requests[0]
        np.testing.assert_array_equal(request["queries"], k0[start:arrival+1])
        assert len(request["prefix"]) == start
        seen.append(arrival)
        return [request["prefix"]+tuple(f"word-{arrival}-{i}" for i in range(arrival+1-start))]
    monkeypatch.setattr(stable, "search_many", search)
    result = stable.replay_shared_initial([dict(predictions={0: k0}, weight=.1, open_vocabulary=True)],
                                         [dict(arrival=0, words=["shared-first"])], [], None, None, None)[0]
    assert seen == [1, 2, 3, 4, 5]
    assert result["trace"][1]["B"][0] != "shared-first", "The shared first word may still be revised by B"
    base.validate_b_result(dict(event_ids=list(range(6))), result)


def test_one_word_group_only_reuses_its_actual_first_output(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A one-word group has no later window")
    monkeypatch.setattr(stable, "search_many", forbidden)
    result = stable.replay_shared_initial([dict(predictions={0: np.eye(1)}, weight=.1, open_vocabulary=True)],
                                         [dict(arrival=0, words=["first"])], [], None, None, None)
    assert result == [dict(final=dict(B=["first"]), trace=[dict(arrival=0, B=["first"])])]


def test_rejects_updated_neural_inputs_hidden_references_and_later_shared_arrivals():
    case = dict(predictions={0: np.eye(3)}, weight=.1, open_vocabulary=True)
    initial = dict(arrival=0, words=["first"])
    variants = [(dict(case, predictions={0: np.eye(3), 1: np.eye(3)}), initial, "K0 only"),
                (dict(case, reference=["secret"]), initial, "undeclared"),
                (case, dict(initial, reference="secret"), "undeclared or future"),
                (case, dict(arrival=1, words=["first", "future"]), "undeclared or future"),
                (case, dict(arrival=0, words=["first", "future"]), "first output")]
    for malformed_case, malformed_initial, message in variants:
        with pytest.raises(ValueError, match=message):
            stable.replay_shared_initial([malformed_case], [malformed_initial], [], None, None, None)


def test_malformed_search_population_or_prefix_fails(monkeypatch):
    case = dict(predictions={0: np.eye(2)}, weight=.1, open_vocabulary=True)
    monkeypatch.setattr(stable, "search_many", lambda *a, **k: [])
    with pytest.raises(ValueError, match="population"):
        stable.replay_shared_initial([case], [dict(arrival=0, words=["first"])], [], None, None, None)
    monkeypatch.setattr(stable, "search_many", lambda *a, **k: [("wrong-length",)])
    with pytest.raises(ValueError, match="known word count"):
        stable.replay_shared_initial([case], [dict(arrival=0, words=["first"])], [], None, None, None)


def test_adoption_preserves_records_original_files_and_strict_initial_agreement(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "ROOT", tmp_path)
    job = dict(dataset="d", seed=2)
    group = dict(group="g", event_ids=["e0", "e1", "e2", "e3"])
    words = ["first", "second", "third", "last"]
    result = dict(final=dict(B=words), trace=[dict(arrival=i, B=words[:i+1]) for i in range(4)])
    old_m = dict(manifest_sha256="old")
    m = dict(manifest_sha256="new", scientific_protocol_manifest_sha256="old")
    old = dict(manifest_sha256="old", dataset="d", seed=2, conditions=["B"], split="test",
               test_evaluated=True, neural_representation="K0_only", branch="open", alignment="real",
               selected_weight=.1, seconds=1., records=[dict(group="g", result=result)])
    ac = [dict(group="g", result=dict(trace=[dict(arrival=0, A=["first"])]))]
    path = tmp_path / "old.json"
    base.save(path, old)
    checksum = base.sha256(path)
    migrated = stable.adopt(path, checksum, old_m, m, job, [group], ac)
    assert migrated["records"] == old["records"]
    assert migrated["adopted_from"]["records_sha256"] == base.digest(old["records"])
    assert base.sha256(path) == checksum
    stable.validate_chunk(migrated, m, job, [group], ac)
    with pytest.raises(ValueError, match="changed before adoption"):
        stable.adopt(path, "stale", old_m, m, job, [group], ac)
    with pytest.raises(ValueError, match="execution provenance"):
        stable.validate_chunk(dict(migrated, shared_initial_arrivals_maximum=2), m, job, [group], ac)
    altered = copy.deepcopy(migrated)
    altered["records"][0]["result"]["trace"][0]["B"] = ["different"]
    with pytest.raises(ValueError, match="B and A disagree"):
        stable.validate_chunk(altered, m, job, [group], ac)
    altered = copy.deepcopy(migrated)
    altered["records"][0]["result"]["trace"][3]["B"][0] = "changed-after-freezing"
    with pytest.raises(ValueError, match="frozen prefix"):
        stable.validate_chunk(altered, m, job, [group], ac)


def test_export_restores_original_reporting_globals_on_error(tmp_path, monkeypatch):
    from braindecoding import fixed_k0_rolling_search_test_v2_results as reporting
    monkeypatch.setattr(stable, "verify_preserved", lambda m: 0)
    previous = {key: getattr(reporting, key) for key in ("ID", "OUT", "REPORT")}
    def fail(*args):
        assert reporting.ID == stable.ID
        assert reporting.OUT == stable.OUT
        assert reporting.REPORT == stable.REPORT
        raise ValueError("export failure")
    monkeypatch.setattr(reporting, "export", fail)
    with pytest.raises(ValueError, match="export failure"):
        stable.export(dict(jobs=[]), {})
    assert {key: getattr(reporting, key) for key in previous} == previous
