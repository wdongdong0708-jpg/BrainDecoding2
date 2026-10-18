"""Recovery must share only pre-divergence arrivals and preserve all old records."""
import copy

import numpy as np
import pytest

from braindecoding import single_revision_c1_test_v2 as base
from braindecoding import single_revision_c1_test_v2_stable as stable
from braindecoding import fixed_k0_rolling_search_test_v2 as shared


pytestmark = pytest.mark.current_protocol


class NoLM:
    def score_extensions(self, prefixes, words):
        return np.zeros(len(words))


def test_sharing_pre_divergence_stage_equals_independent_c1_when_scores_are_stable():
    cases = [dict(predictions={0: np.array([[1., 0.]]*n), 1: np.array([[0., 1.]]*n)},
                  weight=0., open_vocabulary=False) for n in (1, 2, 5)]
    original = base.replay_c1_many(cases, ["a", "b"], np.eye(2), NoLM(), None, candidate_count=2)
    initial = [[dict(arrival=i, words=r["trace"][i]["C1"]) for i in range(min(2, len(r["trace"])))]
               for r in original]
    recovered = stable.replay_shared_initial(cases, initial, ["a", "b"], np.eye(2), NoLM(), None, candidate_count=2)
    assert recovered == original
    for n, result in zip((1, 2, 5), recovered):
        base.validate_c1_result(dict(event_ids=list(range(n))), result)


def test_extraction_never_reads_reference_final_or_future_c_arrivals():
    class EarlyTrace:
        def __getitem__(self, i):
            assert i in (0, 1), "Future C trajectory must not enter C1"
            return dict(arrival=i, C=["early"]*(i+1))
    class Result(dict):
        def __getitem__(self, key):
            assert key == "trace", "Final C output must not enter C1"
            return EarlyTrace()
    class Group(dict):
        def __getitem__(self, key):
            assert key in ("group", "event_ids"), "Reference words must not enter C1"
            return super().__getitem__(key)
    group = Group(group="g", event_ids=list(range(5)))
    ac = dict(group="g", result=Result())
    assert stable.shared_initial([group], [ac]) == [[dict(arrival=0, words=["early"]),
                                                    dict(arrival=1, words=["early", "early"])]]


def test_independent_replay_starts_at_arrival_two_and_never_revises_the_frozen_first_word(monkeypatch):
    k0, k1 = np.eye(5), np.eye(5)*10
    seen = []
    def search(requests, *args, **kwargs):
        arrival = len(seen)+2
        r = requests[0]
        np.testing.assert_array_equal(r["queries"], np.stack([k1[arrival-1], k0[arrival]]))
        assert r["prefix"][0] == "frozen"
        assert len(r["prefix"]) == arrival-1
        seen.append(arrival)
        return [r["prefix"]+tuple(f"new-{arrival}-{i}" for i in range(2))]
    monkeypatch.setattr(stable, "search_many", search)
    initial = [[dict(arrival=0, words=["initial"]), dict(arrival=1, words=["frozen", "second"])]]
    output = stable.replay_shared_initial([dict(predictions={0: k0, 1: k1}, weight=.1, open_vocabulary=True)],
                                         initial, [], None, None, None)[0]
    assert seen == [2, 3, 4]
    assert len(output["trace"]) == 5
    assert output["final"]["C1"][0] == "frozen"
    base.validate_c1_result(dict(event_ids=list(range(5))), output)


def test_shared_input_rejects_extra_future_arrivals_and_hidden_reference_fields():
    case = dict(predictions={0: np.eye(3), 1: np.eye(3)}, weight=0., open_vocabulary=False)
    entries = [dict(arrival=0, words=["a"]), dict(arrival=1, words=["a", "a"])]
    with pytest.raises(ValueError, match="Only actual initial"):
        stable.replay_shared_initial([case], [entries+[dict(arrival=2, words=["future"]*3)]], [], None, None, None)
    hidden = copy.deepcopy(entries)
    hidden[0]["reference"] = "secret"
    with pytest.raises(ValueError, match="undeclared or future"):
        stable.replay_shared_initial([case], [hidden], [], None, None, None)


def test_adoption_preserves_raw_record_content_and_retains_strict_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "ROOT", tmp_path)
    job = dict(dataset="d", seed=0)
    group = dict(group="g", event_ids=["e0", "e1", "e2"])
    result = dict(final=dict(C1=["b", "b", "a"]), trace=[dict(arrival=0, C1=["a"]),
        dict(arrival=1, C1=["b", "a"]), dict(arrival=2, C1=["b", "b", "a"])])
    ac = dict(group="g", result=dict(trace=[dict(arrival=0, A=["a"], C=["a"]),
                                           dict(arrival=1, C=["b", "a"])]))
    old_m = dict(manifest_sha256="old")
    m = dict(manifest_sha256="new", scientific_protocol_manifest_sha256="old")
    path = tmp_path / "old_chunk.json"
    old = dict(**base.provenance(old_m, job), seconds=1., records=[dict(group="g", result=result)])
    base.save(path, old)
    checksum = base.sha256(path)
    migrated = stable.adopt(path, checksum, old_m, m, job, [group], [ac])
    assert migrated["records"] == old["records"]
    assert base.digest(migrated["records"]) == migrated["adopted_from"]["records_sha256"]
    assert migrated["manifest_sha256"] == "new"
    assert migrated["adopted_from"]["manifest_sha256"] == "old"
    assert base.sha256(path) == checksum
    with pytest.raises(ValueError, match="changed before adoption"):
        stable.adopt(path, "stale", old_m, m, job, [group], [ac])
    bad = copy.deepcopy(migrated)
    bad["records"][0]["result"]["trace"][2]["C1"][0] = "second-revision"
    with pytest.raises(ValueError, match="frozen prefix"):
        stable.validate_chunk(bad, m, job, [group], [ac])
    with pytest.raises(ValueError, match="execution provenance"):
        stable.validate_chunk({**migrated, "execution_id": "wrong"}, m, job, [group], [ac])
    altered = copy.deepcopy(migrated)
    altered["records"][0]["result"]["trace"][1]["C1"][0] = "different"
    altered["records"][0]["result"]["trace"][2]["C1"][0] = "different"
    altered["records"][0]["result"]["final"]["C1"][0] = "different"
    with pytest.raises(ValueError, match="first K1"):
        stable.validate_chunk(altered, m, job, [group], [ac])
