"""Scientific contracts for the fixed-K0 B ablation and paired decomposition."""
import copy
import csv

import numpy as np
import pytest

from braindecoding import fixed_k0_rolling_search_test_v2 as worker
from braindecoding import fixed_k0_rolling_search_test_v2_results as results
from braindecoding.unified_sentence_batch import replay_many
from braindecoding.unified_sentence_v2 import read, save, sha256
from braindecoding.unified_sentence_v2_results import METRICS, metric_row

pytestmark = pytest.mark.current_protocol


def test_k0_only_search_can_revise_a_word_with_unchanged_neural_representation():
    class LM:
        def score_extensions(self, prefixes, words):
            return np.array([.5 if p and p[-1] == "a" and w == "a" else 0.
                             for p, w in zip(prefixes, words)])
    k0 = np.array([[.4, .6], [1., 0.]])
    case = dict(predictions={0: k0}, weight=1., open_vocabulary=False)
    actual = replay_many([case], ["a", "b"], np.eye(2), LM(), None,
                         conditions="AB", candidate_count=2, neural_scale=1.)[0]
    assert actual["trace"][0] == dict(arrival=0, A=["b"], B=["b"])
    assert actual["final"] == dict(A=["b", "a"], B=["a", "a"])
    assert np.array_equal(k0, [[.4, .6], [1., 0.]])


def test_b_loader_and_cases_never_read_updated_horizons_or_pass_reference(monkeypatch):
    class Archive:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __getitem__(self, key):
            assert key in ("k0", "event_ids"), "K1/K2 must never be accessed"
            return np.eye(4) if key == "k0" else np.array(["a", "b", "c", "d"])
    monkeypatch.setattr(worker.np, "load", lambda *args, **kwargs: Archive())
    k0, ids = worker.load_k0("predictions.npz")
    group = dict(event_ids=["d", "a"], reference=["secret", "reference"])
    cases = worker.k0_cases([group], k0, ids)
    assert set(cases[0]) == {"predictions", "weight", "open_vocabulary"}
    assert set(cases[0]["predictions"]) == {0}
    assert np.array_equal(cases[0]["predictions"][0], k0[[3, 0]])
    with pytest.raises(ValueError, match="Duplicate"):
        worker.k0_cases([group], k0, ["a"] * 4)


def test_every_b_arrival_uses_the_same_k0_vectors_in_exactly_three_word_window(monkeypatch):
    from braindecoding import unified_sentence_batch as batch
    seen = []
    def search(requests, *args, **kwargs):
        assert len(requests) == 1
        r = requests[0]
        arrival = len(seen)
        start = max(0, arrival - 2)
        np.testing.assert_array_equal(r["queries"], np.eye(6)[start:arrival+1])
        assert len(r["prefix"]) == start
        seen.append(arrival)
        return [r["prefix"] + tuple(f"w-{i}" for i in range(start, arrival+1))]
    monkeypatch.setattr(batch, "search_many", search)
    case = worker.k0_cases([dict(event_ids=[str(i) for i in range(6)])], np.eye(6), list(map(str, range(6))))
    actual = replay_many(case, [], None, None, None, conditions="B")[0]
    assert seen == list(range(6))
    worker.validate_b_result(dict(event_ids=list(range(6))), actual)


def b_result():
    words = ["bad", "y", "z", "last", "end"]
    return dict(final=dict(B=words), trace=[dict(arrival=i, B=words[:i+1]) for i in range(len(words))])


def test_resume_rejects_non_k0_conditions_wrong_population_and_frozen_prefix_changes():
    group = dict(group="g", event_ids=list(range(5)))
    job = dict(dataset="data", seed=0)
    m = dict(manifest_sha256="frozen")
    chunk = dict(manifest_sha256="frozen", dataset="data", seed=0, conditions=["B"], split="test",
                 test_evaluated=True, neural_representation="K0_only", branch="open", alignment="real",
                 selected_weight=.1, records=[dict(group="g", result=b_result())])
    worker.validate_b_chunk(chunk, m, job, [group])
    for key, value in (("conditions", ["C"]), ("neural_representation", "K2"), ("seed", 1),
                       ("manifest_sha256", "stale"), ("selected_weight", .5), ("split", "val")):
        with pytest.raises(ValueError, match="provenance"):
            worker.validate_b_chunk({**chunk, key: value}, m, job, [group])
    with pytest.raises(ValueError, match="population"):
        worker.validate_b_chunk(chunk, m, job, [{**group, "group": "different"}])
    changed = copy.deepcopy(b_result())
    changed["trace"][3]["B"][0] = "changed-after-freezing"
    with pytest.raises(ValueError, match="frozen prefix"):
        worker.validate_b_result(group, changed)


def test_aggregation_keeps_subjects_and_seeds_equal_and_pools_secondary_word_counts():
    rows = []
    for seed, subject, scores in ((0, "s1", [(1, 1.), (9, 0.)]), (0, "s2", [(2, 1.)]),
                                  (1, "s1", [(1, 0.)]), (1, "s2", [(2, 0.)])):
        for words, score in scores:
            for c in "ABC":
                value = score if c == "C" else score/2 if c == "B" else 0.
                rows.append(dict(seed=seed, subject=subject, condition=c, words=words,
                                 sentence_bleu1_mean=value, bleu1=value, aligned_word_accuracy=value, wer=1-value))
    summary = results.summarize(rows)
    c = next(r for r in summary["summary"] if r["condition"] == "C")
    assert c["sentence_bleu1_mean"] == .375
    assert c["aligned_word_accuracy"] == .275
    assert c["wer"] == .725
    with pytest.raises(ValueError, match="Subjects differ"):
        results.summarize([r for r in rows if not (r["seed"] == 1 and r["subject"] == "s2")])


def test_gain_shares_preserve_negative_or_over_100_percent_and_suppress_zero_total():
    summary = [dict(condition=c, **{m: 1-score if m == "wer" else score for m in METRICS})
               for c, score in zip("ABC", [.2, .5, .4])]
    gains = results.gain_decomposition(summary)
    for g in gains.values():
        assert g["search_gain_B_minus_A"] == pytest.approx(.3)
        assert g["neural_update_gain_C_minus_B"] == pytest.approx(-.1)
        assert g["search_share"] == pytest.approx(1.5)
        assert g["neural_update_share"] == pytest.approx(-.5)
    for total in (.2, .1, .2+1e-14):
        same = [dict(condition=c, **{m: score for m in METRICS}) for c, score in zip("ABC", [.2, .3, total])]
        assert results.gain_decomposition(same)["sentence_bleu1_mean"]["search_share"] is None


def test_paired_inference_uses_shared_material_blocks_and_requires_all_three_conditions():
    rows = []
    for seed in range(3):
        for subject in ("s1", "s2"):
            for block in range(4):
                for condition, score in zip("ABC", [.1, .3, .6]):
                    rows.append(dict(seed=seed, subject=subject, group=f"{subject}-{block}", block=f"b{block}",
                                     scope="r", words=3, condition=condition, sentence_bleu1_mean=score))
    cfg = dict(seed=1, paired_material_block_sign_flips=10000, material_block_bootstrap_repetitions=128)
    stats = results.paired_statistics(rows, cfg)
    assert stats["B_minus_A"]["effect"] == pytest.approx(.2)
    assert stats["C_minus_B"]["effect"] == pytest.approx(.3)
    assert all(s["blocks"] == 4 and s["subjects"] == 2 and s["seeds"] == 3 for s in stats.values())
    with pytest.raises(ValueError, match="Unpaired"):
        results.paired_statistics([r for r in rows if r["condition"] != "B"], cfg)


def test_online_counts_exclude_the_last_word_and_final_net_gains_add_up():
    group = dict(reference=["a", "b", "c"])
    ac = dict(final=dict(A=["bad", "b", "c"], C=["a", "b", "bad"]),
              trace=[dict(C=["bad"]), dict(C=["a", "bad"]), dict(C=["a", "b", "bad"])])
    b = dict(final=dict(B=["a", "bad", "c"]), trace=[dict(B=["bad"]), dict(B=["a", "bad"]), dict(B=["a", "bad", "c"])])
    changes = {r["comparison"]: r for r in results.change_rows(group, ac, b, {})}
    assert changes["B_first_to_final"]["eligible"] == 2
    assert changes["B_first_to_final"]["repaired"] == 1
    assert changes["C_first_to_final"]["repaired"] == 2
    assert changes["A_to_B"]["harmed"] == 1
    assert changes["B_to_C"]["net_correct_words"] == 0
    assert changes["A_to_B"]["net_correct_words"] + changes["B_to_C"]["net_correct_words"] == changes["A_to_C"]["net_correct_words"]
    assert results.transition_counts([], [], []) == dict(eligible=0, changed=0, repaired=0, harmed=0, net_correct_words=0)


def test_complete_export_reproduces_original_ac_and_refuses_incomplete_b(tmp_path, monkeypatch):
    for module in (worker, results):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "OUT", tmp_path / "output")
        monkeypatch.setattr(module, "REPORT", tmp_path / "report")
        monkeypatch.setattr(module, "ORIGINAL_SUMMARY", tmp_path / "original_summary.json")
    parent = dict(manifest_sha256="parent", datasets={})
    m = dict(manifest_sha256="new", parent_manifest_sha256="parent", total_segment_seed_units=12,
             evidence="supplementary", source_sha256={}, jobs=[], settings=dict(
                 protocol=dict(datasets=["d1", "d2"], seeds=[0, 1, 2]), statistics=dict(
                     seed=1, paired_material_block_sign_flips=128, material_block_bootstrap_repetitions=128)))
    original = dict(datasets={})
    for dataset in ("d1", "d2"):
        groups = [dict(group=f"g{i}", subject="s", event_ids=[f"e{i}-{j}" for j in range(3)],
                       reference=["x", "y", "z"], material=f"m{i}", block=f"b{i}", scope="r") for i in range(2)]
        parent["datasets"][dataset] = dict(groups=groups, language="fr", counts=dict(groups=2))
        a, b, c = ["bad", "bad", "z"], ["bad", "y", "z"], ["x", "y", "z"]
        summary = [dict(condition=condition, **metric_row(groups[0]["reference"], prediction, False))
                   for condition, prediction in (("A", a), ("C", c))]
        original["datasets"][dataset] = dict(summary=summary, primary_statistics=dict(effect=2/3, p_holm=.5))
        for seed in range(3):
            old = dict(final=dict(A=a, C=c), trace=[dict(arrival=0, A=["bad"], C=["bad"]),
                  dict(arrival=1, A=a[:2], C=c[:2]), dict(arrival=2, A=a, C=c)])
            new = dict(final=dict(B=b), trace=[dict(arrival=i, B=b[:i+1]) for i in range(3)])
            ac_path, ac_chunk = tmp_path / f"{dataset}/{seed}/ac.json", tmp_path / f"{dataset}/{seed}/ac_chunk.json"
            ac_fields = dict(manifest_sha256="parent", dataset=dataset, seed=seed, conditions=["A", "C"],
                             split="test", test_evaluated=True, selected_weight=.1)
            ac_records = [dict(group=g["group"], result=old) for g in groups]
            save(ac_chunk, dict(**ac_fields, records=ac_records))
            save(ac_path, dict(**ac_fields, status="completed", records=ac_records,
                               sources=[dict(path=worker.source_key(ac_chunk), sha256=sha256(ac_chunk))]))
            job = dict(dataset=dataset, seed=seed, original_ac=worker.source_key(ac_path))
            m["jobs"].append(job)
            path = results.OUT / dataset / f"seed-{seed:03d}" / "main.json"
            chunk_path = path.parent / "chunk.json"
            fields = dict(manifest_sha256="new", dataset=dataset, seed=seed, conditions=["B"], split="test",
                          test_evaluated=True, neural_representation="K0_only", branch="open", alignment="real", selected_weight=.1)
            records = [dict(group=g["group"], result=new) for g in groups]
            save(chunk_path, dict(**fields, records=records))
            save(path, dict(**fields, records=records, status="completed",
                            sources=[dict(path=worker.source_key(chunk_path), sha256=sha256(chunk_path))]))
            m["source_sha256"][worker.source_key(ac_path)] = sha256(ac_path)
    save(results.ORIGINAL_SUMMARY, original)
    monkeypatch.setitem(results.NAMES, "d1", "Data 1")
    monkeypatch.setitem(results.NAMES, "d2", "Data 2")
    report = results.export(m, parent)
    assert report["status"] == "completed" and report["original_AC_reproduced"]
    assert report["computed_conditions"] == ["B"] and report["segment_seed_units"] == 12
    assert all(len(d["supplementary_statistics"]) == 2 for d in report["datasets"].values())
    with (tmp_path / f"data/results/exports/{worker.ID}_by_group.csv").open(encoding="utf-8-sig") as handle:
        assert len(list(csv.DictReader(handle))) == 36
    missing = results.OUT / "d1/seed-000/main.json"
    incomplete = read(missing)
    incomplete["records"] = incomplete["records"][:-1]
    save(missing, incomplete)
    with pytest.raises(ValueError, match="population"):
        results.export(m, parent)
