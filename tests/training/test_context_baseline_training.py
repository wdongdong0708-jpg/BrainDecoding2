"""Scientific contracts for new context runs; existing v2 files stay frozen."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from braindecoding import unified_context_baselines_v2 as worker
from braindecoding.models import GroupedTransformerEncoder
from braindecoding.training.word import SentenceBatchSampler


@pytest.fixture(autouse=True)
def roots(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv("BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models")


@pytest.mark.parametrize("dataset", worker.base.DATASETS)
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_k2_alias_including_selection_and_sampling(dataset, seed):
    assert worker.build_config(dataset, seed, "local_r2") == worker.build_config(dataset, seed, "independent_k2")
    for name in worker.settings()["protocol"]["conditions"]:
        cfg = worker.build_config(dataset, seed, name)
        assert "local_window_revision" not in cfg["model"]
        assert cfg["model"]["use_transformer"]
        assert cfg["model"]["transformer"]["depth"] == cfg["model"]["transformer"]["heads"] == 16
        assert cfg["training"]["preserve_complete_groups"]
        assert worker.base.group_column(cfg) == worker.base.config()["adapters"][dataset]["group_column"]


def test_common_query_alignment_uses_ids_not_encoded_order():
    table = pd.DataFrame({"event_id": list("abcdefg"), "group": [1, 1, 1, 1, 2, 2, 2]})
    assert worker.common_mask(table, list("gfedcba"), "group").tolist() == [False, False, True, False, False, True, True]
    with pytest.raises(ValueError):
        worker.common_mask(table, list("abcdefa"), "group")
    with pytest.raises(ValueError):
        worker.common_mask(table, list("abcdef"), "group")


def test_every_training_word_once_without_splitting_groups():
    groups = [1] * 3 + [2] * 4 + [3] * 2
    sampler = SentenceBatchSampler(groups, 5, True, 2, preserve_complete_groups=True)
    sampler.set_epoch(8)
    batches = list(sampler)
    assert sorted(sum(batches, [])) == list(range(9))
    for group in set(groups):
        assert sum(any(groups[i] == group for i in batch) for batch in batches) == 1
    loader = worker.CoverageLoader([{"event_id": [str(i) for i in batch]} for batch in batches],
                                  map(str, range(9)), lambda **kw: None)
    list(loader)
    with pytest.raises(ValueError, match="repeated"):
        list(worker.CoverageLoader([{"event_id": ["a"]}, {"event_id": ["a"]}], ["a"], lambda **kw: None))


def test_radius_two_equals_k2_and_ignores_outside_context():
    config = {"depth": 2, "heads": 2, "attention_dropout": 0, "feedforward_dropout": 0}
    torch.manual_seed(123)
    radius = GroupedTransformerEncoder(8, {**config, "query_window_radius": 2}).eval()
    delay = GroupedTransformerEncoder(8, {**config, "query_window_left_radius": 2,
                                               "query_window_right_radius": 2}).eval()
    delay.load_state_dict(radius.state_dict())
    features, groups = torch.randn(12, 8), torch.tensor([0] * 9 + [1] * 3)
    expected = radius(features, groups)
    torch.testing.assert_close(expected, delay(features, groups), rtol=0, atol=0)
    changed = features.clone()
    changed[0] += 20
    changed[9:] -= 10
    torch.testing.assert_close(expected[4], delay(changed, groups)[4], rtol=0, atol=0)


def test_strict_bank_keeps_absent_candidate():
    encoded = {"event_ids": ["a", "b", "c"], "words": ["one"] * 3,
               "predictions": torch.tensor([[1., 0.]] * 3), "targets": torch.tensor([[1., 0.]] * 3)}
    ds = SimpleNamespace(table=pd.DataFrame({"event_id": ["a", "b", "c"], "sentence_uid": [1] * 3,
         "subject_id": ["s"] * 3}), embedding_map={"one": np.array([1., 0.]), "absent": np.array([0., 1.])})
    wrapped = {**encoded, "revision_predictions": {0: encoded["predictions"]},
               "revision_availability_masks": {0: worker.common_mask(ds.table, encoded["event_ids"], "sentence_uid")}}
    strict = worker.base.strict_bank_metrics(wrapped, ds, ["one", "absent"])["0"]
    assert strict["common_targets"]["candidate_count"] == 2
    assert strict["common_targets"]["query_count"] == 1
