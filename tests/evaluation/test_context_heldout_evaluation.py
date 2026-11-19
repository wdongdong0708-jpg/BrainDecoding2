from types import SimpleNamespace

import numpy as np
import pandas as pd

from braindecoding.evaluation.context_heldout_evaluation import subject_order
from braindecoding.training.context_baseline_training import common_mask


def test_subject_parameter_order_follows_dataset_mapping():
    pallier = SimpleNamespace(subject_map={'sub-03': 1, 'sub-01': 0})
    chinese = SimpleNamespace(subject_ids=('sub-01', 'sub-03'))
    assert subject_order(pallier) == subject_order(chinese) == ['sub-01', 'sub-03']


def test_common_population_aligns_by_event_id_and_clips_each_segment():
    table = pd.DataFrame({'event_id': ['a', 'b', 'c', 'd', 'e', 'f'], 'group': ['long'] * 4 + ['short'] * 2})
    ids = ['f', 'b', 'a', 'e', 'd', 'c']
    observed = common_mask(table, ids, 'group').numpy()
    assert np.array_equal(observed, [False, True, True, False, False, False])
