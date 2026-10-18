import numpy as np

from braindecoding.jayalath_significance import (
    cluster_bleu_differences,
    cluster_sentence_bleu_differences,
    holm_adjust,
    paired_cluster_test,
)


def test_repeated_stimulus_is_one_cluster_and_effect_is_paired():
    rows = [
        {"reference": ["甲", "乙"], "real": ["甲", "乙"], "control": ["丙", "乙"]},
        {"reference": ["甲", "乙"], "real": ["甲", "乙"], "control": ["丙", "乙"]},
        {"reference": ["丁"], "real": ["丁"], "control": ["戊"]},
    ]
    counts = cluster_bleu_differences(rows, "real", "control")
    assert counts.shape == (2, 3)
    assert sorted(counts.tolist()) == [[1, 0, 1], [4, 2, 4]]
    result = paired_cluster_test(counts, permutations=100, bootstraps=100, seed=1)
    assert np.isclose(result["delta_bleu1"], 3 / 5)


def test_holm_adjustment_preserves_original_order():
    assert np.allclose(holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])


def test_author_style_sentence_bleu_weights_segments_equally():
    rows = [
        {"reference": ["a"], "real": ["a"], "control": ["x"]},
        {"reference": ["b", "c", "d"], "real": ["b", "x", "x"],
         "control": ["x", "x", "x"]},
    ]
    counts = cluster_sentence_bleu_differences(rows, "real", "control")
    result = paired_cluster_test(counts, permutations=100, bootstraps=100,
                                 denominator_label="reference_groups")
    assert result["reference_groups"] == 2
    assert np.isclose(result["delta_bleu1"], 2 / 3)
