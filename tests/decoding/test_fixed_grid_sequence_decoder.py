import unittest

import numpy as np

from braindecoding.fixed_grid_ctc import (
    aggregate_counts, average_same_times, continuous_gaussian, fixed_groups,
    input_windows, merge_probabilities, neural_probabilities, sentence_counts,
)


class FixedGridTests(unittest.TestCase):
    def test_absolute_rounding_and_separate_slid_groups(self):
        grid = fixed_groups(13., 15., 13., 400, 5)
        np.testing.assert_array_equal(grid['starts'][:6, 0], [0, 8, 15, 23, 30, 38])
        self.assertTrue((np.diff(grid['starts'], axis=1) == 15).all())
        self.assertEqual(grid['starts'][2, 0], grid['starts'][0, 1])
        self.assertFalse(grid['scored'][:, -2:].any())
        self.assertTrue((grid['ticks'][grid['scored']] < grid['count']).all())

    def test_unavailable_future_fails_without_padding(self):
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            fixed_groups(0., .2, 0., 100, 5)
        with self.assertRaises(ValueError):
            input_windows(np.zeros((2, 100)), [51])

    def test_continuous_noise_shares_samples_before_window_transform(self):
        a = continuous_gaussian((2, 200), [0, 0], [1, 2], 20261012, 'sub-01|run-07')
        b = continuous_gaussian(a.shape, [0, 0], [1, 2], 20261012, 'sub-01|run-07')
        np.testing.assert_array_equal(a, b)
        first, second = a[:, :50], a[:, 15:65]
        np.testing.assert_array_equal(first[:, 15:], second[:, :35])
        self.assertFalse(np.array_equal(a, continuous_gaussian(a.shape, [0, 0], [1, 2], 20261013, 'sub-01|run-07')))
        windows = input_windows(a, [0, 15])
        expected = np.stack([first, second]).copy()
        expected -= expected[:, :, :25].mean(axis=-1, keepdims=True)
        expected = np.clip(expected, -5., 5.)
        np.testing.assert_array_equal(windows, expected)

    def test_duplicate_average_then_pool_not_average_argmax(self):
        raw = np.asarray([[.9, .1], [.2, .8], [.3, .7]], dtype=np.float32)
        averaged, counts, ready = average_same_times(raw, [0, 0, 1], [50, 80, 58], 2)
        np.testing.assert_allclose(averaged, [[.55, .45], [.3, .7]])
        self.assertEqual(counts.tolist(), [2, 1])
        self.assertEqual(ready.tolist(), [80, 58])
        self.assertEqual(merge_probabilities(averaged, ready, smoothing=False)['word_ids'], [0, 1])
        with self.assertRaisesRegex(ValueError, 'Missing'):
            average_same_times(raw[:1], [0], [50], 2)

    def test_valid_pool_and_short_timeline_rule(self):
        p = np.asarray([[.8, .2]] * 5 + [[.1, .9]] * 4)
        merged = merge_probabilities(p, np.arange(9) + 50, smoothing=True)
        self.assertEqual(merged['pool_spans'], [[0, 5], [3, 8]])
        self.assertEqual(merged['pooled_ready_samples'], [54, 57])
        self.assertEqual(merged['word_centre_half_ticks'], [2., 5.])
        short = merge_probabilities(p[:2], [50, 58], smoothing=True)
        self.assertTrue(short['short_pool'])
        self.assertEqual(short['pool_spans'], [[0, 2]])

    def test_no_pool_does_not_use_future_predictions(self):
        p = np.asarray([[.8, .2], [.7, .3], [.1, .9]])
        a = merge_probabilities(p, [50, 58, 65], smoothing=False)
        self.assertEqual(a['word_ids'], [0, 1])
        self.assertEqual(a['word_ready_samples'], [50, 65])
        self.assertEqual(a['pool_spans'], [[0, 1], [1, 2], [2, 3]])

    def test_variable_length_metrics_and_brevity_penalty(self):
        row = sentence_counts(['a', 'b', 'c', 'd'], ['a', 'b'])
        self.assertEqual(row['word_errors'], 2)
        self.assertAlmostEqual(row['sentence_bleu1'], np.exp(-1))
        self.assertEqual(row['reference_characters'], 7)
        inserted = sentence_counts(['a'], ['a', 'x', 'x'])
        self.assertEqual(inserted['word_errors'], 2)
        self.assertAlmostEqual(aggregate_counts([inserted])['wer'], 2.)
        self.assertEqual(sentence_counts(['a'], [])['sentence_bleu1'], 0.)

    def test_probabilities_keep_all_candidates(self):
        p = neural_probabilities(np.eye(2), np.asarray([[1, 0], [0, 1], [0, 0]]))
        self.assertEqual(p.shape, (2, 3))
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-7)
        self.assertEqual(p.argmax(axis=1).tolist(), [0, 1])


if __name__ == '__main__':
    unittest.main()
