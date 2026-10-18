import unittest
import numpy as np
import torch

from braindecoding.models import BrainEmbeddingModel, GroupedTransformerEncoder
from braindecoding.fixed_grid_fast_inference import install


class FastInferenceTests(unittest.TestCase):
    def test_context_exact_for_interleaved_labels_and_variable_lengths(self):
        torch.manual_seed(4)
        model = GroupedTransformerEncoder(16, dict(depth=2, heads=2)).eval()
        features = torch.randn(9, 16)
        groups = torch.tensor([4, 2, 4, 7, 2, 2, 7, 4, 4])
        with torch.inference_mode():
            native = model(features, groups)
            from braindecoding.fixed_grid_fast_inference import vectorized_context
            fast = vectorized_context(model, features, groups)
        torch.testing.assert_close(native, fast, rtol=0, atol=0)

    def test_rolling_exact_with_natural_edges_and_query_batch_splits(self):
        torch.manual_seed(8)
        config = dict(embedding_dimension=16, use_transformer=True,
                      conv=dict(merger_channels=4, merger_position_dimension=8,
                                initial_linear=8, hidden=8, depth=1,
                                temporal_attention_hidden=8),
                      transformer=dict(depth=2, heads=2),
                      local_window_revision=dict(enabled=True, history_radius=2,
                                                 horizons=[0, 1, 2], validation_query_batch_size=4))
        model = BrainEmbeddingModel(np.zeros((3, 2), dtype=np.float32), 1, config).eval()
        signals, subjects = torch.randn(12, 3, 50), torch.zeros(12, dtype=torch.long)
        groups = torch.tensor([0] * 5 + [1] * 2 + [2] * 5)
        with torch.inference_mode():
            native = model.forward_prefix_revisions(signals, subjects, groups)
            install(model)
            fast = model.forward_prefix_revisions(signals, subjects, groups)
        for horizon in range(3):
            torch.testing.assert_close(native[horizon], fast[horizon], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
