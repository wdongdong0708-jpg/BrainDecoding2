import copy
import math

import numpy as np
import pytest
import torch

from braindecoding.models import (
    AdaptiveDirectionalBiasGate,
    BrainEmbeddingModel,
    EventTimeInteractionBias,
    PreTransformerTrajectoryResidual,
    RelationDiscriminationHead,
    GroupedTransformerEncoder,
    LocalRelativePositionBias,
    PhysicalTimeGapBias,
    ResidualSubjectLinear,
    SpatialChannelMerger,
    SubjectLinear,
    build_strided_conv_encoder,
    build_word_decoding_model,
)


def test_residual_subject_linear_has_exact_shared_zero_shot_fallback():
    layer = ResidualSubjectLinear(
        subject_count=2,
        input_channels=3,
        bias=False,
        initialization="identity",
    )
    features = torch.randn(3, 3, 5)
    indices = torch.tensor([0, 1, -1])
    initial = layer(features, indices)
    assert torch.equal(initial, features)

    with torch.no_grad():
        layer.subject_residuals[1, 0, 0] = 0.5
    changed = layer(features, indices)
    assert torch.equal(changed[0], initial[0])
    assert not torch.equal(changed[1], initial[1])
    assert torch.equal(changed[2], initial[2])


def test_residual_spatial_merger_uses_shared_path_for_minus_one_subject():
    positions = torch.tensor([[0.1, 0.2], [0.4, 0.7], [0.8, 0.3]])
    layer = SpatialChannelMerger(
        positions,
        output_channels=2,
        position_dimension=8,
        subject_count=2,
        per_subject=False,
        residual_per_subject=True,
        dropout_radius=0.0,
    ).eval()
    signals = torch.randn(3, 3, 7)
    indices = torch.tensor([0, 1, -1])
    initial = layer(signals, indices)
    shared = SpatialChannelMerger(
        positions,
        output_channels=2,
        position_dimension=8,
        subject_count=1,
        per_subject=False,
        dropout_radius=0.0,
    ).eval()
    with torch.no_grad():
        shared.coefficients.copy_(layer.coefficients)
    assert torch.allclose(initial, shared(signals), rtol=0.0, atol=0.0)

    with torch.no_grad():
        layer.subject_residuals[1, 0, 0] = 2.0
    changed = layer(signals, indices)
    assert torch.equal(changed[0], initial[0])
    assert not torch.equal(changed[1], initial[1])
    assert torch.equal(changed[2], initial[2])


def _small_dilated_config(readout="brain_embedding"):
    return {
        "embedding_dimension": 8,
        "use_transformer": False,
        "readout": readout,
        "candidate_evidence": {
            "evidence_dimension": 2,
            "attention_dimension": 2,
            "attention_gate_init": 0.0,
            "prefix_samples": [4, 8, 12, 16, 20],
            "prefix_loss_weights": [0.0625, 0.0625, 0.0625, 0.0625, 1.0],
            "state_normalizer_samples": 20,
        },
        "state_3": {
            "projection_dimension": 2,
            "trajectory_hidden_dimension": 4,
            "condition_hidden_dimension": 3,
            "concat_hidden_dimension": 3,
            "dropout": 0.0,
            "residual_gate_init": 0.0,
        },
        "state_5": {
            "projection_dimension": 2,
            "hidden_dimension": 4,
            "dropout": 0.0,
            "residual_gate_init": 0.0,
        },
        "conv": {
            "merger_channels": 4,
            "merger_position_dimension": 8,
            "merger_dropout": 0,
            "initial_linear": 6,
            "hidden": 4,
            "depth": 2,
            "kernel_size": 3,
            "dilation_growth": 2,
            "dilation_period": 2,
            "dropout_input": 0,
            "batch_norm": False,
            "gelu": True,
            "skip": True,
            "glu_every": 0,
            "temporal_attention_hidden": 4,
        },
    }


def test_dilated_encoder_can_physically_remove_subject_projection():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = _small_dilated_config()
    config["conv"].update(
        {
            "subject_projection_enabled": False,
            "subject_projection_per_subject": False,
            "subject_projection_residual_per_subject": False,
        }
    )
    model = build_word_decoding_model(6, positions, 2, config).eval()
    encoder = model.brain_encoder

    assert encoder.subject_projection is None
    assert not any("subject_projection" in key for key in model.state_dict())
    signals = torch.randn(3, 6, 20)
    subjects = torch.tensor([0, 1, 0])
    output = model(signals, subjects)
    assert output.shape == (3, 8)
    assert torch.isfinite(output).all()


def test_removed_subject_projection_requires_matching_channel_dimensions():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = _small_dilated_config()
    config["conv"].update(
        {
            "subject_projection_enabled": False,
            "subject_projection_per_subject": False,
            "branch_channels": 5,
        }
    )
    with pytest.raises(ValueError, match="initial_linear and branch_channels"):
        build_word_decoding_model(6, positions, 2, config)


def test_pre_transformer_trajectory_is_exact_identity_and_keeps_boundaries():
    module = PreTransformerTrajectoryResidual(
        8, {"hidden_dimension": 4}
    ).eval()
    features = torch.randn(7, 8)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    initial = module(features, groups)
    assert torch.equal(initial, features)
    initial.square().sum().backward()
    assert module.alpha.grad is not None
    assert torch.isfinite(module.alpha.grad)

    with torch.no_grad():
        module.alpha.fill_(0.5)
    changed = module(features, groups)
    assert torch.equal(changed[[0, 2, 3, 6]], features[[0, 2, 3, 6]])
    assert not torch.equal(changed[[1, 4, 5]], features[[1, 4, 5]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pre_transformer_trajectory_supports_cuda_amp():
    module = PreTransformerTrajectoryResidual(
        8, {"hidden_dimension": 4}
    ).cuda()
    features = torch.randn(7, 8, device="cuda", dtype=torch.float32)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = module(features, groups)
    assert output.dtype == features.dtype
    assert torch.equal(output, features)


def test_pre_transformer_trajectory_preserves_baseline_weights_rng_and_output():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    baseline_config = _small_dilated_config()
    baseline_config.update(
        {
            "use_transformer": True,
            "context_mode": "grouped",
            "transformer": {
                "depth": 1,
                "heads": 2,
                "attention_dropout": 0.1,
                "feedforward_dropout": 0.0,
            },
        }
    )
    trajectory_config = copy.deepcopy(baseline_config)
    trajectory_config["pre_transformer_trajectory"] = {
        "enabled": True,
        "hidden_dimension": 4,
    }

    torch.manual_seed(31)
    baseline = build_word_decoding_model(
        6, positions, 1, baseline_config
    ).eval()
    baseline_next_random = torch.rand(5)
    torch.manual_seed(31)
    trajectory = build_word_decoding_model(
        6, positions, 1, trajectory_config
    ).eval()
    trajectory_next_random = torch.rand(5)

    trajectory_state = trajectory.state_dict()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, trajectory_state[key])
    torch.testing.assert_close(baseline_next_random, trajectory_next_random)

    signals = torch.randn(7, 6, 20)
    subjects = torch.zeros(7, dtype=torch.long)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    assert torch.equal(
        baseline(signals, subjects, groups),
        trajectory(signals, subjects, groups),
    )


def test_grouped_transformer_causal_attention_cannot_read_future_states():
    torch.manual_seed(47)
    encoder = GroupedTransformerEncoder(
        8,
        {
            "depth": 2,
            "heads": 2,
            "attention_dropout": 0.0,
            "feedforward_dropout": 0.0,
            "causal_attention": True,
        },
    ).eval()
    features = torch.randn(5, 8)
    groups = torch.zeros(5, dtype=torch.long)
    changed = features.clone()
    changed[-1] = changed[-1] + 10.0

    original_output = encoder(features, groups)
    changed_output = encoder(changed, groups)

    torch.testing.assert_close(
        original_output[:-1], changed_output[:-1], rtol=0.0, atol=1e-6
    )
    assert not torch.equal(original_output[-1], changed_output[-1])


def test_causal_transformer_adds_no_parameters_or_rng_initialization_drift():
    base_config = {
        "depth": 1,
        "heads": 2,
        "attention_dropout": 0.0,
        "feedforward_dropout": 0.0,
    }
    causal_config = {**base_config, "causal_attention": True}

    torch.manual_seed(53)
    baseline = GroupedTransformerEncoder(8, base_config)
    baseline_next_random = torch.rand(5)
    torch.manual_seed(53)
    causal = GroupedTransformerEncoder(8, causal_config)
    causal_next_random = torch.rand(5)

    assert baseline.state_dict().keys() == causal.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, causal.state_dict()[key])
    torch.testing.assert_close(baseline_next_random, causal_next_random)


def test_prefix_revision_uses_physically_truncated_shared_transformer():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = _small_dilated_config()
    config.update(
        {
            "use_transformer": True,
            "context_mode": "grouped",
            "transformer": {
                "depth": 1,
                "heads": 2,
                "attention_dropout": 0.0,
                "feedforward_dropout": 0.0,
            },
            "prefix_revision": {
                "enabled": True,
                "horizons": [0, 1, 2],
                "validation_query_batch_size": 2,
            },
        }
    )
    model = build_word_decoding_model(6, positions, 1, config).eval()
    signals = torch.randn(7, 6, 20)
    subjects = torch.zeros(7, dtype=torch.long)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    revisions = model.forward_prefix_revisions(signals, subjects, groups)

    assert tuple(revisions) == (0, 1, 2)
    assert all(value.shape == (7, 8) for value in revisions.values())
    with torch.no_grad():
        brain = torch.nn.functional.normalize(
            model.brain_encoder(signals, subjects), dim=-1
        )
        first_prefix = model.context_transformer(
            brain[:1], torch.zeros(1, dtype=torch.long)
        )
        expected = torch.nn.functional.normalize(first_prefix, dim=-1)[0]
    torch.testing.assert_close(revisions[0][0], expected)

    estimates, target_rows, delays = model.forward_prefix_revision_training(
        signals,
        subjects,
        groups,
        cutoff_generator=torch.Generator(device="cpu").manual_seed(5),
    )
    assert estimates.shape == (len(target_rows), 8)
    assert len(target_rows) == len(delays)
    assert int((delays == 0).sum()) == 2
    assert set(delays.tolist()) <= {0, 1, 2}


def test_local_window_rolling_revision_uses_fixed_history_and_growing_future():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = _small_dilated_config()
    config.update(
        {
            "use_transformer": True,
            "context_mode": "grouped",
            "transformer": {
                "depth": 1,
                "heads": 2,
                "attention_dropout": 0.0,
                "feedforward_dropout": 0.0,
            },
            "local_window_revision": {
                "enabled": True,
                "history_radius": 2,
                "horizons": [0, 1, 2],
                "validation_query_batch_size": 7,
            },
        }
    )
    model = build_word_decoding_model(6, positions, 1, config).eval()
    signals = torch.randn(7, 6, 20)
    subjects = torch.zeros(7, dtype=torch.long)
    groups = torch.zeros(7, dtype=torch.long)

    assert model.prefix_revision_protocol == "local_window"
    assert model.prefix_revision_separate_horizon_losses is True
    assert model._revision_window_bounds(3, 0, 7) == (1, 4, 2)
    assert model._revision_window_bounds(3, 1, 7) == (1, 5, 2)
    assert model._revision_window_bounds(3, 2, 7) == (1, 6, 2)
    assert model._revision_window_bounds(0, 2, 7) == (0, 3, 0)

    revisions = model.forward_prefix_revisions(signals, subjects, groups)
    changed = signals.clone()
    changed[0] += 100.0
    changed[6] -= 100.0
    changed_revisions = model.forward_prefix_revisions(changed, subjects, groups)
    for delay in (0, 1, 2):
        torch.testing.assert_close(
            revisions[delay][3], changed_revisions[delay][3]
        )

    availability = model.revision_availability_masks(groups)
    assert availability[0].tolist() == [True] * 7
    assert availability[1].tolist() == [True] * 6 + [False]
    assert availability[2].tolist() == [True] * 5 + [False, False]

    estimates, target_rows, delays = model.forward_prefix_revision_training(
        signals, subjects, groups, cutoff_policy="all"
    )
    for delay in (0, 1, 2):
        delay_mask = delays == delay
        rows = target_rows[delay_mask]
        torch.testing.assert_close(
            rows, torch.nonzero(availability[delay], as_tuple=True)[0]
        )
        torch.testing.assert_close(estimates[delay_mask], revisions[delay][rows])


def test_local_window_and_full_prefix_revision_are_mutually_exclusive():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = _small_dilated_config()
    config.update(
        {
            "use_transformer": True,
            "context_mode": "grouped",
            "transformer": {"depth": 1, "heads": 2},
            "prefix_revision": {"enabled": True},
            "local_window_revision": {"enabled": True},
        }
    )
    with pytest.raises(ValueError, match="不能同时启用"):
        build_word_decoding_model(6, positions, 1, config)


def test_relation_discriminator_uses_strict_same_group_derangements():
    head = RelationDiscriminationHead(
        4, {"hidden_dimension": 3, "minimum_group_size": 3}
    )
    features = torch.arange(36, dtype=torch.float32).reshape(9, 4)
    groups = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    generator = torch.Generator(device="cpu").manual_seed(17)

    corrupted = head.derange_within_groups(features, groups, generator)

    assert torch.equal(corrupted[3:5], features[3:5])
    for indices in ([0, 1, 2], [5, 6, 7, 8]):
        assert not torch.any(
            torch.all(corrupted[indices] == features[indices], dim=1)
        )
        original_rows = {tuple(row.tolist()) for row in features[indices]}
        corrupted_rows = {tuple(row.tolist()) for row in corrupted[indices]}
        assert original_rows == corrupted_rows
    assert head.grouped_scores(features, groups).shape == (2,)


def test_relation_auxiliary_preserves_baseline_initialization_and_inference():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    baseline_config = _small_dilated_config()
    baseline_config.update(
        {
            "use_transformer": True,
            "context_mode": "grouped",
            "transformer": {
                "depth": 1,
                "heads": 2,
                "attention_dropout": 0.1,
                "feedforward_dropout": 0.0,
            },
        }
    )
    auxiliary_config = copy.deepcopy(baseline_config)
    auxiliary_config["relation_auxiliary"] = {
        "enabled": True,
        "hidden_dimension": 4,
        "minimum_group_size": 3,
    }

    torch.manual_seed(43)
    baseline = build_word_decoding_model(
        6, positions, 1, baseline_config
    ).eval()
    baseline_next_random = torch.rand(5)
    torch.manual_seed(43)
    auxiliary = build_word_decoding_model(
        6, positions, 1, auxiliary_config
    ).eval()
    auxiliary_next_random = torch.rand(5)

    auxiliary_state = auxiliary.state_dict()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, auxiliary_state[key])
    torch.testing.assert_close(baseline_next_random, auxiliary_next_random)

    signals = torch.randn(7, 6, 20)
    subjects = torch.zeros(7, dtype=torch.long)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    baseline_output = baseline(signals, subjects, groups)
    assert torch.equal(baseline_output, auxiliary(signals, subjects, groups))
    clean, positive, negative = auxiliary.forward_with_relation_auxiliary(
        signals,
        subjects,
        groups,
        corruption_generator=torch.Generator(device="cpu").manual_seed(9),
    )
    assert torch.equal(clean, baseline_output)
    assert positive.shape == negative.shape == (2,)


























def _small_context_config(readout="brain_embedding"):
    config = _small_dilated_config(readout)
    config["use_transformer"] = True
    config["context_mode"] = "grouped"
    config["transformer"] = {
        "depth": 1,
        "heads": 2,
        "attention_dropout": 0,
        "feedforward_dropout": 0,
    }
    config["candidate_evidence"]["residual_gate_init"] = 0.0
    return config


def test_local_relative_position_bias_is_directional_and_strictly_local():
    bias = LocalRelativePositionBias(max_distance=2, mode="directed")
    with torch.no_grad():
        bias.past_bias.copy_(torch.tensor([1.0, 2.0]))
        bias.future_bias.copy_(torch.tensor([3.0, 4.0]))

    matrix = bias(6)[0, 0]

    assert matrix[3, 2].item() == 1.0
    assert matrix[3, 1].item() == 2.0
    assert matrix[2, 3].item() == 3.0
    assert matrix[1, 3].item() == 4.0
    assert matrix[0, 0].item() == 0.0
    assert matrix[0, 3].item() == 0.0
    assert matrix[3, 0].item() == 0.0


def test_undirected_local_relative_position_bias_is_symmetric():
    bias = LocalRelativePositionBias(max_distance=2, mode="undirected")
    with torch.no_grad():
        bias.past_bias.copy_(torch.tensor([0.5, -0.25]))

    matrix = bias(5)[0, 0]

    torch.testing.assert_close(matrix, matrix.T)
    assert matrix[2, 1].item() == matrix[1, 2].item() == 0.5
    assert matrix[3, 1].item() == matrix[1, 3].item() == -0.25
    assert sum(parameter.numel() for parameter in bias.parameters()) == 2


def test_physical_time_gap_bias_uses_directed_boundary_exact_buckets():
    bias = PhysicalTimeGapBias(
        bin_edges_ms=[200, 400, 600, 1000], mode="directed"
    )
    with torch.no_grad():
        bias.past_bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        bias.future_bias.copy_(
            torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        )

    times = torch.tensor(
        [[0.0, 0.199, 0.2, 0.4, 0.6, 1.0]], dtype=torch.float64
    )
    matrix = bias(times, valid_mask=torch.ones_like(times, dtype=torch.bool))[0, 0]

    assert matrix[0, 1].item() == 10.0
    assert matrix[0, 2].item() == 20.0
    assert matrix[0, 3].item() == 30.0
    assert matrix[0, 4].item() == 40.0
    assert matrix[0, 5].item() == 50.0
    assert matrix[5, 0].item() == 5.0
    assert matrix.diag().count_nonzero().item() == 0
    assert sum(parameter.numel() for parameter in bias.parameters()) == 10


def test_event_time_interaction_bias_is_sparse_in_event_distance_and_time():
    bias = EventTimeInteractionBias(
        max_event_distance=2,
        bin_edges_ms=[200, 400, 1000],
    )
    with torch.no_grad():
        bias.correction.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [10.0, 20.0, 30.0, 40.0],
                    [50.0, 60.0, 70.0, 80.0],
                ]
            )
        )
    times = torch.tensor(
        [[0.0, 0.15, 0.35, 0.9, 2.0]], dtype=torch.float64
    )
    matrix = bias(times, valid_mask=torch.ones_like(times, dtype=torch.bool))[0, 0]

    assert matrix[0, 1].item() == 10.0
    assert matrix[0, 2].item() == 60.0
    assert matrix[1, 3].item() == 70.0
    assert matrix[3, 4].item() == 40.0
    assert matrix[1, 0].item() == 1.0
    assert matrix[2, 0].item() == 6.0
    assert matrix[3, 1].item() == 7.0
    assert matrix[4, 3].item() == 4.0
    assert matrix[0, 3].item() == 0.0
    assert matrix.diag().count_nonzero().item() == 0
    assert sum(parameter.numel() for parameter in bias.parameters()) == 16


def test_zero_event_time_interaction_is_exact_directed_event_bias_baseline():
    directed_config = {
        "depth": 2,
        "heads": 2,
        "attention_dropout": 0.0,
        "feedforward_dropout": 0.0,
        "local_relative_position_bias": {
            "enabled": True,
            "max_distance": 4,
            "mode": "directed",
        },
    }
    joint_config = {
        **directed_config,
        "event_time_interaction_bias": {
            "enabled": True,
            "max_event_distance": 2,
            "bin_edges_ms": [200, 400, 1000],
        },
    }
    torch.manual_seed(53)
    directed = GroupedTransformerEncoder(8, directed_config).eval()
    directed_next_random = torch.rand(4)
    torch.manual_seed(53)
    joint = GroupedTransformerEncoder(8, joint_config).eval()
    joint_next_random = torch.rand(4)

    for key, value in directed.encoder.state_dict().items():
        torch.testing.assert_close(value, joint.encoder.state_dict()[key])
    for key, value in directed.local_relative_position_bias.state_dict().items():
        torch.testing.assert_close(
            value, joint.local_relative_position_bias.state_dict()[key]
        )
    torch.testing.assert_close(directed_next_random, joint_next_random)
    assert joint.event_time_interaction_bias.correction.count_nonzero().item() == 0

    features = torch.randn(7, 8)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    times = torch.tensor(
        [0.0, 0.1, 0.4, 10.0, 10.2, 10.6, 11.7], dtype=torch.float64
    )
    torch.testing.assert_close(
        joint(features, groups, event_times=times),
        directed(features, groups),
        rtol=1e-6,
        atol=1e-7,
    )


def test_event_and_physical_time_bias_are_additive():
    encoder = GroupedTransformerEncoder(
        2,
        {
            "depth": 1,
            "heads": 1,
            "attention_dropout": 0.0,
            "feedforward_dropout": 0.0,
            "local_relative_position_bias": {
                "enabled": True,
                "max_distance": 2,
                "mode": "directed",
            },
            "physical_time_gap_bias": {
                "enabled": True,
                "bin_edges_ms": [200, 400, 600, 1000],
                "mode": "directed",
            },
        },
    ).eval()

    class CaptureIdentity(torch.nn.Module):
        def forward(self, features, **kwargs):
            self.attention_bias = kwargs["attn_bias"].detach().clone()
            return features

    capture = CaptureIdentity()
    encoder.encoder = capture
    with torch.no_grad():
        encoder.local_relative_position_bias.past_bias.copy_(
            torch.tensor([1.0, 2.0])
        )
        encoder.local_relative_position_bias.future_bias.copy_(
            torch.tensor([3.0, 4.0])
        )
        encoder.physical_time_gap_bias.past_bias.copy_(
            torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        )
        encoder.physical_time_gap_bias.future_bias.copy_(
            torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])
        )

    features = torch.zeros(3, 2)
    groups = torch.zeros(3, dtype=torch.long)
    times = torch.tensor([0.0, 0.1, 0.35], dtype=torch.float64)
    encoder(features, groups, event_times=times)
    matrix = capture.attention_bias[0, 0]

    assert matrix[1, 0].item() == 11.0
    assert matrix[1, 2].item() == 203.0
    assert matrix[0, 2].item() == 204.0
    assert matrix[0, 0].item() == 0.0


def test_zero_physical_time_bias_is_exact_directed_event_bias_baseline():
    fixed_config = {
        "depth": 1,
        "heads": 2,
        "attention_dropout": 0.0,
        "feedforward_dropout": 0.0,
        "local_relative_position_bias": {
            "enabled": True,
            "max_distance": 2,
            "mode": "directed",
        },
    }
    event_time_config = {
        **fixed_config,
        "physical_time_gap_bias": {
            "enabled": True,
            "bin_edges_ms": [200, 400, 600, 1000],
            "mode": "directed",
        },
    }
    torch.manual_seed(47)
    fixed = GroupedTransformerEncoder(8, fixed_config).eval()
    fixed_next_random = torch.rand(4)
    torch.manual_seed(47)
    event_time = GroupedTransformerEncoder(8, event_time_config).eval()
    event_time_next_random = torch.rand(4)

    for key, value in fixed.encoder.state_dict().items():
        torch.testing.assert_close(value, event_time.encoder.state_dict()[key])
    for key, value in fixed.local_relative_position_bias.state_dict().items():
        torch.testing.assert_close(
            value, event_time.local_relative_position_bias.state_dict()[key]
        )
    torch.testing.assert_close(fixed_next_random, event_time_next_random)
    assert sum(
        parameter.numel()
        for parameter in event_time.physical_time_gap_bias.parameters()
    ) == 10

    features = torch.randn(7, 8)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    times = torch.tensor(
        [0.0, 0.1, 0.4, 10.0, 10.2, 10.6, 11.7], dtype=torch.float64
    )
    torch.testing.assert_close(
        event_time(features, groups, event_times=times),
        fixed(features, groups),
        rtol=1e-6,
        atol=1e-7,
    )


def test_physical_time_bias_requires_strictly_increasing_group_times():
    encoder = GroupedTransformerEncoder(
        8,
        {
            "depth": 1,
            "heads": 2,
            "physical_time_gap_bias": {
                "enabled": True,
                "bin_edges_ms": [200, 400, 600, 1000],
                "mode": "directed",
            },
        },
    )
    features = torch.randn(3, 8)
    groups = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="严格递增"):
        encoder(
            features,
            groups,
            event_times=torch.tensor([0.0, 0.2, 0.2], dtype=torch.float64),
        )


def test_adaptive_directional_gate_starts_at_identity_without_rng_draws():
    torch.manual_seed(41)
    expected_next_random = torch.rand(4)
    torch.manual_seed(41)
    gate = AdaptiveDirectionalBiasGate(8)
    actual_next_random = torch.rand(4)

    features = torch.randn(3, 5, 8)
    torch.testing.assert_close(gate(features), torch.ones(3, 5, 2))
    torch.testing.assert_close(actual_next_random, expected_next_random)
    assert sum(parameter.numel() for parameter in gate.parameters()) == 18


def test_adaptive_directional_bias_scales_past_and_future_per_target():
    encoder = GroupedTransformerEncoder(
        2,
        {
            "depth": 1,
            "heads": 1,
            "attention_dropout": 0.0,
            "feedforward_dropout": 0.0,
            "local_relative_position_bias": {
                "enabled": True,
                "max_distance": 2,
                "mode": "directed",
            },
            "adaptive_directional_position_bias": {"enabled": True},
        },
    ).eval()

    class CaptureIdentity(torch.nn.Module):
        def forward(self, features, **kwargs):
            self.attention_bias = kwargs["attn_bias"].detach().clone()
            return features

    capture = CaptureIdentity()
    encoder.encoder = capture
    with torch.no_grad():
        encoder.local_relative_position_bias.past_bias.copy_(
            torch.tensor([1.0, 2.0])
        )
        encoder.local_relative_position_bias.future_bias.copy_(
            torch.tensor([3.0, 4.0])
        )
        encoder.adaptive_directional_bias_gate.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        )

    features = torch.tensor(
        [[0.0, 0.0], [0.5, -0.5], [-0.5, 0.5], [0.0, 0.0]]
    )
    encoder(features, torch.zeros(4, dtype=torch.long))
    bias = capture.attention_bias[0, 0]
    target_one_gates = 1.0 + torch.tanh(features[1])
    target_two_gates = 1.0 + torch.tanh(features[2])

    torch.testing.assert_close(bias[1, 0], target_one_gates[0] * 1.0)
    torch.testing.assert_close(bias[1, 2], target_one_gates[1] * 3.0)
    torch.testing.assert_close(bias[2, 0], target_two_gates[0] * 2.0)
    torch.testing.assert_close(bias[2, 3], target_two_gates[1] * 3.0)
    assert bias[0, 3].item() == 0.0


def test_adaptive_directional_bias_starts_as_exact_fixed_directed_bias():
    fixed_config = {
        "depth": 2,
        "heads": 2,
        "attention_dropout": 0.0,
        "feedforward_dropout": 0.0,
        "local_relative_position_bias": {
            "enabled": True,
            "max_distance": 2,
            "mode": "directed",
        },
    }
    adaptive_config = {
        **fixed_config,
        "adaptive_directional_position_bias": {"enabled": True},
    }
    torch.manual_seed(43)
    fixed = GroupedTransformerEncoder(8, fixed_config).eval()
    fixed_next_random = torch.rand(4)
    torch.manual_seed(43)
    adaptive = GroupedTransformerEncoder(8, adaptive_config).eval()
    adaptive_next_random = torch.rand(4)

    for key, value in fixed.encoder.state_dict().items():
        torch.testing.assert_close(value, adaptive.encoder.state_dict()[key])
    for key, value in fixed.local_relative_position_bias.state_dict().items():
        torch.testing.assert_close(
            value, adaptive.local_relative_position_bias.state_dict()[key]
        )
    torch.testing.assert_close(fixed_next_random, adaptive_next_random)
    assert sum(p.numel() for p in adaptive.parameters()) - sum(
        p.numel() for p in fixed.parameters()
    ) == 18

    features = torch.randn(7, 8)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    torch.testing.assert_close(
        adaptive(features, groups), fixed(features, groups),
        rtol=1e-6, atol=1e-7,
    )


def test_adaptive_directional_bias_requires_directed_local_bias():
    with pytest.raises(ValueError, match="要求启用 directed"):
        GroupedTransformerEncoder(
            8,
            {
                "depth": 1,
                "heads": 2,
                "adaptive_directional_position_bias": {"enabled": True},
            },
        )


def test_zero_local_relative_bias_is_exact_transformer_baseline():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    baseline_config = _small_context_config()
    biased_config = _small_context_config()
    biased_config["transformer"]["local_relative_position_bias"] = {
        "enabled": True,
        "max_distance": 2,
        "mode": "directed",
    }
    torch.manual_seed(43)
    baseline = build_word_decoding_model(
        6, positions, 1, baseline_config
    ).eval()
    baseline_next_random = torch.rand(4)
    torch.manual_seed(43)
    biased = build_word_decoding_model(6, positions, 1, biased_config).eval()
    biased_next_random = torch.rand(4)

    for key, value in baseline.brain_encoder.state_dict().items():
        torch.testing.assert_close(value, biased.brain_encoder.state_dict()[key])
    for key, value in baseline.context_transformer.encoder.state_dict().items():
        torch.testing.assert_close(
            value, biased.context_transformer.encoder.state_dict()[key]
        )
    torch.testing.assert_close(baseline_next_random, biased_next_random)
    assert sum(
        parameter.numel()
        for parameter in biased.context_transformer.local_relative_position_bias.parameters()
    ) == 4

    signals = torch.randn(7, 6, 20)
    subjects = torch.zeros(7, dtype=torch.long)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    torch.testing.assert_close(
        biased(signals, subjects, groups),
        baseline(signals, subjects, groups),
        rtol=1e-6,
        atol=1e-7,
    )


def test_local_global_attention_has_head_specific_mask_and_bias():
    encoder = GroupedTransformerEncoder(
        8,
        {
            "depth": 1,
            "heads": 4,
            "attention_dropout": 0.0,
            "feedforward_dropout": 0.0,
            "rotary_position_embedding": True,
            "local_relative_position_bias": {
                "enabled": True,
                "max_distance": 2,
                "mode": "directed",
            },
            "local_global_attention": {
                "enabled": True,
                "local_heads": 2,
                "radius": 2,
            },
        },
    ).eval()
    with torch.no_grad():
        encoder.local_relative_position_bias.past_bias.copy_(
            torch.tensor([1.0, 2.0])
        )
        encoder.local_relative_position_bias.future_bias.copy_(
            torch.tensor([3.0, 4.0])
        )

    mask = encoder.local_global_attention_mask(6, torch.device("cpu"))
    bias = encoder.local_relative_position_bias(
        6, heads=4, active_heads=2
    )

    assert mask.shape == (4, 6, 6)
    assert mask[:2, 3, 0].tolist() == [False, False]
    assert mask[:2, 3, 1].tolist() == [True, True]
    assert mask[2:, 3, 0].tolist() == [True, True]
    assert torch.count_nonzero(bias[:, 2:]).item() == 0
    assert bias[0, 0, 3, 2].item() == 1.0
    assert bias[0, 1, 2, 3].item() == 3.0

    features = torch.randn(7, 8)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    output = encoder(features, groups)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()


def test_local_global_attention_matches_directed_bias_parameter_budget():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    directed_config = _small_context_config()
    directed_config["transformer"]["local_relative_position_bias"] = {
        "enabled": True,
        "max_distance": 2,
        "mode": "directed",
    }
    mixed_config = _small_context_config()
    mixed_config["transformer"]["local_relative_position_bias"] = {
        "enabled": True,
        "max_distance": 2,
        "mode": "directed",
    }
    mixed_config["transformer"]["local_global_attention"] = {
        "enabled": True,
        "local_heads": 1,
        "radius": 2,
    }

    torch.manual_seed(47)
    directed = build_word_decoding_model(
        6, positions, 1, directed_config
    )
    torch.manual_seed(47)
    mixed = build_word_decoding_model(6, positions, 1, mixed_config)

    assert sum(p.numel() for p in directed.parameters()) == sum(
        p.numel() for p in mixed.parameters()
    )
    for key, value in directed.context_transformer.encoder.state_dict().items():
        torch.testing.assert_close(
            value, mixed.context_transformer.encoder.state_dict()[key]
        )


def test_hierarchical_attention_uses_exact_layer_schedule_without_parameters():
    baseline_config = {
        "depth": 4,
        "heads": 2,
        "attention_dropout": 0.0,
        "feedforward_dropout": 0.0,
        "rotary_position_embedding": True,
    }
    hierarchy_config = {
        **baseline_config,
        "hierarchical_attention": {
            "enabled": True,
            "stages": [
                {"layers": 1, "radius": 1},
                {"layers": 1, "radius": 2},
                {"layers": 1, "radius": 3},
                {"layers": 1, "radius": "global"},
            ],
        },
    }
    torch.manual_seed(53)
    baseline = GroupedTransformerEncoder(8, baseline_config).eval()
    baseline_next_random = torch.rand(4)
    torch.manual_seed(53)
    hierarchy = GroupedTransformerEncoder(8, hierarchy_config).eval()
    hierarchy_next_random = torch.rand(4)

    assert hierarchy.hierarchical_attention_radii == (1, 2, 3, None)
    assert baseline.state_dict().keys() == hierarchy.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, hierarchy.state_dict()[key])
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in hierarchy.parameters()
    )
    torch.testing.assert_close(baseline_next_random, hierarchy_next_random)

    attention_layers = [
        layer[1]
        for layer_type, layer in zip(
            hierarchy.encoder.layer_types, hierarchy.encoder.layers
        )
        if layer_type == "a"
    ]
    captured_masks = []

    def capture_mask(module, args, kwargs):
        captured_masks.append(kwargs.get("attn_mask"))

    handles = [
        layer.register_forward_pre_hook(capture_mask, with_kwargs=True)
        for layer in attention_layers
    ]
    features = torch.randn(6, 8)
    groups = torch.zeros(6, dtype=torch.long)
    output = hierarchy(features, groups)
    for handle in handles:
        handle.remove()

    assert output.shape == features.shape
    assert len(captured_masks) == 4
    assert captured_masks[0][2, 0].item() is False
    assert captured_masks[0][2, 1].item() is True
    assert captured_masks[1][3, 1].item() is True
    assert captured_masks[1][3, 0].item() is False
    assert captured_masks[2][4, 1].item() is True
    assert captured_masks[2][4, 0].item() is False
    assert captured_masks[3] is None


def test_hierarchical_attention_rejects_incomplete_schedule():
    with pytest.raises(ValueError, match="层数之和"):
        GroupedTransformerEncoder(
            8,
            {
                "depth": 4,
                "heads": 2,
                "hierarchical_attention": {
                    "enabled": True,
                    "stages": [{"layers": 3, "radius": 2}],
                },
            },
        )


def test_shared_transformer_two_passes_equal_manual_reapplication_without_parameters():
    baseline_config = {
        "depth": 2,
        "heads": 2,
        "attention_dropout": 0.0,
        "feedforward_dropout": 0.0,
        "rotary_position_embedding": True,
    }
    two_pass_config = {**baseline_config, "shared_passes": 2}

    torch.manual_seed(59)
    baseline = GroupedTransformerEncoder(8, baseline_config).eval()
    baseline_next_random = torch.rand(4)
    torch.manual_seed(59)
    two_pass = GroupedTransformerEncoder(8, two_pass_config).eval()
    two_pass_next_random = torch.rand(4)

    assert baseline.shared_passes == 1
    assert two_pass.shared_passes == 2
    assert baseline.state_dict().keys() == two_pass.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, two_pass.state_dict()[key])
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in two_pass.parameters()
    )
    torch.testing.assert_close(baseline_next_random, two_pass_next_random)

    features = torch.randn(7, 8)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    first_pass = baseline(features, groups)
    manual_second_pass = baseline(first_pass, groups)
    actual_second_pass = two_pass(features, groups)

    torch.testing.assert_close(
        actual_second_pass, manual_second_pass, rtol=1e-6, atol=1e-7
    )
    assert not torch.allclose(actual_second_pass, first_pass)


def test_shared_transformer_rejects_nonpositive_pass_count():
    with pytest.raises(ValueError, match="执行次数必须为正数"):
        GroupedTransformerEncoder(
            8,
            {
                "depth": 1,
                "heads": 2,
                "shared_passes": 0,
            },
        )


def test_chunk_summary_tokens_are_inserted_after_each_complete_local_chunk():
    encoder = GroupedTransformerEncoder(
        2,
        {
            "depth": 1,
            "heads": 1,
            "chunk_summary_tokens": {
                "enabled": True,
                "chunk_size": 4,
                "minimum_chunk_size": 3,
            },
        },
    ).eval()

    class CaptureIdentity(torch.nn.Module):
        def forward(self, features, **kwargs):
            self.features = features.detach().clone()
            self.mask = kwargs["mask"].detach().clone()
            return features

    capture = CaptureIdentity()
    encoder.encoder = capture
    features = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    groups = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])

    output = encoder(features, groups)

    torch.testing.assert_close(output, features)
    torch.testing.assert_close(capture.features[0, :4], features[:4])
    torch.testing.assert_close(
        capture.features[0, 4], features[:4].mean(dim=0)
    )
    torch.testing.assert_close(capture.features[0, 5], features[4])
    torch.testing.assert_close(capture.features[1, :3], features[5:])
    torch.testing.assert_close(
        capture.features[1, 3], features[5:].mean(dim=0)
    )
    assert capture.mask.tolist() == [
        [True, True, True, True, True, True],
        [True, True, True, True, False, False],
    ]


def test_chunk_summary_tokens_add_no_parameters_or_rng_draws():
    baseline_config = {
        "depth": 2,
        "heads": 2,
        "attention_dropout": 0.0,
        "feedforward_dropout": 0.0,
    }
    chunk_config = {
        **baseline_config,
        "chunk_summary_tokens": {
            "enabled": True,
            "chunk_size": 4,
            "minimum_chunk_size": 3,
        },
    }

    torch.manual_seed(61)
    baseline = GroupedTransformerEncoder(8, baseline_config).eval()
    baseline_next_random = torch.rand(4)
    torch.manual_seed(61)
    chunked = GroupedTransformerEncoder(8, chunk_config).eval()
    chunked_next_random = torch.rand(4)

    assert baseline.state_dict().keys() == chunked.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, chunked.state_dict()[key])
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in chunked.parameters()
    )
    torch.testing.assert_close(baseline_next_random, chunked_next_random)


@pytest.mark.parametrize(
    "chunk_size,minimum_chunk_size",
    [(0, 0), (4, 0), (4, 5)],
)
def test_chunk_summary_rejects_invalid_sizes(
    chunk_size, minimum_chunk_size
):
    with pytest.raises(ValueError, match="组块摘要"):
        GroupedTransformerEncoder(
            8,
            {
                "depth": 1,
                "heads": 2,
                "chunk_summary_tokens": {
                    "enabled": True,
                    "chunk_size": chunk_size,
                    "minimum_chunk_size": minimum_chunk_size,
                },
            },
        )


















def test_singleton_context_ignores_external_group_assignments():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = {
        "embedding_dimension": 8,
        "use_transformer": True,
        "context_mode": "singleton",
        "conv": {
            "merger_channels": 4,
            "merger_position_dimension": 8,
            "merger_dropout": 0,
            "initial_linear": 6,
            "hidden": 4,
            "depth": 2,
            "kernel_size": 3,
            "dilation_growth": 2,
            "dilation_period": 2,
            "dropout_input": 0,
            "batch_norm": False,
            "gelu": True,
            "skip": True,
            "glu_every": 0,
            "temporal_attention_hidden": 4,
        },
        "transformer": {
            "depth": 1,
            "heads": 2,
            "attention_dropout": 0,
            "feedforward_dropout": 0,
        },
    }
    model = BrainEmbeddingModel(positions, subject_count=1, config=config).eval()
    signals = torch.randn(3, 6, 20)
    subjects = torch.zeros(3, dtype=torch.long)
    grouped = model(signals, subjects, torch.zeros(3, dtype=torch.long))
    separated = model(signals, subjects, torch.arange(3))
    assert torch.allclose(grouped, separated, atol=1e-6)


def test_brain_embedding_model_migrates_legacy_subject_projection_orientation():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = {
        "embedding_dimension": 8,
        "use_transformer": False,
        "conv": {
            "merger_channels": 4,
            "merger_position_dimension": 8,
            "merger_dropout": 0,
            "initial_linear": 6,
            "hidden": 4,
            "depth": 2,
            "kernel_size": 3,
            "dilation_growth": 2,
            "dilation_period": 2,
            "dropout_input": 0,
            "batch_norm": False,
            "gelu": True,
            "skip": True,
            "glu_every": 0,
            "temporal_attention_hidden": 4,
        },
    }
    source = BrainEmbeddingModel(positions, subject_count=1, config=config)
    current_state = source.state_dict()
    legacy_state = {key: value.detach().clone() for key, value in current_state.items()}
    coefficients = legacy_state.pop("brain_encoder.channel_merger.coefficients")
    legacy_state.pop("brain_encoder.channel_merger.basis")
    legacy_state["brain_encoder.channel_merger.heads"] = coefficients
    legacy_state["brain_encoder.channel_merger.channel_positions"] = (
        source.brain_encoder.channel_merger.channel_positions.detach().clone()
    )
    subject_key = "brain_encoder.subject_projection.weights"
    legacy_state[subject_key] = legacy_state[subject_key].transpose(-1, -2).contiguous()
    restored = BrainEmbeddingModel(positions, subject_count=1, config=config)

    restored.load_state_dict(legacy_state, strict=True)

    assert torch.equal(restored.state_dict()[subject_key], current_state[subject_key])


def test_subject_linear_identity_initialization():
    layer = SubjectLinear(
        subject_count=2,
        input_channels=3,
        bias=True,
        initialization="identity",
    )
    features = torch.randn(2, 3, 7)
    output = layer(features, torch.tensor([0, 1]))
    assert torch.equal(output, features)


def test_spatial_channel_merger_supports_subject_specific_2d_positions():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    merger = SpatialChannelMerger(
        positions,
        output_channels=4,
        position_dimension=8,
        subject_count=2,
        per_subject=True,
        dropout_radius=0,
        frequency_start=0,
        fourier_margin=0.2,
        scale_fourier_coordinates=True,
    )
    output = merger(torch.randn(3, 6, 20), torch.tensor([0, 1, 0]))
    assert output.shape == (3, 4, 20)
    assert torch.isfinite(output).all()


def test_subject_independent_encoder_is_invariant_to_subject_indices():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = {
        "embedding_dimension": 8,
        "use_transformer": False,
        "conv": {
            "merger_channels": 4,
            "merger_position_dimension": 8,
            "merger_per_subject": False,
            "merger_dropout": 0,
            "subject_projection_per_subject": False,
            "initial_linear": 6,
            "hidden": 4,
            "depth": 2,
            "kernel_size": 3,
            "dilation_growth": 2,
            "dilation_period": 2,
            "dropout_input": 0,
            "convolution_dropout": 0,
            "batch_norm": False,
            "gelu": True,
            "skip": True,
            "glu_every": 0,
            "temporal_attention_hidden": 4,
        },
    }
    model = BrainEmbeddingModel(positions, subject_count=2, config=config).eval()
    signal = torch.randn(1, 6, 20).repeat(2, 1, 1)
    output = model(signal, torch.tensor([0, 1]))

    assert model.brain_encoder.channel_merger.coefficients.ndim == 2
    assert model.brain_encoder.subject_projection.weights.shape[0] == 1
    assert torch.equal(output[0], output[1])


def test_spatial_channel_merger_loads_legacy_state_keys_strictly():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    source = SpatialChannelMerger(
        positions,
        output_channels=4,
        position_dimension=8,
        subject_count=1,
        per_subject=True,
        frequency_start=0,
        fourier_margin=0.2,
        scale_fourier_coordinates=True,
    )
    legacy_state = {
        "heads": source.coefficients.detach().clone(),
        "channel_positions": source.channel_positions.detach().clone(),
    }
    restored = SpatialChannelMerger(
        positions,
        output_channels=4,
        position_dimension=8,
        subject_count=1,
        per_subject=True,
        frequency_start=0,
        fourier_margin=0.2,
        scale_fourier_coordinates=True,
    )

    restored.load_state_dict(legacy_state, strict=True)

    frequencies_y = torch.arange(2, dtype=restored.channel_positions.dtype)
    frequencies_x = frequencies_y[:, None]
    shifted = restored.channel_positions + 0.2
    phase = (
        shifted[..., None, None, 0] * (2.0 * math.pi * frequencies_x / 1.4)
        + shifted[..., None, None, 1] * (2.0 * math.pi * frequencies_y / 1.4)
    ).reshape(len(positions), -1)
    expected_basis = torch.cat([phase.cos(), phase.sin()], dim=-1)
    assert torch.equal(restored.coefficients, source.coefficients)
    assert torch.equal(restored.basis, expected_basis)


def test_strided_encoder_supports_3d_scalp_positions():
    positions = np.asarray(
        [
            [-0.5, 0.2, 0.8],
            [0.0, 0.7, 0.7],
            [0.5, 0.2, 0.8],
            [-0.4, -0.4, 0.8],
            [0.4, -0.4, 0.8],
            [0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )
    model = build_strided_conv_encoder(
        channel_count=6,
        channel_positions=positions,
        subject_count=2,
        config={
            "embedding_dimension": 8,
            "virtual_channels": 4,
            "fourier_harmonics": 2,
            "dropout": 0,
        },
    )
    output = model(torch.randn(2, 6, 64), torch.tensor([0, 1]))
    assert output.shape == (2, 8)
    assert torch.allclose(
        torch.linalg.vector_norm(output, dim=1), torch.ones(2), atol=1e-5
    )


def test_stage23_full_history_prefix_revision_window_and_shared_cutoff():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = _small_dilated_config()
    config.update(
        {
            "use_transformer": True,
            "context_mode": "grouped",
            "transformer": {
                "depth": 2,
                "heads": 2,
                "attention_dropout": 0.0,
                "feedforward_dropout": 0.0,
            },
            "local_window_revision": {
                "enabled": True,
                "history_radius": None,
                "horizons": [0, 1, 2],
                "validation_query_batch_size": 16,
            },
        }
    )
    model = build_word_decoding_model(6, positions, 1, config).eval()
    assert model.prefix_revision_enabled is True
    assert model.prefix_revision_protocol == "full_prefix"
    assert model.prefix_revision_separate_horizon_losses is True
    assert model.prefix_revision_history_radius is None
    assert model._revision_window_bounds(5, 2, 8) == (0, 8, 5)

    signals = torch.randn(7, 6, 20)
    subjects = torch.zeros(7, dtype=torch.long)
    groups = torch.tensor([0, 0, 0, 0, 0, 1, 1], dtype=torch.long)

    rev_preds, rev_rows, rev_delays = model.forward_prefix_revision_training(
        signals, subjects, groups, cutoff_policy="all"
    )
    assert set(rev_delays.tolist()) == {0, 1, 2}
    assert int((rev_delays == 0).sum().item()) == 7
    assert int((rev_delays == 1).sum().item()) == 5
    assert int((rev_delays == 2).sum().item()) == 3

    eval_out = model.forward_prefix_revisions(signals, subjects, groups)
    assert set(eval_out) == {0, 1, 2}
    # For group 0 (len 5), word i=2 with delay r=2 uses prefix [0..4], which is the same
    # physical prefix window as word i=3 with delay r=1 and word i=4 with delay r=0.
    word_features = torch.nn.functional.normalize(
        model.brain_encoder(signals, subjects), dim=-1
    )
    prefix_0_to_4 = model.context_transformer(
        word_features[:5], torch.zeros(5, dtype=torch.long), causal_override=False
    )
    expected_i2_k2 = torch.nn.functional.normalize(prefix_0_to_4[2:3], dim=-1)[0]
    expected_i3_k1 = torch.nn.functional.normalize(prefix_0_to_4[3:4], dim=-1)[0]
    expected_i4_k0 = torch.nn.functional.normalize(prefix_0_to_4[4:5], dim=-1)[0]
    torch.testing.assert_close(
        eval_out[2][2], expected_i2_k2, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        eval_out[1][3], expected_i3_k1, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        eval_out[0][4], expected_i4_k0, rtol=1e-5, atol=1e-5
    )
    mask_i2_k2 = (rev_rows == 2) & (rev_delays == 2)
    assert int(mask_i2_k2.sum().item()) == 1
    torch.testing.assert_close(
        rev_preds[mask_i2_k2][0], expected_i2_k2, rtol=1e-5, atol=1e-5
    )

