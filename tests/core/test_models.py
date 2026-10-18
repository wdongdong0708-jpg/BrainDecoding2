import copy
import math

import numpy as np
import pytest
import torch

from braindecoding.models import (
    AdaptiveDirectionalBiasGate,
    BrainEmbeddingModel,
    CandidateConditionedAttentionEvidenceReadout,
    CandidateConditionedEvidenceReadout,
    EventTimeInteractionBias,
    PreTransformerTrajectoryResidual,
    RelationDiscriminationHead,
    FiveStateTrajectoryReadout,
    GroupedTransformerEncoder,
    LocalRelativePositionBias,
    PhysicalTimeGapBias,
    ResidualSubjectLinear,
    SpatialChannelMerger,
    SubjectLinear,
    ThreeStateTrajectoryReadout,
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


def test_three_state_zero_gate_preserves_cnn_and_boundaries_exactly():
    torch.manual_seed(7)
    readout = ThreeStateTrajectoryReadout(
        8,
        {
            "projection_dimension": 2,
            "trajectory_hidden_dimension": 4,
            "dropout": 0.0,
            "residual_gate_init": 0.0,
        },
        mode="trajectory",
    ).eval()
    features = torch.nn.functional.normalize(torch.randn(7, 8), dim=-1)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    output, internal = readout(
        features, groups, return_internal_mask=True
    )

    assert internal.tolist() == [False, True, False, False, True, True, False]
    torch.testing.assert_close(output, features, rtol=1e-6, atol=1e-7)
    assert torch.equal(output[~internal], features[~internal])
    assert float(readout.residual_gate.detach()) == 0.0


def test_three_state_trajectory_keeps_incoming_and_outgoing_directional():
    torch.manual_seed(11)
    readout = ThreeStateTrajectoryReadout(
        8,
        {
            "projection_dimension": 2,
            "trajectory_hidden_dimension": 4,
            "dropout": 0.0,
            "residual_gate_init": 0.0,
        },
        mode="trajectory",
    )
    previous = torch.randn(3, 8)
    center = torch.randn(3, 8)
    following = torch.randn(3, 8)

    forward = readout.trajectory_coordinates(previous, center, following)
    mirrored = readout.trajectory_coordinates(following, center, previous)
    forward_parts = forward.chunk(4, dim=-1)
    mirrored_parts = mirrored.chunk(4, dim=-1)

    torch.testing.assert_close(forward_parts[0], mirrored_parts[0])
    torch.testing.assert_close(forward_parts[1], -mirrored_parts[2])
    torch.testing.assert_close(forward_parts[2], -mirrored_parts[1])
    torch.testing.assert_close(forward_parts[3], mirrored_parts[3])
    assert not torch.equal(forward, mirrored)


def test_three_state_ablations_zero_slots_without_changing_parameter_budget():
    base = {
        "projection_dimension": 2,
        "trajectory_hidden_dimension": 4,
        "dropout": 0.0,
        "residual_gate_init": 0.0,
    }
    previous = torch.randn(3, 8)
    center = torch.randn(3, 8)
    following = torch.randn(3, 8)
    models = {}
    for name, components in {
        "full": ["state", "incoming", "outgoing", "curvature"],
        "no_curvature": ["state", "incoming", "outgoing"],
        "outgoing_only": ["state", "outgoing"],
        "incoming_only": ["state", "incoming"],
    }.items():
        torch.manual_seed(17)
        models[name] = ThreeStateTrajectoryReadout(
            8,
            {**base, "trajectory_components": components},
            mode="trajectory",
        )

    coordinates = {
        name: model.trajectory_coordinates(previous, center, following).chunk(
            4, dim=-1
        )
        for name, model in models.items()
    }
    torch.testing.assert_close(coordinates["full"][0], coordinates["no_curvature"][0])
    torch.testing.assert_close(coordinates["full"][1], coordinates["no_curvature"][1])
    torch.testing.assert_close(coordinates["full"][2], coordinates["no_curvature"][2])
    assert torch.count_nonzero(coordinates["no_curvature"][3]).item() == 0
    assert torch.count_nonzero(coordinates["outgoing_only"][1]).item() == 0
    assert torch.count_nonzero(coordinates["outgoing_only"][3]).item() == 0
    assert torch.count_nonzero(coordinates["incoming_only"][2]).item() == 0
    assert torch.count_nonzero(coordinates["incoming_only"][3]).item() == 0
    counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    }
    assert len(counts) == 1


def test_three_state_rejects_curvature_that_leaks_an_ablated_direction():
    with pytest.raises(ValueError, match="曲率会泄漏"):
        ThreeStateTrajectoryReadout(
            8,
            {
                "projection_dimension": 2,
                "trajectory_hidden_dimension": 4,
                "dropout": 0.0,
                "residual_gate_init": 0.0,
                "trajectory_components": ["state", "outgoing", "curvature"],
            },
            mode="trajectory",
        )


def test_three_state_concat_control_matches_trajectory_parameter_budget():
    config = {
        "projection_dimension": 256,
        "trajectory_hidden_dimension": 512,
        "concat_hidden_dimension": 320,
        "dropout": 0.1,
        "residual_gate_init": 0.0,
    }
    trajectory = ThreeStateTrajectoryReadout(
        1024, config, mode="trajectory"
    )
    control = ThreeStateTrajectoryReadout(
        1024, config, mode="concat_control"
    )
    trajectory_count = sum(p.numel() for p in trajectory.parameters())
    control_count = sum(p.numel() for p in control.parameters())

    assert trajectory_count == 1_312_513
    assert control_count == 1_312_065
    assert abs(trajectory_count - control_count) / trajectory_count < 0.001


def test_three_state_conditional_trajectory_starts_as_identity_modulation():
    readout = ThreeStateTrajectoryReadout(
        2,
        {
            "projection_dimension": 2,
            "condition_hidden_dimension": 3,
            "trajectory_hidden_dimension": 4,
            "dropout": 0.0,
            "residual_gate_init": 0.0,
        },
        mode="conditional_trajectory",
    )
    with torch.no_grad():
        for projection in (
            readout.center_projection,
            readout.incoming_projection,
            readout.outgoing_projection,
        ):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
    previous = torch.tensor([[1.0, 2.0]])
    center = torch.tensor([[4.0, 6.0]])
    following = torch.tensor([[9.0, 7.0]])

    coordinates, condition = readout.conditional_coordinates(
        previous, center, following, return_condition=True
    )
    state, incoming, outgoing, conditioned_outgoing = coordinates.chunk(
        4, dim=-1
    )

    torch.testing.assert_close(state, center)
    torch.testing.assert_close(incoming, center - previous)
    torch.testing.assert_close(outgoing, following - center)
    assert torch.count_nonzero(condition).item() == 0
    torch.testing.assert_close(conditioned_outgoing, outgoing)
    assert torch.count_nonzero(readout.condition_network[-1].weight).item() == 0
    assert torch.count_nonzero(readout.condition_network[-1].bias).item() == 0


def test_three_state_conditional_trajectory_applies_elementwise_condition():
    readout = ThreeStateTrajectoryReadout(
        2,
        {
            "projection_dimension": 2,
            "condition_hidden_dimension": 3,
            "trajectory_hidden_dimension": 4,
            "dropout": 0.0,
            "residual_gate_init": 0.0,
        },
        mode="conditional_trajectory",
    )
    previous = torch.randn(3, 2)
    center = torch.randn(3, 2)
    following = torch.randn(3, 2)
    with torch.no_grad():
        readout.condition_network[-1].bias.copy_(
            torch.tensor([0.5, -0.25])
        )

    coordinates, condition = readout.conditional_coordinates(
        previous, center, following, return_condition=True
    )
    outgoing = coordinates.chunk(4, dim=-1)[2]
    conditioned_outgoing = coordinates.chunk(4, dim=-1)[3]

    torch.testing.assert_close(
        conditioned_outgoing, outgoing * (1.0 + condition)
    )


def test_three_state_conditional_zero_gate_preserves_cnn_and_boundaries():
    readout = ThreeStateTrajectoryReadout(
        8,
        {
            "projection_dimension": 2,
            "condition_hidden_dimension": 3,
            "trajectory_hidden_dimension": 4,
            "dropout": 0.0,
            "residual_gate_init": 0.0,
        },
        mode="conditional_trajectory",
    ).eval()
    features = torch.nn.functional.normalize(torch.randn(7, 8), dim=-1)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    output, internal = readout(
        features, groups, return_internal_mask=True
    )

    assert internal.tolist() == [False, True, False, False, True, True, False]
    torch.testing.assert_close(output, features, rtol=1e-6, atol=1e-7)
    assert float(readout.residual_gate.detach()) == 0.0


def test_three_state_paired_models_preserve_cnn_and_global_rng_stream():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    trajectory_config = _small_dilated_config("three_state_trajectory")
    control_config = _small_dilated_config("three_state_concat_control")
    for config in (trajectory_config, control_config):
        config["context_mode"] = "grouped"

    torch.manual_seed(29)
    trajectory = build_word_decoding_model(
        6, positions, 1, trajectory_config
    )
    trajectory_next_random = torch.rand(5)
    torch.manual_seed(29)
    control = build_word_decoding_model(6, positions, 1, control_config)
    control_next_random = torch.rand(5)

    for key, value in trajectory.brain_encoder.state_dict().items():
        torch.testing.assert_close(value, control.brain_encoder.state_dict()[key])
    torch.testing.assert_close(trajectory_next_random, control_next_random)


def test_five_state_multiscale_coordinates_and_concat_control_are_exact():
    config = {
        "projection_dimension": 2,
        "hidden_dimension": 4,
        "dropout": 0.0,
        "residual_gate_init": 0.0,
    }
    trajectory = FiveStateTrajectoryReadout(
        2, config, mode="multiscale_trajectory"
    )
    control = FiveStateTrajectoryReadout(
        2, config, mode="concat_control"
    )
    with torch.no_grad():
        for readout in (trajectory, control):
            readout.state_projection.weight.copy_(torch.eye(2))
            readout.state_projection.bias.zero_()
    far_previous = torch.tensor([[0.0, 2.0]])
    previous = torch.tensor([[2.0, 3.0]])
    center = torch.tensor([[5.0, 7.0]])
    following = torch.tensor([[9.0, 8.0]])
    far_following = torch.tensor([[13.0, 11.0]])

    coordinates = trajectory.joint_coordinates(
        far_previous, previous, center, following, far_following
    ).chunk(5, dim=-1)
    concatenated = control.joint_coordinates(
        far_previous, previous, center, following, far_following
    )

    torch.testing.assert_close(coordinates[0], center)
    torch.testing.assert_close(coordinates[1], center - previous)
    torch.testing.assert_close(coordinates[2], following - center)
    torch.testing.assert_close(coordinates[3], (center - far_previous) / 2)
    torch.testing.assert_close(coordinates[4], (far_following - center) / 2)
    torch.testing.assert_close(
        concatenated,
        torch.cat(
            (far_previous, previous, center, following, far_following), dim=-1
        ),
    )


def test_five_state_zero_gate_uses_radius_two_and_preserves_boundaries():
    readout = FiveStateTrajectoryReadout(
        8,
        {
            "projection_dimension": 2,
            "hidden_dimension": 4,
            "dropout": 0.0,
            "residual_gate_init": 0.0,
        },
        mode="multiscale_trajectory",
    ).eval()
    features = torch.nn.functional.normalize(torch.randn(11, 8), dim=-1)
    groups = torch.tensor([0] * 7 + [1] * 4)

    output, internal = readout(
        features, groups, return_internal_mask=True
    )

    assert internal.tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    torch.testing.assert_close(output, features, rtol=1e-6, atol=1e-7)
    assert torch.equal(output[~internal], features[~internal])


def test_five_state_pair_has_exactly_matched_parameters_and_rng():
    config = {
        "projection_dimension": 256,
        "hidden_dimension": 512,
        "dropout": 0.1,
        "residual_gate_init": 0.0,
    }
    trajectory = FiveStateTrajectoryReadout(
        1024, config, mode="multiscale_trajectory"
    )
    control = FiveStateTrajectoryReadout(
        1024, config, mode="concat_control"
    )
    trajectory_count = sum(p.numel() for p in trajectory.parameters())
    control_count = sum(p.numel() for p in control.parameters())
    assert trajectory_count == control_count == 1_443_585

    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    trajectory_config = _small_dilated_config(
        "five_state_multiscale_trajectory"
    )
    control_config = _small_dilated_config("five_state_concat_control")
    for model_config in (trajectory_config, control_config):
        model_config["context_mode"] = "grouped"
    torch.manual_seed(37)
    trajectory_model = build_word_decoding_model(
        6, positions, 1, trajectory_config
    )
    trajectory_random = torch.rand(5)
    torch.manual_seed(37)
    control_model = build_word_decoding_model(
        6, positions, 1, control_config
    )
    control_random = torch.rand(5)
    for key, value in trajectory_model.brain_encoder.state_dict().items():
        torch.testing.assert_close(
            value, control_model.brain_encoder.state_dict()[key]
        )
    torch.testing.assert_close(trajectory_random, control_random)


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


def test_candidate_evidence_is_signed_candidate_conditioned_and_additive():
    readout = CandidateConditionedEvidenceReadout(2, 2, evidence_dimension=2)
    with torch.no_grad():
        readout.feature_projection.weight.copy_(torch.eye(2))
        readout.candidate_projection.weight.copy_(torch.eye(2))
    features = torch.tensor([[[2.0, -1.0], [0.0, 0.0]]])
    candidates = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    scores, evidence = readout(features, candidates, return_evidence=True)

    assert scores.shape == (1, 2)
    assert evidence.shape == (1, 2, 2)
    assert torch.allclose(scores, evidence.sum(dim=-1))
    assert evidence[0, 0, 0] > 0
    assert evidence[0, 0, 1] < 0
    assert torch.allclose(evidence[0, 0], -evidence[0, 1])
    assert readout.feature_projection.bias is None
    assert readout.candidate_projection.bias is None
    zero_scores = readout(torch.zeros_like(features), candidates)
    assert torch.count_nonzero(zero_scores).item() == 0


def test_candidate_attention_starts_uniform_then_can_select_time_by_candidate():
    uniform = CandidateConditionedEvidenceReadout(2, 2, evidence_dimension=2)
    attention = CandidateConditionedAttentionEvidenceReadout(
        2,
        2,
        evidence_dimension=2,
        attention_dimension=2,
        attention_gate_init=0.0,
    )
    with torch.no_grad():
        for projection in (
            uniform.feature_projection,
            uniform.candidate_projection,
            attention.feature_projection,
            attention.candidate_projection,
            attention.attention_feature_projection,
            attention.attention_candidate_projection,
        ):
            projection.weight.copy_(torch.eye(2))
    features = torch.tensor([[[2.0, -1.0, 0.5], [0.0, 1.0, -0.5]]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    uniform_scores, uniform_contributions = uniform(
        features, candidates, return_evidence=True
    )
    scores, contributions, weights, signed_evidence = attention(
        features,
        candidates,
        return_attention=True,
    )

    assert attention.attention_gate.item() == 0.0
    torch.testing.assert_close(
        weights, torch.full_like(weights, 1.0 / features.shape[-1])
    )
    torch.testing.assert_close(scores, uniform_scores)
    torch.testing.assert_close(contributions, uniform_contributions)
    torch.testing.assert_close(contributions, weights * signed_evidence)
    assert attention.attention_feature_projection.bias is None
    assert attention.attention_candidate_projection.bias is None

    scores[0, 0].backward()
    assert attention.attention_gate_logit.grad.abs().item() > 0
    with torch.no_grad():
        attention.attention_gate_logit.fill_(0.5)
    changed_scores, _, changed_weights, _ = attention(
        features,
        candidates,
        return_attention=True,
    )
    assert not torch.allclose(changed_weights[0, 0], changed_weights[0, 1])
    assert not torch.allclose(changed_scores, uniform_scores)


def test_candidate_evidence_model_preserves_trunk_initialization_and_parameter_budget():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    torch.manual_seed(7)
    baseline = build_word_decoding_model(
        6, positions, 1, _small_dilated_config()
    )
    torch.manual_seed(7)
    candidate = build_word_decoding_model(
        6, positions, 1, _small_dilated_config("candidate_evidence")
    )

    baseline_trunk = {
        key: value
        for key, value in baseline.brain_encoder.state_dict().items()
        if not key.startswith("temporal_attention.")
    }
    candidate_trunk = candidate.brain_encoder.state_dict()
    assert baseline_trunk.keys() == candidate_trunk.keys()
    assert all(
        torch.equal(value, candidate_trunk[key])
        for key, value in baseline_trunk.items()
    )
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    assert candidate_parameters <= baseline_parameters

    signals = torch.randn(3, 6, 20)
    candidates = torch.randn(4, 8)
    scores, evidence = candidate(
        signals,
        candidates,
        torch.zeros(3, dtype=torch.long),
        return_evidence=True,
    )
    assert scores.shape == (3, 4)
    assert evidence.shape == (3, 4, 20)
    assert torch.allclose(scores, evidence.sum(dim=-1), atol=1e-6)


def test_soft_competition_uses_strict_prefixes_and_additive_state_updates():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    torch.manual_seed(23)
    model = build_word_decoding_model(
        6,
        positions,
        1,
        _small_dilated_config("candidate_evidence_competition_soft"),
    ).eval()
    signals = torch.randn(3, 6, 20)
    changed_future = signals.clone()
    changed_future[..., 16:] += 100
    candidates = torch.randn(4, 8)
    subjects = torch.zeros(3, dtype=torch.long)

    states, increments, supervision_scores = model.prefix_candidate_states(
        signals,
        candidates,
        subjects,
        return_increments=True,
        return_supervision_scores=True,
    )
    changed_states = model.prefix_candidate_states(
        changed_future, candidates, subjects
    )
    full_scores = model(signals, candidates, subjects)

    assert states.shape == (5, 3, 4)
    torch.testing.assert_close(states[-1], full_scores)
    torch.testing.assert_close(increments.sum(dim=0), states[-1])
    torch.testing.assert_close(changed_states[:-1], states[:-1])
    assert not torch.allclose(changed_states[-1], states[-1])
    for index, prefix in enumerate(model.prefix_samples):
        torch.testing.assert_close(
            supervision_scores[index],
            states[index] * model.state_normalizer_samples / prefix,
        )
    assert model.prefix_loss_weights == (
        0.0625,
        0.0625,
        0.0625,
        0.0625,
        1.0,
    )


def test_soft_competition_updates_batch_norm_from_full_window_once():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = _small_dilated_config("candidate_evidence_competition_soft")
    config["conv"]["batch_norm"] = True
    model = build_word_decoding_model(6, positions, 1, config).train()
    batch_norm = next(
        module
        for module in model.brain_encoder.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    )
    before = int(batch_norm.num_batches_tracked)

    model.prefix_candidate_states(
        torch.randn(3, 6, 20),
        torch.randn(4, 8),
        torch.zeros(3, dtype=torch.long),
    )

    assert int(batch_norm.num_batches_tracked) - before == 1
    assert batch_norm.training is True


def test_soft_competition_preserves_v0_full_branch_and_rng_stream():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    v0_config = _small_dilated_config("candidate_evidence")
    soft_config = _small_dilated_config("candidate_evidence_competition_soft")
    for config in (v0_config, soft_config):
        config["conv"]["batch_norm"] = True
        config["conv"]["dropout_input"] = 0.2
        config["conv"]["convolution_dropout"] = 0.2

    torch.manual_seed(31)
    v0 = build_word_decoding_model(6, positions, 1, v0_config).train()
    torch.manual_seed(31)
    soft = build_word_decoding_model(6, positions, 1, soft_config).train()
    signals = torch.randn(3, 6, 20)
    candidates = torch.randn(4, 8)
    subjects = torch.zeros(3, dtype=torch.long)

    torch.manual_seed(41)
    v0_scores = v0(signals, candidates, subjects)
    v0_next_random = torch.rand(4)
    torch.manual_seed(41)
    _, supervision_scores = soft.prefix_candidate_states(
        signals,
        candidates,
        subjects,
        return_supervision_scores=True,
    )
    soft_next_random = torch.rand(4)

    torch.testing.assert_close(supervision_scores[-1], v0_scores)
    torch.testing.assert_close(soft_next_random, v0_next_random)
    for v0_module, soft_module in zip(v0.modules(), soft.modules()):
        if isinstance(v0_module, torch.nn.BatchNorm1d):
            torch.testing.assert_close(
                soft_module.running_mean, v0_module.running_mean
            )
            torch.testing.assert_close(
                soft_module.running_var, v0_module.running_var
            )


def test_candidate_attention_preserves_v0_initialization_and_rng_stream():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    torch.manual_seed(19)
    uniform = build_word_decoding_model(
        6, positions, 1, _small_dilated_config("candidate_evidence")
    ).eval()
    uniform_next_random = torch.rand(4)
    torch.manual_seed(19)
    attention = build_word_decoding_model(
        6, positions, 1, _small_dilated_config("candidate_evidence_attention")
    ).eval()
    attention_next_random = torch.rand(4)

    assert uniform.brain_encoder.state_dict().keys() == (
        attention.brain_encoder.state_dict().keys()
    )
    assert all(
        torch.equal(value, attention.brain_encoder.state_dict()[key])
        for key, value in uniform.brain_encoder.state_dict().items()
    )
    for name in ("feature_projection", "candidate_projection"):
        assert torch.equal(
            getattr(uniform.evidence_readout, name).weight,
            getattr(attention.evidence_readout, name).weight,
        )
    torch.testing.assert_close(attention_next_random, uniform_next_random)

    signals = torch.randn(3, 6, 20)
    candidates = torch.randn(4, 8)
    subjects = torch.zeros(3, dtype=torch.long)
    uniform_scores, uniform_contributions = uniform(
        signals, candidates, subjects, return_evidence=True
    )
    scores, contributions, weights, signed_evidence = attention(
        signals,
        candidates,
        subjects,
        return_attention=True,
    )
    torch.testing.assert_close(scores, uniform_scores)
    torch.testing.assert_close(contributions, uniform_contributions)
    torch.testing.assert_close(
        weights, torch.full_like(weights, 1.0 / signals.shape[-1])
    )
    torch.testing.assert_close(contributions, weights * signed_evidence)


def test_candidate_context_residual_starts_as_exact_context_baseline():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    torch.manual_seed(11)
    baseline = build_word_decoding_model(
        6, positions, 1, _small_context_config()
    ).eval()
    baseline_next_random = torch.rand(4)
    torch.manual_seed(11)
    candidate = build_word_decoding_model(
        6, positions, 1, _small_context_config("candidate_context_residual")
    ).eval()
    candidate_next_random = torch.rand(4)

    assert baseline.brain_encoder.state_dict().keys() == (
        candidate.brain_encoder.state_dict().keys()
    )
    assert all(
        torch.equal(value, candidate.brain_encoder.state_dict()[key])
        for key, value in baseline.brain_encoder.state_dict().items()
    )
    assert baseline.context_transformer.state_dict().keys() == (
        candidate.context_transformer.state_dict().keys()
    )
    assert all(
        torch.equal(value, candidate.context_transformer.state_dict()[key])
        for key, value in baseline.context_transformer.state_dict().items()
    )
    assert candidate.evidence_gate.item() == 0.0
    torch.testing.assert_close(candidate_next_random, baseline_next_random)

    signals = torch.randn(4, 6, 20)
    candidates = torch.randn(5, 8)
    subjects = torch.zeros(4, dtype=torch.long)
    groups = torch.tensor([0, 0, 1, 1])
    baseline_embeddings = baseline(signals, subjects, groups)
    expected_scores = baseline_embeddings @ torch.nn.functional.normalize(
        candidates, dim=-1
    ).T
    actual_scores, evidence = candidate(
        signals,
        candidates,
        subjects,
        groups,
        return_evidence=True,
    )

    torch.testing.assert_close(actual_scores, expected_scores)
    assert evidence.shape == (4, 5, 20)
    with torch.no_grad():
        candidate.evidence_gate_logit.fill_(0.25)
    changed_scores = candidate(signals, candidates, subjects, groups)
    assert not torch.allclose(changed_scores, expected_scores)


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
