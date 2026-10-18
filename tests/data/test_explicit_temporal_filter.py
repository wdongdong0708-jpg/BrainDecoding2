"""Checks for the Pallier d'Ascoli-to-LISA temporal front end."""

import numpy as np
import pytest
import torch

from braindecoding.models import DilatedConvEncoder, ExplicitTemporalFilter


def test_shared_filter_starts_as_identity_and_can_move_a_peak():
    layer = ExplicitTemporalFilter(2, 3, 7).eval()
    signal = torch.zeros(1, 2, 11)
    signal[0, 0, 5] = 1
    signal[0, 1, 6] = 2
    assert torch.equal(layer(signal), signal)
    with torch.no_grad():
        layer.weight[0, 0, 3] = 0
        layer.weight[0, 0, 4] = 1
    moved = layer(signal)
    assert moved[0, 0].argmax().item() == 4
    assert torch.equal(moved[0, 1], signal[0, 1])


def test_subject_residual_changes_only_selected_subject_and_has_gradients():
    layer = ExplicitTemporalFilter(2, 3, 7, mode="subject_residual")
    signal = torch.randn(3, 2, 11, requires_grad=True)
    indices = torch.tensor([0, 1, 2])
    assert torch.allclose(layer(signal, indices), signal)
    with torch.no_grad():
        layer.subject_residuals[1, 0, 4] = 0.5
    changed = layer(signal, indices)
    assert torch.equal(changed[0], signal[0])
    assert not torch.equal(changed[1], signal[1])
    assert torch.equal(changed[2], signal[2])
    unknown = layer(signal, torch.tensor([-1, 1, -1]))
    assert torch.equal(unknown[0], signal[0])
    assert torch.equal(unknown[2], signal[2])
    changed.square().sum().backward()
    assert layer.weight.grad is not None
    assert layer.subject_residuals.grad is not None
    assert layer.subject_residuals.grad[1].abs().sum() > 0


def test_temporal_filter_rejects_invalid_configuration_and_subjects():
    with pytest.raises(ValueError):
        ExplicitTemporalFilter(2, 3, 8)
    layer = ExplicitTemporalFilter(2, 3, 7, mode="subject_residual")
    with pytest.raises(ValueError):
        layer(torch.randn(2, 2, 11), torch.tensor([0]))
    with pytest.raises(IndexError):
        layer(torch.randn(2, 2, 11), torch.tensor([0, 3]))


def test_encoder_places_filter_after_subject_projection():
    positions = np.array([[0.1, 0.2], [0.4, 0.7], [0.8, 0.3]], dtype="float32")
    encoder = DilatedConvEncoder(
        positions,
        subject_count=2,
        output_dimension=8,
        config={
            "merger_channels": 4,
            "merger_position_dimension": 8,
            "merger_dropout": 0,
            "initial_linear": 6,
            "hidden": 4,
            "depth": 2,
            "batch_norm": False,
            "glu_every": 0,
            "temporal_filter": {"mode": "subject_residual", "kernel_size": 7},
        },
    ).eval()
    assert encoder.temporal_filter.channels == 6
    output = encoder(torch.randn(2, 3, 50), torch.tensor([0, 1]))
    assert output.shape == (2, 8)


@pytest.mark.parametrize("branch_channels", [2, 4])
def test_encoder_uses_configured_subject_branch_count(branch_channels):
    positions = np.array([[0.1, 0.2], [0.4, 0.7], [0.8, 0.3]], dtype="float32")
    encoder = DilatedConvEncoder(
        positions,
        subject_count=2,
        output_dimension=8,
        config={
            "merger_channels": 4,
            "merger_position_dimension": 8,
            "merger_dropout": 0,
            "initial_linear": 4,
            "branch_channels": branch_channels,
            "hidden": 4,
            "depth": 2,
            "batch_norm": False,
            "glu_every": 0,
            "temporal_filter": {"mode": "shared", "kernel_size": 7},
        },
    )
    assert encoder.subject_projection.weights.shape == (2, branch_channels, 4)
    assert encoder.temporal_filter.channels == branch_channels
    assert encoder.blocks[0][1].in_channels == branch_channels
    output = encoder(torch.randn(2, 3, 50), torch.tensor([0, 1]))
    assert output.shape == (2, 8)
    output.square().sum().backward()
    assert encoder.subject_projection.weights.grad is not None


def test_shallow_models_have_identical_outputs_at_filter_initialization():
    positions = np.array([[0.1, 0.2], [0.4, 0.7], [0.8, 0.3]], dtype="float32")
    base = {
        "merger_channels": 4,
        "merger_position_dimension": 8,
        "merger_dropout": 0,
        "initial_linear": 6,
        "hidden": 4,
        "depth": 2,
        "batch_norm": False,
        "glu_every": 0,
    }
    models = []
    for temporal_filter in (None, {"mode": "shared"}, {"mode": "subject_residual"}):
        config = dict(base)
        if temporal_filter is not None:
            config["temporal_filter"] = temporal_filter
        torch.manual_seed(123)
        models.append(DilatedConvEncoder(positions, 2, 8, config).eval())
    signal = torch.randn(2, 3, 50)
    indices = torch.tensor([0, 1])
    reference = models[0](signal, indices)
    for model in models[1:]:
        assert torch.allclose(model(signal, indices), reference, atol=1e-6)
