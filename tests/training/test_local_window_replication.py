import json

import pytest
import torch

from braindecoding.models import GroupedTransformerEncoder
from scripts.paper.local_window_figures import (
    RADIUS_VALUES,
    evaluation_path,
    load_curve,
    load_multiseed_curve,
    write_curve,
    write_multiseed_curve,
)


@pytest.mark.parametrize("radius", (0, 1, 2))
def test_query_window_cannot_read_outside_radius(radius):
    torch.manual_seed(127)
    model = GroupedTransformerEncoder(
        8,
        {
            "depth": 2,
            "heads": 2,
            "attention_dropout": 0,
            "feedforward_dropout": 0,
            "query_window_radius": radius,
        },
    ).eval()
    features = torch.randn(11, 8)
    groups = torch.tensor([0] * 7 + [1] * 4)
    original = model(features, groups)
    changed = features.clone()
    changed[0] += 13.0
    changed[7] -= 11.0
    revised = model(changed, groups)
    # Query 3 is outside the changed event's radius, including the two-layer case.
    torch.testing.assert_close(original[3], revised[3], rtol=0, atol=1e-6)
    # A change in the second group must never affect the first group.
    torch.testing.assert_close(original[:7], model(features.index_copy(0, torch.tensor([7]), changed[7:8]), groups)[:7], rtol=0, atol=1e-6)


def test_query_window_trains_without_changing_parameter_contract():
    config = {"depth": 1, "heads": 2, "attention_dropout": 0, "feedforward_dropout": 0}
    torch.manual_seed(81)
    full = GroupedTransformerEncoder(8, config)
    torch.manual_seed(81)
    local = GroupedTransformerEncoder(8, {**config, "query_window_radius": 2})
    assert full.state_dict().keys() == local.state_dict().keys()
    for key in full.state_dict():
        torch.testing.assert_close(full.state_dict()[key], local.state_dict()[key])
    features = torch.randn(5, 8, requires_grad=True)
    output = local(features, torch.zeros(5, dtype=torch.long))
    output.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize(
    ("left_radius", "right_radius", "outside_rows", "inside_rows"),
    (
        (2, 0, (0, 4), (1, 2)),
        (2, 1, (0, 5), (1, 2, 4)),
    ),
)
def test_asymmetric_query_window_uses_only_requested_offsets(
    left_radius, right_radius, outside_rows, inside_rows
):
    torch.manual_seed(211)
    model = GroupedTransformerEncoder(
        8,
        {
            "depth": 2,
            "heads": 2,
            "attention_dropout": 0,
            "feedforward_dropout": 0,
            "query_window_left_radius": left_radius,
            "query_window_right_radius": right_radius,
        },
    ).eval()
    features = torch.randn(7, 8)
    groups = torch.zeros(7, dtype=torch.long)
    original = model(features, groups)[3]
    outside = features.clone()
    outside[list(outside_rows)] += 17.0
    torch.testing.assert_close(
        original, model(outside, groups)[3], rtol=0, atol=1e-6
    )
    inside = features.clone()
    inside[list(inside_rows)] += 17.0
    assert not torch.allclose(original, model(inside, groups)[3])


def test_asymmetric_query_window_requires_both_sides_and_no_radius_mix():
    base = {"depth": 1, "heads": 2}
    with pytest.raises(ValueError, match="必须同时设置"):
        GroupedTransformerEncoder(8, {**base, "query_window_left_radius": 2})
    with pytest.raises(ValueError, match="不能与左右独立"):
        GroupedTransformerEncoder(
            8,
            {
                **base,
                "query_window_radius": 2,
                "query_window_left_radius": 2,
                "query_window_right_radius": 0,
            },
        )


def test_curve_refuses_missing_validation_runs(tmp_path):
    with pytest.raises(FileNotFoundError, match="缺少完整验证结果"):
        load_curve(tmp_path)


def test_curve_uses_only_matched_complete_validation_results(tmp_path):
    for radius in (*RADIUS_VALUES, None):
        path = evaluation_path(tmp_path, radius)
        path.parent.mkdir(parents=True)
        name = "main_context_warmstart" if radius is None else f"local_window_r{radius}"
        experiment = {
            "task": "word_decoding",
            "dataset": "pallier2025",
            "subject_scope": "sub01-10",
            "experiment_id": name,
            "seed": 0,
        }
        path.write_text(
            json.dumps(
                {
                    "status": "evaluation_completed",
                    "experiment": experiment,
                    "data": {
                        "split": "val",
                        "query_count": 16960,
                        "event_table_sha256": "events",
                        "subjects": ["sub-01"],
                    },
                    "checkpoint": {"sha256": name},
                    "vocabularies": {
                        "50": {
                            "candidate_count": 50,
                            "manifest_sha256": "vocabulary",
                            "retrieval": {"macro_top10": 0.6},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (path.parents[1] / "training_summary.json").write_text(
            json.dumps(
                {
                    "status": "training_completed",
                    "experiment": experiment,
                    "checkpoint": {"best": {"sha256": name}},
                    "selection": {
                        "metric": "retrieval_acc10_vocab=pallier2025_50_macro"
                    },
                    "training": {"budget": {"type": "epochs", "value": 30}},
                    "details": {
                        "train_rows": 115290,
                        "validation_rows": 16960,
                        "best_validation": {"query_count": 8260},
                        "initialization_audit": {
                            "source_checkpoint_sha256": "same_cnn"
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
    csv_path, figure_path = write_curve(tmp_path, tmp_path / "figures")
    assert csv_path.is_file() and figure_path.is_file()
    assert len(load_curve(tmp_path)) == 6
    for seed in (1, 2):
        for radius in (*RADIUS_VALUES, None):
            source = evaluation_path(tmp_path, radius)
            destination = evaluation_path(tmp_path, radius, seed)
            destination.parent.mkdir(parents=True)
            evaluation = json.loads(source.read_text(encoding="utf-8"))
            summary = json.loads(
                (source.parents[1] / "training_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            evaluation["experiment"]["seed"] = seed
            summary["experiment"]["seed"] = seed
            destination.write_text(json.dumps(evaluation), encoding="utf-8")
            (destination.parents[1] / "training_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
    multi_csv, multi_figure = write_multiseed_curve(
        tmp_path, tmp_path / "figures", seeds=(0, 1, 2)
    )
    assert multi_csv.is_file() and multi_figure.is_file()
    assert len(load_multiseed_curve(tmp_path)) == 6
    altered = json.loads(evaluation_path(tmp_path, 4).read_text(encoding="utf-8"))
    altered["data"]["event_table_sha256"] = "different"
    evaluation_path(tmp_path, 4).write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="验证事件"):
        load_curve(tmp_path)
