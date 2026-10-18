"""论文协议资产、来源边界和稳定哈希测试。"""

import json
from pathlib import Path

import pandas as pd
import pytest

from braindecoding import protocols as manifests


from braindecoding.config import PROJECT_ROOT
MANIFEST_ROOT = PROJECT_ROOT / "data" / "manifests"


def _load(relative_path):
    return json.loads((MANIFEST_ROOT / relative_path).read_text(encoding="utf-8"))


def _vocabularies(dataset):
    return {
        size: _load(f"{dataset}/vocabulary_N{size}.json")
        for size in manifests.PRIMARY_VOCABULARY_SIZES
    }


def test_all_machine_manifests_have_stable_self_hashes():
    from braindecoding.unified_sentence_v2 import digest

    paths = sorted(MANIFEST_ROOT.rglob("*.json"))
    assert (
        MANIFEST_ROOT / "pallier2025/implementation_diagnostics.json"
    ) in paths
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "manifest_sha256" in payload and "profile" not in payload:
            body = {k: v for k, v in payload.items() if k != "manifest_sha256"}
            if manifests.manifest_sha256(payload) == payload["manifest_sha256"]:
                manifests.validate_manifest_sha256(payload)
                assert manifests.with_manifest_sha256(payload) == payload
            else:
                assert digest(body) == payload["manifest_sha256"], path


def test_vocabulary_assets_use_only_train_units_and_stable_order():
    split_files = {
        "chineseeeg2": _load("chineseeeg2/split_manifest.json"),
    }
    for dataset, split_manifest in split_files.items():
        vocabularies = _vocabularies(dataset)
        train_units = set(split_manifest["trainable_split_units"]["train"])
        previous = []
        for size in manifests.PRIMARY_VOCABULARY_SIZES:
            manifest = vocabularies[size]
            assert manifest["source_split"] == "train"
            assert manifest["created_from_test"] is False
            assert manifest["evaluation_support_used_for_selection"] is False
            assert set(manifest["source_units"]) == train_units
            assert manifest["vocabulary_size"] == size
            assert len(manifest["vocabulary"]) == size
            assert manifest["vocabulary"][: len(previous)] == previous

            ranked = [
                (word, manifest["word_counts"][word])
                for word in manifest["vocabulary"]
            ]
            assert ranked == sorted(ranked, key=lambda item: (-item[1], item[0]))
            previous = manifest["vocabulary"]


def test_vocabulary_selection_does_not_read_test_rows(tmp_path):
    base = pd.DataFrame(
        {
            "unit": ["train-a", "train-b", "test-a"],
            "position": [0, 0, 0],
            "subject_id": ["sub-01"] * 3,
            "split": ["train", "train", "test"],
            "is_trainable": [True, True, True],
            "normalized_word": ["乙", "甲", "测试词"],
        }
    )
    changed_test = base.copy()
    changed_test.loc[changed_test["split"].eq("test"), "normalized_word"] = "另一个测试词"

    def vocabulary_payload(table):
        train = manifests.canonical_words(
            table,
            deduplicate_by=("unit", "position"),
            order_by=("unit", "position", "subject_id"),
            split="train",
            trainable_only=True,
        )
        return manifests.build_vocabulary_manifest(
            dataset="fixture",
            size=2,
            canonical_train_words=train,
            source_units=("train-a", "train-b"),
            event_table_path=tmp_path / "events.csv",
            event_table_sha256="fixture",
            generator_sha256="fixture",
            status="test",
            normalization="fixture",
            scope="fixture",
        )

    first = vocabulary_payload(base)
    second = vocabulary_payload(changed_test)
    assert first["vocabulary"] == second["vocabulary"] == ["乙", "甲"]
    assert first["word_counts"] == second["word_counts"]


def test_frequency_order_is_stable_under_row_shuffle_and_ties():
    words = pd.Series(["乙", "甲", "乙", "甲", "丙"])
    expected = (["乙", "甲", "丙"], {"乙": 2, "甲": 2, "丙": 1})
    # Unicode 词典序中“乙”位于“甲”之前；这里只锁定 Python 当前排序合同。
    assert manifests.ranked_word_counts(words) == expected
    assert manifests.ranked_word_counts(words.sample(frac=1, random_state=7)) == expected


def test_split_unit_cannot_cross_splits():
    table = pd.DataFrame(
        {"_split_unit": ["story-01", "story-01"], "split": ["train", "test"]}
    )
    with pytest.raises(ValueError, match="跨 split"):
        manifests.validate_split_units(table)


def test_subject_order_and_multisubject_status_are_explicit():
    chinese = _load("chineseeeg2/split_manifest.json")
    assert chinese["subject_order"] == [f"sub-{index:02d}" for index in range(1, 9)]
    assert chinese["status"] == "frozen"
    assert set(chinese["shared_excluded_units"]) == {
        "ChineseEEG2|littleprince|chapter-14",
        "ChineseEEG2|littleprince|chapter-27",
    }
    assert all(
        item["trainable_event_count"] == 0
        for item in chinese["shared_excluded_units"].values()
    )


def test_formal_context_and_test_history_are_machine_readable():
    conditions = _load("conditions.json")
    chinese = conditions["datasets"]["ChineseEEG2"]
    assert chinese["word"]["model.use_transformer"] is False
    assert chinese["neural_context"] == "bounded_semantic_v1"
    assert chinese["row_context_role"] == "historical_structure_prior_control"
    assert chinese["test_status"] == "previously_used_for_exploratory_analysis"
    assert chinese["confirmatory_status"] == "exploratory_or_replication_only"


def test_primary_sizes_and_control_protocols_are_frozen_without_implementation():
    conditions = _load("conditions.json")
    sizes = conditions["candidate_vocabulary_sizes"]
    assert sizes["primary"] == [20, 50, 100, 150]
    assert sizes["supplementary_exploratory"] == [200, 300, 500]
    assert sizes["test_support_may_change_sizes"] is False

    statuses = {
        name: value["status"] for name, value in conditions["conditions"].items()
    }
    assert statuses == {
        "neural_context.clean": "implemented",
        "neural_context.donor_following": "exploratory",
        "neural_context.donor_swap": "exploratory",
        "neural_context.structure_only": "exploratory",
        "neural_context.temporal_shift": "missing",
        "word.clean": "implemented",
        "word.donor_swap": "missing",
        "word.temporal_shift": "missing",
    }
    temporal = conditions["conditions"]["word.temporal_shift"]["protocol"]
    assert temporal["parameter_status"] == "parameters_not_frozen"
    assert temporal["offset_selection"] == "train_val_only"
    donor = conditions["conditions"]["word.donor_swap"]["protocol"]
    assert donor["seeds"] == list(range(20))
    assert donor["same_subject"] and donor["same_recording"]
    assert donor["cross_recording_primary"] is False


def test_vocabulary_support_and_missing_words_are_explicit():
    expected_missing = {
        "chineseeeg2": {
            "N20": (0, 0),
            "N50": (3, 4),
            "N100": (12, 18),
            "N150": (28, 42),
        },
    }
    for dataset, expected in expected_missing.items():
        support = _load(f"{dataset}/vocabulary_support.json")
        assert support["asset_type"] == "evaluation_vocabulary_support"
        assert support["support_audit_used_for_vocabulary_selection"] is False
        for name, (val_missing, test_missing) in expected.items():
            splits = support["vocabularies"][name]["splits"]
            assert splits["val"]["missing_candidate_count"] == val_missing
            assert splits["test"]["missing_candidate_count"] == test_missing
            for split in ("val", "test"):
                current = splits[split]
                assert current["missing_candidate_count"] == len(current["missing_words"])
                assert current["supported_candidate_count"] + current["missing_candidate_count"] == int(name[1:])
                assert current["support_fraction"] == pytest.approx(
                    current["supported_candidate_count"] / int(name[1:])
                )


def test_write_and_check_are_byte_stable(tmp_path):
    documents = {"fixture.json": manifests.with_manifest_sha256({"value": "词"})}
    paths = manifests.write_or_check_documents(documents, tmp_path, check=False)
    first = paths[0].read_bytes()
    manifests.write_or_check_documents(documents, tmp_path, check=True)
    manifests.write_or_check_documents(documents, tmp_path, check=False)
    assert paths[0].read_bytes() == first
