import ast
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

from braindecoding.data import text
from braindecoding.data import chineseeeg2 as ChineseEEG2


def _install_fake_transformers(monkeypatch, calls):
    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["tokenizer_load"] = (model_name, kwargs)
            return cls()

        def __call__(self, words, **kwargs):
            calls.setdefault("batches", []).append(list(words))
            calls.setdefault("tokenizer_kwargs", []).append(kwargs)
            input_ids = torch.tensor(
                [[len(word), len(word) + 1, 0] for word in words],
                dtype=torch.long,
            )
            attention_mask = torch.tensor(
                [[1, 1, 0] for _ in words], dtype=torch.long
            )
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["model_load"] = (model_name, kwargs)
            return cls()

        def to(self, device):
            calls["device"] = str(device)
            return self

        def eval(self):
            calls["eval"] = True
            return self

        def __call__(self, input_ids, attention_mask, output_hidden_states):
            assert output_hidden_states is True
            base = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 3)
            return types.SimpleNamespace(
                hidden_states=tuple(base + offset for offset in (0, 10, 20, 30))
            )

    module = types.ModuleType("transformers")
    module.AutoTokenizer = FakeTokenizer
    module.AutoModelForTextEncoding = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_text_cache_signature_keeps_the_existing_contract():
    config = {
        "model_name": "synthetic-t5",
        "layer_fraction": 0.25,
        "token_aggregation": "last",
        "batch_size": 7,
        "embedding_dimension": 3,
        "device": "cpu",
        "local_files_only": True,
    }
    assert text.text_embedding_signature(config) == {
        "model_name": "synthetic-t5",
        "layer_fraction": 0.25,
        "token_aggregation": "last",
        "add_special_tokens": False,
        "padding_aggregation": "attention_masked",
    }


def test_embedding_cache_generation_locks_order_shape_values_and_keys(
    tmp_path, monkeypatch
):
    calls = {}
    _install_fake_transformers(monkeypatch, calls)
    path = tmp_path / "text_embeddings.npz"
    config = {
        "model_name": "synthetic-t5",
        "layer_fraction": 0.5,
        "token_aggregation": "mean",
        "batch_size": 1,
        "embedding_dimension": 3,
        "device": "cpu",
        "local_files_only": True,
    }

    result = text.ensure_text_embedding_cache(
        [" BETA! ", "alpha", "beta", ""], config, path
    )

    assert result == path
    assert calls["batches"] == [["alpha"], ["beta"]]
    assert calls["tokenizer_load"] == (
        "synthetic-t5",
        {"truncation_side": "left", "local_files_only": True},
    )
    assert calls["model_load"] == ("synthetic-t5", {"local_files_only": True})
    assert calls["device"] == "cpu"
    assert calls["eval"] is True
    assert calls["tokenizer_kwargs"] == [
        {
            "add_special_tokens": False,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        },
        {
            "add_special_tokens": False,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        },
    ]
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {"words", "embeddings"}
        assert payload["words"].astype(str).tolist() == ["alpha", "beta"]
        assert payload["embeddings"].shape == (2, 3)
        assert payload["embeddings"].dtype == np.float32
        np.testing.assert_array_equal(
            payload["embeddings"],
            np.asarray([[15.5, 15.5, 15.5], [14.5, 14.5, 14.5]], dtype=np.float32),
        )
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(metadata) == {
        "status",
        "signature",
        "word_count",
        "embedding_dimension",
    }
    assert metadata == {
        "status": "materialized",
        "signature": text.text_embedding_signature(config),
        "word_count": 2,
        "embedding_dimension": 3,
    }

    cache_bytes = path.read_bytes()
    metadata_text = path.with_suffix(".json").read_text(encoding="utf-8")
    text.ensure_text_embedding_cache(["alpha", "beta", "alpha"], config, path)
    assert calls["batches"] == [["alpha"], ["beta"]]
    assert path.read_bytes() == cache_bytes
    assert path.with_suffix(".json").read_text(encoding="utf-8") == metadata_text


def test_cache_reading_preserves_word_order_and_embeddings(
    tmp_path,
):
    path = tmp_path / "ordered.npz"
    embeddings = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    np.savez(path, words=np.asarray(["second", "first"]), embeddings=embeddings)
    signature = text.text_embedding_signature({})
    path.with_suffix(".json").write_text(
        json.dumps({"signature": signature}), encoding="utf-8"
    )

    current = text.load_text_embedding_cache(path, expected_signature=signature)

    assert list(current) == ["second", "first"]
    np.testing.assert_array_equal(current["second"], embeddings[0])
    np.testing.assert_array_equal(current["first"], embeddings[1])


def test_word_datasets_use_the_public_text_implementation():
    assert ChineseEEG2.normalize_word is text.normalize_word
    assert (
        ChineseEEG2.text_embedding_signature is text.text_embedding_signature
    )
    assert (
        ChineseEEG2.ensure_text_embedding_cache
        is text.ensure_text_embedding_cache
    )
    assert (
        ChineseEEG2.load_text_embedding_cache is text.load_text_embedding_cache
    )


def test_dataset_production_modules_do_not_import_each_other():
    project_root = Path(__file__).resolve().parents[2]
    dataset_paths = tuple((project_root / "datasets").glob("*.py"))
    dataset_modules = {path.stem for path in dataset_paths}
    for path in dataset_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_module = node.module.split(".")[-1]
                is_dataset_import = node.module.startswith("datasets.") or (
                    node.level > 0 and imported_module in dataset_modules
                )
                assert not is_dataset_import, f"{path.name}: {node.module}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("datasets."), (
                        f"{path.name}: {alias.name}"
                    )
