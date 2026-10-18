"""跨数据集复用的文本表示与缓存工具。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch


def normalize_word(value: str) -> str:
    """按照现有实验规则，将单词清洗为小写字母数字形式。"""
    return "".join(
        character
        for character in str(value).strip()
        if character.isalnum() or character in {"-", "'"}
    ).lower()


def text_embedding_signature(config) -> dict:
    """返回与现有文本向量缓存完全一致的配置签名。"""
    signature = {
        "model_name": str(config.get("model_name", "t5-large")),
        "layer_fraction": float(config.get("layer_fraction", 0.5)),
        "token_aggregation": str(config.get("token_aggregation", "mean")),
        "add_special_tokens": False,
        "padding_aggregation": "attention_masked",
    }
    # 旧数据集未声明此项时保持既有 cache signature 完全不变。
    if config.get("word_normalization") is not None:
        signature["word_normalization"] = str(config["word_normalization"])
    # 新协议显式启用多语表示；未声明时保持旧缓存签名不变。
    for key in ("revision", "backend", "normalize_embeddings", "max_length", "dtype"):
        if key in config:
            signature[key] = config[key]
    if "add_special_tokens" in config:
        signature["add_special_tokens"] = bool(config["add_special_tokens"])
    return signature


def prepared_words(words, config) -> list[str]:
    """按文本合同去重并稳定排序待编码词型。"""
    normalization = config.get("word_normalization")
    if normalization == "canonical_standard_word":
        normalize = lambda value: str(value).strip()
    elif normalization in (None, "legacy_alnum_hyphen_apostrophe"):
        normalize = normalize_word
    else:
        raise ValueError(f"未知文本词规范化合同：{normalization}")
    prepared = {normalize(word) for word in words}
    prepared.discard("")
    return sorted(prepared)


def load_text_model(config):
    """按现有配置加载 tokenizer 与文本编码模型，不隐式联网。"""
    try:
        from transformers import AutoModelForTextEncoding, AutoTokenizer
        if config.get("backend") == "auto_model":
            from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("生成词向量需要 transformers。") from exc

    signature = text_embedding_signature(config)
    model_name = signature["model_name"]
    local_only = bool(config.get("local_files_only", False))
    pinned = {"revision": config["revision"]} if config.get("revision") else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        truncation_side="left",
        local_files_only=local_only,
        **pinned,
    )
    model_class = AutoModel if config.get("backend") == "auto_model" else AutoModelForTextEncoding
    model_kwargs = dict(pinned)
    if config.get("dtype"):
        model_kwargs["dtype"] = getattr(torch, config["dtype"])
    model = model_class.from_pretrained(model_name, local_files_only=local_only, **model_kwargs)
    requested_device = str(config.get("device", "cpu"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model.to(device).eval()
    return tokenizer, model, device


def encode_words(
    words, config, *, tokenizer=None, model=None, device=None
) -> np.ndarray:
    """逐词批量编码，并保持输入词序与既有层/聚合语义。"""
    words = [str(word) for word in words]
    if not words:
        return np.empty(
            (0, int(config.get("embedding_dimension", 1024))),
            dtype=np.float32,
        )
    if tokenizer is None or model is None or device is None:
        tokenizer, model, device = load_text_model(config)

    signature = text_embedding_signature(config)
    batch_size = int(config.get("batch_size", 32))
    expected_dimension = int(config.get("embedding_dimension", 1024))
    batches = []
    for start in range(0, len(words), batch_size):
        batch_words = words[start : start + batch_size]
        print(f"文本向量 {min(start + len(batch_words), len(words))}/{len(words)}", flush=True)
        inputs = tokenizer(
            batch_words,
            add_special_tokens=signature["add_special_tokens"],
            return_tensors="pt",
            padding=True,
            truncation=not bool(config.get("max_length")),
        )
        if config.get("max_length") and inputs["input_ids"].shape[1] > int(config["max_length"]):
            raise ValueError("完整词超过文本编码长度上限，不能静默截断。")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        states = outputs.hidden_states
        layer_index = int(signature["layer_fraction"] * len(states) - 1e-6)
        layer_index = min(max(layer_index, 0), len(states) - 1)
        hidden = states[layer_index]
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if signature["token_aggregation"] == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        elif signature["token_aggregation"] == "sum":
            pooled = (hidden * mask).sum(dim=1)
        elif signature["token_aggregation"] == "first":
            pooled = hidden[:, 0]
        elif signature["token_aggregation"] == "last":
            last = inputs["attention_mask"].sum(dim=1).sub(1).clamp_min(0)
            pooled = hidden[torch.arange(len(hidden), device=device), last]
        else:
            raise ValueError(
                f"未知 token aggregation：{signature['token_aggregation']}"
            )
        pooled = pooled.float()
        if config.get("normalize_embeddings", False):
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        pooled = pooled.cpu().numpy().astype(np.float32, copy=False)
        if pooled.shape[1] != expected_dimension:
            raise ValueError(
                f"文本向量维数为 {pooled.shape[1]}，配置预期 {expected_dimension}。"
            )
        batches.append(pooled)
    return np.concatenate(batches, axis=0)


def save_text_embedding_cache(path, words, embeddings, config) -> Path:
    """原子保存公共 NPZ 格式及 sidecar。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    words = [str(word) for word in words]
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(words):
        raise ValueError("文本词序与 embedding 矩阵 shape 不一致。")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.savez(file, words=np.asarray(words), embeddings=matrix)
    os.replace(temporary_path, path)
    metadata = {
        "status": "materialized",
        "signature": text_embedding_signature(config),
        "word_count": len(words),
        "embedding_dimension": int(matrix.shape[1]),
    }
    temporary_metadata = path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_metadata, path.with_suffix(".json"))
    return path


def load_text_embedding_cache(path, expected_signature=None) -> dict[str, np.ndarray]:
    """读取现有 NPZ 文本向量缓存并验证配置签名。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文本向量缓存不存在：{path}")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"文本向量缓存缺少元数据：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if expected_signature is not None and metadata.get("signature") != expected_signature:
        raise ValueError("文本向量缓存与当前模型/层配置不一致，请使用新的缓存路径。")
    with np.load(path, allow_pickle=False) as payload:
        words = payload["words"].astype(str).tolist()
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if len(words) != len(embeddings) or len(words) != len(set(words)):
        raise ValueError(f"文本向量缓存索引无效：{path}")
    return {word: embeddings[index] for index, word in enumerate(words)}


def ensure_text_embedding_cache(words, config, path) -> Path:
    """增量缓存所选数据划分需要的 T5 词向量。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = text_embedding_signature(config)
    existing = {}
    if path.exists():
        existing = load_text_embedding_cache(path, expected_signature=signature)
    requested = prepared_words(words, config)
    missing = [word for word in requested if word not in existing]
    if not missing:
        return path

    tokenizer, model, device = load_text_model(config)
    encoded = encode_words(
        missing, config, tokenizer=tokenizer, model=model, device=device
    )
    for word, embedding in zip(missing, encoded):
        existing[word] = embedding

    ordered_words = sorted(existing)
    matrix = np.stack([existing[word] for word in ordered_words]).astype(np.float32)
    return save_text_embedding_cache(path, ordered_words, matrix, config)
