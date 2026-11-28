"""第三阶梯 Stage-1 抗坍缩多试次原型训练单元测试与 CUDA bf16 端到端冒烟校验。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from braindecoding.config import load_yaml_with_extends
from braindecoding.losses import MultiTrialPrototypeSigLipLoss, build_siglip_loss
from braindecoding.models import build_brain_embedding_model
from braindecoding.optimizers import build_adamw_for_modules
from braindecoding.training.word_encoder_training import (
    MultiTrialPrototypeBatchSampler,
    encode_loader,
    make_loader,
    train_one_epoch,
)


class _ToyWordMegDataset(Dataset):
    """构造带多录音、多词重复的轻量 MEG 词窗数据集供单测与 CUDA 冒烟使用。"""

    def __init__(
        self,
        *,
        num_recordings: int = 8,
        words_per_recording: int = 24,
        channels: int = 16,
        samples: int = 50,
        embed_dim: int = 32,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        vocab = [f"word_{i:02d}" for i in range(18)]
        vocab_embs = rng.standard_normal((len(vocab), embed_dim)).astype(np.float32)
        vocab_embs /= np.linalg.norm(vocab_embs, axis=-1, keepdims=True)
        self.embedding_map = {w: vocab_embs[i] for i, w in enumerate(vocab)}

        rows = []
        signals = []
        idx = 0
        for rec_i in range(num_recordings):
            rec_id = f"rec-{rec_i:02d}"
            onset = 0.0
            for w_i in range(words_per_recording):
                # 偶发保留 1-2 个仅出现 1 次的罕见词，验证单试次兜底
                if rec_i == 0 and w_i in (0, 1):
                    word = f"hapax_{w_i}"
                    if word not in self.embedding_map:
                        vec = rng.standard_normal(embed_dim).astype(np.float32)
                        vec /= np.linalg.norm(vec)
                        self.embedding_map[word] = vec
                else:
                    word = vocab[(rec_i * 5 + w_i) % len(vocab)]
                rows.append(
                    {
                        "event_id": f"{rec_id}:{w_i}",
                        "recording_id": rec_id,
                        "sentence_uid": f"{rec_id}:s{w_i // 8}",
                        "sentence_index": rec_i * 3 + (w_i // 8),
                        "subject_index": 0,
                        "normalized_word": word,
                        "onset_seconds": onset,
                    }
                )
                sig = rng.standard_normal((channels, samples)).astype(np.float32)
                signals.append(sig)
                onset += 3.0
                idx += 1

        self.table = pd.DataFrame(rows)
        self.signals = np.stack(signals, axis=0)
        self.channel_count = channels
        self.subject_count = 1
        self.channel_positions = rng.uniform(0.0, 1.0, size=(channels, 2)).astype(
            np.float32
        )

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        word = str(row["normalized_word"])
        return {
            "meg": torch.from_numpy(self.signals[index]),
            "text_embedding": torch.from_numpy(self.embedding_map[word].copy()),
            "subject_index": torch.tensor(int(row["subject_index"]), dtype=torch.long),
            "sentence_index": torch.tensor(int(row["sentence_index"]), dtype=torch.long),
            "word": word,
            "event_id": str(row["event_id"]),
            "recording_id": str(row["recording_id"]),
            "onset_seconds": torch.tensor(float(row["onset_seconds"]), dtype=torch.float32),
        }


def test_multi_trial_prototype_batch_sampler_invariants_and_speed() -> None:
    """验证 100% 覆盖、零重复、同录音 2.5s 与同词 10.0s 硬间隔、零单样本批及 50k 样本构建速度。"""
    rng = np.random.default_rng(123)
    n_recs = 40
    per_rec = 1000
    n_total = n_recs * per_rec
    # 模拟 Zipf 词频分布（含高频停用词与大量只出现 1 次的罕见词）
    words: list[str] = []
    rec_ids: list[str] = []
    onsets = np.empty(n_total, dtype=np.float64)
    cursor = 0
    for r in range(n_recs):
        rec_name = f"rec-{r:03d}"
        t = 0.0
        for j in range(per_rec):
            u = rng.random()
            if u < 0.06:
                w = "the"
            elif u < 0.12:
                w = "and"
            elif u < 0.90:
                w = f"w_{int(rng.integers(0, 600))}"
            else:
                w = f"rare_{cursor}"
            words.append(w)
            rec_ids.append(rec_name)
            onsets[cursor] = t
            t += float(rng.uniform(0.2, 0.5))
            cursor += 1

    t0 = time.perf_counter()
    sampler = MultiTrialPrototypeBatchSampler(
        words,
        rec_ids,
        onsets,
        batch_size=256,
        proto_group_size=4,
        min_gap_seconds=2.5,
        proto_min_gap_seconds=10.0,
        shuffle=True,
        seed=7,
    )
    batches = list(sampler)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"40k 样本批采样构建过慢：{elapsed:.2f}s"
    assert len(batches) == len(sampler)

    seen: list[int] = []
    multi_trial_word_count = 0
    for batch in batches:
        assert len(batch) >= 2, "出现单样本尾批，可能引发 BatchNorm1d 异常"
        assert len(batch) <= 256
        seen.extend(batch)
        by_rec: dict[str, list[float]] = {}
        by_word_rec: dict[tuple[str, str], list[float]] = {}
        word_counts: dict[str, int] = {}
        for idx in batch:
            rec = rec_ids[idx]
            w = words[idx]
            onset = float(onsets[idx])
            word_counts[w] = word_counts.get(w, 0) + 1
            for prev_t in by_rec.get(rec, ()):
                assert abs(onset - prev_t) >= 2.5 - 1e-9
            for prev_t in by_word_rec.get((w, rec), ()):
                assert abs(onset - prev_t) >= 10.0 - 1e-9
            by_rec.setdefault(rec, []).append(onset)
            by_word_rec.setdefault((w, rec), []).append(onset)
        multi_trial_word_count += sum(1 for c in word_counts.values() if c >= 2)

    assert len(seen) == n_total
    assert len(set(seen)) == n_total
    assert multi_trial_word_count > 1000

    # 验证 set_epoch 会更新批序并清缓存
    sampler.set_epoch(1)
    batches_ep1 = list(sampler)
    assert batches_ep1 != batches
    assert len({i for b in batches_ep1 for i in b}) == n_total


def test_multi_trial_prototype_siglip_loss_math_and_gradients() -> None:
    """验证唯一词原型池化数学正确性、无重复词回退与梯度完整回传至 h_pre / z / proto_pre_head。"""
    torch.manual_seed(0)
    loss_mod = build_siglip_loss(
        {
            "name": "multi_trial_prototype_siglip",
            "single_weight": 1.0,
            "proto_pre_weight": 0.5,
            "proto_z_weight": 0.25,
            "min_group_size": 2,
            "pre_head_dimension": 24,
            "output_dimension": 16,
        }
    )
    assert isinstance(loss_mod, MultiTrialPrototypeSigLipLoss)
    assert loss_mod.requires_word_ids is True
    assert loss_mod.requires_pre_head is True

    words = ["cat", "cat", "cat", "dog", "dog", "bird", "fish", "fish"]
    loss_mod.set_vocabulary(["cat", "dog", "bird", "fish"])
    word_ids = loss_mod.map_words_to_ids(words, torch.device("cpu"))

    bank = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    targets = bank[word_ids]
    h_pre = torch.randn(8, 24, requires_grad=True)
    z_raw = torch.randn(8, 16, requires_grad=True)
    z_norm = torch.nn.functional.normalize(z_raw, dim=-1)

    loss = loss_mod(z_norm, targets, target_word_ids=word_ids, pre_head=h_pre)
    assert torch.isfinite(loss)
    assert loss_mod.last_components["active_prototypes"] == 3.0  # cat, dog, fish
    assert float(loss_mod.last_components["proto_pre_loss"]) > 0.0
    assert float(loss_mod.last_components["proto_z_loss"]) > 0.0

    loss.backward()
    assert h_pre.grad is not None and float(h_pre.grad.abs().sum()) > 0.0
    assert z_raw.grad is not None and float(z_raw.grad.abs().sum()) > 0.0
    assert loss_mod.proto_pre_head is not None
    assert loss_mod.proto_pre_head.weight.grad is not None
    assert float(loss_mod.proto_pre_head.weight.grad.abs().sum()) > 0.0

    # 全部为单试次词（无 >=2 组）时优雅回退为仅单试次损失
    uniq_words = ["cat", "dog", "bird", "fish"]
    uniq_ids = loss_mod.map_words_to_ids(uniq_words, torch.device("cpu"))
    loss_singleton = loss_mod(
        z_norm[:4].detach(),
        bank,
        target_word_ids=uniq_ids,
        pre_head=h_pre[:4].detach(),
    )
    assert torch.isfinite(loss_singleton)
    assert loss_mod.last_components["active_prototypes"] == 0.0
    assert float(loss_mod.last_components["proto_pre_loss"]) == 0.0


def test_stage1_prototype_yaml_state_dict_parity_and_stage2_warm_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证第三阶梯 YAML 与基准 MultiBinResConvEncoder state_dict 100% 键名一致，可 strict=True 热启动 Op3A-v2。"""
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    base_cfg = load_yaml_with_extends(
        "configs/training/libribrain100_word_encoder_multibin_resconv_butterworth.yaml"
    )
    proto_cfg = load_yaml_with_extends(
        "configs/training/libribrain100_word_encoder_multibin_resconv_prototype_anticollapse.yaml"
    )
    op3a_cfg = load_yaml_with_extends(
        "configs/training/siamese_debiased/libribrain100_multibin_signed_decollapse_prefix_revision.yaml"
    )
    assert proto_cfg["loss"]["name"] == "multi_trial_prototype_siglip"
    assert proto_cfg["training"]["word_sampling"] == "multi_trial_prototype_sampler"
    assert proto_cfg["model"]["conv"]["channel_offset_std"] == pytest.approx(0.10)

    positions = np.zeros((16, 2), dtype=np.float32)
    # 使用小尺寸通道快速构建同构检查
    for cfg in (base_cfg, proto_cfg, op3a_cfg):
        cfg["model"]["conv"]["merger_channels"] = 16
        cfg["model"]["conv"]["merger_position_dimension"] = 32
        cfg["model"]["conv"]["hidden"] = 32
        cfg["model"]["conv"]["depth"] = 2
        cfg["model"]["conv"]["mlp_dim"] = 64
        cfg["model"]["conv"]["mlp_hidden"] = 128
        cfg["model"]["conv"]["mlp_depth"] = 2
        cfg["model"]["embedding_dimension"] = 64
    op3a_cfg["model"]["conv"] = dict(proto_cfg["model"]["conv"])
    op3a_cfg["model"]["transformer"]["heads"] = 4
    op3a_cfg["model"]["transformer"]["head_dimension"] = 8
    op3a_cfg["model"]["transformer"]["self_decollapse_rank"] = 16
    op3a_cfg["model"]["transformer"]["h_bins_pca_dim"] = 16

    m_base = build_brain_embedding_model(16, positions, 1, base_cfg["model"])
    m_proto = build_brain_embedding_model(16, positions, 1, proto_cfg["model"])
    m_stage2 = build_brain_embedding_model(16, positions, 1, op3a_cfg["model"])

    assert list(m_base.state_dict().keys()) == list(m_proto.state_dict().keys())
    enc_weights = {
        k.removeprefix("brain_encoder."): v.clone()
        for k, v in m_proto.state_dict().items()
        if k.startswith("brain_encoder.")
    }
    incompat = m_stage2.brain_encoder.load_state_dict(enc_weights, strict=True)
    assert not incompat.missing_keys
    assert not incompat.unexpected_keys


def test_stage1_prototype_train_one_epoch_and_cuda_bf16_smoke() -> None:
    """端到端校验 make_loader -> train_one_epoch -> encode_loader（优先在 CUDA bf16 上真跑）。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = _ToyWordMegDataset(
        num_recordings=6, words_per_recording=24, channels=16, samples=50, embed_dim=32
    )
    model_cfg = {
        "embedding_dimension": 32,
        "use_transformer": False,
        "conv": {
            "architecture": "multibin_resconv",
            "merger_channels": 16,
            "merger_position_dimension": 32,
            "hidden": 24,
            "depth": 2,
            "kernel_size": 3,
            "num_temporal_bins": 5,
            "bin_attention_hidden": 16,
            "mlp_dim": 32,
            "mlp_hidden": 64,
            "mlp_depth": 2,
            "mlp_dropout": 0.1,
            "gaussian_noise_std": 0.05,
            "channel_offset_std": 0.10,
            "channel_dropout": 0.05,
        },
    }
    tc = {
        "seed": 0,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "decoupled_weight_decay": True,
        "max_grad_norm": 1.0,
        "amp": device.type == "cuda",
        "amp_dtype": "bfloat16" if device.type == "cuda" else "float32",
        "word_sampling": "multi_trial_prototype_sampler",
        "proto_group_size": 4,
        "min_gap_seconds": 2.5,
        "proto_min_gap_seconds": 10.0,
    }
    train_loader, sampler = make_loader(ds, tc, shuffle=True)
    assert isinstance(sampler, MultiTrialPrototypeBatchSampler)
    eval_loader, _ = make_loader(ds, tc, shuffle=False)

    model = build_brain_embedding_model(16, ds.channel_positions, 1, model_cfg).to(device)
    loss_module = build_siglip_loss(
        {
            "name": "multi_trial_prototype_siglip",
            "single_weight": 1.0,
            "proto_pre_weight": 0.5,
            "proto_z_weight": 0.25,
            "min_group_size": 2,
            "pre_head_dimension": 32,
            "output_dimension": 32,
        }
    ).to(device)
    loss_module.set_vocabulary(sorted(ds.embedding_map.keys()))

    optimizer = build_adamw_for_modules([model, loss_module], tc)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    mean_loss, updates, batches_seen, diag = train_one_epoch(
        model,
        loss_module,
        train_loader,
        optimizer,
        scaler,
        device,
        tc,
        return_stats=True,
        return_diagnostics=True,
        epoch_index=0,
    )
    assert np.isfinite(mean_loss)
    assert updates == batches_seen and updates > 0
    p_diag = diag["prototype_anti_collapse"]
    assert p_diag is not None
    assert p_diag["mean_active_prototypes"] > 0.0
    assert p_diag["proto_pre_loss"] > 0.0
    assert p_diag["proto_z_loss"] > 0.0

    # 推理期仍严格 k=1 单试次，完全不依赖 loss_module
    encoded = encode_loader(
        model,
        eval_loader,
        device,
        amp=(device.type == "cuda"),
        amp_dtype=("bfloat16" if device.type == "cuda" else "float32"),
    )
    assert encoded["predictions"].shape == (len(ds), 32)
    assert torch.isfinite(encoded["predictions"]).all()
