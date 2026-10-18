import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from braindecoding.data import chineseeeg2 as dataset_module
from braindecoding.models import build_brain_embedding_model
from braindecoding.tasks.word_decoding.chineseeeg2_littleprince.train import (
    设置脑编码器可训练,
    载入配置,
    载入预训练脑编码器,
)


通道名 = ["E1", "E2", "E3", "E4"]



def 创建记录文件(root, subject, run, rate=100, duration=10):
    """创建足以验证元数据合同的最小 BrainVision 旁车结构。"""
    directory = (
        root
        / "derivatives"
        / "preprocessed"
        / subject
        / "ses-littleprince"
        / "eeg"
    )
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{subject}_ses-littleprince_task-lis_run-{run}_eeg"
    vhdr_path = directory / f"{stem}.vhdr"
    channel_lines = "\n".join(
        f"Ch{index}={name},,1.0,µV"
        for index, name in enumerate(通道名, start=1)
    )
    vhdr_path.write_text(
        "\n".join(
            [
                "Brain Vision Data Exchange Header File Version 1.0",
                "[Common Infos]",
                f"DataFile={stem}.eeg",
                "DataFormat=BINARY",
                "DataOrientation=MULTIPLEXED",
                f"NumberOfChannels={len(通道名)}",
                f"SamplingInterval={1e6 / rate}",
                "[Binary Infos]",
                "BinaryFormat=IEEE_FLOAT_32",
                "[Channel Infos]",
                channel_lines,
            ]
        ),
        encoding="utf-8",
    )
    vhdr_path.with_suffix(".eeg").write_bytes(
        b"\0" * (4 * len(通道名) * rate * duration)
    )
    vhdr_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "SamplingFrequency": rate,
                "RecordingDuration": duration,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "name": 通道名,
            "type": ["EEG"] * len(通道名),
            "low_cutoff": [1.0] * len(通道名),
            "high_cutoff": [40.0] * len(通道名),
            "sampling_frequency": [rate] * len(通道名),
            "status": ["good"] * len(通道名),
        }
    ).to_csv(
        directory / f"{stem.replace('_eeg', '')}_channels.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        {
            "name": 通道名,
            "x": [0.0, 1.0, 0.0, -1.0],
            "y": [1.0, 0.0, -1.0, 0.0],
            "z": [1.0, 1.0, 1.0, 1.0],
        }
    ).to_csv(
        directory / f"{subject}_ses-littleprince_space-CapTrak_electrodes.tsv",
        sep="\t",
        index=False,
    )
    return vhdr_path


def 创建合成任务(tmp_path):
    """建立只使用 actual_reading 工作簿的合成任务。"""
    root = tmp_path / "PassiveListening"
    subject = "sub-01"
    for chapter, run in ((1, 11), (2, 12), (3, 13)):
        创建记录文件(root, subject, run)
    workbook = tmp_path / "actual.xlsx"
    workbook.write_bytes(b"x")
    return {
        "root": str(root),
        "alignment_path": str(workbook),
        "event_source_type": "actual_reading_workbook",
        "actual_reading_sheet_name": "词级时间戳",
        "expected_source_event_count": 6,
        "timestamp_to_eeg_offset_seconds": 0.0,
        "preprocessed_dir": "derivatives/preprocessed",
        "subjects": [subject],
        "split": {
            "seed": 42,
            "train_chapters": [1],
            "val_chapters": [2],
            "test_chapters": [3],
        },
        "vocabulary_size": 2,
        "training_vocabulary_policy": "all_primary_complete_window_words",
        "evaluation_vocabulary_policy": "frozen_train_chapter_top2_unique_voice_word_events",
        "source_sampling_rate_hz": 100,
        "target_sampling_rate_hz": 50,
        "window_start_offset_seconds": 0.0,
        "window_seconds": 1.0,
        "eligibility_window_seconds": 1.0,
        "source_preprocessed_filter_hz": [1.0, 40.0],
        "additional_filter": None,
        "scaler": "RobustScaler",
        "clamp": 5,
        "materialization_channel_chunk": 2,
        "context_grouping": "actual_reading_row_then_contiguous_chunks",
        "max_context_words": 4,
        "expected_trainable_counts": {"train": 2, "val": 2, "test": 2},
    }


def test_候选词只由训练章节冻结():
    frame = pd.DataFrame({
        "normalized_word": ["甲", "甲", "乙", "验证专用词", "验证专用词", "乙"],
        "split": ["train", "train", "train", "val", "val", "test"],
        "is_trainable": [True] * 6,
        "voice_version": ["f1"] * 6,
        "source_word_id": [f"w-{index}" for index in range(6)],
        "chapter": [1, 1, 1, 2, 2, 3],
        "word_index": [1, 2, 3, 1, 2, 1],
        "chapter_word_index": [1, 2, 3, 1, 2, 1],
        "subject_id": ["sub-01"] * 6,
    })

    vocabulary, counts = dataset_module.训练集高频词(frame, 2)

    assert vocabulary == ("甲", "乙")
    assert counts == {"甲": 2, "乙": 1}
    assert "验证专用词" not in vocabulary


def test_实际朗读一秒事件保留警告和零时长词(tmp_path, monkeypatch):
    """工作簿警告和零时长不能变成删词条件。"""
    config = 创建合成任务(tmp_path)
    for chapter, run in ((1, 11), (2, 12), (3, 13)):
        创建记录文件(Path(config["root"]), "sub-01", run)
    workbook = tmp_path / "actual.xlsx"
    workbook.write_bytes(b"x")
    config.update(
        alignment_path=str(workbook), event_source_type="actual_reading_workbook",
        expected_source_event_count=6, timestamp_to_eeg_offset_seconds=0.0,
        eligibility_window_seconds=1.0, context_grouping="actual_reading_row_then_contiguous_chunks",
    )
    source = pd.DataFrame([
        {"音频编号": chapter, "音频文件": f"audio_{chapter}.wav", "原表行号": 10 + chapter,
         "词": word, "文本起始位置": index, "文本结束位置": index,
         "开始时间（秒）": 1.0 + index, "结束时间（秒）": 1.0 + index,
         "质检提示": "非正时长" if index == 0 else np.nan,
         "时间戳来源": "首字开始至末字结束"}
        for chapter, words in ((1, ["甲", "乙"]), (2, ["甲", "乙"]), (3, ["甲", "乙"]))
        for index, word in enumerate(words)
    ])
    monkeypatch.setattr(dataset_module.pd, "read_excel", lambda *args, **kwargs: source.copy())
    table = dataset_module.构建实际朗读事件表(config)
    audit = dataset_module.审计事件表(table, config)
    assert len(table) == 6
    assert table["is_trainable"].all()
    assert table["window_complete"].all()
    assert table.loc[table["timestamp_qc_prompt"].notna(), "automatic_qc_pass"].eq(False).all()
    assert table.groupby("chapter")["run"].first().to_dict() == {1: 11, 2: 12, 3: 13}
    assert table["timing_status"].eq("actual_reading_audio_clock_used_as_eeg_clock").all()
    assert audit["raw_eeg_loaded"] is False
    assert audit["test_eeg_loaded"] is False
    assert set(audit["evaluation_vocabulary"]) == {"甲", "乙"}


def test_实际朗读事件按配置展开到两名女声一受试者(tmp_path, monkeypatch):
    """同一套 f1 词时间戳应为每名受试者生成独立脑电事件。"""
    config = 创建合成任务(tmp_path)
    config["subjects"] = ["sub-01", "sub-02"]
    for subject in config["subjects"]:
        for chapter, run in ((1, 11), (2, 12), (3, 13)):
            创建记录文件(Path(config["root"]), subject, run)
    workbook = tmp_path / "actual.xlsx"
    workbook.write_bytes(b"x")
    config.update(
        alignment_path=str(workbook), event_source_type="actual_reading_workbook",
        expected_source_event_count=6, timestamp_to_eeg_offset_seconds=0.0,
        eligibility_window_seconds=1.0, context_grouping="actual_reading_row_then_contiguous_chunks",
    )
    source = pd.DataFrame([
        {"音频编号": chapter, "音频文件": f"audio_{chapter}.wav", "原表行号": 10 + chapter,
         "词": word, "文本起始位置": index, "文本结束位置": index,
         "开始时间（秒）": 1.0 + index, "结束时间（秒）": 1.0 + index,
         "质检提示": np.nan, "时间戳来源": "首字开始至末字结束"}
        for chapter, words in ((1, ["甲", "乙"]), (2, ["甲", "乙"]), (3, ["甲", "乙"]))
        for index, word in enumerate(words)
    ])
    monkeypatch.setattr(dataset_module.pd, "read_excel", lambda *args, **kwargs: source.copy())

    table = dataset_module.构建实际朗读事件表(config)

    assert len(table) == 12
    assert table.groupby("subject_id").size().to_dict() == {"sub-01": 6, "sub-02": 6}
    assert table["event_id"].is_unique
    assert all(
        sentence_uid.startswith(subject_id + "|")
        for sentence_uid, subject_id in zip(table["sentence_uid"], table["subject_id"])
    )
    assert table.groupby("subject_id")["source_word_id"].nunique().eq(6).all()


def test_两种声音事件源合并后统一冻结候选词(tmp_path, monkeypatch):
    """f1和m1应各自匹配受试者，并在合并训练集上统一统计词频。"""
    config = 创建合成任务(tmp_path)
    config["subjects"] = ["sub-01", "sub-05"]
    for chapter, run in ((1, 11), (2, 12), (3, 13)):
        创建记录文件(Path(config["root"]), "sub-05", run)
    config["actual_reading_sources"] = [
        {
            "alignment_path": str(tmp_path / "f1.xlsx"),
            "expected_source_event_count": 6,
            "voice_version": "f1",
            "subjects": ["sub-01"],
        },
        {
            "alignment_path": str(tmp_path / "m1.xlsx"),
            "expected_source_event_count": 6,
            "voice_version": "m1",
            "subjects": ["sub-05"],
        },
    ]
    for source in config["actual_reading_sources"]:
        Path(source["alignment_path"]).write_bytes(b"x")
    source = pd.DataFrame([
        {"音频编号": chapter, "音频文件": f"audio_{chapter}.wav", "原表行号": 10 + chapter,
         "词": word, "文本起始位置": index, "文本结束位置": index,
         "开始时间（秒）": 1.0 + index, "结束时间（秒）": 1.0 + index,
         "质检提示": np.nan, "时间戳来源": "模型对齐估计"}
        for chapter, words in ((1, ("甲", "乙")), (2, ("甲", "乙")), (3, ("甲", "乙")))
        for index, word in enumerate(words)
    ])
    monkeypatch.setattr(dataset_module.pd, "read_excel", lambda *args, **kwargs: source.copy())

    table = dataset_module.构建实际朗读事件表(config)

    assert len(table) == 12
    assert table.groupby("voice_version")["subject_id"].unique().map(tuple).to_dict() == {
        "f1": ("sub-01",), "m1": ("sub-05",)
    }
    assert set(table.loc[table["vocabulary_rank"].gt(0), "normalized_word"]) == {"甲", "乙"}


def test_对齐文件只迁移目录不改变事件科学内容(tmp_path, monkeypatch):
    """同一工作簿位于 outputs 或 artifacts 时，事件表必须逐值相同。"""
    config = 创建合成任务(tmp_path)
    old_path = tmp_path / "outputs" / "女声一小王子时间戳" / "actual.xlsx"
    new_path = tmp_path / "artifacts" / "alignments" / "f1" / "actual.xlsx"
    old_path.parent.mkdir(parents=True)
    new_path.parent.mkdir(parents=True)
    old_path.write_bytes(b"same-workbook")
    new_path.write_bytes(old_path.read_bytes())
    source = pd.DataFrame([
        {"音频编号": chapter, "音频文件": f"audio_{chapter}.wav", "原表行号": 10 + chapter,
         "词": word, "文本起始位置": index, "文本结束位置": index,
         "开始时间（秒）": 1.0 + index, "结束时间（秒）": 1.25 + index,
         "质检提示": np.nan, "时间戳来源": "首字开始至末字结束"}
        for chapter, words in ((1, ("甲", "乙")), (2, ("甲", "乙")), (3, ("甲", "乙")))
        for index, word in enumerate(words)
    ])
    monkeypatch.setattr(
        dataset_module.pd,
        "read_excel",
        lambda *args, **kwargs: source.copy(),
    )

    old_config = dict(config, alignment_path=str(old_path))
    new_config = dict(config, alignment_path=str(new_path))
    old_table = dataset_module.构建实际朗读事件表(old_config)
    new_table = dataset_module.构建实际朗读事件表(new_config)

    pd.testing.assert_frame_equal(
        old_table,
        new_table,
        check_dtype=True,
        check_exact=True,
    )


def test_长度受控语义片段跨行合并并在自然边界切分():
    """短行应跨行合并，达到优先长度后才在分句边界切分。"""
    frame = pd.DataFrame(
        {
            "subject_id": ["sub-01"] * 7,
            "run": [11] * 7,
            "chapter": [1] * 7,
            "material_line": [10, 10, 11, 11, 12, 12, 12],
            "aligned_start_seconds": np.arange(7, dtype=float),
            "aligned_stop_seconds": np.arange(7, dtype=float) + 0.5,
            "sentence_uid": ["原分组"] * 7,
            "context_chunk_index": [0] * 7,
        }
    )
    texts = {
        (1, 10): "我六岁那年，",
        (1, 11): "在一本书上，",
        (1, 12): "看见一幅插图。",
    }

    grouped = dataset_module.构建长度受控语义片段(
        frame,
        texts,
        {"preferred_words": 4, "maximum_words": 6, "maximum_seconds": 15.0},
    )

    groups = list(grouped.groupby("sentence_uid", sort=False))
    assert [len(group) for _, group in groups] == [4, 3]
    assert groups[0][1]["material_line"].nunique() == 2
    assert groups[0][1]["context_boundary_reason"].eq(
        "clause_end_after_preferred_length"
    ).all()
    assert groups[1][1]["context_boundary_reason"].eq("sentence_end").all()
    assert grouped["context_grouping"].eq("bounded_semantic_v1").all()


def test_记录缓存按合同降采样且不重复滤波(tmp_path):
    config = 创建合成任务(tmp_path)
    source_path = dataset_module.记录路径(config, "sub-01", 11)
    samples = np.arange(4 * 1000, dtype=np.float32).reshape(4, 1000)
    source_path.with_suffix(".eeg").write_bytes(
        samples.T.astype("<f4").tobytes()
    )
    cache_path = dataset_module.物化记录缓存(
        source_path, config, tmp_path / "cache"
    )
    output = np.load(cache_path)
    metadata = json.loads(
        cache_path.with_suffix(".json").read_text(encoding="utf-8")
    )

    assert output.shape == (4, 500)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert metadata["signature"]["additional_filter"] is None
    assert metadata["signature"]["source_is_preprocessed"] is True


def test_actual_reading配置保持单一八人数据合同(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )
    project_root = Path(__file__).resolve().parents[2]
    root = project_root / "configs/word_decoding/chineseeeg2_littleprince/sub01-08"
    word = 载入配置(root / "main_word.yaml")
    context = 载入配置(root / "main_context_warmstart.yaml")

    expected_subjects = [f"sub-{index:02d}" for index in range(1, 9)]
    expected_counts = {"train": 81620, "val": 14132, "test": 8480}
    for config in (word, context):
        dataset = config["dataset"]
        assert dataset["event_source_type"] == "actual_reading_workbook"
        assert dataset["subjects"] == expected_subjects
        assert dataset["expected_trainable_counts"] == expected_counts
        assert dataset["window_seconds"] == 1.0
        assert dataset["eligibility_window_seconds"] == 1.0
        assert [
            source["voice_version"] for source in dataset["actual_reading_sources"]
        ] == ["f1", "m1"]
        assert dataset["excluded_chapters"] == [14, 27]
        assert config["model"]["transformer"]["depth"] == 4
        assert config["model"]["transformer"]["heads"] == 8
        assert config["training"]["epochs"] == 30
        assert config["training"]["patience"] == 10
        assert "max_updates" not in config["training"]
        assert "minimum_updates_before_early_stopping" not in config["training"]
        assert "scheduler_total_updates" not in config["training"]
        assert all(
            Path(value).resolve().is_relative_to((project_root / "data/cache/derived").resolve())
            for value in config["cache"].values()
        )

    assert word["model"]["use_transformer"] is False
    assert context["model"]["use_transformer"] is True
    assert context["experiment"]["id"] == "main_context_warmstart"
    for config in (word, context):
        assert config["dataset"]["context_grouping"] == "bounded_semantic_v1"
        assert config["dataset"]["semantic_context"] == {
            "actual_reading_line_sheet_name": "逐行对照",
            "preferred_words": 16,
            "maximum_words": 32,
            "maximum_seconds": 15.0,
        }
    assert context["training"]["freeze_brain_encoder_epochs"] == 1
    assert "freeze_brain_encoder_updates" not in context["training"]
    assert context["training"]["warm_start_from"] == "main_word"
    for section in ("text_embedding", "loss", "evaluation"):
        assert context[section] == word[section]

def test_预热检查点只载入脑编码器(tmp_path):
    positions = np.asarray(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    conv = {
        "merger_channels": 4,
        "merger_position_dimension": 8,
        "merger_per_subject": True,
        "merger_dropout": 0,
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
    }
    source_model_config = {
        "embedding_dimension": 8,
        "use_transformer": False,
        "conv": conv,
    }
    target_model_config = {
        **source_model_config,
        "use_transformer": True,
        "transformer": {"depth": 1, "heads": 2},
    }
    source = build_brain_embedding_model(4, positions, 1, source_model_config)
    target = build_brain_embedding_model(4, positions, 1, target_model_config)
    with torch.no_grad():
        for parameter in source.brain_encoder.parameters():
            parameter.fill_(0.125)
    transformer_before = {
        key: value.detach().clone()
        for key, value in target.context_transformer.state_dict().items()
    }
    split = {
        "seed": 42,
        "train_chapters": [1],
        "val_chapters": [2],
        "test_chapters": [3],
    }
    dataset_contract = {
        "split": split,
        "window_start_offset_seconds": 0.0,
        "window_seconds": 1.0,
        "eligibility_window_seconds": 1.0,
        "source_sampling_rate_hz": 100,
        "target_sampling_rate_hz": 50,
        "training_vocabulary_policy": "all_primary_complete_window_words",
        "evaluation_vocabulary_policy": "train_only_top2",
        "event_source_type": "actual_reading_workbook",
        "timestamp_to_eeg_offset_seconds": 0.0,
        "evaluation_vocabulary": ["甲", "乙"],
    }
    text_config = {"model_name": "synthetic", "embedding_dimension": 8}
    checkpoint_path = tmp_path / "cnn.pt"
    torch.save(
        {
            "task": "word_decoding/ChineseEEG2_LittlePrince",
            "epoch": 3,
            "optimizer_updates": 20,
            "model_state": source.state_dict(),
            "model_config": source_model_config,
            "text_embedding_config": text_config,
            "dataset_contract": dataset_contract,
            "channel_names": 通道名,
            "channel_positions": positions.tolist(),
            "subject_ids": ["sub-01"],
            "metrics": {
                "retrieval_acc10_vocab=chineseeeg2_littleprince50_macro": 0.3
            },
        },
        checkpoint_path,
    )
    config = {
        "dataset": dataset_contract,
        "text_embedding": text_config,
        "model": target_model_config,
    }

    audit = 载入预训练脑编码器(
        target,
        checkpoint_path,
        config,
        通道名,
        positions,
        ["sub-01"],
        ["甲", "乙"],
    )

    for key, value in source.brain_encoder.state_dict().items():
        torch.testing.assert_close(target.brain_encoder.state_dict()[key], value)
    for key, value in transformer_before.items():
        torch.testing.assert_close(target.context_transformer.state_dict()[key], value)
    assert audit["loaded_loss_state"] is False
    assert audit["loaded_transformer_state"] is False
    设置脑编码器可训练(target, False)
    assert not any(
        parameter.requires_grad for parameter in target.brain_encoder.parameters()
    )
    assert target.brain_encoder.training is False
    设置脑编码器可训练(target, True)
    assert all(
        parameter.requires_grad for parameter in target.brain_encoder.parameters()
    )
