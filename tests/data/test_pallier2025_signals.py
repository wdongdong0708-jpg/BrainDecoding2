from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler

from braindecoding.data import pallier2025
from braindecoding.data.derived import sha256_file


class _FakeRaw:
    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float64)
        self.info = {"sfreq": 1000.0}
        self.first_samp = 8000
        self.n_times = self._data.shape[1]
        self.calls = []

    def load_data(self):
        self.calls.append("load")
        return self

    def filter(self, l_freq, h_freq, **kwargs):
        self.calls.append(("filter", l_freq, h_freq, kwargs))
        return self

    def resample(self, sfreq, **kwargs):
        self.calls.append(("resample", sfreq, kwargs))
        self.info["sfreq"] = float(sfreq)
        self.first_samp = 400
        return self


def _signal_config():
    return {
        "source_sampling_rate_hz": 1000.0,
        "target_sampling_rate_hz": 50.0,
        "filter_low_hz": 0.1,
        "filter_high_hz": 40.0,
        "mne_n_jobs": 1,
    }


def test_frozen_preprocessing_contract_order_and_explicit_exclusions(monkeypatch):
    # 静态合同测试不需要为读取版本字符串初始化完整 MNE。
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace(__version__="test"))
    contract = pallier2025.preprocessing_contract(_signal_config())
    assert contract["operation_order"] == [
        "mne_read_raw_fif_allow_maxshield",
        "mne_pick_meg",
        "mne_filter_0.1_40_hz",
        "mne_resample_50_hz",
        "sklearn_robust_scaler_per_channel_full_recording",
        "float32",
    ]
    assert contract["filter"]["l_freq_hz"] == 0.1
    assert contract["filter"]["h_freq_hz"] == 40.0
    assert contract["resample"]["implementation"] == "mne.io.Raw.resample"
    assert contract["notch_filter"] is False
    assert contract["sss_or_maxfilter"] is False
    assert contract["baseline_correction"] is False
    assert contract["clamp"] is None
    assert contract["reference"]["commit"] == pallier2025.DASCOLI_REFERENCE_COMMIT


def test_processing_filters_then_resamples_then_scales_full_recording():
    source = np.array([[1.0, 2.0, 100.0, 4.0], [9.0, 6.0, 3.0, 0.0]])
    raw = _FakeRaw(source.copy())
    output, processed = pallier2025._process_picked_meg(
        raw, _signal_config()
    )
    assert raw.calls[0] == "load"
    assert raw.calls[1][0:3] == ("filter", 0.1, 40.0)
    assert raw.calls[2][0:2] == ("resample", 50.0)
    expected = RobustScaler().fit_transform(source.T).T.astype(np.float32)
    np.testing.assert_array_equal(output, expected)
    assert output.dtype == np.float32
    assert processed is raw


def test_channel_digest_uses_frozen_order_and_is_order_sensitive():
    names = ["MEG001", "MEG002", "MEG003"]
    assert pallier2025.channel_names_sha256(names) != pallier2025.channel_names_sha256(
        list(reversed(names))
    )
    assert len(pallier2025.CHANNEL_NAMES_SHA256) == 64
    assert pallier2025.MEG_CHANNEL_COUNT == 306
    assert pallier2025.MAGNETOMETER_COUNT == 102
    assert pallier2025.GRADIOMETER_COUNT == 204


@pytest.mark.parametrize(
    ("first_samp", "onset", "expected"),
    [(8000, 8.0, 0), (38000, 39.0, 50), (462000, 465.5, 175)],
)
def test_first_samp_recording_clock_to_output_sample(first_samp, onset, expected):
    assert pallier2025.recording_onset_to_sample(
        onset, source_first_samp=first_samp
    ) == expected


def test_sub09_run03_inferred_times_are_before_recording_start():
    # 阶段 2A QC 证明这两个推算异常词时间早于该 FIF 的有效起点。
    first_samp = 152000
    inferred = [151.726, 151.936]
    assert all(
        pallier2025.recording_onset_to_sample(
            onset, source_first_samp=first_samp
        ) < 0
        for onset in inferred
    )


def test_atomic_npy_write_and_sha_validated_resume_contract(tmp_path):
    output = tmp_path / "recording.npy"
    array = np.arange(306 * 4, dtype=np.float32).reshape(306, 4)
    pallier2025._write_array_atomic(output, array)
    assert not output.with_suffix(".npy.tmp").exists()
    source_contract = {
        "fingerprint": "fixed",
        "source_sample_count": 80,
        "source_sampling_rate_hz": 1000.0,
    }
    sidecar = {
        "source_contract": source_contract,
        "shape": [306, 4],
        "dtype": "float32",
        "output_sha256": sha256_file(output),
    }
    pallier2025._write_json_atomic(output.with_suffix(".json"), sidecar)
    assert pallier2025._valid_signal_product(output, source_contract)
    sidecar["source_contract"] = {"fingerprint": "changed"}
    pallier2025._write_json_atomic(output.with_suffix(".json"), sidecar)
    assert not pallier2025._valid_signal_product(output, source_contract)


def test_incomplete_signal_set_cannot_create_complete_manifest(tmp_path):
    table = __import__("pandas").DataFrame(
        {
            "记录编号": ["pallier2025|sub-01|run-01"],
            "是否可训练": [False],
            "开始时间": [10.0],
        }
    )
    config = {
        **_signal_config(),
        "subjects": ["sub-01"],
        "event_table": str(tmp_path / "events.csv"),
    }
    (tmp_path / "events.csv").write_text("fixture\n", encoding="utf-8")
    with pytest.raises(ValueError, match="覆盖不完整|90/90"):
        pallier2025.build_signal_manifest(
            table, config, tmp_path / "signals/meg_50hz", [], project_root=tmp_path
        )
    assert not (tmp_path / "signals/manifest.json").exists()


def test_local_reference_gate_records_bitwise_equivalence():
    root = Path(__file__).resolve().parents[2]
    path = root / "data/cache/derived/pallier2025/provenance/preprocessing_reference.json"
    if not path.exists():
        pytest.skip("本机尚未运行 Pallier2025 真实 recording reference gate。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["recording_id"] == "pallier2025|sub-01|run-01"
    assert payload["comparison"]["bitwise_equal"] is True
    assert payload["comparison"]["different_element_count"] == 0
    assert payload["comparison"]["max_abs_error"] == 0.0
    assert payload["channel_names_sha256"] == pallier2025.CHANNEL_NAMES_SHA256
    assert payload["model_or_checkpoint_loaded"] is False
    assert payload["model_evaluation_performed"] is False


def test_local_time_origin_audit_uses_three_train_recordings():
    root = Path(__file__).resolve().parents[2]
    path = root / "data/cache/derived/pallier2025/provenance/time_origin_audit.json"
    if not path.exists():
        pytest.skip("本机尚未运行 Pallier2025 first_samp 审计。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload["train_recording_examples"]
    assert len(examples) == 3
    assert [item["source_first_samp"] for item in examples] == sorted(
        item["source_first_samp"] for item in examples
    )
    assert payload["mne_time_as_index_agrees"] is True
    negative = payload["sub-09_run-03_negative_boundary"]
    assert negative["recording_excluded_from_training"] is True
    assert all(
        item["canonical_output_sample_index"] < 0
        for item in negative["anomalous_words"]
    )


def test_local_complete_pallier_signals_and_qc_boundary(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "data/cache/derived/pallier2025/signals/manifest.json"
    event_path = root / "data/cache/derived/pallier2025/events/events.csv"
    if not manifest_path.exists():
        pytest.skip("本机尚未完成 Pallier2025 90 条 signals。")
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    table = pallier2025.load_event_table(event_path)
    payload = pallier2025.validate_signal_manifest(
        manifest_path, event_table=table
    )
    assert payload["status"] == "complete"
    assert payload["recording_count"] == 90
    assert payload["coverage"] == {
        "expected_recordings": 90,
        "signal_products": 90,
        "missing": 0,
        "extra": 0,
        "channel_count": 306,
        "trainable_windows_in_range": True,
    }
    assert payload["test_neural_data_status"] == (
        "raw_accessed_for_deterministic_preprocessing_only"
    )
    assert payload["test_model_evaluation"] == "not_run"
    assert payload["test_predictions_generated"] is False
    assert payload["test_metrics_inspected"] is False
    excluded = next(
        item
        for item in payload["recordings"]
        if item["recording_id"] == "pallier2025|sub-09|run-03"
    )
    assert excluded["recording_signal_status"] == "valid"
    assert excluded["event_training_status"] == "excluded_by_annotation_qc"
