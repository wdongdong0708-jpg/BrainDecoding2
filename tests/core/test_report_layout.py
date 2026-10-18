from pathlib import Path

import pytest

from braindecoding.reporting import (
    AUDIT_CACHE_ROOT,
    REPORT_AUDITS_ROOT,
    REPORT_EXPORTS_ROOT,
    REPORT_PROTOCOLS_ROOT,
    REPORT_STUDIES_ROOT,
    REPORTS_ROOT,
    RUNTIME_CACHE_ROOT,
    runtime_cache_dir,
    study_report_dir,
)
from braindecoding.report_layout import check_layout


def test_report_root_has_only_contract_and_known_sections():
    assert {path.name for path in REPORTS_ROOT.iterdir() if path.is_file()} == set()
    assert {path.name for path in REPORTS_ROOT.iterdir() if path.is_dir()} <= {
        "assessments",
        "audits",
        "exports",
        "protocols",
        "studies",
        "tables",
    }


def test_reports_contain_no_runtime_or_model_artifacts():
    forbidden_names = {"best.pt", "last.pt", "run.pid"}
    forbidden_suffixes = {".log", ".pid", ".pt", ".tmp"}
    exempt_studies = {"fixed_grid_ctc_pallier_seed000_v1", "noise_seed_stability_v1"}
    for path in REPORTS_ROOT.rglob("*"):
        if any(part in exempt_studies for part in path.parts):
            continue
        assert path.name != "feature_cache"
        if path.is_file():
            assert path.name not in forbidden_names
            assert path.suffix not in forbidden_suffixes


def test_reporting_paths_are_partitioned_by_lifecycle():
    assert study_report_dir(None, "unified_sentence_v2_preview") == (
        REPORT_STUDIES_ROOT / "unified_sentence_v2_preview"
    )
    assert study_report_dir("pallier2025", "local_window") == (
        REPORT_STUDIES_ROOT / "pallier2025" / "local_window"
    )
    assert REPORT_AUDITS_ROOT == REPORTS_ROOT / "audits"
    assert REPORT_EXPORTS_ROOT == REPORTS_ROOT / "exports"
    assert REPORT_PROTOCOLS_ROOT == REPORTS_ROOT / "protocols"
    assert AUDIT_CACHE_ROOT == REPORTS_ROOT.parents[1] / ".cache" / "audits"
    assert runtime_cache_dir("example_job") == RUNTIME_CACHE_ROOT / "example_job"


@pytest.mark.parametrize("value", ("", ".", "..", "a/b", "a\\b", "C:\\tmp"))
def test_reporting_paths_reject_unsafe_segments(value):
    with pytest.raises(ValueError):
        study_report_dir(value, "study")
    with pytest.raises(ValueError):
        study_report_dir(None, value)


def test_report_figures_and_sources_follow_the_study_layout():
    assert check_layout() == []
