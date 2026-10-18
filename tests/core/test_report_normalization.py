import hashlib
import json
from types import SimpleNamespace

import pytest

from braindecoding import report_layout as layout


def digest(content):
    return hashlib.sha256(content).hexdigest()


def legacy_figure(root, study="example"):
    destination = root / "reports/studies" / study
    destination.mkdir(parents=True, exist_ok=True)
    report = destination / "results.json"
    report.write_bytes(b'{"metric": 0.125}\n')
    folder = root / "reports/figures" / study
    folder.mkdir(parents=True, exist_ok=True)
    media = folder / "chart.png"
    media.write_bytes(b"existing figure bytes")
    source = folder / "sources.json"
    payload = {"report": {"path": report.relative_to(root).as_posix(), "sha256": digest(report.read_bytes())},
               "implementation_sha256": "original implementation", "test_evaluated": False,
               "files": [{"path": media.relative_to(root).as_posix(), "sha256": digest(media.read_bytes())}]}
    source.write_text(json.dumps(payload), encoding="utf-8")
    return folder, destination


def test_migration_preserves_media_results_and_original_provenance(tmp_path):
    folder, destination = legacy_figure(tmp_path)
    report_bytes = (destination / "results.json").read_bytes()
    original_source = (folder / "sources.json").read_bytes()
    original_markdown = b"Results: 12.5%\n![chart](../../figures/example/chart.png)\n"
    (destination / "results.md").write_bytes(original_markdown)

    result = layout.normalize_legacy_figures(tmp_path)

    assert (destination / "chart.png").read_bytes() == b"existing figure bytes"
    assert (destination / "results.json").read_bytes() == report_bytes
    assert (destination / "results.md").read_bytes() == b"Results: 12.5%\n![chart](chart.png)\n"
    source = json.loads((destination / "sources.json").read_text())
    assert source["implementation_sha256"] == "original implementation"
    assert source["test_evaluated"] is False
    assert source["files"] == [{"path": "reports/studies/example/chart.png",
                                "original_path": "reports/figures/example/chart.png",
                                "sha256": digest(b"existing figure bytes")}]
    originals = tmp_path / result["audit"] / "originals"
    assert (originals / "reports/figures/example/sources.json").read_bytes() == original_source
    assert (originals / "reports/studies/example/results.md").read_bytes() == original_markdown
    assert not (tmp_path / "reports/figures").exists()
    assert layout.check_layout(tmp_path) == []
    assert layout.normalize_legacy_figures(tmp_path) == {"moved_files": [], "updated_files": [], "audit": None}


def test_conflicts_are_checked_before_any_files_move(tmp_path):
    first, _ = legacy_figure(tmp_path, "first")
    second, destination = legacy_figure(tmp_path, "second")
    (destination / "chart.png").write_bytes(b"keep existing figure")
    with pytest.raises(FileExistsError):
        layout.normalize_legacy_figures(tmp_path)
    assert (first / "chart.png").exists() and (second / "chart.png").exists()
    assert (destination / "chart.png").read_bytes() == b"keep existing figure"
    assert not (tmp_path / ".cache").exists()


def test_modified_media_is_rejected_without_rewriting_its_source(tmp_path):
    folder, destination = legacy_figure(tmp_path)
    source = (folder / "sources.json").read_bytes()
    (folder / "chart.png").write_bytes(b"unexpected changed bytes")
    with pytest.raises(ValueError, match="哈希"):
        layout.normalize_legacy_figures(tmp_path)
    assert (folder / "sources.json").read_bytes() == source
    assert not (destination / "chart.png").exists()
    assert not (tmp_path / ".cache").exists()


def test_unknown_artifacts_are_not_moved_or_removed(tmp_path):
    folder, destination = legacy_figure(tmp_path)
    (folder / "checkpoint.pt").write_bytes(b"do not move")
    with pytest.raises(ValueError, match="未知产物"):
        layout.normalize_legacy_figures(tmp_path)
    assert (folder / "checkpoint.pt").read_bytes() == b"do not move"
    assert not (destination / "chart.png").exists()


def test_only_markdown_links_can_be_normalized_again(tmp_path):
    destination = tmp_path / "reports/studies/example"
    destination.mkdir(parents=True)
    (destination / "chart.png").write_bytes(b"already moved")
    (destination / "results.md").write_bytes(b"![chart](../../figures/example/chart.png)\n")
    result = layout.normalize_legacy_figures(tmp_path)
    assert result["moved_files"] == []
    assert (destination / "results.md").read_bytes() == b"![chart](chart.png)\n"


def test_missing_image_is_rejected_before_report_links_are_changed(tmp_path):
    folder, destination = legacy_figure(tmp_path)
    original = b"![missing](../../figures/example/not_present.png)\n"
    (destination / "results.md").write_bytes(original)
    with pytest.raises(ValueError, match="图表未找到"):
        layout.normalize_legacy_figures(tmp_path)
    assert (destination / "results.md").read_bytes() == original
    assert (folder / "chart.png").exists()


def test_check_is_read_only_and_reports_source_damage(tmp_path):
    legacy_figure(tmp_path)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert layout.check_layout(tmp_path)
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    layout.normalize_legacy_figures(tmp_path)
    (tmp_path / "reports/studies/example/chart.png").write_bytes(b"modified")
    assert any("来源无法核对" in problem for problem in layout.check_layout(tmp_path))


def test_render_only_calls_the_selected_plotter_and_replaces_its_figures(tmp_path, monkeypatch):
    study = "unified_sentence_v2_revision_gain_full"
    destination = tmp_path / "reports/studies" / study
    destination.mkdir(parents=True)
    payload = {"status": "completed_full_revision_gain", "scientific_value": 0.125}
    path = destination / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report_bytes = path.read_bytes()
    (destination / "chart.png").write_bytes(b"old figure")
    calls = []

    def plot(report):
        assert report == payload
        calls.append("render")
        folder = tmp_path / "reports/figures" / study
        (folder / "chart.png").write_bytes(b"redrawn figure")
        (folder / "sources.json").write_text(json.dumps({"files": [{
            "path": f"reports/figures/{study}/chart.png", "sha256": digest(b"redrawn figure")}]}), encoding="utf-8")

    def import_plotter(name):
        assert name == "braindecoding.unified_sentence_v2_revision_gain_full_results"
        return SimpleNamespace(render=plot)

    monkeypatch.setattr(layout, "import_module", import_plotter)
    result = layout.render_figures("revision_gain_full", root=tmp_path)
    assert calls == ["render"]
    assert path.read_bytes() == report_bytes
    assert (destination / "chart.png").read_bytes() == b"redrawn figure"
    assert not (tmp_path / "reports/figures").exists()
    assert (tmp_path / result["audit"] / "originals" / destination.relative_to(tmp_path) / "chart.png").read_bytes() == b"old figure"


def test_incomplete_report_cannot_trigger_rendering(tmp_path, monkeypatch):
    destination = tmp_path / "reports/studies/unified_sentence_v2_revision_gain_full"
    destination.mkdir(parents=True)
    (destination / "results.json").write_text('{"status": "partial"}', encoding="utf-8")
    monkeypatch.setattr(layout, "import_module", lambda name: pytest.fail("Incomplete report imported a plotter"))
    with pytest.raises(ValueError, match="已完成"):
        layout.render_figures("revision_gain_full", root=tmp_path)
    assert not (tmp_path / "reports/figures").exists()
