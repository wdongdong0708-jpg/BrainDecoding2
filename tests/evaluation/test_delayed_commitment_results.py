"""Exercise the final report contract in a temporary synthetic test workspace."""
import copy
import json

import pytest

from braindecoding.unified_sentence_v2 import settings


def test_complete_population_export_and_missing_group_rejection(tmp_path, monkeypatch):
    import braindecoding.unified_sentence_v2 as runner
    import braindecoding.unified_sentence_v2_results as reports
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUT", tmp_path / "outputs")
    monkeypatch.setattr(reports, "PROJECT_ROOT", tmp_path)
    cfg = settings()
    cfg["statistics"]["material_block_bootstrap_repetitions"] = 1000
    cfg["controls"]["uniform_random_repetitions"] = 10
    groups = [dict(group=f"s{subject}-b{block}", subject=f"s{subject}", material=f"m{block}",
                   block=f"b{block}", scope="r", partition="evaluation", reference=["x"])
              for subject in range(2) for block in range(4)]
    m = dict(settings=cfg, datasets={d: dict(language=lang, vocabulary=["x", "y"], groups=groups)
             for d, lang in (("fr_data", "fr"), ("zh_data", "zh"))}, manifest_sha256="fixture")
    for dataset in m["datasets"]:
        root = runner.OUT / dataset
        runner.save(root / "completion.json", dict(status="completed", manifest_sha256="fixture"))
        runner.save(root / "calibration.json", dict(selected_weight=.1))
        runner.save(root / "pure_language.json", dict(manifest_sha256="fixture", no_brain_shortlist=True,
            result=dict(trace=[dict(arrival=0, C=["y"])])))
        for seed in range(3):
            for branch in ("open", "closed"):
                for alignment in ("real", "shift-00", "shift-01", "shift-02"):
                    cword = "x" if alignment == "real" else "y"
                    result = dict(final=dict(A=["y"], B=["y"], C=[cword]),
                                  trace=[dict(arrival=0, A=["y"], B=["y"], C=[cword])])
                    runner.save(root / f"seed-{seed:03d}" / branch / alignment / "chunk-0000.json",
                        dict(manifest_sha256="fixture", test_evaluated=False,
                             records=[dict(group=g["group"], result=result) for g in groups]))
    reports.export(m)
    completed = runner.read(tmp_path / "data/results/studies/unified_sentence_v2/results.json")
    assert completed["status"] == "completed" and completed["test_evaluated"] is False
    assert len(completed["primary_statistics"]) == 6
    assert all(r["effect"] == 1. and r["blocks"] == 4 for r in completed["primary_statistics"])
    row = next(r for r in completed["summary"] if r["alignment"] == "real" and r["condition"] == "C")
    assert row["sentence_bleu1_mean_mean"] == 1. and row["wer_mean"] == 0.
    missing = runner.OUT / "fr_data/seed-000/open/real/chunk-0000.json"
    chunk = runner.read(missing)
    chunk["records"].pop()
    runner.save(missing, chunk)
    with pytest.raises(ValueError, match="population incomplete"):
        reports.export(m)
