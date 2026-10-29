"""CPU coverage of legacy and source-call-only noise-level-aligned report contracts."""
from pathlib import Path

import numpy as np
import pytest

from scripts.patch import summarize_ar_patch_rollout as report
from scripts.patch.audit_ar_patch_report import audit
from scripts.patch.summarize_ar_patch_aligned_index import write_index


def make_case(root, matched=False):
    data, output = root / "data", root / "report"
    data.mkdir(parents=True)
    output.mkdir()
    regions = [{"id": row * 2 + col, "row": row, "col": col, "y0": row * 4, "y1": (row + 1) * 4,
                "x0": col * 4, "x1": (col + 1) * 4} for row in range(2) for col in range(2)]
    protocol = {"prompt_index": 0, "seed": 0, "prompt": "synthetic report regression", "grid_rows": 2, "grid_cols": 2,
                "regions": regions, "shape": [1, 6, 16, 8, 8], "chunks": [3, 3], "steps": [1],
                "source_chunk": 0, "source_frame": 1, "sigma_scales": [.75, .5, .25, 0.],
                "schedule": [1000., 937.5, 750., 500.], "sigmas": [1., .9375, .75, .5], "expected_branches": 16}
    if matched:
        protocol.update(intervention_mode="matched", timestep_policy="source_call_only")
    report.atomic_json(data / "manifest.json", {"protocol": protocol, "protocol_sha256": "fixture"})
    coordinates = [(chunk, step) for chunk in range(2) for step in range(4)]

    def save(directory, source=None, scale=0):
        directory.mkdir(parents=True)
        branch = source is not None
        entries = [{"chunk_index": chunk, "step_index": step, "global_t": protocol["schedule"][step],
                    "sigma": protocol["sigmas"][step], "preview_frame_index": 4} for chunk, step in coordinates[int(branch):]]
        pixels = np.broadcast_to(np.arange(5, dtype=np.uint8)[:, None, None, None] * 15 + 45 + (source + 1 if branch else 0), (5, 16, 16, 3)).copy()
        np.savez_compressed(directory / "final_pixels.npz", pixels=pixels)
        maps = np.broadcast_to(np.arange(len(entries), dtype=np.float32)[:, None, None, None] / 100 + np.arange(3, dtype=np.float32)[None, :, None, None] / 1000, (len(entries), 3, 8, 8)).copy()
        np.savez_compressed(directory / "trajectory_maps.npz", noisy_rms=maps, clean_rms=maps + .1,
                            noisy_delta_rms=np.full_like(maps, .01 if branch else 0), clean_delta_rms=np.full_like(maps, .02 if branch else 0))
        previews = np.full((len(entries), 16, 16, 3), 80 + 5 * branch, np.uint8)
        np.savez_compressed(directory / "trajectory_preview.npz", noisy=previews, clean=previews + 30)
        record = {"protocol_sha256": "fixture", "baseline_sha256": "baseline", "trajectory": entries,
                  "artifacts": {"pixels_npz": "final_pixels.npz", "trajectory_maps_npz": "trajectory_maps.npz", "trajectory_preview_npz": "trajectory_preview.npz"}}
        if branch:
            sigma = .9375 * scale
            record.update(step_index=1, source_id=source, sigma_scale=scale, source_sigma=sigma, base_sigma=.9375, global_t=937.5,
                          pixel={"all_video_mse": .04}, one_step={"input_energy": 3.},
                          rollout={"next_chunk_mse": .09, "normalized_next_chunk_energy": 1.1, "input_energy": 3.})
            np.savez_compressed(directory / "impact_maps.npz", latent_mse=np.full((6, 8, 8), .09, np.float32), pixel_mse_downsample=np.full((5, 2, 2), .04, np.float32))
            record["artifacts"]["impact_maps_npz"] = "impact_maps.npz"
            if matched:
                record.update(intervention_mode="matched", timestep_policy="source_call_only", local_timestep=sigma * 1000)
                entries[0].update(mode="matched", source_sigma=sigma, source_frame=1, region=regions[source],
                                  local_timestep=sigma * 1000, timestep_policy="source_call_only")
        report.atomic_json(directory / "record.json", record)
    save(data / "baseline")
    for source, scale in [(0, .75), (0, 0), (3, 0)]:
        save(data / "step_1" / f"source_{source:03d}" / f"scale_{report.scale_key(scale).replace('.', 'p')}", source, scale)
    return data, output


def render(data, output):
    summary, pixels = report.collect(data, output, workers=1)
    report.write_heatmaps(summary, output)
    report.write_posters(summary, pixels, output)
    report.write_html(summary, output)
    return summary


@pytest.mark.parametrize("matched", [False, True], ids=["legacy-noise-only", "source-call-aligned"])
def test_report_preserves_mode_metrics_missing_and_trace_prefix(tmp_path, matched):
    data, output = make_case(tmp_path, matched)
    summary = render(data, output)
    assert summary["completed"] == 3 and summary["expected"] == 16 and not summary["complete"]
    assert "1:1:0" not in summary["branches"]
    branch = summary["branches"]["1:0:0.75"]
    assert branch["metrics"] == {"pixel_rmse": .2, "pixel_mae": None, "next_chunk_mse": .09, "normalized_next_chunk_energy": 1.1, "input_energy": 3.}
    assert [(entry["chunk"], entry["step"]) for entry in branch["trajectory"]] == [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    assert branch["media"]["noisy_rms"]["count"] == 21 and branch["media"]["noisy_pixels"]["count"] == 7
    if matched:
        assert branch["local_timestep"] == 703.125
        assert branch["trajectory"][0]["mode"] == "matched"
        assert all("local_timestep" not in entry for entry in branch["trajectory"][1:])
    else:
        assert "local_timestep" not in branch
        assert all("mode" not in entry for entry in branch["trajectory"])
    assert len(summary["static_heatmaps"]) == 12 and len(summary["posters"]) == 2
    assert audit(output, allow_partial=True, workers=1)["status"] == "passed"


@pytest.mark.parametrize("fault", ["policy", "wrong_timestep", "wrong_frame", "wrong_region", "later_override", "missing_override"])
def test_rejects_mislabelled_matched_conditioning(tmp_path, fault):
    data, _ = make_case(tmp_path, True)
    protocol = report.load_json(data / "manifest.json")["protocol"]
    record = report.load_json(data / "step_1/source_000/scale_0p75/record.json")
    if fault == "policy":
        protocol["timestep_policy"] = "all_calls"
    elif fault == "wrong_timestep":
        record["local_timestep"] = 937.5
    elif fault == "wrong_frame":
        record["trajectory"][0]["source_frame"] = 2
    elif fault == "wrong_region":
        record["trajectory"][0]["region"]["x0"] = 4
    elif fault == "later_override":
        record["trajectory"][1]["local_timestep"] = 703.125
    else:
        record["trajectory"][0].pop("mode")
    with pytest.raises(ValueError):
        report.validate_matched_intervention(protocol, record)


def test_only_explicit_protocol_marker_enables_metadata(tmp_path):
    data, output = make_case(tmp_path, matched=True)
    path = data / "manifest.json"
    manifest = report.load_json(path)
    manifest["protocol"].pop("intervention_mode")
    report.atomic_json(path, manifest)
    summary, _ = report.collect(data, output, workers=1)
    assert "local_timestep" not in summary["branches"]["1:0:0.75"]
    assert "mode" not in summary["branches"]["1:0:0.75"]["trajectory"][0]


def test_aligned_multi_prompt_navigation_and_legacy_rejection(tmp_path):
    data, output = make_case(tmp_path / "cases/prompt_000000_seed_000000", matched=True)
    cases = [{"prompt_index": prompt, "seed": 0, "prompt": f"fixture {prompt}", "data_dir": f"cases/prompt_{prompt:06d}_seed_000000/data",
              "report": f"cases/prompt_{prompt:06d}_seed_000000/report/report.html", "expected_branches": 16} for prompt in (0, 42, 158, 300)]
    report.atomic_json(tmp_path / "experiment.json", {"cases": cases})
    write_index(tmp_path)
    render(data, output)
    write_index(tmp_path)
    assert audit(output, allow_partial=True, workers=1)["status"] == "passed"
    index = report.load_json(tmp_path / "report_index.json")
    assert index["completed"] == 3 and index["expected"] == 64 and len(index["cases"]) == 4
    assert "仅 source 调用" in (tmp_path / "report.html").read_text()
    assert "timestep 条件变化" in (tmp_path / "report.html").read_text()
    for name in ("paired_comparison.json", "audit_summary.json", "completion_validation.json"):
        report.atomic_json(tmp_path / name, {"status": "passed"})
    report.atomic_json(tmp_path / "experiment.json", {"cases": cases, "status": "cancelled"})
    write_index(tmp_path)
    document = (tmp_path / "report.html").read_text()
    assert 'href="paired_comparison.json"' in document and 'href="completion_validation.json"' in document
    assert "后台生成已停止" in document and "后台持续生成" not in document
    path = output / "report_data.json"
    summary = report.load_json(path)
    summary["protocol"].pop("intervention_mode")
    report.atomic_json(path, summary)
    with pytest.raises(ValueError, match="another intervention policy"):
        write_index(tmp_path)


def test_pilot_cli_keeps_protocol_coverage(tmp_path, monkeypatch):
    data, output = make_case(tmp_path, matched=True)
    monkeypatch.setattr("sys.argv", ["report", "--input-dir", str(data), "--output-dir", str(output), "--workers", "1", "--pilot"])
    report.main()
    summary = report.load_json(output / "report_data.json")
    assert summary["report_context"] == {"phase": "pilot", "source_ids": [0, 3], "full_scan_url": "../report.html"}
    assert summary["completed"] == 3 and summary["expected"] == 16 and summary["complete"] is False
    assert audit(output, allow_partial=True, workers=1)["status"] == "passed"
