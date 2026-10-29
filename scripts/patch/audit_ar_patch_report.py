#!/usr/bin/env python3
"""Check a standalone AR patch report against its protocol and runner records."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote

from PIL import Image

if __package__:
    from .summarize_ar_patch_rollout import normalize_trajectory, validate_matched_intervention
else:
    from summarize_ar_patch_rollout import normalize_trajectory, validate_matched_intervention


def audit(root, allow_partial=False, workers=16):
    data = json.loads((root / "report_data.json").read_text())
    document = (root / "report.html").read_text()
    embedded = json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>', document, re.S).group(1))
    assert embedded == data, "HTML embeds a different report snapshot"
    assert 'id="patch-id"' not in document and 'id="patch-go"' not in document
    assert 'id="selection-mode"' in document and 'id="selector-hover"' in document
    protocol = data["protocol"]
    steps = protocol["steps"]
    scales = protocol["sigma_scales"]
    assert data["steps"] == steps and data["scales"] == scales
    expected = len(protocol["regions"]) * len(steps) * len(scales)
    assert len(data["branches"]) == data["completed"]
    assert data["expected"] == expected == protocol["expected_branches"]
    if not allow_partial:
        assert data["completed"] == expected and data["complete"]
        assert len(data["posters"]) == len(steps) * len(protocol["regions"])

    def file(href):
        path = root / unquote(href)
        assert path.is_file(), f"Missing link: {path}"
        return path

    def media(owner):
        for name, descriptor in owner["media"].items():
            path = file(descriptor["url"])
            with Image.open(path) as image:
                assert image.size == (descriptor["columns"] * descriptor["width"],
                                      math.ceil(descriptor["count"] / descriptor["columns"]) * descriptor["height"]), path
            expected_frames = (sum(protocol["chunks"][item["chunk"]] for item in owner["trajectory"]) if name.endswith("rms") else
                               len(owner["trajectory"]) if name.endswith("pixels") else
                               protocol["shape"][1] if name == "latent_mse" else data["display"]["frames"])
            assert descriptor["count"] == expected_frames, (name, descriptor["count"], expected_frames)
        for name in ("video_url", "record_url", "latent_url"):
            if owner.get(name):
                file(owner[name])

    baseline_keys = [(item["chunk"], item["step"]) for item in data["baseline"]["trajectory"]]
    expected_keys = [(chunk, step) for chunk in range(len(protocol["chunks"]))
                     for step in range(len(protocol["schedule"]))]
    assert baseline_keys == expected_keys
    media(data["baseline"])

    def branch(item):
        key, value = item
        media(value)
        record = json.loads(file(value["record_url"]).read_text())
        if protocol.get("intervention_mode") == "matched":
            validate_matched_intervention(protocol, record)
            assert value["local_timestep"] == record["local_timestep"], key
            assert value["intervention_mode"] == "matched" and value["timestep_policy"] == "source_call_only", key
            assert value["trajectory"] == normalize_trajectory(record["trajectory"], matched=True), key
        assert value["step"] in steps
        assert value["metrics"]["pixel_rmse"] == math.sqrt(record["pixel"]["all_video_mse"]), key
        for name in ("next_chunk_mse", "normalized_next_chunk_energy"):
            assert value["metrics"][name] == record["rollout"][name], (key, name)
        assert value["metrics"]["input_energy"] == record["one_step"]["input_energy"], key
        trajectory_keys = [(entry["chunk"], entry["step"]) for entry in value["trajectory"]]
        start = expected_keys.index((protocol["source_chunk"], value["step"]))
        assert trajectory_keys == expected_keys[start:], key
        return len(value["media"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        media_count = sum(pool.map(branch, data["branches"].items())) + len(data["baseline"]["media"])
    for href in data["static_heatmaps"].values():
        file(href)
    for href in data["posters"].values():
        path = file(href)
        expected_prefix = f"prompt_{protocol['prompt_index']:06d}_seed_{protocol['seed']:06d}_chunk_{protocol['source_chunk']}_"
        assert path.name.startswith(expected_prefix), path
    assert len(data["static_heatmaps"]) == len(steps) * len(scales) * len(data["metrics"])
    file(data["manifest_url"])
    if data.get("navigation"):
        file(data["navigation"]["overview_url"])
        for item in data["navigation"]["cases"]:
            file(item["url"])
        assert any(item["prompt_index"] == protocol["prompt_index"] and item["seed"] == protocol["seed"]
                   for item in data["navigation"]["cases"])
    return {"status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_index": protocol["prompt_index"], "seed": protocol["seed"], "steps": steps,
            "completed": data["completed"], "expected": expected, "complete": data["complete"],
            "checked_atlas_images": media_count, "checked_posters": len(data["posters"]),
            "checked_heatmaps": len(data["static_heatmaps"]),
            "checks": ["embedded_index_matches_json", "no_numeric_patch_selector", "all_link_targets_exist",
                       "atlas_dimensions_and_frame_counts", "float_metrics_match_runner_records",
                       "trajectory_alignment", "protocol_derived_coverage", "prompt_seed_poster_labels", "prompt_navigation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.report_dir, args.allow_partial, args.workers)
    destination = args.output or args.report_dir / "report_audit.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
