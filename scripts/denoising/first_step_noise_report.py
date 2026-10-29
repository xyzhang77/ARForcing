"""Render the local, self-contained first-step noise comparison report."""

from __future__ import annotations

import html
import json
from pathlib import Path


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_report(root, summary, manifest, sheets):
    """Write report.html using validated summary data and the image manifest."""
    root = Path(root)
    assets = Path(__file__).with_name("first_step_noise_report_assets")
    payload = {
        "summary": summary,
        "manifest": manifest,
        "sheets": sheets,
        "assessment": _read_json(root / "diversity_assessment.json"),
        "qualitative": _read_json(root / "qualitative_observations.json"),
        "validation": _read_json(root / "validation.json"),
        "latent": _read_json(root / "latent_summary.json"),
        "downloads": [
            {"path": name, "label": label}
            for name, label in (
                ("summary.csv", "RGB 指标 CSV"),
                ("summary.json", "完整统计 JSON"),
                ("diversity_assessment.json", "Diversity 评估 JSON"),
                ("latent_summary.json", "Latent 统计 JSON"),
                ("validation.json", "实验校验 JSON"),
                ("frozen_config.yaml", "冻结配置 YAML"),
            )
            if (root / name).is_file()
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = (assets / "report.html").read_text(encoding="utf-8")
    document = template.replace("__REPORT_CSS__", (assets / "report.css").read_text(encoding="utf-8"))
    document = document.replace("__REPORT_DATA__", encoded)
    document = document.replace("__REPORT_JS__", (assets / "report.js").read_text(encoding="utf-8"))
    if (root / "denoising_steps" / "report.html").is_file():
        document = document.replace(
            '<nav aria-label="报告目录">',
            '<nav aria-label="报告目录"><a id="denoising-steps-report-link" '
            'href="denoising_steps/report.html">四步变化分析</a>',
            1,
        )
    # A useful error-free reading experience is retained when scripts are disabled.
    document = document.replace("__CASE_COUNT__", html.escape(str(summary.get("n_completed_cases", "—"))))
    temporary = root / "report.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(root / "report.html")
