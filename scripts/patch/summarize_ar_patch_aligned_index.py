#!/usr/bin/env python3
"""Publish the independent multi-prompt index for noise-level-aligned AR scans."""
import argparse
from datetime import datetime, timezone
import html
from pathlib import Path

from scripts.patch.summarize_ar_patch_rollout import atomic_json, load_json, url


LABELS = {0: "基准场景", 42: "沙漠绿洲", 158: "人物行进", 300: "猫喝水"}


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_index(root):
    experiment = load_json(root / "experiment.json")
    stopped = experiment.get("status") in ("cancelled", "failed", "error", "stopped")
    cases = experiment["cases"]
    statuses, cards = [], []
    for case in cases:
        report = root / case["report"]
        report.parent.mkdir(parents=True, exist_ok=True)
        navigation = {"overview_url": url(root / "report.html", report.parent), "cases": [
            {"prompt_index": other["prompt_index"], "seed": other.get("seed", 0),
             "label": f"Prompt {other['prompt_index']} · {LABELS.get(other['prompt_index'], '生成样例')}",
             "url": url(root / other["report"], report.parent)} for other in cases]}
        atomic_json(report.parent.parent / "report_navigation.json", navigation)
        data_path = report.parent / "report_data.json"
        data = load_json(data_path) if data_path.is_file() else None
        if data and (data["protocol"].get("intervention_mode") != "matched" or data["protocol"].get("timestep_policy") != "source_call_only"):
            raise ValueError(f"Aligned index cannot include another intervention policy: {report}")
        count = data["completed"] if data else 0
        expected = data["expected"] if data else int(case.get("expected_branches", case.get("expected_full_rollouts", 768)))
        statuses.append({"prompt_index": case["prompt_index"], "seed": case.get("seed", 0), "completed": count,
                         "expected": expected, "complete": count == expected, "report": case["report"]})
        preview = root / case["data_dir"] / "baseline" / "final.jpg"
        if not preview.is_file():
            preview = report.parent / "media" / "baseline" / "last.jpg"
        picture = f'<img src="{url(preview, root)}" alt="baseline" loading="lazy">' if preview.is_file() else '<div class="placeholder">等待 baseline</div>'
        prompt_index = case["prompt_index"]
        old = case.get("noise_only_report")
        if old is None:
            previous = (root.parent / "ar_patch_rollout_20261022/report/report.html" if prompt_index == 0 else
                        root.parent / "ar_patch_rollout_step1_multiprompt_20261022/cases" / f"prompt_{prompt_index:06d}_seed_{case.get('seed', 0):06d}" / "report/report.html")
            old = url(previous, root) if previous.is_file() else None
        old_link = f'<p class="note"><a href="{html.escape(old)}" target="_blank" rel="noopener">旧结果：仅改噪声 ↗</a> · 比较时选择 source_step 1</p>' if old else ""
        prompt = data["prompt"] if data else case.get("prompt", "")
        cards.append(f'<article><a class="preview" href="{html.escape(case["report"])}">{picture}</a><div class="content"><span class="tag">PROMPT {prompt_index} · SEED {case.get("seed",0)}</span><h2>{LABELS.get(prompt_index,"生成样例")}</h2><p class="prompt">{html.escape(prompt)}</p><p>{count:,} / {expected:,} 个完整生成分支已发布</p><a class="button" href="{html.escape(case["report"])}">点击画面选择 patch →</a>{old_link}</div></article>')
        if not report.is_file():
            write_text(report, f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AR · Noise-level aligned</title><body style="font:17px system-ui;padding:40px"><a href="{navigation["overview_url"]}">← 多 prompt 总览</a><h1>Prompt {prompt_index} · 噪声等级对齐</h1><p>等待 baseline 与第一批结果。</p><button onclick="location.reload()">刷新结果</button></body>')
    total, expected = sum(item["completed"] for item in statuses), sum(item["expected"] for item in statuses)
    complete = total == expected
    options = '<option value="">切换 prompt…</option>' + ''.join(f'<option value="{html.escape(case["report"])}">Prompt {case["prompt_index"]} · {LABELS.get(case["prompt_index"],"生成样例")}</option>' for case in cases)
    css = '*{box-sizing:border-box}body{margin:0;background:#f2f6f5;color:#16373b;font:15px/1.65 system-ui,sans-serif}main{max-width:1380px;margin:auto;padding:36px 28px}h1{font-size:32px;line-height:1.3;margin:12px 0}h2{font-size:23px;margin:5px 0}p{margin:9px 0}.tag{font-size:11px;letter-spacing:1px;color:#087f7a;font-weight:bold}.intro,.note{color:#63787b}.toolbar{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:22px 0}select,button{font:inherit;padding:8px;border:1px solid #bcd0cb;border-radius:7px;background:white;color:inherit}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}article{background:white;border:1px solid #dce7e5;border-radius:13px;overflow:hidden}.preview{display:block;aspect-ratio:832/480;background:#e6efeb}.preview img{width:100%;height:100%;object-fit:cover}.placeholder{display:grid;place-items:center;height:100%;color:#7b8e88}.content{padding:20px}.prompt,.note,footer{font-size:12px}.prompt{min-height:55px;color:#63787b}a{color:#087f7a;text-decoration:none}a:hover{text-decoration:underline}.button{display:inline-block;padding:8px 12px;border-radius:6px;background:#087f7a;color:white}footer{margin-top:24px;padding-top:16px;border-top:1px solid #dce7e5;color:#63787b}@media(max-width:760px){.grid{grid-template-columns:1fr}main{padding:24px 16px}h1{font-size:27px}.prompt{min-height:0}}'
    document = f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AR Patch Atlas · Noise-level aligned</title><style>{css}</style><body><main><div class="tag">SELF-FORCING · CHECKPOINT 600 · SOURCE_STEP 1 · NOISE-LEVEL ALIGNED</div><h1>逐 patch 扫描 · 噪声等级与 timestep 对齐</h1><p class="intro">选择场景，点击画面中的 patch，对比完整生成的原始结果、干预结果与像素差，并查看各 step 的加噪输入和预测。</p><p class="intro">仅 source 调用中，将选中 patch 的实际 σ、token timestep（σ × 1000）和 flow→x0 的 sigma_map 同步对齐；背景 token 保留原值，后续调用按原 schedule 继续。</p><div class="toolbar"><strong>{"全部发布" if complete else "已发布"} · {total:,} / {expected:,}</strong><select id="prompt-picker">{options}</select>'
    if stopped:
        document += f'<span class="note">任务状态：{html.escape(experiment["status"])}。后台生成已停止，已发布结果保留。</span>'
    elif not complete:
        document += '<button onclick="location.reload()">刷新结果 ↻</button><span class="note">后台持续生成，刷新页面可查看新结果。</span>'
    summaries = [f'<a href="{name}" target="_blank" rel="noopener">{label} ↗</a>' for name, label in
                 (("paired_comparison.json", "新旧干预成对比较 JSON"), ("audit_summary.json", "数据完整性审计"),
                  ("completion_validation.json", "流水线完成验证"),
                  ("aligned_comparison_summary.json", "新旧干预对照 JSON"), ("aligned_comparison_summary.txt", "新旧干预对照摘要"),
                  ("cross_prompt_summary.json", "跨 prompt 统计 JSON"), ("report_validation.json", "报告完整性验证"),
                  ("final_report_validation.json", "报告完整性验证")) if (root / name).is_file()]
    document += '</div><div class="grid">' + ''.join(cards) + '</div><footer><p>source_step 1 是第 2 次去噪。σ × 0 保留前一步模型预测，不是真值或删除信息。每个场景 192 patches × 4 强度，21 帧完整生成；本轮共 4 prompts、每个 1 seed。</p><p>旧结果只改变 latent 噪声，仍使用全局 timestep。两种设定使用独立结果目录；新旧变化不代表生成质量提升。归一化响应的分母只计 latent 输入扰动，不包含 timestep 条件变化。</p><p>不同 prompt 的同编号 patch 仅共享空间位置，未作语义对齐；独立干预分数不是可相加的因果贡献。</p>' + ('<p>' + ' · '.join(summaries) + '</p>' if summaries else '') + '<a href="experiment.json">实验设置 JSON ↗</a></footer></main><script>document.getElementById("prompt-picker").onchange=e=>{if(e.target.value)location.href=e.target.value;};</script></body></html>'
    write_text(root / "report.html", document)
    atomic_json(root / "report_index.json", {"created_at": datetime.now(timezone.utc).isoformat(), "intervention_mode": "matched",
                "timestep_policy": "source_call_only", "status": experiment.get("status"), "completed": total, "expected": expected, "complete": complete, "cases": statuses})
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    write_index(args.root)


if __name__ == "__main__":
    main()
