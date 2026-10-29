"""Sample prompts and concat videos from several output dirs side by side (with a prompt caption) into one mp4."""
import argparse
import json
import os
import random


import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument("--dirs", nargs="+", default=["outputs/vbench/vbench_14b", "outputs/vbench/vbench_14b_350"])
p.add_argument("--num_prompts", type=int, default=20)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="video index suffixes (-N.mp4), stacked as rows")
p.add_argument("--rand_seed", type=int, default=0)
p.add_argument("--out", default="outputs/compare_14b_vs_350", help="output folder")
p.add_argument("--prompt_path", default="prompts/vbench/all_dimension.txt")
p.add_argument("--extended_prompt_path", default="prompts/vbench/all_dimension_extended.txt")
p.add_argument("--fps", type=int, default=16)
p.add_argument("--dimension", default=None, help="only sample prompts of this VBench dimension, e.g. dynamic_degree")
p.add_argument("--info_path", default="prompts/VBench_full_info.json")
args = p.parse_args()

# inference.py names files by the short prompt but feeds the line-aligned extended prompt; first occurrence wins
with open(args.prompt_path) as f1, open(args.extended_prompt_path) as f2:
    extended = {}
    for s, e in zip(f1.read().splitlines(), f2.read().splitlines()):
        extended.setdefault(s, e)

# file name format: "<prompt>-<idx>.mp4"
prompt_sets = [{f.rsplit("-", 1)[0] for f in os.listdir(d) if f.endswith(".mp4")} for d in args.dirs]
common = sorted(set.intersection(*prompt_sets))
if args.dimension:
    with open(args.info_path) as f:
        dim_prompts = {e["prompt_en"] for e in json.load(f) if args.dimension in e["dimension"]}
    assert dim_prompts, f"no prompts for dimension {args.dimension}"
    common = sorted(set(common) & dim_prompts)
prompts = random.Random(args.rand_seed).sample(common, min(args.num_prompts, len(common)))

try:
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
except OSError:
    font = ImageFont.load_default()


LABEL_W, LINE_H, NAME_H = 96, 24, 34


def text_bar(w, h, items):
    img = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(img)
    for xy, s, color in items:
        draw.text(xy, s, fill=color, font=font)
    return np.asarray(img)


def wrap_px(text, max_w):
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if cur and font.getlength(cand) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    return lines + [cur] if cur else lines


# all.mp4 needs one frame size, so the prompt bar uses the max line count over all sampled prompts
w0 = imageio.get_reader(os.path.join(args.dirs[0], f"{prompts[0]}-{args.seeds[0]}.mp4")).get_meta_data()["size"][0]
W = LABEL_W + len(args.dirs) * w0
captions = [wrap_px(f"[{k:02d}] {extended.get(pr, pr)}", W - 16) for k, pr in enumerate(prompts)]
PROMPT_H = 8 + LINE_H * max(map(len, captions)) + 4

os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)
mk = lambda path: imageio.get_writer(path, fps=args.fps, codec="libx264", quality=None, macro_block_size=1)
writer = mk(os.path.join(args.out, "all.mp4"))
n = 0
for k, pr in enumerate(prompts):
    paths = [[os.path.join(d, f"{pr}-{s}.mp4") for d in args.dirs] for s in args.seeds]
    if not all(os.path.exists(x) for row in paths for x in row):
        print(f"skip (missing seed): {pr}")
        continue
    vids = [[np.stack(imageio.mimread(x, memtest=False)) for x in row] for row in paths]
    flat = [v for row in vids for v in row]
    assert len({v.shape[1:] for v in flat}) == 1 and flat[0].shape[2] == w0, f"resolution mismatch for {pr}"
    t = min(v.shape[0] for v in flat)
    _, h, w, _ = flat[0].shape
    # grid: rows = seeds, cols = dirs; seed label strip on the left
    rows = [
        np.concatenate([np.broadcast_to(text_bar(LABEL_W, h, [((8, h // 2 - 10), f"seed {s}", "yellow")]), (t, h, LABEL_W, 3))]
                       + [v[:t] for v in row], axis=2)
        for s, row in zip(args.seeds, vids)
    ]
    grid = np.concatenate(rows, axis=1)
    top = np.concatenate([
        text_bar(W, PROMPT_H, [((8, 8 + LINE_H * j), line, "white") for j, line in enumerate(captions[k])]),
        text_bar(W, NAME_H, [((LABEL_W + i * w + 8, 6), os.path.basename(d.rstrip("/")), "yellow") for i, d in enumerate(args.dirs)]),
    ], axis=0)
    clip = mk(os.path.join(args.out, "clips", f"{k:02d}_{pr[:80].replace('/', '_')}.mp4"))
    for fr in grid:
        out_fr = np.concatenate([top, fr], axis=0)
        writer.append_data(out_fr)
        clip.append_data(out_fr)
    clip.close()
    n += 1
writer.close()

with open(os.path.join(args.out, "prompts.txt"), "w") as f:
    for k, pr in enumerate(prompts):
        f.write(f"[{k:02d}] {pr}\n  extended: {extended.get(pr, '(not found)')}\n\n")
print(f"wrote {args.out}/ (all.mp4, {n} clips, prompts.txt)")
