"""Select N frames with one selector and answer with a frozen backbone.

    python -m evigraph.answer --dataset lvb --selector evigraph --frames 32 --backbone qwen

Selectors: evigraph (needs Phase-1 graphs), uniform, aks, videotree, lvnet. Every
selector feeds the same backbone with the same prompt; only the frames differ.
Results: <EVIGRAPH_RESULTS_DIR>/<tag>.json (per-question rows with the selected
frame indices, the raw output and correctness). Interrupted runs resume from
<tag>.jsonl.
"""
import argparse
import json
import os
import re

import numpy as np

from . import baselines, data, paths
from .backbones import BACKBONES, load_backbone
from .selection import evigraph_select
from .video import sample_frames

SELECTORS = ("evigraph", "uniform", "aks", "videotree", "lvnet")


def select_frames(item, selector, n, graph_tag="gpt-5.4-mini", k_rel=None, snaps=2, radius=8.0):
    """Native frame indices chosen by `selector`, or None if its inputs are missing."""
    if selector == "evigraph":
        return evigraph_select(item, n, graph_tag, k_rel=k_rel, snaps=snaps, radius=radius)
    return {"uniform": baselines.uniform_select, "aks": baselines.aks_select,
            "videotree": baselines.videotree_select, "lvnet": baselines.lvnet_select}[selector](item, n)


def parse_letter(text):
    """First option letter in a reply ("B", "B.", "(B) ...", "Answer: B")."""
    t = str(text).strip().upper()
    m = re.match(r"^\(?([A-G])(?![A-Z])", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-G])\b", t)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=data.DATASETS)
    ap.add_argument("--selector", required=True, choices=SELECTORS)
    ap.add_argument("--frames", type=int, default=32, help="frame budget N")
    ap.add_argument("--backbone", default="qwen", choices=BACKBONES)
    ap.add_argument("--graph_tag", default="gpt-5.4-mini", help="evigraph: which graphs to use")
    ap.add_argument("--k_rel", type=int, default=None, help="evigraph: relevance frames (default N/2)")
    ap.add_argument("--snaps", type=int, default=2, help="evigraph: detail frames per node")
    ap.add_argument("--radius", type=float, default=8.0, help="evigraph: detail window half-width (s)")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="paper", choices=["paper", "all"],
                    help="paper: the questions evaluated in the paper; all: every question with a local video")
    ap.add_argument("--shard", default="0/1", help="i/n: answer every n-th question starting at i")
    ap.add_argument("--tag", default=None, help="results name (default: dataset_selector_backbone_fN)")
    a = ap.parse_args()
    a.graph_tag = paths.model_tag(a.graph_tag)       # accept a model id or a tag
    tag = a.tag or f"{a.dataset}_{a.selector}_{a.backbone}_f{a.frames}"
    if a.selector == "evigraph" and a.graph_tag != "gpt-5.4-mini" and not a.tag:
        tag += f"_{a.graph_tag}"

    items, build_query = data.load(a.dataset, limit=a.limit, split=a.split)
    si, sn = map(int, a.shard.split("/"))
    items = items[si::sn]
    os.makedirs(paths.RESULTS, exist_ok=True)
    cache = os.path.join(paths.RESULTS, tag + ".jsonl")
    done = {}
    if os.path.exists(cache):
        for line in open(cache):
            if line.strip():
                r = json.loads(line)
                done[r["_qid"]] = r
    print(f"[answer] {tag}: {len(items)} questions ({len(done)} cached), selector={a.selector}, "
          f"N={a.frames}, backbone={a.backbone}", flush=True)

    vlm = None
    rows = []
    with open(cache, "a") as fh:
        for i, it in enumerate(items):
            if it["_qid"] in done:
                rows.append(done[it["_qid"]])
                continue
            sel = select_frames(it, a.selector, a.frames, a.graph_tag, a.k_rel, a.snaps, a.radius)
            if not sel:
                rows.append({"_qid": it["_qid"], "task": it.get("task", ""), "correct": False,
                             "skip": "missing selector inputs"})
                continue
            if vlm is None:
                vlm = load_backbone(a.backbone)
            frames = sample_frames(it["_abs_video"], sel, max_side=512)
            out = vlm.answer_mc(frames, build_query(it), max_new_tokens=a.max_new_tokens)
            pred = parse_letter(out)
            row = {"_qid": it["_qid"], "task": it.get("task", ""), "n": len(frames),
                   "frames": [int(x) for x in sel], "output": out, "pred": pred,
                   "correct": pred == str(it["correct_answer"]).strip().upper()}
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(items)}] acc={np.mean([r['correct'] for r in rows]):.3f}", flush=True)
    acc = float(np.mean([r["correct"] for r in rows])) if rows else 0.0
    skipped = sum(1 for r in rows if r.get("skip"))
    json.dump({"tag": tag, "cfg": vars(a), "acc": acc, "n": len(rows), "skipped": skipped, "rows": rows},
              open(os.path.join(paths.RESULTS, tag + ".json"), "w"), ensure_ascii=False)
    print(f"[answer] {tag}: acc={100 * acc:.1f} (n={len(rows)}, skipped={skipped})", flush=True)


if __name__ == "__main__":
    main()
