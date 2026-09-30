"""Where inputs and intermediate results live. Every location can be overridden
with an environment variable (see scripts/env.example.sh)."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("EVIGRAPH_OUT", os.path.join(ROOT, "outputs"))

# Per-question BLIP-ITM relevance of every 1 fps frame: <qid>.json
BLIP_SCORES = os.environ.get("EVIGRAPH_BLIP_DIR", os.path.join(OUT, "blip_scores"))
# Per-video CLIP-L image features of the same 1 fps frames: <videoID>.npz
CLIP_FEATS = os.environ.get("EVIGRAPH_CLIP_DIR", os.path.join(OUT, "clip_feats"))
# LVNet baseline: option keywords and their BLIP-ITM scores: <qid>.json
LVNET = os.environ.get("EVIGRAPH_LVNET_DIR", os.path.join(OUT, "lvnet"))
# Phase 1: coarse 20 s clip scores (<model>/<qid>_ci20.json) and graphs (<tag>/<qid>.json)
COARSE = os.environ.get("EVIGRAPH_COARSE_DIR", os.path.join(OUT, "coarse_scores"))
GRAPHS = os.environ.get("EVIGRAPH_GRAPH_DIR", os.path.join(OUT, "graphs"))
# Answering results: <tag>.json (+ <tag>.jsonl resume cache)
RESULTS = os.environ.get("EVIGRAPH_RESULTS_DIR", os.path.join(OUT, "results"))


def model_tag(model):
    """Filesystem-safe name of a model id, e.g. Qwen/Qwen3-VL-8B-Instruct -> Qwen--Qwen3-VL-8B-Instruct."""
    return re.sub(r"[^A-Za-z0-9._-]", "", model.replace("/", "--"))


def safe_qid(qid):
    return str(qid).replace("/", "_")
