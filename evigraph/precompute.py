"""Per-frame features shared by EviGraph and the baselines (1 fps frames, AKS-style).

    python -m evigraph.precompute blip  --dataset lvb   # BLIP-ITM(question, frame) per question
    python -m evigraph.precompute clip  --dataset lvb   # CLIP-L image features per video
    python -m evigraph.precompute lvnet --dataset lvb   # LVNet: option keywords (LLM) + BLIP-ITM

The video is decoded once and its ViT features are reused by all of its questions.
Existing outputs are skipped; use --shard i/n to split the videos across GPUs.
"""
import argparse
import json
import os
import time

import numpy as np

from . import data, paths
from .video import read_1fps

BLIP_ID = os.environ.get("BLIP_ITM_MODEL", "Salesforce/blip-itm-large-coco")


def _encode_vision(model, proc, frames, device, bs=48):
    import torch
    embs = []
    with torch.no_grad():
        for k in range(0, len(frames), bs):
            pv = proc(images=frames[k:k + bs], return_tensors="pt")["pixel_values"].to(device, torch.float16)
            embs.append(model.vision_model(pv)[0].cpu())
    return torch.cat(embs, 0)                     # [F, seq, dim] fp16 on CPU


def _itm_scores(model, proc, image_embeds, text, device, bs=64):
    """BLIP image-text-matching probability of `text` for every frame."""
    import torch
    tok = proc(text=text, return_tensors="pt", truncation=True, max_length=35).to(device)
    out = np.empty(image_embeds.shape[0], dtype=np.float32)
    with torch.no_grad():
        for k in range(0, image_embeds.shape[0], bs):
            ie = image_embeds[k:k + bs].to(device, torch.float16)
            iatt = torch.ones(ie.shape[:-1], dtype=torch.long, device=device)
            te = model.text_encoder(input_ids=tok.input_ids.expand(ie.shape[0], -1),
                                    attention_mask=tok.attention_mask.expand(ie.shape[0], -1),
                                    encoder_hidden_states=ie, encoder_attention_mask=iatt,
                                    return_dict=True)
            logits = model.itm_head(te.last_hidden_state[:, 0, :])
            out[k:k + ie.shape[0]] = logits.float().softmax(-1)[:, 1].cpu().numpy()
    return out.tolist()


def _by_video(items, shard):
    by_vid = {}
    for it in items:
        by_vid.setdefault(data.video_key(it), []).append(it)
    si, sn = map(int, shard.split("/"))
    return [(v, by_vid[v]) for i, v in enumerate(sorted(by_vid)) if i % sn == si]


def run_blip(items, a, out_dir, text_fn):
    import torch
    from transformers import AutoProcessor, BlipForImageTextRetrieval
    os.makedirs(out_dir, exist_ok=True)
    todo = _by_video(items, a.shard)
    # the Hub checkpoint ships a .bin; use_safetensors picks its safetensors conversion
    # (transformers refuses .bin files with torch < 2.6)
    model = BlipForImageTextRetrieval.from_pretrained(BLIP_ID, torch_dtype=torch.float16,
                                                      use_safetensors=True).to(a.device).eval()
    proc = AutoProcessor.from_pretrained(BLIP_ID)
    for n, (v, its) in enumerate(todo):
        pend = [it for it in its if not os.path.exists(os.path.join(out_dir, paths.safe_qid(it["_qid"]) + ".json"))]
        if not pend:
            continue
        t0 = time.time()
        frames, fidx = read_1fps(its[0]["_abs_video"])
        if not frames:
            print(f"  !! no frames for {v}", flush=True)
            continue
        ie = _encode_vision(model, proc, frames, a.device)
        for it in pend:
            text, extra = text_fn(it)
            json.dump({"qid": it["_qid"], "videoID": v, "frames": fidx, "fps": 1,
                       "scores": _itm_scores(model, proc, ie, text, a.device), **extra},
                      open(os.path.join(out_dir, paths.safe_qid(it["_qid"]) + ".json"), "w"),
                      ensure_ascii=False)
        print(f"  [{n + 1}/{len(todo)}] {v}: {len(frames)} frames, {len(pend)} questions, "
              f"{time.time() - t0:.0f}s", flush=True)


def run_clip(items, a):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from .features import CLIP_ID, clip_image_feats
    os.makedirs(paths.CLIP_FEATS, exist_ok=True)
    todo = _by_video(items, a.shard)
    model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(a.device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    for n, (v, its) in enumerate(todo):
        out = os.path.join(paths.CLIP_FEATS, str(v) + ".npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        frames, fidx = read_1fps(its[0]["_abs_video"])
        if not frames:
            print(f"  !! no frames for {v}", flush=True)
            continue
        feats = clip_image_feats(model, proc, frames, a.device)
        np.savez_compressed(out, feats=feats, frames=np.asarray(fidx, np.int64))
        print(f"  [{n + 1}/{len(todo)}] {v}: {len(frames)} frames, {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["blip", "clip", "lvnet"])
    ap.add_argument("--dataset", required=True, choices=data.DATASETS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default="0/1", help="i/n over videos")
    ap.add_argument("--keyword_model", default="gpt-5.4-mini", help="lvnet: keyword extractor")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="paper", choices=["paper", "all"],
                    help="paper: the questions evaluated in the paper; all: every question with a local video")
    a = ap.parse_args()
    items, _ = data.load(a.dataset, limit=a.limit, split=a.split)
    print(f"[precompute {a.what}] {a.dataset}: {len(items)} questions", flush=True)
    if a.what == "blip":
        run_blip(items, a, paths.BLIP_SCORES, lambda it: (it["question"], {"model": BLIP_ID}))
    elif a.what == "clip":
        run_clip(items, a)
    else:
        from .baselines import lvnet_keywords, lvnet_text

        def text_fn(it):
            kw = lvnet_keywords(it, a.keyword_model)
            return (", ".join(kw) if kw else lvnet_text(it)), {"keywords": kw, "model": BLIP_ID}
        run_blip(items, a, paths.LVNET, text_fn)


if __name__ == "__main__":
    main()
