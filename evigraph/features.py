"""CLIP-L (openai/clip-vit-large-patch14) features.

Image features of the 1 fps frames are precomputed once per video
(`python -m evigraph.precompute clip`) and shared by EviGraph's detail-frame
retrieval and the VideoTree baseline.
"""
import os

import numpy as np

from . import paths

CLIP_ID = "openai/clip-vit-large-patch14"
_CLIP = {"model": None, "proc": None, "device": None}
_FEATS = {}   # video key -> (L2-normalized feats [F, 768], seconds [F]) or None


def _clip_load():
    if _CLIP["model"] is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = os.environ.get("EVIGRAPH_CLIP_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        _CLIP["model"] = CLIPModel.from_pretrained(CLIP_ID).to(dev).eval()
        _CLIP["proc"] = CLIPProcessor.from_pretrained(CLIP_ID)
        _CLIP["device"] = dev
    return _CLIP["model"], _CLIP["proc"], _CLIP["device"]


def clip_text_feat(text):
    """L2-normalized CLIP-L text embedding of `text` (first 300 characters)."""
    import torch
    model, proc, dev = _clip_load()
    tok = proc(text=[text[:300]], return_tensors="pt", padding=True, truncation=True).to(dev)
    with torch.no_grad():
        out = model.get_text_features(**tok)
        if not torch.is_tensor(out):        # some transformers versions return a ModelOutput
            po = model.text_model(**tok)
            out = model.text_projection(po.pooler_output)
        f = out.float().cpu().numpy()[0]
    return f / (np.linalg.norm(f) + 1e-8)


def load_clip_feats(video_key, fps):
    """Cached 1 fps image features of a video and their timestamps in seconds
    (native frame index / fps), or None if not precomputed."""
    if video_key not in _FEATS:
        vf = os.path.join(paths.CLIP_FEATS, str(video_key) + ".npz")
        if not os.path.exists(vf):
            _FEATS[video_key] = None
        else:
            z = np.load(vf)
            fn = z["feats"].astype(np.float32)
            fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
            secs = z["frames"].astype(np.float64) / (fps or 30.0)
            _FEATS[video_key] = (fn, secs)
    return _FEATS[video_key]


def detail_times(video_key, qfeat, lo, hi, n, fps):
    """Timestamps (s) of the `n` frames in [lo, hi] that best match `qfeat`.
    Falls back to `n` evenly spaced times if the features are missing or the
    window holds no more than `n` frames."""
    cached = load_clip_feats(video_key, fps)
    if cached is None or qfeat is None:
        return list(np.linspace(lo, hi, n))
    fn, secs = cached
    idx = np.where((secs >= lo) & (secs <= hi))[0]
    if len(idx) <= n:
        return list(np.linspace(lo, hi, n))
    sims = fn[idx] @ qfeat
    top = idx[np.argsort(-sims)[:n]]
    return sorted(float(secs[t]) for t in top)


def clip_image_feats(model, proc, frames, device, bs=64):
    """CLIP-L image embeddings (float16, unnormalized) of a list of PIL frames."""
    import torch
    embs = []
    with torch.no_grad():
        for k in range(0, len(frames), bs):
            px = proc(images=frames[k:k + bs], return_tensors="pt")["pixel_values"].to(device, torch.float16)
            embs.append(model.get_image_features(pixel_values=px).float().cpu().numpy())
    return np.concatenate(embs, 0).astype(np.float16)
