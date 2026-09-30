"""Training-free baseline selectors, re-implemented on the same 1 fps frame grid,
BLIP-ITM scores and CLIP-L features as EviGraph so that only the selection differs.

  uniform    evenly spaced frames over the whole video
  aks        AKS (Tang et al., CVPR 2025): adaptive split of the BLIP-ITM relevance curve
  videotree  VideoTree (Wang et al., CVPR 2025): KMeans breadth + relevance-weighted depth
  lvnet      LVNet (Park et al., 2024): BLIP-ITM relevance to a keyword bag from the options
"""
import heapq
import json
import os

import numpy as np

from . import paths, prompts
from .data import video_key
from .llm import chat_json
from .video import frame_count


def uniform_select(item, n):
    total = frame_count(item["_abs_video"])
    if total <= 0:
        return None
    return np.linspace(0, total - 1, min(n, total), dtype=int).tolist()


# ── AKS: `meanstd` verbatim from ncTimTang/AKS (frame_select.py) ──────────────
def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []
    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score['score']
        depth = dic_score['depth']
        mean = np.mean(score)
        std = np.std(score)
        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        mean_diff = np.mean(top_score) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            score1 = score[:len(score) // 2]
            score2 = score[len(score) // 2:]
            fn1 = fn[:len(score) // 2]
            fn2 = fn[len(score) // 2:]
            split_scores.append(dict(score=score1, depth=depth + 1))
            split_scores.append(dict(score=score2, depth=depth + 1))
            split_fn.append(fn1)
            split_fn.append(fn2)
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    if len(split_scores) > 0:
        all_split_score, all_split_fn = meanstd(
            len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        all_split_score = []
        all_split_fn = []
    all_split_score = no_split_scores + all_split_score
    all_split_fn = no_split_fn + all_split_fn
    return all_split_score, all_split_fn


def aks_frames(scores, frames, n=64, t1=0.8, t2=-100, all_depth=5):
    """AKS's main(): normalize -> meanstd -> depth-weighted top-k per leaf, then a
    hard budget of n frames (highest normalized score) so every selector gets N."""
    scores = list(scores)
    frames = list(frames)
    if len(scores) <= n:
        return sorted(frames)
    arr = np.asarray(scores, dtype=np.float64)
    rng = float(arr.max() - arr.min())
    norm = (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)
    a, b = meanstd(len(norm), [dict(score=norm, depth=0)], n, [list(frames)], t1, t2, all_depth)
    pairs = []
    for s, f in zip(a, b):
        sc = s['score']
        f_num = max(1, int(n / 2 ** (s['depth'])))
        topk = heapq.nlargest(f_num, range(len(sc)), sc.__getitem__)
        pairs.extend((f[t], float(sc[t])) for t in topk)
    if len(pairs) > n:
        pairs.sort(key=lambda p: -p[1])
        pairs = pairs[:n]
    return sorted({fr for fr, _ in pairs})


def aks_select(item, n):
    f = os.path.join(paths.BLIP_SCORES, paths.safe_qid(item["_qid"]) + ".json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return aks_frames(d["scores"], d["frames"], n=n)


# ── VideoTree (CLIP-L features + BLIP-ITM relevance instead of EVA-CLIP + captions) ─
def _l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _subcluster_reps(feats, gframes, k):
    """Split a cluster into k sub-clusters -> the frame closest to each centroid."""
    from sklearn.cluster import KMeans
    k = int(min(k, len(gframes)))
    if k <= 1:
        Xn = _l2n(feats)
        c = Xn.mean(0, keepdims=True)
        return [gframes[int(np.linalg.norm(Xn - c, axis=1).argmin())]]
    Xn = _l2n(feats)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(Xn)
    reps = []
    for c in range(k):
        m = np.where(km.labels_ == c)[0]
        if len(m) == 0:
            continue
        d = np.linalg.norm(Xn[m] - km.cluster_centers_[c], axis=1)
        reps.append(gframes[int(m[d.argmin()])])
    return reps


def videotree_frames(feats, frames, scores, n=64, init_clusters=8):
    from sklearn.cluster import KMeans
    F = len(frames)
    feats = np.asarray(feats, np.float32)
    scores = np.asarray(scores, np.float32)
    frames = list(frames)
    if F <= n:
        return sorted(frames)
    # breadth: cluster all frames for visual coverage
    C = int(min(init_clusters, F))
    km = KMeans(n_clusters=C, n_init=4, random_state=0).fit(_l2n(feats))
    labels = km.labels_
    rel = np.array([scores[labels == c].mean() if (labels == c).any() else 0.0
                    for c in range(C)], np.float64)
    rel = np.clip(rel, 1e-3, None)
    # depth: one frame per cluster + the remainder weighted by cluster relevance
    alloc = np.ones(C, int)
    extra = n - C
    if extra > 0:
        add = np.floor(rel / rel.sum() * extra).astype(int)
        alloc += add
        for c in np.argsort(-rel)[:extra - int(add.sum())]:
            alloc[c] += 1
    out = []
    for c in range(C):
        m = np.where(labels == c)[0]
        if len(m) == 0:
            continue
        out.extend(_subcluster_reps(feats[m], [frames[i] for i in m], alloc[c]))
    out = sorted(set(out))
    if len(out) > n:                   # hard budget by relevance
        sc = {frames[i]: scores[i] for i in range(F)}
        out = sorted(sorted(out, key=lambda f: -sc.get(f, 0))[:n])
    return out


def videotree_select(item, n):
    af = os.path.join(paths.BLIP_SCORES, paths.safe_qid(item["_qid"]) + ".json")
    vf = os.path.join(paths.CLIP_FEATS, str(video_key(item)) + ".npz")
    if not (os.path.exists(af) and os.path.exists(vf)):
        return None
    d = json.load(open(af))
    z = np.load(vf)
    fmap = {int(f): i for i, f in enumerate(z["frames"].tolist())}
    keep = [k for k, f in enumerate(d["frames"]) if int(f) in fmap]
    feats = z["feats"][[fmap[int(d["frames"][k])] for k in keep]]
    scores = [d["scores"][k] for k in keep]
    fr_idx = [d["frames"][k] for k in keep]
    return videotree_frames(feats, fr_idx, scores, n=n)


# ── LVNet ────────────────────────────────────────────────────────────────────
def lvnet_text(item):
    """Question text the keywords are extracted from (options included)."""
    if item.get("options"):
        return item["question"] + "\n" + "\n".join(item["options"])
    return item["question"]            # LVBench: options are already in the question


def lvnet_keywords(item, model):
    """Bag of visual keywords from the options, cached per question."""
    f = os.path.join(paths.LVNET, "keywords", paths.safe_qid(item["_qid"]) + ".json")
    if os.path.exists(f):
        return json.load(open(f)).get("keywords", [])
    r = chat_json(model, prompts.LVNET_KEYWORDS, lvnet_text(item), max_tokens=160) or {}
    kw = [k.strip() for k in (r.get("keywords") or []) if isinstance(k, str) and k.strip()][:12]
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump({"keywords": kw}, open(f, "w"), ensure_ascii=False)
    return kw


def lvnet_select(item, n):
    f = os.path.join(paths.LVNET, paths.safe_qid(item["_qid"]) + ".json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    sc = np.asarray(d["scores"], float)
    fr = np.asarray(d["frames"])
    return sorted(int(t) for t in fr[np.argsort(-sc)[:n]])
