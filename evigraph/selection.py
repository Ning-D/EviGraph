"""Phase 2: EviGraph frame selection by evidence coverage.

Half of the budget comes from the BLIP-ITM question-relevance ranking of the 1 fps
frames, the other half from the evidence graph. Graph nodes are ranked in strict
priority tiers -- (1) the most core-relevant node of every sub-need, (2) the
strongest supporter of every answer option, (3) the remaining nodes by
discriminability (top-1 minus top-2 option support) -- and every selected node
contributes `snaps` detail frames retrieved with CLIP inside +/-`radius` seconds.
Frames closer than ~1 s to an already selected frame are skipped; any shortfall
is back-filled from the relevance ranking.
"""
import json
import os

import numpy as np

from . import paths
from .data import split_options, video_key
from .features import clip_text_feat, detail_times


def load_blip_scores(qid):
    """(native frame indices, BLIP-ITM scores) of the 1 fps frames, or None."""
    f = os.path.join(paths.BLIP_SCORES, paths.safe_qid(qid) + ".json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return np.array(d["frames"], float), np.array(d["scores"], float)


def load_graph(qid, graph_tag):
    f = os.path.join(paths.GRAPHS, graph_tag, paths.safe_qid(qid) + ".json")
    return json.load(open(f)) if os.path.exists(f) else None


def rank_nodes(nodes, support, nopt):
    """Tiered node order: sub-need coverage, option coverage, discriminability."""
    def gs(i):                        # option-support vector padded/truncated to nopt
        v = support.get(i)
        if not v:
            return [0] * nopt
        return (list(v) + [0] * nopt)[:nopt]

    def disc(i):
        s = np.sort(np.array(gs(i), dtype=float))[::-1]
        return (s[0] - s[1]) if len(s) > 1 else (s[0] if len(s) else 0.0)

    by_sq = {}                        # tier 1: most core-relevant node per sub-need
    for i, n in enumerate(nodes):
        sq = str(n.get("sub_q", "?"))
        if sq not in by_sq or n["_str"] > nodes[by_sq[sq]]["_str"]:
            by_sq[sq] = i
    order = sorted(by_sq.values(), key=lambda i: -nodes[i]["_str"])
    chosen = set(order)
    for k in range(nopt):             # tier 2: strongest supporter of each option
        best = max(range(len(nodes)), key=lambda i: gs(i)[k])
        if best not in chosen and gs(best)[k] > 0:
            order.append(best)
            chosen.add(best)
    order += sorted([i for i in range(len(nodes)) if i not in chosen], key=lambda i: -disc(i))
    return order, gs


def evigraph_select(item, n_frames, graph_tag, k_rel=None, snaps=2, radius=8.0):
    """Sorted native frame indices of the N selected frames, or None if the BLIP
    scores of this question are missing. Questions without a graph fall back to
    the relevance ranking."""
    k_rel = n_frames // 2 if k_rel is None else k_rel
    qid = item["_qid"]
    # "A. ..." strings; LVBench options are parsed out of the question text (no letters)
    opts = item.get("options") or split_options(item)[1]
    nopt = len(opts)
    blip = load_blip_scores(qid)
    if blip is None:
        return None
    fr, sc = blip
    fps = float(np.median(np.diff(fr))) if len(fr) > 1 else 30.0   # 1 fps grid -> native fps
    gap = max(1.0, fps)                                            # ~1 s de-duplication
    relevance = [int(fr[i]) for i in np.argsort(-sc)]
    sel = []

    def add(idxs, cap):
        c = 0
        for x in idxs:
            x = int(round(x))
            if x >= 0 and all(abs(x - y) >= gap for y in sel):
                sel.append(x)
                c += 1
                if c >= cap:
                    return

    add(relevance, k_rel)                                          # relevance half

    g = load_graph(qid, graph_tag)
    nodes = [n for n in (g.get("active", []) if g else []) if n.get("ts") not in (None, "")]
    for n in nodes:
        try:
            n["_ts"] = float(n["ts"])
            n["_str"] = float(n.get("str_to_core", 0) or 0)
        except (TypeError, ValueError):
            n["_ts"] = None
    nodes = [n for n in nodes if n["_ts"] is not None]
    if nodes:
        support = {i: [float(v) for v in n["option_support"]] for i, n in enumerate(nodes)
                   if isinstance(n.get("option_support"), list) and n["option_support"]}
        order, gs = rank_nodes(nodes, support, nopt)
        subqs = (g.get("subqs") or []) if g else []
        vkey = video_key(item) or qid
        stem = item.get("question", "")

        def node_query(i):               # CLIP query: the option the node backs, else its sub-need
            v = np.array(gs(i), dtype=float)
            if opts and v.size and v.max() > 0:
                return (stem + " " + str(opts[int(v.argmax())]))[:300]
            sq = nodes[i].get("sub_q")
            if isinstance(sq, int) and 0 <= sq < len(subqs):
                return str(subqs[sq])[:300]
            return (str(nodes[i].get("observation") or stem))[:300]

        for i in order:                                            # graph half
            if len(sel) >= n_frames:
                break
            ts = nodes[i]["_ts"]
            qf = clip_text_feat(node_query(i))
            for t in detail_times(vkey, qf, max(0.0, ts - radius), ts + radius, snaps, fps):
                add([t * fps], 1)
    if len(sel) < n_frames:                                        # relevance back-fill
        add(relevance, n_frames - len(sel))
    return sorted(sel[:n_frames])
