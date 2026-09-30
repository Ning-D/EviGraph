"""Phase 1: question-conditioned evidence-graph construction.

1. Query-conditioned clip scoring: the video is cut into `clip_interval`-second clips,
   the builder VLM scores each clip's center frame (1-5) against the question. The
   best clip becomes the core node; the next ones (and a uniform grid) become
   candidate timestamps.
2. Sub-need decomposition of the question (2-4 sub-questions).
3. Gap-driven policy loop (<= max_steps): the builder sees the graph state (sub-need
   coverage, per-option for/against support, collected nodes) and picks an action
   aimed at the biggest gap. Every visited moment becomes a node: a +/-snap_sec
   snapshot is captioned, scored for each option (-2..+2) and related to the core
   with a typed relation and a relevance-to-core strength.

The graph is written to <paths.GRAPHS>/<graph_tag>/<qid>.json and reused by every
frame budget and answering backbone in Phase 2.
"""
import json
import os
import re

import numpy as np

from . import paths, prompts
from .data import LETTERS, question_with_options, split_options
from .llm import chat_json, hms, image_part
from .video import grab, open_video, snapshot

RELATIONS = {"supports", "causes", "precedes", "contradicts", "follows"}
DEAD_RADIUS = 8.0          # seconds around a dead (unproductive) moment
USEFUL_STRENGTH = 0.35     # relevance-to-core above which a related node is "useful"


def _num(x, default=0.0):
    """Lenient float parse (open-source builders sometimes answer with strings)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+(?:\.\d+)?", str(x))
        return float(m.group()) if m else default


# ── 1. query-conditioned clip scoring ─────────────────────────────────────────
def _score_clips(cap, fps, clips, qtext, batch, max_side, model):
    payloads = []
    for s in range(0, len(clips), batch):
        chunk = clips[s:s + batch]
        content = [{"type": "text", "text":
                    f"Question: {qtext}\n\nRate each of the {len(chunk)} "
                    f"clips below for relevance to the question:"}]
        for i, (t0, t1) in enumerate(chunk):
            content.append({"type": "text", "text": f"\n--- Clip {i} [{hms(t0)} - {hms(t1)}] ---"})
            fr = grab(cap, fps, (t0 + t1) / 2, max_side)
            content.append(image_part(fr) if fr is not None else {"type": "text", "text": "(no frame)"})
        payloads.append((chunk, content))

    def _do(payload):
        chunk, content = payload
        obj = chat_json(model, prompts.SCORE_CLIPS, content, max_tokens=900)
        results = (obj or {}).get("clips", []) if isinstance(obj, dict) else []
        out = [{"score": 1, "observation": ""} for _ in chunk]
        for r in results:
            try:
                idx = int(r.get("clip_idx", 0))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(out):
                out[idx] = {"score": int(r.get("score", 1)), "observation": str(r.get("observation", ""))}
        return [{"t0": t0, "t1": t1, "score": o["score"], "observation": o["observation"]}
                for (t0, t1), o in zip(chunk, out)]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, len(payloads) or 1)) as ex:
        per = list(ex.map(_do, payloads))
    return [s for batch_scores in per for s in batch_scores]


def score_clips_cached(item, model, cache_file, clip_interval=20.0, batch=10, max_side=448):
    """[{t0, t1, score, observation}] for every clip, and the video duration."""
    if os.path.exists(cache_file):
        try:
            j = json.load(open(cache_file))
            return j["scores"], j["dur"]
        except Exception:
            pass
    cap, fps, dur = open_video(item["_abs_video"])
    if dur <= 0:
        cap.release()
        return [], 0
    clips, t = [], 0.0
    while t < dur - 1.0:
        clips.append((t, min(t + clip_interval, dur)))
        t += clip_interval
    cs = _score_clips(cap, fps, clips, question_with_options(item), batch, max_side, model)
    cap.release()
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    json.dump({"scores": cs, "dur": dur}, open(cache_file, "w"), ensure_ascii=False)
    return cs, dur


# ── 2./3. sub-needs, graph state and node creation ────────────────────────────
def decompose(qtext, model):
    r = chat_json(model, prompts.DECOMPOSE, qtext, max_tokens=220) or {}
    sq = r.get("sub_questions") or []
    return [s.strip() for s in sq if isinstance(s, str) and s.strip()][:4]


def _node_desc(n):
    if n.get("type") == "core":
        return f"[core @{hms(n['ts'])}] {n.get('observation', '')}"
    d = f"[@{hms(n['ts'])}] {n.get('observation', '')}"
    if n.get("change") and n["change"] not in ("static", ""):
        d += f" | change: {n['change']}"
    if n.get("causal_inference") and n["causal_inference"] != "unclear":
        d += f" | causal: {n['causal_inference']}"
    return d


def graph_state(core, active, edges, visited, dead, hints, qtext, dur, budget_left,
                subqs=None, opts=None):
    """Text rendering of the current graph that the policy sees at each step."""
    lines = [f"Question:\n{qtext}", f"\nVideo length: {hms(dur)}",
             f"Budget left: {budget_left}", "\nCore node:",
             f"  {core['node_id']}: {_node_desc(core)}"]
    if subqs:
        cnt = {}
        for n in active:
            cnt[n.get("sub_q")] = cnt.get(n.get("sub_q"), 0) + 1
        lines.append("\nSub-questions to cover (target the least-covered):")
        for i, s in enumerate(subqs):
            lines.append(f"  [{i}] ({cnt.get(i, 0)} nodes) {s}")
    if opts:
        no = len(opts)
        fa, ag = [0.0] * no, [0.0] * no
        for n in active:
            sp = n.get("option_support") or []
            for k in range(min(no, len(sp))):
                if sp[k] > 0:
                    fa[k] += sp[k]
                elif sp[k] < 0:
                    ag[k] += -sp[k]
        lines.append("\nOptions to discriminate (target one still undecided):")
        for k, o in enumerate(opts):
            lines.append(f"  {LETTERS[k]} (for={fa[k]:.0f}, against={ag[k]:.0f}) {str(o)[:70]}")
    if active:
        lines.append("\nCollected evidence nodes:")
        for n in active:
            lines.append(f"  {n['node_id']} (rel_to_core={n.get('rel_to_core', '?')},"
                         f" str={n.get('str_to_core', 0):.2f}): {_node_desc(n)}")
    if edges:
        lines.append("\nTyped relations between nodes:")
        for e in edges[-12:]:
            lines.append(f"  {e['from']} --{e['relation']}({e['strength']:.2f})--> {e['to']}")
    unexplored = [h for h in hints
                  if all(abs(h - v) > 5 for v in visited)
                  and all(abs(h - d) > DEAD_RADIUS for d in dead)]
    if unexplored:
        lines.append("\nUnexplored candidate timestamps: " + ", ".join(hms(h) for h in unexplored[:8]))
    return "\n".join(lines)


def collect_node(cap, fps, tgt, core, active, edges, qtext, dur, snap_n, snap_sec,
                 model, max_side, opts=None):
    """Snapshot at `tgt` -> caption, per-option support, typed relation to the core.
    Appends the node to `active` (and its edge to `edges`). Returns
    {nid, status, explore_further}, or None if no frame could be read."""
    frames = snapshot(cap, fps, tgt, dur, snap_sec, snap_n, max_side)
    if not frames:
        return None
    capj = chat_json(model, prompts.CAPTION,
                     [image_part(f) for f in frames] + [{"type": "text", "text": "Describe."}],
                     max_tokens=220) or {}
    nid = f"n{len(active):02d}"
    node = {"node_id": nid, "type": "evidence", "ts": tgt,
            "observation": capj.get("observation", ""),
            "change": capj.get("change", "static"),
            "objects": capj.get("objects", []),
            "actions": capj.get("actions", "none"),
            "causal_inference": capj.get("causal_inference", "unclear")}
    if opts:
        oj = chat_json(model, prompts.OPTION_SUPPORT,
                       [image_part(f) for f in frames] +
                       [{"type": "text", "text": "Question: " + qtext + "\nOptions:\n" + "\n".join(opts)}],
                       max_tokens=110) or {}
        sup = oj.get("support")
        if isinstance(sup, list) and all(isinstance(v, (int, float))
                                         or re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", str(v))
                                         for v in sup[:len(opts)]):
            node["option_support"] = [float(v) for v in sup[:len(opts)]]
    rr = chat_json(model, prompts.RELATION,
                   f"Question: {qtext}\n\nNode A ({nid}):\n{_node_desc(node)}"
                   f"\n\nNode B ({core['node_id']}):\n{_node_desc(core)}", max_tokens=120) or {}
    rel = rr.get("relation", "irrelevant")
    rel = rel if rel in RELATIONS else "irrelevant"
    strength = _num(rr.get("strength", 0) or 0)
    # The temporal direction (precedes vs follows) is decided by the timestamps.
    try:
        nts, cts = float(node.get("ts")), float(core.get("ts"))
        if rel == "precedes" and nts >= cts:
            rel = "follows"
        elif rel == "follows" and nts < cts:
            rel = "precedes"
    except (TypeError, ValueError):
        pass
    node["rel_to_core"], node["str_to_core"] = rel, strength
    if rel != "irrelevant":
        edges.append({"from": nid, "to": core["node_id"], "relation": rel, "strength": strength})
    explore_further = True
    if node.get("rel_to_core", "irrelevant") != "irrelevant" and node.get("str_to_core", 0) >= USEFUL_STRENGTH:
        status = "useful"
    else:
        ev = chat_json(model, prompts.NODE_EVAL, f"Node: {_node_desc(node)}\nQuestion: {qtext}",
                       max_tokens=120) or {}
        status = ev.get("status", "irrelevant")
        explore_further = bool(ev.get("explore_further", True))
    active.append(node)          # every visited moment is kept as a node
    return {"nid": nid, "status": status, "explore_further": explore_further}


# ── main entry ────────────────────────────────────────────────────────────────
def build_graph(item, model, graph_file, coarse_file, max_steps=24, min_explore=15,
                clip_interval=20.0, snap_sec=5.0, snap_n=3, max_side=448, batch=10):
    """Build (or load) the evidence graph of one question.
    Returns the graph dict {active, edges, chain, log, subqs}, or None on failure."""
    if os.path.exists(graph_file):
        try:
            return json.load(open(graph_file))
        except Exception:
            pass
    if not item.get("options"):        # LVBench: options are embedded in the question text
        stem, raw = split_options(item)
        if raw:
            item = {**item, "question": stem,
                    "options": [f"{LETTERS[i]}. {o}" for i, o in enumerate(raw)]}
    qtext = question_with_options(item)

    cs, dur = score_clips_cached(item, model, coarse_file, clip_interval, batch, max_side)
    if not cs or dur <= 0:
        return None
    cs = sorted(cs, key=lambda c: -c.get("score", 1))
    mid = lambda c: (c["t0"] + c["t1"]) / 2
    core = {"node_id": "core", "type": "core", "ts": mid(cs[0]),
            "observation": cs[0].get("observation", ""), "score": cs[0].get("score", 1)}
    # candidate timestamps: high-scoring clip centers, then a uniform grid as fallback
    hints = [mid(c) for c in cs[1:14] if c.get("score", 1) >= 2]
    for g in [float(t) for t in np.linspace(0, dur, 14)[1:-1]]:
        if all(abs(g - h) > clip_interval for h in hints):
            hints.append(g)

    cap, fps, dur2 = open_video(item["_abs_video"])
    dur = dur or dur2
    if dur2 <= 0:
        cap.release()
        return None

    active, edges, log = [], [], []
    opts = list(item.get("options") or [])
    subqs = decompose(qtext, model)
    policy = prompts.POLICY if subqs else prompts.POLICY_NO_SUBNEEDS
    visited, dead = {core["ts"]}, set()
    for step in range(max_steps):
        budget_left = max_steps - step
        state = graph_state(core, active, edges, visited, dead, hints, qtext, dur,
                            budget_left, subqs=subqs, opts=opts)
        pol = chat_json(model, policy, state, max_tokens=260) or {}
        action = pol.get("action", "EXPLORE_SLOT")
        if action not in ("FOLLOW_EDGE", "EXPLORE_SLOT", "BACKTRACK", "ANSWER"):
            action = "EXPLORE_SLOT"
        # keep exploring until at least `min_explore` nodes are collected
        if action == "ANSWER" and len(active) < min_explore and budget_left > 1:
            action = "EXPLORE_SLOT"
        if action == "ANSWER" or budget_left == 1:
            log.append({"step": step, "action": "ANSWER",
                        "reasoning": pol.get("reasoning", ""), "policy_raw": pol})
            break
        if action == "BACKTRACK":
            nid = pol.get("node_id", "")
            nd = next((n for n in active if n["node_id"] == nid), None)
            if nd:
                dead.add(nd["ts"])
            log.append({"step": step, "action": "BACKTRACK", "node": nid,
                        "reasoning": pol.get("reasoning", ""), "policy_raw": pol})
            continue
        tgt = None
        if action == "FOLLOW_EDGE":           # temporal expansion around an existing node
            nid = pol.get("node_id", "core")
            base = next((n["ts"] for n in [core] + active if n["node_id"] == nid), core["ts"])
            delta = snap_sec * 3
            tgt = (max(0, base - delta * 2) if pol.get("direction") == "before"
                   else min(dur, base + delta * 2))
        else:                                  # EXPLORE_SLOT: jump to a timestamp
            ts_raw = pol.get("timestamp")
            if ts_raw is not None:
                try:
                    tgt = float(max(0, min(dur, ts_raw)))
                except (TypeError, ValueError):
                    tgt = None
            if tgt is None:
                tgt = next((h for h in hints if all(abs(h - v) > snap_sec for v in visited)), None)
        if tgt is None:
            log.append({"step": step, "action": action, "skip": "no_target", "policy_raw": pol})
            continue
        # already visited / dead -> fall back to an unexplored candidate
        if (any(abs(tgt - v) <= snap_sec for v in visited) or
                any(abs(tgt - d) <= DEAD_RADIUS for d in dead)):
            tgt = next((h for h in hints
                        if all(abs(h - v) > snap_sec for v in visited)
                        and all(abs(h - d) > DEAD_RADIUS for d in dead)), None)
            if tgt is None:
                log.append({"step": step, "action": action, "skip": "exhausted", "policy_raw": pol})
                continue
        visited.add(tgt)
        res = collect_node(cap, fps, tgt, core, active, edges, qtext, dur, snap_n, snap_sec,
                           model, max_side, opts=opts)
        if res is None:
            dead.add(tgt)
            continue
        if res["status"] != "useful" and not res["explore_further"]:
            dead.add(tgt)
        if subqs:                              # tag the node with the targeted sub-need
            sq_i = pol.get("sub_q")
            if isinstance(sq_i, int) and 0 <= sq_i < len(subqs):
                nd = next((n for n in active if n["node_id"] == res["nid"]), None)
                if nd is not None:
                    nd["sub_q"] = sq_i
        log.append({"step": step, "action": action, "ts": round(tgt),
                    "nid": res["nid"], "status": res["status"],
                    "reasoning": pol.get("reasoning", ""),
                    "missing": pol.get("missing_info", ""),
                    "sub_q": pol.get("sub_q"), "policy_raw": pol})
    cap.release()
    chain = [n["node_id"] for n in sorted(active, key=lambda n: -n.get("str_to_core", 0))[:4]]
    g = {"active": active, "edges": edges, "chain": chain, "log": log}
    if subqs:
        g["subqs"] = subqs
    os.makedirs(os.path.dirname(graph_file), exist_ok=True)
    json.dump(g, open(graph_file, "w"), ensure_ascii=False)
    return g
