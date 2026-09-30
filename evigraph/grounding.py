"""Grounding of the selected frames on LVBench (answer-clue span = `time_reference`).

    python -m evigraph.grounding --frames 32

For every selector (no answering model needed) it reports, over the questions with an
annotated span:
  frame level     share of selected frames inside the span / within 30 s / within 60 s,
                  and the median distance of a selected frame to the span
  question level  share of questions with >= k selected frames inside the span, and
                  with >= 1 frame within 30 s / 60 s of it
"""
import argparse
from statistics import median

import numpy as np

from . import data
from .answer import SELECTORS, select_frames
from .selection import load_blip_scores
from .video import open_video


def dist_to_span(sec, span):
    s, e = span
    if s <= sec <= e:
        return 0.0
    return (s - sec) if sec < s else (sec - e)


def seconds(item, selector, idx, _fps_cache={}):
    """Frame index -> seconds. The 1 fps selectors use the native fps implied by the
    1 fps grid; uniform sampling uses the video's own fps."""
    if selector == "uniform":
        v = item["_abs_video"]
        if v not in _fps_cache:
            cap, fps, _ = open_video(v)
            cap.release()
            _fps_cache[v] = fps
        fps = _fps_cache[v]
    else:
        fr = load_blip_scores(item["_qid"])[0]
        fps = float(np.median(np.diff(fr))) if len(fr) > 1 else 30.0
    return sorted({round(float(x) / fps, 2) for x in idx})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--selectors", nargs="+", default=list(SELECTORS), choices=SELECTORS)
    ap.add_argument("--graph_tag", default="gpt-5.4-mini")
    ap.add_argument("--split", default="paper", choices=["paper", "all"],
                    help="paper: the questions evaluated in the paper; all: every question with a local video")
    a = ap.parse_args()
    items = [it for it in data.load("lvbench", split=a.split)[0] if it.get("gt_interval")]
    print(f"[grounding] LVBench, N={a.frames}: {len(items)} questions with an answer span")
    print(f"{'selector':10s} | frames in span / <=30s / <=60s | median dist | "
          f"questions >=1 / >=2 / >=3 in span | >=1 within 30s / 60s")
    for s in a.selectors:
        dists, per_q = [], []
        for it in items:
            idx = select_frames(it, s, a.frames, a.graph_tag)
            secs = seconds(it, s, idx) if idx else []
            d = [dist_to_span(t, it["gt_interval"]) for t in secs]
            dists += d
            per_q.append(d)
        dists = np.array(dists)
        inspan = [sum(1 for x in d if x == 0.0) for d in per_q]
        near = lambda tol: 100 * np.mean([any(x <= tol for x in d) for d in per_q])
        print(f"{s:10s} | {100 * np.mean(dists == 0):5.1f} / {100 * np.mean(dists <= 30):5.1f} / "
              f"{100 * np.mean(dists <= 60):5.1f} | {median(dists.tolist()):7.0f} s | "
              f"{100 * np.mean([k >= 1 for k in inspan]):5.1f} / {100 * np.mean([k >= 2 for k in inspan]):5.1f} / "
              f"{100 * np.mean([k >= 3 for k in inspan]):5.1f} | {near(30):5.1f} / {near(60):5.1f}")


if __name__ == "__main__":
    main()
