"""Phase 1 CLI: build one evidence graph per question.

    python -m evigraph.build_graphs --dataset lvb --builder gpt-5.4-mini --workers 12

Graphs go to <EVIGRAPH_GRAPH_DIR>/<graph_tag>/<qid>.json (graph_tag defaults to the
builder name); already built graphs are skipped, so the command can be resumed.
Open-source builders: serve the model with vLLM, then
    OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
    python -m evigraph.build_graphs --dataset lvb --builder OpenGVLab/InternVL3-8B-hf
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import data, paths
from .graph_builder import build_graph


def graph_file(graph_tag, qid):
    return os.path.join(paths.GRAPHS, graph_tag, paths.safe_qid(qid) + ".json")


def coarse_file(builder, qid, clip_interval):
    return os.path.join(paths.COARSE, paths.model_tag(builder),
                        f"{paths.safe_qid(qid)}_ci{int(clip_interval)}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=data.DATASETS)
    ap.add_argument("--builder", default="gpt-5.4-mini", help="graph-building VLM (OpenAI-compatible model id)")
    ap.add_argument("--graph_tag", default=None, help="graph directory name (default: builder name)")
    ap.add_argument("--max_steps", type=int, default=24, help="step budget T of the policy loop")
    ap.add_argument("--min_explore", type=int, default=15, help="no early stop before this many nodes")
    ap.add_argument("--clip_interval", type=float, default=20.0, help="coarse clip length (s)")
    ap.add_argument("--snap_sec", type=float, default=5.0, help="snapshot half-window (s)")
    ap.add_argument("--snap_n", type=int, default=3, help="frames per snapshot")
    ap.add_argument("--max_side", type=int, default=448, help="longer image side sent to the builder")
    ap.add_argument("--workers", type=int, default=12, help="questions built in parallel")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="paper", choices=["paper", "all"],
                    help="paper: the questions evaluated in the paper; all: every question with a local video")
    ap.add_argument("--shard", default="0/1", help="i/n: build every n-th question starting at i")
    a = ap.parse_args()
    tag = paths.model_tag(a.graph_tag or a.builder)

    items, _ = data.load(a.dataset, limit=a.limit, split=a.split)
    si, sn = map(int, a.shard.split("/"))
    items = items[si::sn]
    todo = [it for it in items if not os.path.exists(graph_file(tag, it["_qid"]))]
    print(f"[build] {a.dataset}: {len(items)} questions, {len(items) - len(todo)} already built, "
          f"builder={a.builder} -> {os.path.join(paths.GRAPHS, tag)}", flush=True)

    def run(it):
        return build_graph(it, a.builder, graph_file(tag, it["_qid"]),
                           coarse_file(a.builder, it["_qid"], a.clip_interval),
                           max_steps=a.max_steps, min_explore=a.min_explore,
                           clip_interval=a.clip_interval, snap_sec=a.snap_sec,
                           snap_n=a.snap_n, max_side=a.max_side)

    t0, failed = time.time(), []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run, it): it for it in todo}
        for k, fut in enumerate(as_completed(futs)):
            it = futs[fut]
            try:
                g = fut.result()
            except Exception as e:           # keep going; rerun to retry
                g = None
                print(f"  !! {it['_qid']}: {e!r}", flush=True)
            if g is None:
                failed.append(it["_qid"])
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                dt = time.time() - t0
                print(f"  [{k + 1}/{len(todo)}] {dt:.0f}s ({dt / (k + 1):.1f}s/q)", flush=True)
    print(f"[build] done, {len(failed)} failed" + (f": {failed[:10]}" if failed else ""), flush=True)


if __name__ == "__main__":
    main()
