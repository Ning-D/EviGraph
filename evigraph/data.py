"""Benchmark loaders: LVBench, LongVideoBench (val, (900, 3600] s) and Video-MME (Long).

Every item is a dict with at least
    _qid, _abs_video, question, options, correct_answer, task
plus a video key (`videoID` or `video_id`). `options` holds "A. ..." strings. LVBench
embeds the lettered options in the question text, so its `options` list is empty;
`split_options` recovers them.
"""
import glob
import json
import os
import re

LETTERS = "ABCDEFG"
OPT_RE = re.compile(r"^\s*[\(\[]?([A-G])[\)\].:]\s*(.+?)\s*$")
DATASETS = ("lvbench", "lvb", "vmme")


def video_key(item):
    return item.get("videoID") or item.get("video_id")


def split_options(item):
    """(stem, [option texts]) with the letter prefixes removed. LongVideoBench and
    Video-MME keep the options in item['options']; LVBench writes them as '(A) ...'
    lines inside the question."""
    opts = item.get("options") or []
    q = item["question"]
    if opts:
        clean = [(OPT_RE.match(str(o)).group(2) if OPT_RE.match(str(o)) else str(o)) for o in opts]
        return q.strip(), clean
    stem, parsed = [], []
    for ln in q.splitlines():
        m = OPT_RE.match(ln)
        if m and m.group(2):
            parsed.append(m.group(2).strip())
        else:
            stem.append(ln)
    return "\n".join(stem).strip(), parsed


def question_with_options(item):
    """Question text given to the graph-building model."""
    opts = "\n".join(item["options"])
    return f"{item['question']}\nOptions:\n{opts}"


# ── LongVideoBench ───────────────────────────────────────────────────────────
def _lvb_paths():
    root = os.environ.get("LVB_ROOT", "")
    return (os.environ.get("LVB_ANN", os.path.join(root, "lvb_val.json")),
            os.environ.get("LVB_VIDEO_DIR", os.path.join(root, "videos")))


def load_lvb(duration_group=3600, limit=None):
    ann, video_dir = _lvb_paths()
    out = []
    for r in json.load(open(ann)):
        if duration_group and r["duration_group"] != duration_group:
            continue
        vp = os.path.join(video_dir, r["video_path"])
        if not os.path.exists(vp):
            continue
        opts = [f"{chr(65 + i)}. {c}" for i, c in enumerate(r["candidates"])]
        out.append({**r, "_abs_video": vp, "_qid": r["id"],
                    "question": r["question"], "options": opts,
                    "correct_answer": chr(65 + r["correct_choice"]),
                    "task": r.get("question_category", "")})
    return out[:limit] if limit else out


def build_query_lvb(item):
    opts = "\n".join(item["options"])
    last = chr(64 + len(item["options"]))          # A-D or A-E
    return (f"Question: {item['question']}\n{opts}\n"
            f"Answer with the option's letter (A-{last}) only.")


# ── LVBench ──────────────────────────────────────────────────────────────────
def _lvbench_videos(keys):
    """youtube key -> local mp4. Videos are looked up as <LVBENCH_VIDEO_DIR>/<key>.mp4;
    for downloads made with video2dataset (webdataset shards named by index, each with
    a sibling .json holding the youtube url), set LVBENCH_SHARD_DIRS to the shard
    directories (os.pathsep-separated; the first hit wins)."""
    root = os.environ.get("LVBENCH_ROOT", "")
    video_dir = os.environ.get("LVBENCH_VIDEO_DIR", os.path.join(root, "videos"))
    m = {}
    for d in filter(None, os.environ.get("LVBENCH_SHARD_DIRS", "").split(os.pathsep)):
        for jf in glob.glob(os.path.join(d, "*.json")):
            mp4 = jf[:-5] + ".mp4"
            if not os.path.exists(mp4) or os.path.getsize(mp4) < 10000:
                continue
            try:
                url = str(json.load(open(jf)).get("url", ""))
            except Exception:
                continue
            mt = re.search(r"v=([A-Za-z0-9_-]{11})", url)
            if mt:
                m.setdefault(mt.group(1), mp4)
    for k in keys:
        p = os.path.join(video_dir, k + ".mp4")
        if k not in m and os.path.exists(p):
            m[k] = p
    return m


def _parse_time_reference(tr):
    """LVBench 'time_reference' ('mm:ss-mm:ss', the annotated answer clue) -> (start_s, end_s)."""
    if not tr or not isinstance(tr, str) or "-" not in tr:
        return None
    try:
        a, b = tr.split("-")

        def s(t):
            mm, ss = t.strip().split(":")
            return int(mm) * 60 + int(ss)
        return (s(a), s(b))
    except Exception:
        return None


def load_lvbench(limit=None):
    root = os.environ.get("LVBENCH_ROOT", "")
    meta_file = os.environ.get("LVBENCH_META", os.path.join(root, "data/video_info.meta.jsonl"))
    meta = [json.loads(l) for l in open(meta_file)]
    key2mp4 = _lvbench_videos([v["key"] for v in meta])
    out = []
    for v in meta:
        key = v["key"]
        mp4 = key2mp4.get(key)
        if not mp4:
            continue
        dur = float(v["video_info"]["duration_minutes"]) * 60.0
        for qa in v["qa"]:
            ans = re.search(r"[A-E]", str(qa.get("answer", "")).upper())
            if not ans:
                continue
            qtypes = qa.get("question_type") or []
            out.append({
                "_qid": f"{key}_{qa['uid']}",
                "_abs_video": mp4,
                "question": qa["question"],
                "options": [],                       # embedded in the question text
                "correct_answer": ans.group(0),
                "duration": dur,
                "videoID": key,
                "task": qtypes[0] if qtypes else "",
                "time_reference": qa.get("time_reference"),
                "gt_interval": _parse_time_reference(qa.get("time_reference")),
            })
    return out[:limit] if limit else out


def build_query_lvbench(item):
    return item["question"].strip() + "\nAnswer with the option's letter only."


# ── Video-MME ────────────────────────────────────────────────────────────────
def load_vmme(duration="long", limit=None):
    root = os.environ.get("VMME_ROOT", "")
    ann = os.environ.get("VMME_ANN", os.path.join(root, "annotations/test.json"))
    video_dir = os.environ.get("VMME_VIDEO_DIR", os.path.join(root, "videos/extracted/data"))
    if ann.endswith(".parquet"):
        import pandas as pd
        data = pd.read_parquet(ann).to_dict("records")
        for d in data:
            d["options"] = list(d["options"])
    else:
        data = json.load(open(ann))
    out = []
    for item in data:
        if duration and item["duration"] != duration:
            continue
        vp = os.path.join(video_dir, item["videoID"] + ".mp4")
        if not os.path.exists(vp):
            continue
        item["_abs_video"] = vp
        item["_qid"] = item["question_id"]
        item["correct_answer"] = item["answer"]
        item["task"] = item["task_type"]
        out.append(item)
    return out[:limit] if limit else out


def build_query_vmme(item):
    opts = "\n".join(item["options"])
    return (f"Question: {item['question']}\n{opts}\n"
            f"Answer with the option's letter (A, B, C or D) only.")


SPLITS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")


def load(dataset, limit=None, split="paper"):
    """-> (items, build_query). split="paper" keeps the questions evaluated in the paper
    (LVBench 1,242 / LongVideoBench 564 / Video-MME 900, see splits/); "all" keeps every
    question whose video is available locally."""
    if dataset == "lvb":
        items, bq = load_lvb(), build_query_lvb
    elif dataset == "lvbench":
        items, bq = load_lvbench(), build_query_lvbench
    elif dataset == "vmme":
        items, bq = load_vmme(), build_query_vmme
    else:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    if split == "paper":
        keep = {l.strip() for l in open(os.path.join(SPLITS, dataset + ".txt")) if l.strip()}
        items = [it for it in items if it["_qid"] in keep]
        if len(items) < len(keep):
            print(f"[data] {dataset}: {len(keep) - len(items)} of the {len(keep)} paper questions "
                  f"have no local video")
    return (items[:limit] if limit else items), bq
