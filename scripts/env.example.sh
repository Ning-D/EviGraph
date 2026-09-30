# Copy to scripts/env.sh (git-ignored), edit, then `source scripts/env.sh`.

# ── benchmarks ────────────────────────────────────────────────────────────────
# LongVideoBench: <LVB_ROOT>/lvb_val.json and <LVB_ROOT>/videos/
export LVB_ROOT=/data/LongVideoBench
# LVBench: <LVBENCH_ROOT>/data/video_info.meta.jsonl and <LVBENCH_ROOT>/videos/<youtube key>.mp4
# (for video2dataset downloads, list the shard directories in LVBENCH_SHARD_DIRS instead)
export LVBENCH_ROOT=/data/LVBench
# export LVBENCH_SHARD_DIRS=/data/LVBench/scripts/videos/00000:/data/LVBench/scripts/videos1/00000
# Video-MME: annotation json/parquet and a directory of <videoID>.mp4
export VMME_ROOT=/data/Video-MME
# export VMME_ANN=$VMME_ROOT/videomme/test-00000-of-00001.parquet
# export VMME_VIDEO_DIR=$VMME_ROOT/data

# ── outputs (features, graphs, results); every sub-directory can also be set on its own
export EVIGRAPH_OUT=$PWD/outputs

# ── graph-building VLM ────────────────────────────────────────────────────────
# GPT-5.4-mini through the OpenAI API ...
export OPENAI_API_KEY=sk-...
# ... or an open-source VLM served with vLLM (scripts/serve_builder.sh):
# export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY

# ── answering backbones ───────────────────────────────────────────────────────
export QWEN_MODEL=Qwen/Qwen3-VL-8B-Instruct
export MPLUG_MODEL=mPLUG/mPLUG-Owl3-7B-241101
export LLAVAVIDEO_MODEL=lmms-lab/LLaVA-Video-7B-Qwen2
# export LLAVA_NEXT_DIR=/path/to/LLaVA-NeXT   # if the llava package is not pip-installed
