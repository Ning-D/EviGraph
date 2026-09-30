#!/usr/bin/env bash
# Serve an open-source graph-building VLM with vLLM (settings of the builder ablation).
#   bash scripts/serve_builder.sh OpenGVLab/InternVL3-8B-hf [port] [gpu]
# Models used in the paper: OpenGVLab/InternVL3-8B-hf, Qwen/Qwen3-VL-8B-Instruct,
# Qwen/Qwen3-VL-2B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct. Then build graphs with
#   OPENAI_BASE_URL=http://127.0.0.1:<port>/v1 OPENAI_API_KEY=EMPTY \
#   python -m evigraph.build_graphs --dataset lvb --builder <model> --workers 16
set -euo pipefail
MODEL=${1:?model id}
PORT=${2:-8000}
GPU=${3:-0}
CUDA_VISIBLE_DEVICES=$GPU exec vllm serve "$MODEL" --port "$PORT" --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":40}' --max-num-seqs 48 --trust-remote-code
