#!/usr/bin/env bash
# Full pipeline on one benchmark: features -> graphs -> all selectors x budgets.
#   bash scripts/run_benchmark.sh lvb qwen [builder]
# benchmark: lvbench | lvb | vmme;  backbone: qwen | mplug | llava (run each in its own env)
# builder:   gpt-5.4-mini (default, needs OPENAI_API_KEY) or an open model served by vLLM
set -euo pipefail
DS=${1:?benchmark}
BACKBONE=${2:-qwen}
BUILDER=${3:-gpt-5.4-mini}
TAG=$(python -c "import sys; from evigraph.paths import model_tag; print(model_tag(sys.argv[1]))" "$BUILDER")

# 1. per-frame features (GPU) and LVNet keywords (LLM, same endpoint as the builder)
python -m evigraph.precompute blip  --dataset "$DS"
python -m evigraph.precompute clip  --dataset "$DS"
python -m evigraph.precompute lvnet --dataset "$DS" --keyword_model "$BUILDER"

# 2. Phase 1: one evidence graph per question (API / vLLM only, no local GPU needed)
python -m evigraph.build_graphs --dataset "$DS" --builder "$BUILDER"

# 3. Phase 2 + answering
for N in 8 16 32 64; do
  for S in uniform aks videotree lvnet; do
    python -m evigraph.answer --dataset "$DS" --selector "$S" --frames "$N" --backbone "$BACKBONE"
  done
  python -m evigraph.answer --dataset "$DS" --selector evigraph --graph_tag "$TAG" \
    --frames "$N" --backbone "$BACKBONE"
done
