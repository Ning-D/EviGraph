# EviGraph: Evidence-Graph Frame Selection for Long-Video Question Answering

Code for the ACCV 2026 paper by Ning Ding, Keisuke Fujii and Toru Tamaki.

EviGraph is a training-free, backbone-agnostic frame selector for long-video
multiple-choice QA. Instead of ranking frames only by their similarity to the
question, it decomposes the question into sub-needs, lets a VLM search the video
for evidence that covers them and separates the answer options, and then gives a
frozen answering model N frames: half from a BLIP-ITM relevance ranking and half
from the evidence graph.

- **Phase 1 – evidence-graph construction** (`evigraph/graph_builder.py`). A
  graph-building VLM scores 20 s clips against the question (the best clip becomes
  the core node), decomposes the question into 2–4 sub-needs, and runs a gap-driven
  policy loop of at most 24 steps. At every visited moment it captions a ±5 s
  snapshot, scores its support for each option (−2 … +2) and relates it to the core
  with a typed relation and a relevance score. One graph per question is reused for
  every frame budget and answering backbone.
- **Phase 2 – frame selection** (`evigraph/selection.py`). N/2 frames come from the
  BLIP-ITM relevance ranking of 1 fps frames. The rest come from graph nodes ranked
  in strict tiers (best node per sub-need → best supporter per option → the most
  discriminative remaining nodes); each node adds two CLIP-retrieved detail frames
  within ±8 s. The answering model sees only the frames.

## Repository layout

```
evigraph/
  data.py           LVBench / LongVideoBench / Video-MME loaders and prompts; splits/ = paper question lists
  precompute.py     BLIP-ITM scores and CLIP-L features of 1 fps frames; LVNet keywords
  graph_builder.py  Phase 1 (policy loop, node creation); prompts.py holds its fixed prompts
  build_graphs.py   Phase 1 command line (parallel over questions)
  selection.py      Phase 2 EviGraph selection
  baselines.py      uniform, AKS, VideoTree, LVNet selectors on the same features
  backbones.py      Qwen3-VL-8B, mPLUG-Owl3-7B, LLaVA-Video-7B answering wrappers
  answer.py         select frames with one selector and answer with one backbone
  grounding.py      LVBench grounding analysis (selected frames vs. annotated answer span)
scripts/
  env.example.sh    paths and model settings
  serve_builder.sh  vLLM server for an open-source graph builder
  run_benchmark.sh  the full pipeline on one benchmark
```

## Installation

```bash
conda create -n evigraph python=3.10 -y && conda activate evigraph
pip install -r requirements.txt
```

This environment covers feature extraction, graph building through an API, frame
selection and the Qwen3-VL-8B backbone. The other backbones need their own
environments: mPLUG-Owl3 runs with `transformers<4.48` (its remote code), and
LLaVA-Video needs the [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) package
(`LLAVA_NEXT_DIR` if it is not pip-installed). Open-source graph builders are served
with [vLLM](https://github.com/vllm-project/vllm), preferably in a separate environment.

## Data

| benchmark | questions used | setting |
|---|---|---|
| [LVBench](https://github.com/zai-org/LVBench) | 1,242 of 1,549 (videos available in our download) | test |
| [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench) | 564 | val, (900, 3600] s |
| [Video-MME](https://github.com/MME-Benchmarks/Video-MME) | 900 | Long, without subtitles |

Set the dataset locations in `scripts/env.sh` (copy `scripts/env.example.sh`). By
default every command evaluates exactly the questions of the paper
(`evigraph/splits/`); `--split all` uses every question whose video is available.

## Running

```bash
source scripts/env.sh

# 1. 1 fps BLIP-ITM relevance (per question) and CLIP-L features (per video); GPU
python -m evigraph.precompute blip --dataset lvb
python -m evigraph.precompute clip --dataset lvb

# 2. Phase 1: evidence graphs (no local GPU needed)
python -m evigraph.build_graphs --dataset lvb --builder gpt-5.4-mini --workers 12

# 3. Phase 2 + answering with a frozen backbone
python -m evigraph.answer --dataset lvb --selector evigraph --frames 32 --backbone qwen
```

`--selector` is one of `evigraph`, `uniform`, `aks`, `videotree`, `lvnet`; all of them
feed the same backbone with the same prompt. Results (per-question selected frames,
raw output and correctness) are written to `outputs/results/<tag>.json`; interrupted
runs resume. `scripts/run_benchmark.sh` runs every selector and budget.

### Open-source graph builders (no API cost)

Any OpenAI-compatible endpoint can build the graphs. With vLLM:

```bash
bash scripts/serve_builder.sh OpenGVLab/InternVL3-8B-hf 8000 0      # terminal 1
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY  # terminal 2
python -m evigraph.build_graphs --dataset lvb --builder OpenGVLab/InternVL3-8B-hf --workers 16
python -m evigraph.answer --dataset lvb --selector evigraph --graph_tag OpenGVLab/InternVL3-8B-hf \
    --frames 32 --backbone qwen
```

Graphs are stored per builder (`outputs/graphs/<graph_tag>/`, the tag being the model id
with `/` replaced by `--`). The paper evaluates InternVL3-8B, Qwen3-VL-8B, Qwen3-VL-2B and Qwen2.5-VL-7B
as builders.

### LVNet baseline

LVNet extracts a keyword bag from the answer options with an LLM (any OpenAI-compatible
model, e.g. the vLLM server above) and ranks frames by BLIP-ITM relevance to it:

```bash
python -m evigraph.precompute lvnet --dataset lvb --keyword_model gpt-5.4-mini
python -m evigraph.answer --dataset lvb --selector lvnet --frames 32 --backbone qwen
```

### Grounding analysis (LVBench)

```bash
python -m evigraph.grounding --frames 32
```

reports, for every selector, the share of selected frames inside / within 30 s / within
60 s of the annotated answer span, their median distance to it, and the share of
questions with at least k selected frames inside it.

## Reference numbers

Accuracy (%) with the Qwen3-VL-8B answering model and N = 32 frames:

| graph builder | LVBench | LongVideoBench | Video-MME |
|---|---|---|---|
| GPT-5.4-mini | 55.6 | 61.7 | 60.0 |
| InternVL3-8B | – | 62.2 | – |
| Qwen3-VL-8B | – | 59.2 | – |
| Qwen3-VL-2B | – | 58.7 | – |
| Qwen2.5-VL-7B | – | 58.3 | – |

Building a graph takes about 100 builder calls per question (≈100k input and 10k output
tokens; about $0.10 with GPT-5.4-mini; set `EVIGRAPH_COST_LOG=<file>` to count them).
LLM sampling makes rebuilt graphs differ from ours, and BLIP/CLIP features recomputed
on another GPU match ours only to fp16 precision, so expect small deviations unless
the released graphs and features are used.

## Citation

```bibtex
@inproceedings{ding2026evigraph,
  title     = {EviGraph: Evidence-Graph Frame Selection for Long-Video Question Answering},
  author    = {Ding, Ning and Fujii, Keisuke and Tamaki, Toru},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

AKS's `meanstd` selection is taken from [ncTimTang/AKS](https://github.com/ncTimTang/AKS);
VideoTree and LVNet are re-implemented on the same features as EviGraph.
