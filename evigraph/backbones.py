"""Frozen answering backbones. Each exposes answer_mc(frames, query, max_new_tokens)
where frames are HxWx3 RGB arrays; decoding is greedy.

  qwen   Qwen3-VL-8B-Instruct (QWEN_MODEL)          transformers >= 4.57
  mplug  mPLUG-Owl3-7B-241101 (MPLUG_MODEL)         transformers < 4.48 (remote code)
  llava  LLaVA-Video-7B-Qwen2 (LLAVAVIDEO_MODEL)    LLaVA-NeXT package (LLAVA_NEXT_DIR)

The three need different environments, so every import is local to its class.
"""
import copy
import os
import sys

import numpy as np

BACKBONES = ("qwen", "mplug", "llava")


def _to_pil(frame):
    from PIL import Image
    if isinstance(frame, Image.Image):
        return frame
    return Image.fromarray(frame.astype(np.uint8))


class QwenVL:
    def __init__(self, model_id=None, device="cuda"):
        import torch
        from transformers import AutoProcessor
        model_id = model_id or os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        if "3-VL" in model_id or "Qwen3VL" in model_id:
            from transformers import Qwen3VLForConditionalGeneration as cls
        elif "2.5" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration as cls
        else:
            from transformers import Qwen2VLForConditionalGeneration as cls
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device,
                                         attn_implementation="sdpa").eval()

    def answer_mc(self, frames, query, max_new_tokens=8):
        import torch
        content = [{"type": "image", "image": _to_pil(f)} for f in frames]
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [c["image"] for c in content if c["type"] == "image"]
        inputs = self.processor(text=[text], images=images or None, padding=True,
                                return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0].strip()


class MplugOwl3:
    def __init__(self, model_path=None, device="cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        model_path = model_path or os.environ.get("MPLUG_MODEL", "mPLUG/mPLUG-Owl3-7B-241101")
        self.model = AutoModel.from_pretrained(model_path, attn_implementation="sdpa",
                                               torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = self.model.init_processor(self.tokenizer)
        self.device = device

    def answer_mc(self, frames, query, max_new_tokens=8):
        import torch
        messages = [{"role": "user", "content": f"<|video|>\n{query}"},
                    {"role": "assistant", "content": ""}]
        inputs = self.processor(messages, images=None, videos=[[_to_pil(f) for f in frames]])
        inputs.to(self.device)
        inputs.update({"tokenizer": self.tokenizer, "max_new_tokens": max_new_tokens,
                       "decode_text": True})
        with torch.inference_mode():
            g = self.model.generate(**inputs)
        out = g[0] if isinstance(g, (list, tuple)) else g
        return str(out).strip()


class LLaVAVideo:
    def __init__(self, model_path=None, device="cuda"):
        next_dir = os.environ.get("LLAVA_NEXT_DIR")
        if next_dir and next_dir not in sys.path:
            sys.path.insert(0, next_dir)
        from llava.model.builder import load_pretrained_model
        model_path = model_path or os.environ.get("LLAVAVIDEO_MODEL", "lmms-lab/LLaVA-Video-7B-Qwen2")
        self.tok, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", torch_dtype="bfloat16", device_map="auto",
            attn_implementation="sdpa")
        self.model.eval()
        self.device = device

    def answer_mc(self, frames, query, max_new_tokens=8):
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        arr = np.stack([np.asarray(f) for f in frames])
        video = self.image_processor.preprocess(arr, return_tensors="pt")["pixel_values"]
        video = [video.to(self.device).bfloat16()]
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + query)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), self.tok, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(input_ids, images=video, modalities=["video"],
                                      do_sample=False, temperature=0, max_new_tokens=max_new_tokens)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def load_backbone(name):
    return {"qwen": QwenVL, "mplug": MplugOwl3, "llava": LLaVAVideo}[name]()
