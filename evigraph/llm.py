"""JSON chat calls to the graph-building VLM through an OpenAI-compatible API.

GPT models: set OPENAI_API_KEY (or OPENAI_API_KEYS=key1,key2,... to round-robin
several keys). Open-source builders: serve the model with vLLM and point
OPENAI_BASE_URL at it (OPENAI_API_KEY=EMPTY). Set EVIGRAPH_COST_LOG=<file> to
record the number of calls and prompt/completion tokens.
"""
import base64
import itertools
import json
import os
import re
import threading
import time

import cv2

_lock = threading.Lock()
_clients = None
_cost = {"calls": 0, "in": 0, "out": 0}


def _next_client():
    global _clients
    with _lock:
        if _clients is None:
            from openai import OpenAI
            keys = [k.strip() for k in os.environ.get("OPENAI_API_KEYS", "").split(",") if k.strip()]
            _clients = itertools.cycle([OpenAI(api_key=k) for k in keys] if keys else [OpenAI()])
        return next(_clients)


def _log_cost(usage):
    path = os.environ.get("EVIGRAPH_COST_LOG")
    if usage is None or not path:
        return
    with _lock:
        _cost["calls"] += 1
        _cost["in"] += getattr(usage, "prompt_tokens", 0) or 0
        _cost["out"] += getattr(usage, "completion_tokens", 0) or 0
        with open(path, "w") as f:
            json.dump(_cost, f)


def chat_json(model, system, content, max_tokens=800):
    """One chat call; returns the first {...} object parsed from the reply, or None.
    `content` is a string or a list of OpenAI content parts (text / image_url)."""
    for attempt in range(3):
        try:
            r = _next_client().chat.completions.create(
                model=model, max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": content}])
            _log_cost(getattr(r, "usage", None))
            text = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(m.group()) if m else None
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2)


def image_part(frame, quality=80):
    """RGB frame -> low-detail JPEG content part."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}}


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"
