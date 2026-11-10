"""LLM interface. The planner never calls it; only three steps do (doc, part 3):
build the graph and readings, phrase a question about a slot, and write the final OverpassQL.

Real client: any OpenAI-compatible endpoint (DashScope for Qwen, etc.), configured by env vars
  AMBITOPO_LLM_BASE_URL, AMBITOPO_LLM_MODEL, AMBITOPO_LLM_KEY_ENV (name of the env var holding the key)
The key itself is never written to disk or logs. MockLLM keeps every pipeline runnable offline.
"""
from __future__ import annotations

import json
import os
import urllib.request

from .topology import Slot


class LLM:
    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        raise NotImplementedError

    def phrase_question(self, request: str, slot: Slot) -> str:
        opts = "; ".join(r.label for r in slot.readings)
        return self.chat([
            {"role": "system", "content": "You help a non-technical user clarify a map search. Ask ONE short "
             "question in plain language. Do not mention OpenStreetMap tags or query syntax."},
            {"role": "user", "content": "Request: %s\nThe unclear part (%s) could mean: %s\nQuestion:" %
             (request, slot.kind, opts)}])


class OpenAICompatible(LLM):
    def __init__(self, base_url: str | None = None, model: str | None = None, key_env: str | None = None):
        self.base_url = (base_url or os.environ.get("AMBITOPO_LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("AMBITOPO_LLM_MODEL", "")
        self.key_env = key_env or os.environ.get("AMBITOPO_LLM_KEY_ENV", "DASHSCOPE_API_KEY")
        if not (self.base_url and self.model and os.environ.get(self.key_env)):
            raise RuntimeError("LLM not configured: set AMBITOPO_LLM_BASE_URL, AMBITOPO_LLM_MODEL and %s"
                               % self.key_env)

    def chat(self, messages, temperature=0.0, max_tokens=1024):
        req = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if os.environ.get("AMBITOPO_LLM_NOTHINK"):          # Qwen3-style models: skip the thinking block
            req["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(req).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + os.environ[self.key_env]})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()


class MockLLM(LLM):
    def chat(self, messages, temperature=0.0, max_tokens=1024):
        return "[mock] " + messages[-1]["content"].splitlines()[-2][:120]
