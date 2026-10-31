"""Explicit opt-in OpenAI Responses API adapter for role-based JSON proposals.

The same protocol can be implemented by another provider. External calls are
never made in default offline mode. No model may invoke tools or approve acts.
"""
from __future__ import annotations

import json
import os
import re
from urllib import request


class OpenAIResponsesModel:
    ENDPOINT = "https://api.openai.com/v1/responses"

    def __init__(self, *, allow_external_data_transfer: bool = False,
                 model: str | None = None, key: str | None = None,
                 timeout: int = 20):
        if not allow_external_data_transfer:
            raise PermissionError("--allow-external-data-transfer required for API access")
        self.model = model or os.environ.get("AGENT_FORGE_MODEL", "")
        self.key = key or os.environ.get("OPENAI_API_KEY", "")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", self.model):
            raise ValueError("Set an available model ID using AGENT_FORGE_MODEL")
        if not self.key:
            raise ValueError("OPENAI_API_KEY required")
        if type(timeout) is not int or not 1 <= timeout <= 120:
            raise ValueError("timeout must be 1–120 seconds")
        self.timeout = timeout

    def ask(self, role: str, instructions: str, context: dict) -> dict:
        if role not in {"planner", "researcher", "verifier", "reporter"}:
            raise ValueError("unknown model role")
        data = json.dumps({
            "model": self.model,
            "instructions": instructions + " Return a single JSON object and nothing else.",
            "input": json.dumps(context, ensure_ascii=True),
            "store": False,
            "max_output_tokens": 1000,
        }).encode("utf-8")
        if len(data) > 100_000:
            raise ValueError("model input exceeds 100 kB cap")
        req = request.Request(self.ENDPOINT, data=data, method="POST",
                              headers={"Authorization": f"Bearer {self.key}",
                                       "Content-Type": "application/json"})
        # No raw payload, secret or response logging. Transport errors propagate.
        with request.urlopen(req, timeout=self.timeout) as stream:
            raw = stream.read(256_001)
        if len(raw) > 256_000:
            raise ValueError("model response exceeds size cap")
        response = json.loads(raw)
        if not isinstance(response, dict) or response.get("status") != "completed":
            raise ValueError("model response incomplete")
        pieces = [part.get("text", "")
                  for item in response.get("output", []) if isinstance(item, dict)
                  for part in item.get("content", []) if isinstance(part, dict)
                  and part.get("type") == "output_text"]
        text = "\n".join(pieces).strip()
        if not text or len(text) > 16_000:
            raise ValueError("invalid model text output")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("model must return JSON object")
        return parsed
