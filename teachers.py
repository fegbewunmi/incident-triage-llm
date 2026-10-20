"""Thin clients for hosted LLMs, used both to generate alert text (teacher) and
as evaluation baselines. Each returns (text, usage, latency_s).

Gemini goes through Vertex AI with Application Default Credentials, the same
setup as ai-operations-center:
    gcloud auth application-default login
    export GOOGLE_CLOUD_PROJECT=ai-ops-center-eb26   # or your project
    export GOOGLE_CLOUD_LOCATION=us-central1
Set GEMINI_API_KEY instead to use the Gemini Developer API.
Anthropic uses ANTHROPIC_API_KEY.
"""

import json
import os
import time


def _retry(fn, attempts=5):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # network / rate limit / transient server errors
            if i == attempts - 1:
                raise
            wait = 2 ** i + 1
            print(f"  retrying in {wait}s after: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(wait)


class GeminiClient:
    def __init__(self, model="gemini-2.5-flash", thinking_budget: int | None = 0):
        from google import genai
        from google.genai import types

        self.types = types
        if os.environ.get("GEMINI_API_KEY"):
            self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        else:
            self.client = genai.Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
        self.model = model
        self.thinking_budget = thinking_budget  # 0 = off, -1 = dynamic, None = model default

    def complete(self, system: str, user: str, json_mode=True, temperature=0.0, max_tokens=4096):
        t = self.types
        cfg = dict(system_instruction=system, temperature=temperature, max_output_tokens=max_tokens)
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if self.thinking_budget is not None:
            cfg["thinking_config"] = t.ThinkingConfig(thinking_budget=self.thinking_budget)

        def call():
            start = time.perf_counter()
            r = self.client.models.generate_content(model=self.model, contents=user, config=t.GenerateContentConfig(**cfg))
            return r, time.perf_counter() - start

        r, latency = _retry(call)
        u = r.usage_metadata
        usage = {
            "input_tokens": u.prompt_token_count or 0,
            "output_tokens": u.candidates_token_count or 0,
            "thinking_tokens": getattr(u, "thoughts_token_count", 0) or 0,
        }
        return r.text or "", usage, latency


class AnthropicClient:
    def __init__(self, model="claude-haiku-5-5"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, user: str, json_mode=True, temperature=0.0, max_tokens=4096):
        def call():
            start = time.perf_counter()
            r = self.client.messages.create(
                model=self.model, system=system, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": user}],
            )
            return r, time.perf_counter() - start

        r, latency = _retry(call)
        text = "".join(b.text for b in r.content if b.type == "text")
        usage = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens, "thinking_tokens": 0}
        return text, usage, latency


class MockTeacher:
    """Offline stand-in for pipeline tests. Writes plain, template-based alert
    text straight from the spec. Not for real training data."""

    def complete(self, system: str, user: str, json_mode=True, temperature=0.0, max_tokens=4096):
        specs = json.loads(user[user.index("["):])
        out = []
        for s in specs:
            ev = s["evidence"]
            desc = f"{s['service']} alert. " + "; ".join(f"{k}={v}" for k, v in ev.items())
            if s.get("dependency_name"):
                desc += f"; upstream calls to {s['dependency_name']} timing out"
            out.append({
                "id": s["id"],
                "alert_name": "MockAlert",
                "description": desc,
                "alert_metadata": {"region": "us-central1"},
                "rationale": f"Evidence on {s['service']} points to the cause.",
            })
        return json.dumps(out), {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}, 0.0


def make_client(name: str, model: str | None = None, thinking_budget: int | None = 0):
    if name == "gemini":
        return GeminiClient(model or "gemini-2.5-flash", thinking_budget=thinking_budget)
    if name == "anthropic":
        return AnthropicClient(model or "claude-haiku-5-5")
    if name == "mock":
        return MockTeacher()
    raise ValueError(f"unknown client {name}")
