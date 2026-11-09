"""Runtime configuration, read once from environment variables.

This repo never calls a hosted LLM API (Claude/Anthropic, OpenAI, ...): the "weight programmer" works
offline through `wsp.harness`, and nothing in the code reads an API key. The switch below exists so
that any future code that *would* call such an API is off by default and must opt in explicitly.

    WSP_ALLOW_LLM_API=1   enable hosted-LLM API use (default: 0 = disabled)

API credentials are never stored in this repo. If you enable the switch, supply keys through your
shell environment (e.g. ANTHROPIC_API_KEY) or a secrets manager, never through a committed file.
"""
from __future__ import annotations

import os

# Pinned Hugging Face revisions (commit SHAs) so the downloaded weights cannot change under us.
BASE_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BASE_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
PRETRAINED_ID = "nreimers/MiniLM-L6-H384-uncased"
PRETRAINED_REVISION = "3276f0fac9d818781d7a1327b3ff818fc4e643c0"


def _flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


ALLOW_LLM_API = _flag("WSP_ALLOW_LLM_API", False)


class LLMApiDisabled(RuntimeError):
    pass


def require_llm_api(what: str = "this operation") -> None:
    """Call at the top of any code path that would contact a hosted LLM API."""
    if not ALLOW_LLM_API:
        raise LLMApiDisabled(f"{what} needs a hosted LLM API, which is disabled. "
                             "Set WSP_ALLOW_LLM_API=1 to enable it (credentials come from the environment).")


def check() -> dict:
    """Start-up check: report the active configuration, and never print secret values."""
    leaked = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if os.environ.get(k)]
    if leaked and not ALLOW_LLM_API:
        print(f"[wsp.config] note: {', '.join(leaked)} is set in the environment but unused "
              "(WSP_ALLOW_LLM_API=0).")
    return {"allow_llm_api": ALLOW_LLM_API, "base_revision": BASE_REVISION}
