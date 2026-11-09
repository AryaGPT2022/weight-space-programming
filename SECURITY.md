# Security notes

Checks run: secret scan (grep for API keys/tokens, none found), `bandit` static analysis, `pip-audit`
on the locked dependencies (no known vulnerabilities), and the repo's own `wsp.audit`.

## No hosted-LLM API in this repo
The code never calls Claude/Anthropic/OpenAI and never reads an API key. `wsp/config.py` holds a
start-up switch, `WSP_ALLOW_LLM_API` (default `0` = disabled). Any future code that needs a hosted
API must call `config.require_llm_api()`. Keys come from the environment, never from a committed file.

## Mitigated
- Hugging Face models and datasets are pinned to commit SHAs (`wsp/config.py`, `wsp/data.py`).
- `assert` used for a real check was replaced by an explicit exception.

## Accepted / known
- **Pickle (bandit B301/B403).** Intermediate pass results in `logs/passes/` are pickles written by
  this repo. They are git-ignored and must never be loaded from an untrusted source.
- `random` (B311) is used only for seeded benchmark splits, not for security.
- `wsp.audit --rebuild` runs the repo's own `edits/*.py` scripts via `subprocess` (B404/B603).
- Downloaded model weights are `safetensors`, which cannot execute code.
