"""
core/pricing.py

Rough $ cost estimation from token counts, for Anthropic and OpenAI
models. Ollama is always free/local -- no pricing lookup applies.

IMPORTANT: these rates are a snapshot and WILL go stale. Anthropic and
OpenAI both revise pricing periodically (e.g. Anthropic cut Opus pricing
by 67% at the 4.6 launch). Treat every number here as an estimate for
rough budgeting, not a bill. Verify current rates at
https://docs.claude.com/en/docs/about-claude/pricing (Anthropic) or
https://platform.openai.com/docs/pricing (OpenAI) before relying on this
for anything beyond a ballpark.

Rates as of Sept 2026, $ per million tokens, (input, output):
"""
from __future__ import annotations

from typing import Optional

# Order matters: more specific keys (e.g. "gpt-4o-mini") must be checked
# before substrings they contain (e.g. "gpt-4o").
PRICING_TABLE: list[tuple[str, float, float]] = [
    ("opus", 5.00, 25.00),
    ("sonnet", 3.00, 15.00),
    ("haiku", 1.00, 5.00),
    ("gpt-4o-mini", 0.15, 0.60),
    ("gpt-4o", 2.50, 10.00),
]

PRICING_SOURCE_NOTE = (
    "Rates are an approximate snapshot (Sept 2026) and may be out of "
    "date -- verify current pricing at the provider's own pricing page "
    "before treating this as a real budget figure."
)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """
    Returns an estimated $ cost, or None if the model isn't recognized
    (including any Ollama model -- those are always free/local and
    intentionally excluded from this table).
    """
    model_lower = (model or "").lower()
    for key, price_in, price_out in PRICING_TABLE:
        if key in model_lower:
            return (prompt_tokens / 1_000_000) * price_in + (completion_tokens / 1_000_000) * price_out
    return None
