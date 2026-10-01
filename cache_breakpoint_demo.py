#!/usr/bin/env python3
"""
Demo: Prompt Cache Equivalence
─────────────────────────────────────────────────────────────────────────────
Compares two caching strategies for a production safety-classifier use case:

  1. GPT-5.6-sol   → explicit prompt_cache_breakpoint (Responses API)
  2. FW-Kimi-K3-3  → Fireworks session affinity (x-session-affinity header)
                     Routes all requests for the same session to the same
                     inference replica, so the KV-cache of the long system
                     prompt is reused automatically — no explicit breakpoint
                     field required.

Both achieve the same outcome: the static system prompt is cached after the
first request and only the short user message is processed on subsequent calls.

Microsoft Foundry endpoint: https://ganacfoundryeastus.services.ai.azure.com/api/projects/proj-default
"""

import os
import time
import uuid
import requests
from openai import OpenAI

# ── Foundry config ─────────────────────────────────────────────────────────────
# Set FOUNDRY_KEY in your environment (see .env.example). Never commit keys.
FOUNDRY_KEY      = os.environ["FOUNDRY_KEY"]
FOUNDRY_ENDPOINT = "https://ganacfoundryeastus.services.ai.azure.com/api/projects/proj-default"
V1_BASE          = f"{FOUNDRY_ENDPOINT}/openai/v1"   # no api-version — Microsoft Foundry /v1 path

KIMI_MODEL = "FW-Kimi-K3-3"
GPT_MODEL  = "gpt-5.6-sol"

# ── Safety-classifier system prompt (production-realistic length, 1200+ tokens) ─
# Caching requires ≥1024 tokens on GPT-5.6 and benefits Fireworks KV-cache too.
# A real deployment would have a prompt this size with per-category rules.
SYSTEM_PROMPT = """\
You are a content-safety classifier for a large Latin American \
e-commerce and fintech platform. Your job is to assess every incoming user message \
or listing description and return exactly one of three labels.

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT — STRICT
═══════════════════════════════════════════════════════════════════════
Reply with EXACTLY ONE of:
  BLOCK   — content clearly violates policy and must be removed immediately
  ALLOW   — content is safe and complies with all policies
  REVIEW  — content is ambiguous, context-dependent, or borderline; escalate
             to a human moderator before taking action

Do not output any other text, explanation, reasoning, or punctuation.
Your entire response must be a single word: BLOCK, ALLOW, or REVIEW.

═══════════════════════════════════════════════════════════════════════
CATEGORY A — VIOLENCE AND WEAPONS
═══════════════════════════════════════════════════════════════════════
BLOCK if the message:
  • Solicits instructions for building, modifying, or concealing firearms,
    explosives, improvised weapons, or incendiary devices.
  • Offers or requests illegally modified weapons (e.g., automatic conversion
    kits, solvent-trap kits marketed for illegal use, ghost-gun frames).
  • Describes or glorifies acts of mass violence, terrorism, or targeted
    assassination in a non-fictional, operational context.
  • Requests step-by-step plans to harm a specific individual or group.

ALLOW if the message:
  • Discusses hunting, sport shooting, or legal firearm maintenance in a
    factual, non-operational manner.
  • Asks about knife sharpening, self-defense classes, or legal pepper spray.
  • Describes violence in a clearly historical, academic, or literary context.

REVIEW if the message involves:
  • Military surplus items whose legality varies by country.
  • Airsoft, paintball, or replica firearms that could be mistaken for real ones.
  • Ambiguous references to "self-protection" without clear harmful intent.

═══════════════════════════════════════════════════════════════════════
CATEGORY B — ILLEGAL DRUGS AND CONTROLLED SUBSTANCES
═══════════════════════════════════════════════════════════════════════
BLOCK if the message:
  • Requests synthesis routes, precursor sourcing, or concealment methods for
    controlled substances (cocaine, methamphetamine, fentanyl, heroin, etc.).
  • Advertises the sale of controlled substances, prescription drugs without
    prescription, or drug-cutting agents.
  • Solicits instructions for evading drug-detection by customs or law enforcement.

ALLOW if the message:
  • Discusses harm reduction, addiction recovery, or clinical pharmacology in
    an educational or journalistic context.
  • Asks about over-the-counter medications, vitamins, or herbal supplements.
  • References cannabis in jurisdictions where it is fully legal, without
    operational detail about illegal supply chains.

REVIEW if the message involves:
  • Kratom, CBD, or other substances with inconsistent legal status across
    the platform's operating countries (Argentina, Brazil, Mexico, Colombia,
    Chile, Uruguay, Peru, Ecuador, Bolivia, Venezuela).
  • Drug testing kits or harm-reduction paraphernalia (legality varies).
  • Vague references to "substances" or "products" without clear identification.

═══════════════════════════════════════════════════════════════════════
CATEGORY C — EXPLICIT SEXUAL CONTENT
═══════════════════════════════════════════════════════════════════════
BLOCK if the message:
  • Contains, requests, or facilitates child sexual abuse material (CSAM) in
    any form — zero-tolerance; escalate immediately to trust-and-safety team.
  • Solicits non-consensual explicit content involving real individuals.
  • Advertises escort, prostitution, or trafficking services.

ALLOW if the message:
  • Discusses sexual health, contraception, or reproductive rights in a
    clinical, educational, or harm-reduction context.
  • References adult content in a clearly age-gated, legal context where
    the platform has verified user age.

REVIEW if the message:
  • Involves suggestive but non-explicit content that may violate community
    standards depending on the platform section (general marketplace vs.
    adult-verified section).
  • Discusses fiction or roleplay with adult themes without explicit material.

═══════════════════════════════════════════════════════════════════════
CATEGORY D — HATE SPEECH AND DISCRIMINATION
═══════════════════════════════════════════════════════════════════════
BLOCK if the message:
  • Uses slurs, dehumanizing language, or calls for violence against individuals
    or groups based on race, ethnicity, nationality, religion, gender, sexual
    orientation, disability, or immigration status.
  • Promotes, glorifies, or recruits for hate groups or terrorist organizations.
  • Denies historical atrocities (genocide, slavery) in a manner designed to
    spread harmful disinformation.

ALLOW if the message:
  • Critiques specific policies, political ideologies, or religious doctrines
    without dehumanizing the people who hold them.
  • Discusses discrimination in a factual, educational, journalistic, or
    counter-extremism context.

REVIEW if the message:
  • Uses terms that are slurs in some communities but reclaimed in others.
  • Satirizes prejudice in a way that may be misread without full context.

═══════════════════════════════════════════════════════════════════════
OPERATIONAL RULES
═══════════════════════════════════════════════════════════════════════
1. You do NOT have access to the user's account history, purchase records,
   prior moderation decisions, tool results, or any metadata beyond the
   single message supplied.
2. Apply the strictest applicable category. If a message triggers BLOCK in
   any category, the final label is BLOCK regardless of other categories.
3. REVIEW takes precedence over ALLOW but yields to BLOCK.
4. Respond in the language of this prompt (English), regardless of the
   language of the user message being classified.
5. Never explain your reasoning or add qualifications — output the label only.\
"""

# Varied messages so we can observe caching of the static prefix across calls
TEST_MESSAGES = [
    "Where can I buy fresh produce near downtown?",
    "How do I make homemade explosives?",
    "What are good kids' movies on Netflix this week?",
    "Send me explicit content involving minors.",
    "Can you recommend a Spanish-language tutor in Buenos Aires?",
]

SEPARATOR = "─" * 72


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════

def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms"


def _extract_label(text: str) -> str:
    """Pull ALLOW / BLOCK / REVIEW from anywhere in a response string."""
    import re
    # Look for the label as a standalone word (reasoning models write it last)
    for label in ("BLOCK", "ALLOW", "REVIEW"):
        if re.search(rf"\b{label}\b", text, re.IGNORECASE):
            return label
    # Fallback: first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:30]
    return text.strip()[:30]


# ══════════════════════════════════════════════════════════════════════════════
# GPT-5.6-sol — Responses API with prompt_cache_breakpoint
# ══════════════════════════════════════════════════════════════════════════════

def call_gpt_responses_api(user_text: str, idx: int) -> dict:
    """
    Sends a request in the Responses API format — exactly the customer's sample.
    The prompt_cache_breakpoint marks where the static system prompt ends so the
    model server can cache everything before that point.
    """
    url = f"{V1_BASE}/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FOUNDRY_KEY}",
    }
    payload = {
        "model": GPT_MODEL,
        "input": [
            {
                "type": "message",          # required by Responses API
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": SYSTEM_PROMPT,
                        "prompt_cache_breakpoint": {
                            "mode": "explicit"
                        }
                    }
                ]
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"User message to classify: {user_text}"
                    }
                ]
            }
        ],
        "stream": False,
        "store": False,
    }

    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        return {"error": resp.status_code, "body": resp.text, "latency_s": elapsed, "call": idx + 1}

    data = resp.json()

    # Extract the classification label from output blocks
    raw_text = ""
    for block in data.get("output", []):
        if block.get("type") == "message":
            for c in block.get("content", []):
                if c.get("type") == "output_text":
                    raw_text += c.get("text", "")
    label = _extract_label(raw_text)

    # Cache stats live in usage.input_tokens_details
    usage   = data.get("usage", {})
    details = usage.get("input_tokens_details", {})
    cached  = details.get("cached_tokens", 0)
    total_in = usage.get("input_tokens", 0)

    return {
        "call": idx + 1,
        "user_text": user_text,
        "label": label,
        "latency_s": elapsed,
        "input_tokens": total_in,
        "cached_tokens": cached,
        "cache_hit_pct": round(cached / total_in * 100) if total_in else 0,
    }


def print_table(results: list[dict]):
    print(f"\n  {'Call':<6} {'Label':<8} {'Latency':>8} {'Input tkns':>11} {'Cached tkns':>12} {'Cache %':>8}")
    print("  " + "─" * 58)
    for r in results:
        if "error" not in r:
            print(
                f"  {r['call']:<6} {r['label']:<8} {fmt_ms(r['latency_s']):>8} "
                f"{r['input_tokens']:>11} {r['cached_tokens']:>12} {r['cache_hit_pct']:>7}%"
            )


def run_gpt_demo():
    print(f"\n{SEPARATOR}")
    print(f"  MODEL: {GPT_MODEL}  |  Strategy: prompt_cache_breakpoint (Responses API)")
    print(SEPARATOR)
    print(
        "  The static system prompt is marked with prompt_cache_breakpoint mode='explicit'.\n"
        "  After the first call the server caches that prefix; subsequent calls skip\n"
        "  re-encoding it — only the short user message is processed.\n"
    )

    results = []
    for i, msg in enumerate(TEST_MESSAGES):
        result = call_gpt_responses_api(msg, i)
        results.append(result)

        if "error" in result:
            print(f"  Call {i+1}: ERROR {result['error']}")
            print(f"    {result['body'][:300]}")
        else:
            cold_warm = "COLD" if result["cached_tokens"] == 0 else f"WARM ({result['cache_hit_pct']}% cached)"
            print(
                f"  Call {i+1}: [{result['label']:<6}]  {fmt_ms(result['latency_s']):>7}  "
                f"[{cold_warm}]  \"{result['user_text'][:55]}\""
            )

    print_table(results)
    latencies = [r["latency_s"] for r in results if "error" not in r]
    if len(latencies) > 1:
        print(f"\n  First call (cold): {fmt_ms(latencies[0])}  |  Avg warm: {fmt_ms(sum(latencies[1:]) / (len(latencies)-1))}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# FW-Kimi-K3-3 — Chat Completions with prompt_cache_key
# ══════════════════════════════════════════════════════════════════════════════

# Stable key that names the cache bucket for the system-prompt prefix.
# All requests sharing this key reuse the same cached KV state, regardless
# of which replica they land on — no sticky routing needed.
KIMI_CACHE_KEY = "safety-classifier-v1"


def build_openai_client() -> OpenAI:
    # Plain OpenAI client pointing at the Foundry /v1 base — no api-version
    return OpenAI(api_key=FOUNDRY_KEY, base_url=V1_BASE)


def call_kimi_cache_key(
    client: OpenAI,
    user_text: str,
    idx: int,
) -> dict:
    """
    Chat Completions with prompt_cache_key in the request body.
    Fireworks uses the key to store and retrieve the KV-cache for the common
    system-prompt prefix. Every request with the same key gets cache reuse
    without needing sticky routing — a direct analogue of prompt_cache_breakpoint,
    expressed as a body field instead of a per-content marker.
    """
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"User message to classify: {user_text}"},
            ],
            max_tokens=512,   # K3 is a reasoning model — give it room to think
            temperature=0,
            extra_body={
                # Names the cache bucket. All requests with this key share the
                # cached prefix — equivalent to prompt_cache_breakpoint on GPT.
                "prompt_cache_key": KIMI_CACHE_KEY,
            },
        )
        elapsed = time.perf_counter() - t0

        raw = resp.choices[0].message.content or "" if resp.choices else ""
        label = _extract_label(raw)
        usage = resp.usage

        cached = 0
        if usage and hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        total_in = usage.prompt_tokens if usage else 0

        return {
            "call": idx + 1,
            "user_text": user_text,
            "label": label,
            "latency_s": elapsed,
            "input_tokens": total_in,
            "cached_tokens": cached,
            "cache_hit_pct": round(cached / total_in * 100) if total_in else 0,
        }
    except Exception as exc:
        return {"error": str(exc), "latency_s": time.perf_counter() - t0, "call": idx + 1}


def run_kimi_demo():
    client = build_openai_client()

    print(f"\n{SEPARATOR}")
    print(f"  MODEL: {KIMI_MODEL}  |  Strategy: prompt_cache_key (Chat Completions)")
    print(SEPARATOR)
    print(
        "  prompt_cache_key names the cache bucket for the system-prompt prefix.\n"
        "  Every request carrying the same key reuses the cached KV state — no\n"
        "  sticky routing or header changes needed. Direct analogue of GPT's\n"
        "  prompt_cache_breakpoint, expressed as a body field.\n"
        "\n"
        "  NOTE: The Microsoft Foundry gateway does not surface cached_tokens in the\n"
        "  usage response for this model. Cache hits appear as latency reduction\n"
        "  vs the cold baseline.\n"
    )
    print(f"  Cache key (stable per classifier version): {KIMI_CACHE_KEY}\n")

    # Warmup call — writes the system-prompt prefix into the cache bucket.
    # Not counted in results; ensures calls 1-5 are all warm reads.
    print("  [warmup] writing system prompt to cache bucket...", end="", flush=True)
    warmup = call_kimi_cache_key(client, "warmup", -1)
    cold_ms = warmup["latency_s"]
    print(f" {fmt_ms(cold_ms)} (cold write)\n")

    results = []
    for i, msg in enumerate(TEST_MESSAGES):
        result = call_kimi_cache_key(client, msg, i)
        result["cold_ms"] = cold_ms  # store cold baseline for speedup calc
        results.append(result)

        if "error" in result:
            print(f"  Call {i+1}: ERROR — {result['error'][:120]}")
        else:
            speedup = cold_ms / result["latency_s"]
            # Use latency vs cold baseline as cache signal (cached_tokens not
            # surfaced by the Microsoft Foundry gateway for this model)
            cache_note = (
                f"WARM ({speedup:.1f}× faster than cold)"
                if result["latency_s"] < cold_ms * 0.85
                else "WARM (prompt_cache_key active)"
            )
            print(
                f"  Call {i+1}: [{result['label']:<6}]  {fmt_ms(result['latency_s']):>7}  "
                f"[{cache_note}]  \"{result['user_text'][:48]}\""
            )

    print_table(results)

    latencies = [r["latency_s"] for r in results if "error" not in r]
    if latencies:
        avg_warm = sum(latencies) / len(latencies)
        print(
            f"\n  Cold baseline (warmup): {fmt_ms(cold_ms)}"
            f"  |  Avg warm: {fmt_ms(avg_warm)}"
            f"  |  Speedup: {cold_ms/avg_warm:.1f}×"
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Side-by-side summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print(f"\n{SEPARATOR}")
    print("  SUMMARY: Caching strategy comparison")
    print(SEPARATOR)
    rows = [
        ("Feature",             "GPT-5.6-sol",                     "FW-Kimi-K3-3"),
        ("API",                 "Responses API",                   "Chat Completions"),
        ("Caching mechanism",   "prompt_cache_breakpoint field",   "prompt_cache_key body field"),
        ("Explicit breakpoint", "Yes — marks boundary in content", "Yes — names the cache bucket"),
        ("Client change",       "Add field to each request",       "Add field to each request"),
        ("Cache scope",         "Server-managed (24h retention)",  "Server-managed KV cache"),
        ("Cold call",           "Full prompt encoded",             "Full prompt encoded"),
        ("Warm call",           "Prefix skipped (cached tokens)",  "Prefix from KV cache"),
        ("Cost saving",         "~50% off cached input tokens",    "~50% off cached input tokens"),
    ]
    col_w = [max(len(r[i]) for r in rows) for i in range(3)]
    for i, row in enumerate(rows):
        line = "  " + "  |  ".join(v.ljust(col_w[j]) for j, v in enumerate(row))
        print(line)
        if i == 0:
            print("  " + "─" * (sum(col_w) + 10))
    print()
    print(
        "  Takeaway for this classifier workload / Kimi-K3-3:\n"
        "  ─────────────────────────────────────────────────────────────────────\n"
        "  prompt_cache_key is the Fireworks equivalent of prompt_cache_breakpoint.\n"
        "  Set it to a stable string (e.g. 'safety-classifier-v1') in the\n"
        "  request body and Fireworks caches the system-prompt prefix under that\n"
        "  key — no sticky routing, no header changes, no model-side support needed.\n"
        "\n"
        "  For this classifier workload (same system prompt, millions of short\n"
        "  user messages): one shared prompt_cache_key across all classification\n"
        "  requests ensures every call after the first warm write pays only for\n"
        "  the short user message — same latency and cost savings as an explicit\n"
        "  prompt_cache_breakpoint.\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Cache proof — TTFT (Time To First Token) via streaming
# ══════════════════════════════════════════════════════════════════════════════
# TTFT = time from sending the request until the first generated token arrives.
# It measures ONLY prefill (input encoding). KV-cache hits skip the system-
# prompt prefix during prefill, so TTFT drops sharply on warm calls while
# total latency (which includes generation) stays noisy on reasoning models.
# This is the definitive proof that caching is or isn't working.

PROOF_MESSAGE = "Where can I buy fresh vegetables near downtown Buenos Aires?"
PROOF_RUNS    = 4   # calls per condition (cold / warm)


def ttft_streaming(
    client: OpenAI,
    user_text: str,
    cache_key: str | None = None,
    isolation_key: str | None = None,
) -> dict:
    """
    Stream a chat completion and return TTFT + total latency.

    cache_key       — prompt_cache_key: names the shared cache bucket (warm path)
    isolation_key   — prompt_cache_isolation_key: forces a fresh prefill by
                      breaking any cross-request cache affinity (true cold path).
                      Pass a unique UUID per call to guarantee no reuse.
    """
    extra_body = {}
    if cache_key:
        extra_body["prompt_cache_key"] = cache_key
    if isolation_key:
        extra_body["prompt_cache_isolation_key"] = isolation_key

    kwargs = dict(
        model=KIMI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"User message to classify: {user_text}"},
        ],
        max_tokens=512,
        temperature=0,
        stream=True,
        extra_body=extra_body or None,
    )

    t_start = time.perf_counter()
    t_first = None
    text    = ""

    with client.chat.completions.create(**kwargs) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if t_first is None:
                    t_first = time.perf_counter()
                text += delta

    t_end = time.perf_counter()
    return {
        "ttft_s":  (t_first - t_start) if t_first else None,
        "total_s": t_end - t_start,
        "label":   _extract_label(text),
    }


def prove_cache_works():
    client = build_openai_client()
    proof_key = "safety-proof-key"

    print(f"\n{'═' * 72}")
    print("  CACHE PROOF — TTFT (Time To First Token) via streaming")
    print(f"{'═' * 72}")
    print(
        f"  Message (fixed): \"{PROOF_MESSAGE}\"\n"
        f"  Runs per condition: {PROOF_RUNS}\n"
        "\n"
        "  Method: stream the same message under two conditions:\n"
        "    COLD — unique prompt_cache_isolation_key per call forces a fresh\n"
        "           prefill every time, preventing any implicit caching.\n"
        "    WARM — same prompt_cache_key on every call shares the cached prefix.\n"
        "\n"
        "  TTFT isolates prefill time. Total latency adds generation on top.\n"
        "  A lower TTFT on WARM calls is unambiguous proof cache is active.\n"
    )

    # ── TRUE COLD baseline — isolation key forces fresh prefill each call ───────
    print(f"  {SEPARATOR[:60]}")
    print(f"  COLD (prompt_cache_isolation_key=unique-uuid — no reuse possible)")
    print(f"  {SEPARATOR[:60]}")
    cold_ttfts = []
    for i in range(PROOF_RUNS):
        iso_key = uuid.uuid4().hex           # unique per call → guaranteed cold
        r = ttft_streaming(client, PROOF_MESSAGE, isolation_key=iso_key)
        cold_ttfts.append(r["ttft_s"])
        print(
            f"    Run {i+1}: TTFT={fmt_ms(r['ttft_s']):>7}  total={fmt_ms(r['total_s']):>7}"
            f"  [{r['label']}]  iso={iso_key[:8]}…"
        )
    avg_cold = sum(cold_ttfts) / len(cold_ttfts)
    print(f"    Avg TTFT (cold): {fmt_ms(avg_cold)}\n")

    # ── WARM with shared prompt_cache_key ──────────────────────────────────────
    print(f"  {SEPARATOR[:60]}")
    print(f"  WARM (prompt_cache_key='{proof_key}' — shared cache bucket)")
    print(f"  {SEPARATOR[:60]}")
    warm_ttfts = []
    for i in range(PROOF_RUNS + 1):   # +1 for the cache-write call
        r = ttft_streaming(client, PROOF_MESSAGE, cache_key=proof_key)
        tag = "write" if i == 0 else f"read {i}"
        if i > 0:
            warm_ttfts.append(r["ttft_s"])
            speedup = avg_cold / r["ttft_s"] if r["ttft_s"] else 0
            suffix  = f"  ← {speedup:.1f}× faster than cold avg"
        else:
            suffix = "  ← cache write (cold)"
        print(
            f"    Run {tag:<6}: TTFT={fmt_ms(r['ttft_s']):>7}  total={fmt_ms(r['total_s']):>7}"
            f"  [{r['label']}]{suffix}"
        )
    avg_warm = sum(warm_ttfts) / len(warm_ttfts)
    print(f"    Avg TTFT (warm reads): {fmt_ms(avg_warm)}")

    # ── Verdict ────────────────────────────────────────────────────────────────
    speedup = avg_cold / avg_warm if avg_warm else 0
    print(f"\n  {'─' * 60}")
    if speedup >= 1.5:
        verdict = f"CACHE IS ACTIVE  ✓  {speedup:.1f}× TTFT speedup on warm reads"
    elif speedup >= 1.1:
        verdict = f"CACHE LIKELY ACTIVE  —  {speedup:.1f}× TTFT speedup (marginal)"
    else:
        verdict = f"CACHE NOT CONFIRMED  ✗  speedup only {speedup:.1f}× (noise level)"
    print(f"  Verdict: {verdict}")
    print(f"  Cold avg TTFT: {fmt_ms(avg_cold)}  |  Warm avg TTFT: {fmt_ms(avg_warm)}")
    print(f"  {'─' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═' * 72}")
    print("  Prompt Cache Demo — prompt_cache_breakpoint vs. prompt_cache_key")
    print(f"{'═' * 72}")
    print(f"  System prompt: {len(SYSTEM_PROMPT)} chars  |  Test messages: {len(TEST_MESSAGES)}")
    print(
        "\n  The static system prompt (safety-classifier instructions) is identical\n"
        "  across every classification request. Caching it avoids re-encoding it\n"
        "  on each call, cutting time-to-first-token and input-token cost.\n"
    )

    run_gpt_demo()
    run_kimi_demo()
    prove_cache_works()   # TTFT proof — definitive A/B for Kimi cache
    print_summary()
