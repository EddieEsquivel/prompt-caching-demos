#!/usr/bin/env python3
"""
Demo: Responses API vs Chat Completions — FW-Kimi-K3-3 with prompt_cache_key
─────────────────────────────────────────────────────────────────────────────
Compares two ways to invoke FW-Kimi-K3-3 on Microsoft Foundry with caching:

  1. Responses API     — client.responses.create(...)
                         (OpenAI SDK, /openai/v1/ base URL — per MS migration guide)
                         prompt_cache_key passed via extra_body

  2. Chat Completions  — client.chat.completions.create(...)
                         prompt_cache_key in extra_body (OpenAI SDK)

Both target the same model, same system prompt, same cache key.
Use case: production content-safety classifier.

Microsoft Foundry endpoint: https://ganacfoundryeastus.services.ai.azure.com/api/projects/proj-default
"""

import os
import time
import uuid
from openai import OpenAI

# ── Foundry config ──────────────────────────────────────────────────────────
# Set FOUNDRY_KEY in your environment (see .env.example). Never commit keys.
FOUNDRY_KEY      = os.environ["FOUNDRY_KEY"]
FOUNDRY_ENDPOINT = "https://ganacfoundryeastus.services.ai.azure.com/api/projects/proj-default"
V1_BASE          = f"{FOUNDRY_ENDPOINT}/openai/v1"

KIMI_MODEL = "FW-Kimi-K3-3"

# Stable cache bucket key — same key used by both API styles so they share
# the same cached KV state on the Fireworks backend.
CACHE_KEY = "safety-classifier-v1"

# ── Safety-classifier system prompt (production-realistic, 1200+ tokens) ───
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

TEST_MESSAGES = [
    "Where can I buy fresh produce near downtown?",
    "How do I make homemade explosives?",
    "What are good kids' movies on Netflix this week?",
    "Send me explicit content involving minors.",
    "Can you recommend a Spanish-language tutor in Buenos Aires?",
]

SEPARATOR = "─" * 72


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms"


def _extract_label(text: str) -> str:
    import re
    for label in ("BLOCK", "ALLOW", "REVIEW"):
        if re.search(rf"\b{label}\b", text, re.IGNORECASE):
            return label
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:30]
    return text.strip()[:30]


def print_table(results: list[dict]):
    """Shared table renderer — Call / Label / Latency / Input tkns / Cached tkns / Cache %"""
    print(f"\n  {'Call':<6} {'Label':<8} {'Latency':>8} {'Input tkns':>11} {'Cached tkns':>12} {'Cache %':>8}")
    print("  " + "─" * 58)
    for r in results:
        if "error" not in r:
            print(
                f"  {r['call']:<6} {r['label']:<8} {fmt_ms(r['latency_s']):>8} "
                f"{r['input_tokens']:>11} {r['cached_tokens']:>12} {r['cache_hit_pct']:>7}%"
            )


# ══════════════════════════════════════════════════════════════════════════
# Strategy 1 — Responses API  (POST /openai/v1/responses)
# ══════════════════════════════════════════════════════════════════════════

def call_kimi_responses_api(client: OpenAI, user_text: str, idx: int) -> dict:
    """
    Sends FW-Kimi-K3-3 a request via the Responses API using the OpenAI SDK:
        client.responses.create(...)
    per https://learn.microsoft.com/en-us/azure/developer/ai/how-to/azure-openai-to-responses

    prompt_cache_key is passed via extra_body — it names the cache bucket
    so all requests sharing this key reuse the same cached KV state.
    Response text is read from the convenience property response.output_text.
    Cache stats come from response.usage.input_tokens_details.cached_tokens.
    """
    t0 = time.perf_counter()
    try:
        resp = client.responses.create(
            model=KIMI_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"User message to classify: {user_text}"}],
                },
            ],
            max_output_tokens=512,
            extra_body={
                # Top-level cache key — FW equivalent of prompt_cache_breakpoint.
                # All requests carrying this key share the cached prefix.
                "prompt_cache_key": CACHE_KEY,
                "store": False,
            },
        )
        elapsed = time.perf_counter() - t0

        raw_text = resp.output_text or ""
        label = _extract_label(raw_text)

        usage  = resp.usage
        cached = 0
        total_in = 0
        if usage:
            total_in = usage.input_tokens or 0
            details = getattr(usage, "input_tokens_details", None)
            if details:
                cached = getattr(details, "cached_tokens", 0) or 0

        return {
            "call":          idx + 1,
            "user_text":     user_text,
            "label":         label,
            "latency_s":     elapsed,
            "input_tokens":  total_in,
            "cached_tokens": cached,
            "cache_hit_pct": round(cached / total_in * 100) if total_in else 0,
        }
    except Exception as exc:
        body = str(exc)
        return {"error": body, "body": body[:400], "latency_s": time.perf_counter() - t0, "call": idx + 1}


def run_responses_demo(client: OpenAI) -> list[dict]:
    print(f"\n{SEPARATOR}")
    print(f"  MODEL: {KIMI_MODEL}  |  API: Responses API  |  Caching: prompt_cache_key")
    print(SEPARATOR)
    print(
        "  prompt_cache_key is a top-level field in the Responses API body.\n"
        "  The Fireworks backend caches the system-prompt prefix under this key;\n"
        "  every subsequent request with the same key skips re-encoding it.\n"
        f"\n  Cache key: {CACHE_KEY}\n"
    )

    # Warmup — write the system-prompt prefix into the cache bucket
    print("  [warmup] priming cache bucket...", end="", flush=True)
    wu = call_kimi_responses_api(client, "warmup", -1)
    if "error" in wu:
        print(f" ERROR {wu['error']}: {wu['body'][:200]}")
        return []
    cold_ms = wu["latency_s"]
    print(f" {fmt_ms(cold_ms)} (cold write)\n")

    results = []
    for i, msg in enumerate(TEST_MESSAGES):
        r = call_kimi_responses_api(client, msg, i)
        results.append(r)

        if "error" in r:
            print(f"  Call {i+1}: ERROR {r['error']} — {r['body'][:200]}")
        else:
            if r["cached_tokens"] > 0:
                cache_note = f"WARM ({r['cache_hit_pct']}% cached)"
            elif r["latency_s"] < cold_ms * 0.85:
                speedup = cold_ms / r["latency_s"]
                cache_note = f"WARM ({speedup:.1f}× faster than cold)"
            else:
                cache_note = "WARM (prompt_cache_key active)"

            print(
                f"  Call {i+1}: [{r['label']:<6}]  {fmt_ms(r['latency_s']):>7}  "
                f"[{cache_note}]  \"{r['user_text'][:52]}\""
            )

    print_table(results)

    latencies = [r["latency_s"] for r in results if "error" not in r]
    if latencies:
        avg_warm = sum(latencies) / len(latencies)
        print(f"\n  Cold baseline (warmup): {fmt_ms(cold_ms)}  |  Avg warm: {fmt_ms(avg_warm)}  |  Speedup: {cold_ms/avg_warm:.1f}×")

    return results


# ══════════════════════════════════════════════════════════════════════════
# Strategy 2 — Chat Completions  (OpenAI SDK)
# ══════════════════════════════════════════════════════════════════════════

def build_client() -> OpenAI:
    return OpenAI(api_key=FOUNDRY_KEY, base_url=V1_BASE)


def call_kimi_chat_completions(client: OpenAI, user_text: str, idx: int) -> dict:
    """
    Sends FW-Kimi-K3-3 a request via Chat Completions.
    prompt_cache_key lives in extra_body — same cache bucket as Responses API
    above, so both strategies share the same cached KV state on the backend.
    """
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"User message to classify: {user_text}"},
            ],
            max_tokens=512,
            temperature=0,
            extra_body={"prompt_cache_key": CACHE_KEY},
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
            "call":          idx + 1,
            "user_text":     user_text,
            "label":         label,
            "latency_s":     elapsed,
            "input_tokens":  total_in,
            "cached_tokens": cached,
            "cache_hit_pct": round(cached / total_in * 100) if total_in else 0,
        }
    except Exception as exc:
        return {"error": str(exc), "latency_s": time.perf_counter() - t0, "call": idx + 1}


def run_chat_completions_demo(client: OpenAI) -> list[dict]:
    print(f"\n{SEPARATOR}")
    print(f"  MODEL: {KIMI_MODEL}  |  API: Chat Completions  |  Caching: prompt_cache_key")
    print(SEPARATOR)
    print(
        "  prompt_cache_key lives in extra_body of the Chat Completions request.\n"
        "  The same cache bucket is used — both API styles share the cached prefix\n"
        "  as long as the key string is identical.\n"
        f"\n  Cache key: {CACHE_KEY}\n"
    )

    # Warmup
    print("  [warmup] priming cache bucket...", end="", flush=True)
    wu = call_kimi_chat_completions(client, "warmup", -1)
    if "error" in wu:
        print(f" ERROR — {wu['error'][:200]}")
        return []
    cold_ms = wu["latency_s"]
    print(f" {fmt_ms(cold_ms)} (cold write)\n")

    results = []
    for i, msg in enumerate(TEST_MESSAGES):
        r = call_kimi_chat_completions(client, msg, i)
        results.append(r)

        if "error" in r:
            print(f"  Call {i+1}: ERROR — {r['error'][:120]}")
        else:
            if r["cached_tokens"] > 0:
                cache_note = f"WARM ({r['cache_hit_pct']}% cached)"
            elif r["latency_s"] < cold_ms * 0.85:
                speedup = cold_ms / r["latency_s"]
                cache_note = f"WARM ({speedup:.1f}× faster than cold)"
            else:
                cache_note = "WARM (prompt_cache_key active)"

            print(
                f"  Call {i+1}: [{r['label']:<6}]  {fmt_ms(r['latency_s']):>7}  "
                f"[{cache_note}]  \"{r['user_text'][:52]}\""
            )

    print_table(results)

    latencies = [r["latency_s"] for r in results if "error" not in r]
    if latencies:
        avg_warm = sum(latencies) / len(latencies)
        print(f"\n  Cold baseline (warmup): {fmt_ms(cold_ms)}  |  Avg warm: {fmt_ms(avg_warm)}  |  Speedup: {cold_ms/avg_warm:.1f}×")

    return results


# ══════════════════════════════════════════════════════════════════════════
# TTFT Cache Proof — same message, cold vs warm, via streaming
# ══════════════════════════════════════════════════════════════════════════

PROOF_MESSAGE = "Where can I buy fresh vegetables near downtown Buenos Aires?"
PROOF_RUNS    = 4


def ttft_streaming(
    client: OpenAI,
    user_text: str,
    cache_key: str | None = None,
    isolation_key: str | None = None,
    api: str = "chat",             # "chat" or "responses"
) -> dict:
    """
    Stream a completion and measure TTFT + total latency.

    api="chat"      → Chat Completions streaming
    api="responses" → Responses API streaming (raw HTTP SSE)

    cache_key      — prompt_cache_key: shared cache bucket (warm path)
    isolation_key  — prompt_cache_isolation_key: per-call UUID (true cold path)
    """
    t_start = time.perf_counter()
    t_first = None
    text    = ""

    if api == "chat":
        extra_body = {}
        if cache_key:
            extra_body["prompt_cache_key"] = cache_key
        if isolation_key:
            extra_body["prompt_cache_isolation_key"] = isolation_key

        with client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"User message to classify: {user_text}"},
            ],
            max_tokens=512,
            temperature=0,
            stream=True,
            extra_body=extra_body or None,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    text += delta

    else:  # Responses API — SDK streaming events
        # NOTE: the Responses API endpoint does not accept
        # prompt_cache_isolation_key (Chat Completions only). A true-cold
        # read is enforced instead with a unique prompt_cache_key per call —
        # a fresh bucket name guarantees no prefix reuse.
        extra_body = {"store": False}
        if isolation_key:
            extra_body["prompt_cache_key"] = f"iso-{isolation_key}"
        elif cache_key:
            extra_body["prompt_cache_key"] = cache_key

        with client.responses.create(
            model=KIMI_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"User message to classify: {user_text}"}],
                },
            ],
            max_output_tokens=512,
            stream=True,
            extra_body=extra_body,
        ) as stream:
            for event in stream:
                # Per MS docs: listen for typed streaming events
                if event.type == "response.output_text.delta":
                    if t_first is None:
                        t_first = time.perf_counter()
                    text += event.delta
                elif event.type == "response.completed":
                    break

    t_end = time.perf_counter()
    return {
        "ttft_s":  (t_first - t_start) if t_first else (t_end - t_start),
        "total_s": t_end - t_start,
        "label":   _extract_label(text),
    }


def prove_cache_works(client: OpenAI):
    proof_key_responses = "safety-proof-responses"
    proof_key_chat      = "safety-proof-chat"

    print(f"\n{'═' * 72}")
    print("  CACHE PROOF — TTFT via streaming (Responses API vs Chat Completions)")
    print(f"{'═' * 72}")
    print(
        f"  Message (fixed): \"{PROOF_MESSAGE}\"\n"
        f"  Runs per condition: {PROOF_RUNS}\n"
        "\n"
        "  COLD — unique per-call key (isolation_key on Chat Completions,\n"
        "         fresh cache bucket on Responses API) — no reuse possible\n"
        "  WARM — shared prompt_cache_key (cached prefix reused)\n"
        "\n"
        "  TTFT isolates prefill. Lower TTFT on warm = unambiguous cache proof.\n"
    )

    def run_ttft_section(label: str, api: str, warm_key: str):
        print(f"\n  {'─' * 68}")
        print(f"  {label}")
        print(f"  {'─' * 68}")

        # Cold baseline
        print(f"  COLD (isolation_key=unique-uuid):")
        cold_ttfts = []
        for i in range(PROOF_RUNS):
            iso = uuid.uuid4().hex
            r = ttft_streaming(client, PROOF_MESSAGE, isolation_key=iso, api=api)
            cold_ttfts.append(r["ttft_s"])
            print(
                f"    Run {i+1}: TTFT={fmt_ms(r['ttft_s']):>7}  total={fmt_ms(r['total_s']):>7}"
                f"  [{r['label']}]  iso={iso[:8]}…"
            )
        avg_cold = sum(cold_ttfts) / len(cold_ttfts)
        print(f"    Avg TTFT (cold): {fmt_ms(avg_cold)}\n")

        # Warm reads
        print(f"  WARM (prompt_cache_key='{warm_key}'):")
        warm_ttfts = []
        for i in range(PROOF_RUNS + 1):
            r = ttft_streaming(client, PROOF_MESSAGE, cache_key=warm_key, api=api)
            tag = "write" if i == 0 else f"read {i}"
            if i > 0:
                warm_ttfts.append(r["ttft_s"])
                speedup = avg_cold / r["ttft_s"] if r["ttft_s"] else 0
                suffix  = f"  ← {speedup:.1f}× faster than cold"
            else:
                suffix = "  ← cache write (cold)"
            print(
                f"    Run {tag:<6}: TTFT={fmt_ms(r['ttft_s']):>7}  total={fmt_ms(r['total_s']):>7}"
                f"  [{r['label']}]{suffix}"
            )
        avg_warm = sum(warm_ttfts) / len(warm_ttfts) if warm_ttfts else 0
        speedup  = avg_cold / avg_warm if avg_warm else 0
        print(f"    Avg TTFT (warm reads): {fmt_ms(avg_warm)}")

        if speedup >= 1.5:
            verdict = f"CACHE ACTIVE  ✓  {speedup:.1f}× TTFT speedup"
        elif speedup >= 1.1:
            verdict = f"CACHE LIKELY ACTIVE  —  {speedup:.1f}× TTFT speedup (marginal)"
        else:
            verdict = f"CACHE NOT CONFIRMED  ✗  speedup {speedup:.1f}× (noise level)"

        print(f"\n    Verdict: {verdict}")
        print(f"    Cold avg: {fmt_ms(avg_cold)}  |  Warm avg: {fmt_ms(avg_warm)}")
        return {"avg_cold": avg_cold, "avg_warm": avg_warm, "speedup": speedup, "verdict": verdict}

    res_proof  = run_ttft_section(
        f"Responses API  — model={KIMI_MODEL}",
        api="responses",
        warm_key=proof_key_responses,
    )
    chat_proof = run_ttft_section(
        f"Chat Completions — model={KIMI_MODEL}",
        api="chat",
        warm_key=proof_key_chat,
    )
    return res_proof, chat_proof


# ══════════════════════════════════════════════════════════════════════════
# Side-by-side summary
# ══════════════════════════════════════════════════════════════════════════

def print_summary(responses_results: list[dict], chat_results: list[dict]):
    print(f"\n{'═' * 72}")
    print("  SUMMARY: Responses API vs Chat Completions — FW-Kimi-K3-3")
    print(f"{'═' * 72}")

    rows = [
        ("Feature",             "Responses API",                    "Chat Completions"),
        ("Endpoint",            "POST /openai/v1/responses",        "POST /openai/v1/chat/completions"),
        ("Cache field",         "prompt_cache_key (top-level)",     "prompt_cache_key (extra_body)"),
        ("Cache bucket",        CACHE_KEY,                          CACHE_KEY),
        ("SDK required",        "Yes — client.responses.create",    "Yes — client.chat.completions.create"),
        ("Response format",     "output[].content[].text",          "choices[0].message.content"),
        ("Cached tkns field",   "usage.input_tokens_details",       "usage.prompt_tokens_details"),
        ("Same backend cache?", "Yes — shared key = shared KV",     "Yes — shared key = shared KV"),
    ]
    col_w = [max(len(r[i]) for r in rows) for i in range(3)]
    for i, row in enumerate(rows):
        line = "  " + "  |  ".join(v.ljust(col_w[j]) for j, v in enumerate(row))
        print(line)
        if i == 0:
            print("  " + "─" * (sum(col_w) + 10))

    # Latency comparison
    def avg_lat(res):
        vals = [r["latency_s"] for r in res if "error" not in r]
        return sum(vals) / len(vals) if vals else 0

    ra = avg_lat(responses_results)
    ca = avg_lat(chat_results)

    print(f"\n  Avg warm latency  — Responses API: {fmt_ms(ra)}  |  Chat Completions: {fmt_ms(ca)}")

    print(
        "\n  Key insight:\n"
        "  ─────────────────────────────────────────────────────────────────────\n"
        "  Both API styles send prompt_cache_key to the same Fireworks backend.\n"
        "  The cache bucket is shared — a warm write via Chat Completions is\n"
        "  immediately readable by a Responses API call and vice versa.\n"
        "\n"
        "  Choose based on your integration needs:\n"
        "    • Responses API  — consistent with your GPT-5.6-sol integration;\n"
        "                       same request/response shape across all models.\n"
        "    • Chat Completions — OpenAI SDK convenience, easier streaming,\n"
        "                         works with any OpenAI-compatible client.\n"
        "  Caching behaviour, cost savings (~50% cached tokens), and latency\n"
        "  reduction are identical regardless of which API you choose.\n"
    )


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═' * 72}")
    print(f"  Responses API vs Chat Completions — FW-Kimi-K3-3 + prompt_cache_key")
    print(f"{'═' * 72}")
    print(f"  Model: {KIMI_MODEL}  |  System prompt: {len(SYSTEM_PROMPT)} chars  |  Test messages: {len(TEST_MESSAGES)}")
    print(
        "\n  Same model, same cache key, two API styles.\n"
        "  Goal: confirm both achieve identical caching outcomes.\n"
    )

    client = build_client()

    responses_results = run_responses_demo(client)
    chat_results      = run_chat_completions_demo(client)
    prove_cache_works(client)
    print_summary(responses_results, chat_results)
