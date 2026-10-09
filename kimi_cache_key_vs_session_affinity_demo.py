#!/usr/bin/env python3
"""
Demo: prompt_cache_key (body) vs x-session-affinity (header) — direct Fireworks
─────────────────────────────────────────────────────────────────────────────
Runs against the Fireworks serverless endpoint directly (NOT via Foundry).

Tests, on FW-Kimi-K3 serverless:

  1. Mechanism equivalence — three arms (body prompt_cache_key only,
     x-session-affinity header only, no affinity control), each with its
     own key value and a warmup. Both mechanism arms should cache; the
     control arm should not.
  2. Cross-mechanism shared value — prime the cache with the body field,
     warm-read with the header carrying the SAME string (and the reverse).
     A warm read proves both fields feed the same routing session.
  3. Precedence — header vs body with different values; the header should
     win (matches Fireworks gateway precedence: session headers rank above
     body prompt_cache_key/user).

Config via environment: FOUNDRY_KEY unused here; set FIREWORKS_KEY and
FIREWORKS_MODEL (e.g. accounts/fireworks/models/kimi-k3). See .env.example.
"""

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# ── Fireworks config ────────────────────────────────────────────────────────
FIREWORKS_KEY  = os.environ["FIREWORKS_KEY"]
FIREWORKS_BASE = os.environ.get("FIREWORKS_BASE", "https://api.fireworks.ai/inference/v1")
FIREWORKS_MODEL = os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/kimi-k3")

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




def build_client() -> OpenAI:
    return OpenAI(api_key=FIREWORKS_KEY, base_url=FIREWORKS_BASE)


def call_classify(
    client: OpenAI,
    user_text: str,
    idx: int,
    cache_key: str | None = None,
    session_header: str | None = None,
    isolation_key: str | None = None,
) -> dict:
    """
    One classification call. Affinity is set via exactly one mechanism:

      cache_key      — body prompt_cache_key
      session_header — x-session-affinity request header
      isolation_key  — prompt_cache_isolation_key (forces cold; no reuse)
    """
    extra_body = {}
    if cache_key:
        extra_body["prompt_cache_key"] = cache_key
    if isolation_key:
        extra_body["prompt_cache_isolation_key"] = isolation_key

    headers = {}
    if session_header:
        headers["x-session-affinity"] = session_header

    t0 = time.perf_counter()
    try:
        kwargs = dict(
            model=FIREWORKS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"User message to classify: {user_text}"},
            ],
            max_tokens=512,
            temperature=0,
            extra_body=extra_body or None,
        )
        if headers:
            kwargs["extra_headers"] = headers
        raw = client.chat.completions.with_raw_response.create(**kwargs)
        elapsed = time.perf_counter() - t0
        resp = raw.parse()
        # Authoritative cache signal: Fireworks reports cached prompt tokens in
        # the fireworks-cached-prompt-tokens response header; the SDK's
        # usage.prompt_tokens_details.cached_tokens is a fallback (observed 0
        # even when the header reports a full cache hit on serverless kimi-k3).
        cached = int(raw.headers.get("fireworks-cached-prompt-tokens") or 0)

        raw_text = resp.choices[0].message.content or "" if resp.choices else ""
        label = _extract_label(raw_text)
        usage = resp.usage
        if not cached and usage and hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        total_in = usage.prompt_tokens if usage else 0
        return {
            "call": idx + 1, "user_text": user_text, "label": label,
            "latency_s": elapsed, "input_tokens": total_in, "cached_tokens": cached,
            "cache_hit_pct": round(cached / total_in * 100) if total_in else 0,
        }
    except Exception as exc:
        return {"error": str(exc), "latency_s": time.perf_counter() - t0, "call": idx + 1}


def print_table(results):
    print(f"\n  {'Call':<6} {'Label':<8} {'Latency':>8} {'Input tkns':>11} {'Cached tkns':>12} {'Cache %':>8}")
    print("  " + "─" * 58)
    for r in results:
        if "error" not in r:
            print(
                f"  {r['call']:<6} {r['label']:<8} {fmt_ms(r['latency_s']):>8} "
                f"{r['input_tokens']:>11} {r['cached_tokens']:>12} {r['cache_hit_pct']:>7}%"
            )


def arm_summary(name, results):
    ok = [r for r in results if "error" not in r]
    warm = [r for r in ok if r["cached_tokens"] > 0]
    avg_cached_pct = round(sum(r["cache_hit_pct"] for r in ok) / len(ok)) if ok else 0
    print(f"\n  {name}: {len(warm)}/{len(ok)} calls warm | avg cache %: {avg_cached_pct}%")
    return {"name": name, "warm": len(warm), "total": len(ok), "avg_cached_pct": avg_cached_pct}


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — Mechanism equivalence under concurrency
# ══════════════════════════════════════════════════════════════════════════════

# Test 1 runs arms CONCURRENTLY with per-conversation keys and DISTINCT messages
# per arm. Two validity reasons:
#   - Sequential single-user traffic can't differentiate anything: with no
#     session the gateway falls back to account-based affinity, so even the
#     control arm pins to one replica and stays warm. Concurrency + per-
#     conversation keys (arms A/B) spreads conversations across replicas,
#     while the control stays account-pinned — that is the difference.
#   - Distinct messages per arm rule out content-hash session affinity
#     cross-warming between arms (kimi routers may hash the first user
#     message when no session is sent).
WAVE_SIZE = 12
TOPICS = [
    "fresh produce near downtown", "kids movies this week", "a Spanish tutor in Buenos Aires",
    "homemade bread recipes", "winter tires for a sedan", "a dentist appointment",
    "local hiking trails", "a birthday gift for a cyclist", "public transport passes",
    "a gym membership", "leather boot care", "a smartphone for travel",
]

def arm_messages(prefix: str) -> list[str]:
    return [f"Question about {t} (case {prefix}-{i})" for i, t in enumerate(TOPICS)]

def run_concurrent_arm(client, name, msgs, key_mech):
    """One concurrent arm: wave 1 (first touch, cold expected) then wave 2
    (same conversations repeated, warm expected if the mechanism works).
    key_mech(i) → dict(cache_key=...) or dict(session_header=...) or {} (control)."""
    def one(i, msg):
        mech = key_mech(i)
        return call_classify(
            client, msg, i,
            cache_key=mech.get("cache_key"),
            session_header=mech.get("session_header"),
        )
    def run_wave(tag):
        with ThreadPoolExecutor(max_workers=WAVE_SIZE) as ex:
            wave = list(ex.map(lambda t: one(*t), list(enumerate(msgs))))
        ok = [r for r in wave if "error" not in r]
        warm = sum(1 for r in ok if r["cached_tokens"] > 0)
        avg_pct = round(sum(r["cache_hit_pct"] for r in ok) / len(ok)) if ok else 0
        avg_lat = sum(r["latency_s"] for r in ok) / len(ok) if ok else 0
        print(f"  wave {tag}: warm {warm}/{len(ok)} ({round(warm/len(ok)*100) if ok else 0}%) | avg cached {avg_pct}% | avg latency {fmt_ms(avg_lat)}")
        errs = [r for r in wave if "error" in r]
        if errs:
            print(f"    ({len(errs)} errors, first: {errs[0]['error'][:100]})")
        return {"warm": warm, "total": len(ok), "avg_pct": avg_pct, "avg_lat": avg_lat}
    print(f"\n  {SEPARATOR[:60]}")
    print(f"  ARM {name}")
    print(f"  {SEPARATOR[:60]}")
    print("  wave 1 (first touch):")
    w1 = run_wave(1)
    print("  wave 2 (same conversations, repeated):")
    w2 = run_wave(2)
    return {"name": name, "wave1": w1, "wave2": w2}


def test_mechanism_equivalence(client):
    print(f"\n{'═' * 72}")
    print("  TEST 1 — Mechanism equivalence under concurrency")
    print(f"{'═' * 72}")
    print(
        f"  {WAVE_SIZE} concurrent conversations per arm, two waves, distinct\n"
        "  messages per arm (rules out content-hash cross-warming):\n"
        "    A. body prompt_cache_key — one key per conversation\n"
        "    B. header x-session-affinity — one value per conversation\n"
        "    C. control — nothing sent (falls back to account-based affinity)\n"
        "  Expected: A and B warm on wave 2 (conversations spread across replicas);\n"
        "  C stays account-pinned, so under concurrent load it can\n"
        "  overflow its few pinned replicas and go cold.\n"
        "  Note: if C stays fully warm on wave 2, the router is likely applying\n"
        "  content-based session affinity (first-user-message hash) — also a\n"
        "  useful finding.\n"
    )
    out = {}
    out["A"] = run_concurrent_arm(
        client, "A. body prompt_cache_key (per-conversation)", arm_messages("A"),
        key_mech=lambda i: {"cache_key": f"equiv-c-{i}"},
    )
    out["B"] = run_concurrent_arm(
        client, "B. header x-session-affinity (per-conversation)", arm_messages("B"),
        key_mech=lambda i: {"session_header": f"equiv-h-{i}"},
    )
    out["C"] = run_concurrent_arm(
        client, "C. control (no affinity — account fallback)", arm_messages("C"),
        key_mech=lambda i: {},
    )
    a_ok = out["A"]["wave2"]["warm"] > 0 and out["A"]["wave2"]["warm"] >= out["A"]["wave1"]["warm"]
    b_ok = out["B"]["wave2"]["warm"] > 0 and out["B"]["wave2"]["warm"] >= out["B"]["wave1"]["warm"]
    print("\n  VERDICT:", "EQUIVALENT — both mechanisms cached under concurrency"
          if a_ok and b_ok else "CHECK RESULTS — see per-arm waves above")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — Cross-mechanism shared value (same string, both transports)
# ══════════════════════════════════════════════════════════════════════════════

def test_cross_mechanism_shared_value(client):
    print(f"\n{'═' * 72}")
    print("  TEST 2 — Cross-mechanism shared value")
    print(f"{'═' * 72}")
    print(
        "  Prime the cache through ONE mechanism, warm-read through the OTHER,\n"
        "  using the SAME value string. If the read is warm, both mechanisms feed\n"
        "  the same routing session on the Fireworks gateway.\n"
    )
    shared = "shared-session-token-1"
    # Distinct message so no other arm/test can warm this conversation via
    # content-hash affinity; identical across prime+read because it IS the
    # same conversation (the prefix being reused is the system prompt).
    CROSS_MSG = "Cross-mechanism probe message (shared-session-token conversation)"
    out = {}
    for label, priming_mech, reading_mech in [
        ("body → header", "body", "header"),
        ("header → body", "header", "body"),
    ]:
        print(f"  {SEPARATOR[:60]}")
        print(f"  {label}")
        print(f"  {SEPARATOR[:60]}")
        prime = call_classify(
            client, CROSS_MSG + " — prime", -1,
            cache_key=shared if priming_mech == "body" else None,
            session_header=shared if priming_mech == "header" else None,
        )
        if "error" in prime:
            print(f"  prime ERROR: {prime['error'][:150]}")
            continue
        print(f"  [prime via {priming_mech}] {fmt_ms(prime['latency_s'])} (cache write)")
        read = call_classify(
            client, CROSS_MSG, 0,
            cache_key=shared if reading_mech == "body" else None,
            session_header=shared if reading_mech == "header" else None,
        )
        if "error" in read:
            print(f"  read ERROR: {read['error'][:150]}")
            continue
        warm = read["cached_tokens"] > 0
        out[label] = read
        print(
            f"  [read via {reading_mech}]  [{read['label']:<6}] {fmt_ms(read['latency_s']):>7}  "
            f"[{'WARM (' + str(read['cache_hit_pct']) + '% cached)' if warm else 'COLD (0 cached)'}]"
        )
    both_warm = all(r.get("cached_tokens", 0) > 0 for r in out.values()) and len(out) == 2
    print(f"\n  VERDICT: {'SHARED SESSION CONFIRMED — the header read the body-primed cache and vice versa' if both_warm else 'NOT CONFIRMED — see reads above'}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — Precedence: header outranks body
# ══════════════════════════════════════════════════════════════════════════════

def test_precedence(client):
    print(f"\n{'═' * 72}")
    print("  TEST 3 — Precedence (header value vs body value, different strings)")
    print(f"{'═' * 72}")
    print(
        "  Fireworks gateway precedence: session headers rank above body\n"
        "  prompt_cache_key. We prime value P via the header, then send a call\n"
        "  with header=P + body=Q (Q never primed). Warm ⇒ the header value won.\n"
        "  Mirror case: header=R (never primed) + body=P. Cold ⇒ header still won.\n"
    )
    results = {}

    # Case 1: header value primed, body value not → expect WARM (header wins)
    p, q = "prec-header-primed", "prec-body-unprimed"
    prec_msg_1 = "Precedence case 1 probe (header value primed)"
    prime = call_classify(client, prec_msg_1, -1, session_header=p)
    if "error" not in prime:
        print(f"  [prime header={p}] {fmt_ms(prime['latency_s'])}")
        r = call_classify(client, prec_msg_1, 0, cache_key=q, session_header=p)
        if "error" not in r:
            results["header=P + body=Q (P primed)"] = r
            print(f"  call header={p} + body={q}: [{'WARM' if r['cached_tokens'] > 0 else 'COLD'}] ({r['cache_hit_pct']}% cached)")

    # Case 2: header value NOT primed, body value primed → expect COLD (header wins anyway)
    # NOTE: warm here would mean the BODY won — i.e. this router does NOT
    # apply the documented header-over-body precedence (possible on routers
    # with prefer_content_based_session, which rewrite the session post-auth).
    prec_msg_2 = "Precedence case 2 probe (body value primed)"
    r2 = call_classify(client, prec_msg_2, 1, cache_key=p, session_header=q)
    if "error" not in r2:
        results["header=Q + body=P (P primed)"] = r2
        print(f"  call header={q} + body={p}: [{'WARM' if r2['cached_tokens'] > 0 else 'COLD'}] ({r2['cache_hit_pct']}% cached)")

    c1 = results.get("header=P + body=Q (P primed)", {})
    c2 = results.get("header=Q + body=P (P primed)", {})
    confirmed = c1.get("cached_tokens", 0) > 0 and c2.get("cached_tokens", 0) == 0
    print(f"\n  VERDICT: {'HEADER OUTRANKS BODY confirmed' if confirmed else 'INCONCLUSIVE — see cases above'}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═' * 72}")
    print("  prompt_cache_key (body) vs x-session-affinity (header) — direct Fireworks")
    print(f"{'═' * 72}")
    print(f"  Model: {FIREWORKS_MODEL}  |  System prompt: {len(SYSTEM_PROMPT)} chars")
    client = build_client()
    test_mechanism_equivalence(client)
    test_cross_mechanism_shared_value(client)
    test_precedence(client)
    print()
