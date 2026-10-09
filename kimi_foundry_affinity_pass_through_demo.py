#!/usr/bin/env python3
"""
Demo: prompt_cache_key vs x-session-affinity through Microsoft Foundry
─────────────────────────────────────────────────────────────────────────────
Same mechanism-equivalence test as the direct-Fireworks demo, but THROUGH a
Microsoft Foundry endpoint (the FW-Kimi deployment), plus the Foundry-specific
pass-through question: does the x-session-affinity header survive Microsoft's
gateway and reach the Fireworks backend?

Foundry strips the raw fireworks-cached-prompt-tokens response header, but
completed responses surface usage.prompt_tokens_details.cached_tokens. The demo
completes three turns for each mechanism and validates the healthy production
shape observed in the Sept 26 Foundry analysis: first request cold (~1–5%),
later turns warm (>80%; healthy windows were ~88–89%).

Config via environment: FOUNDRY_KEY, FOUNDRY_ENDPOINT, KIMI_MODEL.
See .env.example.
"""

import os
import time
import uuid
from openai import OpenAI

# ── Foundry config ──────────────────────────────────────────────────────────
FOUNDRY_KEY      = os.environ["FOUNDRY_KEY"]
FOUNDRY_ENDPOINT = os.environ.get(
    "FOUNDRY_ENDPOINT",
    "https://ganacfoundryeastus.services.ai.azure.com/api/projects/proj-default",
).rstrip("/")
V1_BASE          = f"{FOUNDRY_ENDPOINT}/openai/v1"
KIMI_MODEL       = os.environ.get("KIMI_MODEL", "FW-Kimi-K3-3")

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
    return OpenAI(api_key=FOUNDRY_KEY, base_url=V1_BASE)


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — definitive multi-turn cache proof (completed responses + usage tokens)
# ══════════════════════════════════════════════════════════════════════════════

MULTI_TURN_MESSAGES = [
    "Where can I buy apples near downtown?",
    "What about oranges in the same area?",
    "And where could I find bananas?",
]


def completed_turn(client, messages, cache_key=None, session_header=None, isolation_key=None):
    """Complete the full response so Foundry emits the final usage event.

    Important: Foundry does not forward the raw Fireworks
    fireworks-cached-prompt-tokens response header, but it DOES surface cache
    usage in usage.prompt_tokens_details.cached_tokens on completed responses.
    Streaming tests that stop at first token cannot observe that final usage.
    """
    kwargs = dict(
        model=KIMI_MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0,
    )
    extra_body = {}
    if cache_key:
        extra_body["prompt_cache_key"] = cache_key
    if isolation_key:
        extra_body["prompt_cache_isolation_key"] = isolation_key
    if extra_body:
        kwargs["extra_body"] = extra_body
    if session_header:
        kwargs["extra_headers"] = {"x-session-affinity": session_header}
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - t0
    usage = resp.usage
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
    total = usage.prompt_tokens or 0 if usage else 0
    text = resp.choices[0].message.content or "" if resp.choices else ""
    return {
        "cached_tokens": cached,
        "input_tokens": total,
        "cache_hit_pct": round(cached / total * 100, 1) if total else 0,
        "latency_s": elapsed,
        "text": text,
    }


def run_multiturn_mechanism(client, name, cache_key=None, session_header=None, isolation_key=None):
    history = []
    rows = []
    print(f"\n  {SEPARATOR[:60]}")
    print(f"  {name}")
    print(f"  {SEPARATOR[:60]}")
    for turn, user_text in enumerate(MULTI_TURN_MESSAGES, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_text},
        ]
        r = completed_turn(client, messages, cache_key=cache_key, session_header=session_header, isolation_key=isolation_key)
        rows.append(r)
        print(
            f"  turn {turn}: cached {r['cached_tokens']}/{r['input_tokens']} "
            f"({r['cache_hit_pct']:.1f}%) | {fmt_ms(r['latency_s'])} | "
            f"[{_extract_label(r['text'])}]"
        )
        history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": r["text"]},
        ])
    return rows


def test_definitive_multiturn_cache_proof(client):
    """Reproduce the Sept 26 Foundry shape across multiple independent sessions.

    A mechanism is proven when at least one later turn reports >80% cached; we
    separately report consistency because production later-turn reuse is known
    to vary with prefix stability, eviction, and cache-domain placement.
    """
    trials = 3
    print(f"\n{'═' * 72}")
    print("  TEST 1 — Multi-turn cache proof across independent Foundry sessions")
    print(f"{'═' * 72}")
    print(
        f"  {trials} independent sessions per mechanism, three turns each.\n"
        "  Healthy Sept 26 production shape: first request ~1–5% cached; later\n"
        "  turns >80% (healthy windows ~88–89%). We report proof and consistency.\n"
    )
    all_results = {}
    for mechanism in ("body", "header"):
        sessions = []
        for trial in range(trials):
            suffix = uuid.uuid4().hex[:8]
            name = f"{mechanism.upper()} trial {trial + 1}/{trials}"
            rows = run_multiturn_mechanism(
                client,
                name,
                cache_key=f"foundry-proof-body-{suffix}" if mechanism == "body" else None,
                session_header=f"foundry-proof-header-{suffix}" if mechanism == "header" else None,
                # Stable within the trial, unique across trials: guarantees turn 1
                # is cold without changing the body/header routing mechanism.
                isolation_key=f"foundry-proof-isolation-{suffix}",
            )
            sessions.append(rows)
        later = [row for rows in sessions for row in rows[1:]]
        hits = [row for row in later if row["cache_hit_pct"] > 80]
        avg = sum(row["cache_hit_pct"] for row in later) / len(later)
        first = [rows[0]["cache_hit_pct"] for rows in sessions]
        works = bool(hits) and all(v < 10 for v in first)
        all_results[mechanism] = {
            "sessions": sessions, "works": works, "hits": len(hits),
            "later_total": len(later), "avg_later_pct": avg,
        }
        print(
            f"\n  {mechanism}: {'WORKS' if works else 'NOT CONFIRMED'} | "
            f"later-turn hits >80%: {len(hits)}/{len(later)} | "
            f"average later-turn cache: {avg:.1f}%"
        )
    print(
        f"\n  VERDICT: body prompt_cache_key "
        f"{'WORKS' if all_results['body']['works'] else 'NOT CONFIRMED'}; "
        f"x-session-affinity header "
        f"{'WORKS' if all_results['header']['works'] else 'NOT CONFIRMED'} through Foundry."
    )
    return all_results


def main():
    print(f"\n{'═' * 72}")
    print("  Definitive cache mechanism proof — via Microsoft Foundry")
    print(f"{'═' * 72}")
    print(f"  Model: {KIMI_MODEL}  |  System prompt: {len(SYSTEM_PROMPT)} chars")
    client = build_client()
    test_definitive_multiturn_cache_proof(client)


if __name__ == "__main__":
    main()
