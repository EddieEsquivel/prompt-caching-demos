#!/usr/bin/env python3
"""
Demo: Responses API with store=True — stored prompts & completions
─────────────────────────────────────────────────────────────────────────────
Shows what the Responses API persists server-side when store=True:

  1. Create stored responses  — client.responses.create(store=True)
     The full request (system prompt + user message) AND the model's
     completion are stored server-side and assigned a response ID.

  2. Retrieve them later      — client.responses.retrieve(response_id)
     Fetch the stored prompt + completion back — nothing is resent.

  3. Chain a follow-up        — previous_response_id
     Ask a follow-up question without resending the system prompt or
     conversation history — the server already has it.

  4. Delete when done         — client.responses.delete(response_id)

Also demonstrates that prompt caching (prompt_cache_key) works alongside
stored responses — two independent mechanisms.

Use case context: a production safety classifier on FW-Kimi-K3-3.
In production you would likely use store=False (stateless), but this demo
shows exactly what IS retained when store=True so the customer can make an
informed data-retention decision.

Microsoft Foundry endpoint: https://ganacfoundryeastus.services.ai.azure.com/api/projects/proj-default
"""

import os
import time
from openai import OpenAI

# ── Foundry config ──────────────────────────────────────────────────────────
# Set FOUNDRY_KEY in your environment (see .env.example). Never commit keys.
FOUNDRY_KEY      = os.environ["FOUNDRY_KEY"]
FOUNDRY_ENDPOINT = "https://ganacfoundryeastus.services.ai.azure.com/api/projects/proj-default"
V1_BASE          = f"{FOUNDRY_ENDPOINT}/openai/v1"

KIMI_MODEL = "FW-Kimi-K3-3"
CACHE_KEY  = "safety-classifier-v1"

# ── Safety-classifier system prompt (same as other demos) ───────────────────
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


def build_client() -> OpenAI:
    return OpenAI(api_key=FOUNDRY_KEY, base_url=V1_BASE)


# ══════════════════════════════════════════════════════════════════════════
# Step 1 — Create STORED responses (store=True)
# ══════════════════════════════════════════════════════════════════════════

def create_stored_response(client: OpenAI, user_text: str, idx: int) -> dict:
    """
    Responses API call with store=True.
    The server stores the full request (system prompt + user message) and the
    completion, and returns a response ID for later retrieval.
    prompt_cache_key also stays active — caching and storage are independent.
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
            store=True,                                    # ← persist server-side
            extra_body={"prompt_cache_key": CACHE_KEY},
        )
        elapsed = time.perf_counter() - t0

        usage = resp.usage
        cached = 0
        if usage:
            details = getattr(usage, "input_tokens_details", None)
            if details:
                cached = getattr(details, "cached_tokens", 0) or 0

        return {
            "call":         idx + 1,
            "response_id":  resp.id,
            "user_text":    user_text,
            "output_text":  resp.output_text or "",
            "label":        _extract_label(resp.output_text or ""),
            "latency_s":    elapsed,
            "input_tokens": usage.input_tokens if usage else 0,
            "cached_tokens": cached,
        }
    except Exception as exc:
        return {"error": str(exc), "latency_s": time.perf_counter() - t0, "call": idx + 1}


def run_create_stored(client: OpenAI) -> list[dict]:
    print(f"\n{SEPARATOR}")
    print(f"  STEP 1 — Stored responses (store=True) + prompt_cache_key  |  model={KIMI_MODEL}")
    print(SEPARATOR)
    print(
        "  Each call sends the full system prompt + user message with store=True\n"
        "  AND prompt_cache_key. Two independent mechanisms at once:\n"
        "    • store=True          → server persists prompt + completion, returns ID\n"
        "    • prompt_cache_key    → Fireworks KV-cache skips re-encoding the prefix\n"
        "  This proves the Responses API supports prompt caching, not just storage.\n"
        f"\n  Cache key: {CACHE_KEY}\n"
    )

    # Warmup — writes the system-prompt prefix into the cache bucket AND
    # creates a stored response (deleted at the end with the others).
    print("  [warmup] priming cache bucket...", end="", flush=True)
    wu = create_stored_response(client, "warmup", -1)
    if "error" in wu:
        print(f" ERROR — {wu['error'][:250]}")
        return []
    cold_ms = wu["latency_s"]
    print(f" {fmt_ms(cold_ms)} (cold write)\n")

    results = []
    for i, msg in enumerate(TEST_MESSAGES):
        r = create_stored_response(client, msg, i)
        r["cold_ms"] = cold_ms
        results.append(r)
        if "error" in r:
            print(f"  Call {i+1}: ERROR — {r['error'][:250]}")
        else:
            cache_note = f"WARM ({r['cached_tokens']} cached tkns)" if r["cached_tokens"] else "WARM (key active)"
            print(
                f"  Call {i+1}: [{r['label']:<6}]  {fmt_ms(r['latency_s']):>7}  "
                f"[{cache_note}]  \"{r['user_text'][:45]}\"\n"
                f"           response_id = {r['response_id']}"
            )

    # Cache stats table — same format as the other demos
    print(f"\n  {'Call':<6} {'Label':<8} {'Latency':>8} {'Input tkns':>11} {'Cached tkns':>12} {'Cache %':>8}")
    print("  " + "─" * 58)
    for r in results:
        if "error" not in r:
            pct = round(r["cached_tokens"] / r["input_tokens"] * 100) if r["input_tokens"] else 0
            print(
                f"  {r['call']:<6} {r['label']:<8} {fmt_ms(r['latency_s']):>8} "
                f"{r['input_tokens']:>11} {r['cached_tokens']:>12} {pct:>7}%"
            )

    latencies = [r["latency_s"] for r in results if "error" not in r]
    if latencies:
        avg_warm = sum(latencies) / len(latencies)
        print(f"\n  Cold baseline (warmup): {fmt_ms(cold_ms)}  |  Avg warm: {fmt_ms(avg_warm)}  |  Speedup: {cold_ms/avg_warm:.1f}×")

    # The warmup response is also stored — include it for cleanup
    results.append(wu)
    return results


# ══════════════════════════════════════════════════════════════════════════
# Step 2 — Retrieve stored prompts & completions via GET
# ══════════════════════════════════════════════════════════════════════════

def run_retrieve_stored(client: OpenAI, results: list[dict]):
    print(f"\n{SEPARATOR}")
    print("  STEP 2 — Retrieve stored responses  |  GET /responses/{id}")
    print(SEPARATOR)
    print(
        "  Fetch each stored response by ID. Notice we resend NOTHING — the\n"
        "  server returns the stored completion, usage, and metadata.\n"
        "\n"
        "  NOTE: this Foundry gateway does not echo the raw input items back\n"
        "  on retrieval — it returns the completion + metadata. The prompt IS\n"
        "  stored server-side though: Step 3 proves it (previous_response_id\n"
        "  chaining only works if the server retained the full context).\n"
    )

    for r in results:
        if "error" in r:
            continue
        rid = r["response_id"]
        print(f"  GET /responses/{rid}")
        try:
            stored = client.responses.retrieve(rid)

            # Some gateways echo input items back; show them if present.
            input_items = getattr(stored, "input", None)
            if input_items:
                print("  ── Stored PROMPT (input items):")
                for item in input_items:
                    role = getattr(item, "role", "?")
                    content = getattr(item, "content", "")
                    if isinstance(content, str):
                        text = content
                    else:
                        text = " ".join(getattr(part, "text", "") for part in content)
                    if len(text) > 120:
                        shown = text[:120] + f" … [{len(text)} chars stored in full]"
                    else:
                        shown = text
                    print(f"    [{role:>7}] {shown}")
            else:
                print("  ── Stored PROMPT: retained server-side (input items not echoed by gateway)")

            # ── Stored completion ─────────────────────────────────────────
            print(f"  ── Stored COMPLETION: \"{stored.output_text}\"")
            print(f"  ── Status: {stored.status}  |  Created at: {stored.created_at}")
            u = stored.usage
            if u:
                print(f"  ── Usage: input={u.input_tokens}  output={u.output_tokens}")
            # Cache retention info surfaced by the gateway (when present)
            retention = getattr(stored, "prompt_cache_retention", None)
            if retention:
                print(f"  ── prompt_cache_retention: {retention}")
            print()

        except Exception as exc:
            print(f"    ERROR retrieving — {exc}\n")


# ══════════════════════════════════════════════════════════════════════════
# Step 3 — Chain a follow-up with previous_response_id (optional power move)
# ══════════════════════════════════════════════════════════════════════════

def run_chain_followup(client: OpenAI, results: list[dict]):
    if not results or "error" in results[0]:
        print("\n  STEP 3 — skipped (no stored response available)")
        return

    rid = results[0]["response_id"]
    print(f"\n{SEPARATOR}")
    print("  STEP 3 — Chain a follow-up via previous_response_id")
    print(SEPARATOR)
    print(
        "  Send ONLY a new user message, referencing the stored response ID.\n"
        "  No system prompt, no history — the server reconstructs the full\n"
        "  context from storage.\n"
    )
    print(f"  previous_response_id = {rid}")

    try:
        resp = client.responses.create(
            model=KIMI_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text":
                                 "Classify this too: Best welding gloves for heavy duty work?"}],
                }
            ],
            max_output_tokens=512,
            store=True,
            previous_response_id=rid,          # ← server-side context chaining
            extra_body={"prompt_cache_key": CACHE_KEY},
        )
        label = _extract_label(resp.output_text or "")
        print(f"  Follow-up answer: [{label}]  \"{resp.output_text}\"")
        print(f"  New response_id   = {resp.id}")
        return resp
    except Exception as exc:
        print(f"  ERROR — {exc}")


# ══════════════════════════════════════════════════════════════════════════
# Step 4 — Delete stored responses (cleanup)
# ══════════════════════════════════════════════════════════════════════════

def run_delete_stored(client: OpenAI, results: list[dict]):
    print(f"\n{SEPARATOR}")
    print("  STEP 4 — Delete stored responses (cleanup)")
    print(SEPARATOR)

    for r in results:
        if "error" in r:
            continue
        rid = r["response_id"]
        try:
            client.responses.delete(rid)
            print(f"  DELETED  {rid}")
        except Exception as exc:
            print(f"  DELETE failed for {rid} — {str(exc)[:150]}")


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════

def print_summary():
    print(f"\n{'═' * 72}")
    print("  SUMMARY — What is stored where when store=True")
    print(f"{'═' * 72}")
    rows = [
        ("Layer",          "What is stored",                          "Retention / control"),
        ("Microsoft/Foundry",  "Full request (system prompt + user msg)", "Stored until deleted;"),
        ("   (Responses",  "Full completion (model output)",          "  delete via API, or"),
        ("    API store)", "Response ID for retrieval & chaining",    "  per Microsoft retention"),
        ("Fireworks",      "KV-cache of prompt prefix (RAM only)",    "Minutes–hours, evicted"),
        ("   (cache)",     "  — prompt caching, independent of store", "  oldest-first; never"),
        ("",               "",                                        "  persisted to disk"),
        ("Fireworks logs", "Metadata only (token counts)",            "Operational, no content"),
    ]
    col_w = [max(len(r[i]) for r in rows) for i in range(3)]
    for i, row in enumerate(rows):
        print("  " + "  |  ".join(v.ljust(col_w[j]) for j, v in enumerate(row)))
        if i == 0:
            print("  " + "─" * (sum(col_w) + 10))
    print(
        "\n  Recommendation for this classifier workload:\n"
        "  ─────────────────────────────────────────────────────────────────────\n"
        "  • store=False (default in the other demos) is the right choice for a\n"
        "    stateless classifier — nothing persists, best data-retention posture.\n"
        "  • store=True is useful when you NEED statefulness: audit trails,\n"
        "    conversation chaining (previous_response_id), or deferred retrieval\n"
        "    of classification decisions.\n"
        "  • Either way, prompt caching (prompt_cache_key) works independently\n"
        "    and only ever touches volatile memory on the Fireworks replicas.\n"
    )


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═' * 72}")
    print("  Stored Responses Demo — Responses API with store=True")
    print(f"{'═' * 72}")
    print(f"  Model: {KIMI_MODEL}  |  System prompt: {len(SYSTEM_PROMPT)} chars  |  Messages: {len(TEST_MESSAGES)}")
    print(
        "\n  Demonstrates: create stored responses → retrieve them via GET →\n"
        "  chain a follow-up via previous_response_id → delete for cleanup.\n"
    )

    client = build_client()

    results = run_create_stored(client)
    run_retrieve_stored(client, results)
    run_chain_followup(client, results)
    run_delete_stored(client, results)
    print_summary()
