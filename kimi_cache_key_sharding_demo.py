#!/usr/bin/env python3
"""
Demo 3: stateless classifier cache-key sharding — direct Fireworks serverless
─────────────────────────────────────────────────────────────────────────────
Production-shaped benchmark for classifiers/moderation/extraction workloads:

  • byte-identical long system prompt on every request
  • unique user content on every request (no conversation, no repeated user msg)
  • high concurrency

Four strategies:
  A. one shared prompt_cache_key for all requests
  B. N pre-warmed sharded prompt_cache_key values (stable hash(item_id) % N)
  C. unique prompt_cache_key per request (true cold control)
  D. no explicit affinity (gateway default/account/content routing control)

Every measured request has unique content and a run-scoped namespace, preventing
content-hash cross-warming and contamination from prior runs. Completed responses
collect the authoritative fireworks-cached-prompt-tokens response header (SDK
usage field is fallback) plus total latency. Cache percentage is the primary
metric; latency p50/p90 is secondary.
"""

import hashlib
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

FIREWORKS_KEY = os.environ["FIREWORKS_KEY"]
FIREWORKS_BASE = os.environ.get("FIREWORKS_BASE", "https://api.fireworks.ai/inference/v1")
FIREWORKS_MODEL = os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/kimi-k3")

WAVE_SIZE = int(os.environ.get("CLASSIFIER_WAVE_SIZE", "12"))
WAVES = int(os.environ.get("CLASSIFIER_WAVES", "3"))
NUM_SHARDS = int(os.environ.get("CLASSIFIER_SHARDS", "12"))
RUN_ID = uuid.uuid4().hex[:10]

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

# Stateless fan-out needs a stable prefix long enough to complete cache blocks
# before the unique user suffix. The tested Kimi K3 path defaults to 1,024-token
# GDN/Mamba cache granularity; use a comfortably larger prefix because chat
# formatting and cache boundaries consume part of the nominal system prompt.
STATIC_APPENDIX = (
    "\nREFERENCE POLICY APPENDIX (stable, production-scale classifier):\n"
    + (
        "Apply every policy category consistently. Preserve the strict output schema. "
        "Treat this reference text as immutable classifier guidance. "
        "Use the highest-severity matching rule and never include explanations.\n"
        * 220
    )
)
CLASSIFIER_SYSTEM_PROMPT = SYSTEM_PROMPT + STATIC_APPENDIX


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




def build_client():
    return OpenAI(api_key=FIREWORKS_KEY, base_url=FIREWORKS_BASE)


def stable_shard(item_id: str, n: int) -> int:
    return int.from_bytes(hashlib.sha256(item_id.encode()).digest()[:8], "big") % n


def classifier_input(strategy: str, wave: int, i: int) -> str:
    # Unique content per strategy, wave, and item. The stable system prompt is
    # the only reusable prefix — exactly the stateless classifier workload.
    nonce = uuid.uuid4().hex[:12]
    return (
        f"Classify marketplace listing {RUN_ID}-{strategy}-w{wave}-i{i}-{nonce}: "
        f"item={100000 + wave * WAVE_SIZE + i}; description=unique product text {nonce}; "
        f"seller_note=unique note {strategy}-{wave}-{i}."
    )


def classify(client, user_text, cache_key=None, isolation_key=None):
    kwargs = dict(
        model=FIREWORKS_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": f"User message to classify: {user_text}"},
        ],
        max_tokens=512, temperature=0,
    )
    extra_body = {}
    if cache_key:
        extra_body["prompt_cache_key"] = cache_key
    if isolation_key:
        extra_body["prompt_cache_isolation_key"] = isolation_key
    if extra_body:
        kwargs["extra_body"] = extra_body
    t0 = time.perf_counter()
    raw = client.chat.completions.with_raw_response.create(**kwargs)
    elapsed = time.perf_counter() - t0
    resp = raw.parse()
    cached = int(raw.headers.get("fireworks-cached-prompt-tokens") or 0)
    usage = resp.usage
    if not cached and usage:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
    total = usage.prompt_tokens or 0 if usage else 0
    return {
        "cached_tokens": cached, "input_tokens": total,
        "cache_pct": cached / total * 100 if total else 0,
        "latency_ms": elapsed * 1000,
    }


def warm_keys(client, keys):
    def one(key):
        # Warmup content is unique to each key; only the stable system prompt is
        # reused by later measured requests.
        return classify(client, f"warmup listing for {RUN_ID}-{key}-{uuid.uuid4().hex[:8]}", key)
    with ThreadPoolExecutor(max_workers=min(len(keys), WAVE_SIZE)) as ex:
        list(ex.map(one, keys))


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0
    return values[max(0, min(len(values)-1, round((p/100)*(len(values)-1))))]


def summarize(rows):
    warm = [r for r in rows if r["cached_tokens"] > 0]
    return {
        "n": len(rows), "warm": len(warm),
        "warm_pct": len(warm)/len(rows)*100 if rows else 0,
        "avg_cache_pct": statistics.mean(r["cache_pct"] for r in rows) if rows else 0,
        "p50_ms": percentile([r["latency_ms"] for r in rows], 50),
        "p90_ms": percentile([r["latency_ms"] for r in rows], 90),
    }


def run_strategy(client, code, name, key_for, warmup_keys=(), isolate=False):
    print(f"\n  {SEPARATOR[:60]}\n  {name}\n  {SEPARATOR[:60]}")
    if warmup_keys:
        print(f"  pre-warming {len(warmup_keys)} key(s)...", end="", flush=True)
        warm_keys(client, warmup_keys); print(" done")
    waves=[]
    for wave in range(WAVES):
        def one(i):
            item_id=f"{RUN_ID}-{code}-w{wave}-i{i}"
            return classify(client, classifier_input(code,wave,i), key_for(item_id,i,wave), uuid.uuid4().hex if isolate else None)
        with ThreadPoolExecutor(max_workers=WAVE_SIZE) as ex:
            rows=list(ex.map(one, range(WAVE_SIZE)))
        s=summarize(rows); waves.append(s)
        print(f"  wave {wave+1}: warm {s['warm']}/{s['n']} ({s['warm_pct']:.0f}%) | avg cache {s['avg_cache_pct']:.1f}% | latency p50 {s['p50_ms']:.0f} ms / p90 {s['p90_ms']:.0f} ms")
    all_summary={
        "name":name,
        "warm_pct":statistics.mean(w["warm_pct"] for w in waves),
        "avg_cache_pct":statistics.mean(w["avg_cache_pct"] for w in waves),
        "p50_ms":statistics.mean(w["p50_ms"] for w in waves),
        "p90_ms":statistics.mean(w["p90_ms"] for w in waves),
        "waves":waves,
    }
    return all_summary


def main():
    print(f"\n{'═'*72}\n  Stateless classifier sharding — direct Fireworks serverless\n{'═'*72}")
    print(f"  model={FIREWORKS_MODEL} | run={RUN_ID} | {WAVES} waves × {WAVE_SIZE} concurrent | shards={NUM_SHARDS}")
    print("  Every measured user input is unique; only the ~6K-token system prefix is reusable.\n")
    client=build_client()
    single=f"classifier-{RUN_ID}-single"
    shard_keys=[f"classifier-{RUN_ID}-shard-{i}" for i in range(NUM_SHARDS)]
    results=[]
    results.append(run_strategy(client,"A","A. single shared key",lambda item,i,w:single,[single]))
    results.append(run_strategy(client,"B",f"B. {NUM_SHARDS} pre-warmed shards",lambda item,i,w:shard_keys[stable_shard(item,NUM_SHARDS)],shard_keys))
    results.append(run_strategy(client,"C","C. unique key + isolation key per request (true cold)",lambda item,i,w:f"classifier-{RUN_ID}-unique-{uuid.uuid4().hex}",isolate=True))
    results.append(run_strategy(client,"D","D. no explicit affinity",lambda item,i,w:None))
    print(f"\n{SEPARATOR}\n  SUMMARY — unique-content stateless classifier\n{SEPARATOR}")
    print(f"  {'Strategy':<42} {'warm %':>8} {'avg cache %':>12} {'p50 ms':>9} {'p90 ms':>9}")
    print("  "+"─"*84)
    for r in results:
        print(f"  {r['name']:<42} {r['warm_pct']:>7.0f}% {r['avg_cache_pct']:>11.1f}% {r['p50_ms']:>9.0f} {r['p90_ms']:>9.0f}")
    print("\n  Interpret A vs B for sharding benefit; C uses a unique isolation key and must be ~0% warm (true cold); D measures gateway defaults.\n")

if __name__ == "__main__": main()
