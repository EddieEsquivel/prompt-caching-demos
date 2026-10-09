#!/usr/bin/env python3
"""
Demo 4: stateless classifier sharding — Microsoft Foundry
─────────────────────────────────────────────────────────────────────────────
Foundry equivalent of Demo 3. Production-shaped classifier workload:
byte-identical system prompt, unique user content every request, no history,
concurrent load. Compares body prompt_cache_key and header x-session-affinity,
with single-key and N-sharded variants plus true-cold and no-affinity controls.

Foundry strips the raw Fireworks cache response header, so completed responses
use usage.prompt_tokens_details.cached_tokens as the authoritative signal.
"""

import hashlib
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

FOUNDRY_KEY=os.environ["FOUNDRY_KEY"]
FOUNDRY_ENDPOINT=os.environ["FOUNDRY_ENDPOINT"].rstrip("/")
KIMI_MODEL=os.environ.get("KIMI_MODEL","FW-Kimi-K3-3")
WAVE_SIZE=int(os.environ.get("FOUNDRY_CLASSIFIER_WAVE_SIZE","3"))
WAVES=int(os.environ.get("FOUNDRY_CLASSIFIER_WAVES","2"))
NUM_SHARDS=int(os.environ.get("FOUNDRY_CLASSIFIER_SHARDS","4"))
RUN_ID=uuid.uuid4().hex[:10]

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
# before the unique user suffix. Kimi K3 reports 2,048-token-aligned cache hits;
# the shared ~1,280-token prompt used by conversational demos is too short.
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




def build_client(): return OpenAI(api_key=FOUNDRY_KEY,base_url=FOUNDRY_ENDPOINT+"/openai/v1")
def stable_shard(x,n): return int.from_bytes(hashlib.sha256(x.encode()).digest()[:8],"big")%n

def classifier_input(strategy,wave,i):
    nonce=uuid.uuid4().hex[:12]
    return f"Classify listing {RUN_ID}-{strategy}-w{wave}-i{i}-{nonce}: unique item {100000+wave*WAVE_SIZE+i}; unique description {nonce}; unique seller note {strategy}-{wave}-{i}."


def classify(client,user_text,cache_key=None,session_header=None,isolation_key=None):
    kw=dict(model=KIMI_MODEL,messages=[{"role":"system","content":CLASSIFIER_SYSTEM_PROMPT},{"role":"user","content":f"User message to classify: {user_text}"}],max_tokens=32,temperature=0)
    extra_body={}
    if cache_key: extra_body["prompt_cache_key"]=cache_key
    if isolation_key: extra_body["prompt_cache_isolation_key"]=isolation_key
    if extra_body: kw["extra_body"]=extra_body
    if session_header: kw["extra_headers"]={"x-session-affinity":session_header}
    for attempt in range(5):
        t0=time.perf_counter()
        try:
            r=client.chat.completions.create(**kw); elapsed=(time.perf_counter()-t0)*1000
            u=r.usage; d=getattr(u,"prompt_tokens_details",None) if u else None
            cached=getattr(d,"cached_tokens",0) or 0 if d else 0; total=u.prompt_tokens or 0 if u else 0
            return {"cached_tokens":cached,"input_tokens":total,"cache_pct":cached/total*100 if total else 0,"latency_ms":elapsed,"errors":0}
        except Exception as exc:
            if "429" in str(exc) or "RateLimit" in type(exc).__name__:
                time.sleep((2**attempt)*3); continue
            raise
    return {"cached_tokens":0,"input_tokens":0,"cache_pct":0,"latency_ms":0,"errors":1}


def warm_keys(client,keys,mechanism):
    def one(key): return classify(client,f"warmup {RUN_ID}-{key}-{uuid.uuid4().hex[:8]}",cache_key=key if mechanism=="body" else None,session_header=key if mechanism=="header" else None)
    # Serial warmup avoids bursting uncached prompt TPM through Foundry.
    for key in keys:
        one(key)


def pct(v,p):
    v=sorted(v)
    return v[max(0,min(len(v)-1,round((p/100)*(len(v)-1))))] if v else 0


def summary(rows):
    ok=[r for r in rows if not r["errors"]]; warm=[r for r in ok if r["cached_tokens"]>0]
    return {"n":len(ok),"warm":len(warm),"warm_pct":len(warm)/len(ok)*100 if ok else 0,"avg_cache_pct":statistics.mean(r["cache_pct"] for r in ok) if ok else 0,"p50_ms":pct([r["latency_ms"] for r in ok],50),"p90_ms":pct([r["latency_ms"] for r in ok],90),"errors":sum(r["errors"] for r in rows)}


def run_strategy(client,code,name,key_for,mechanism=None,warmup_keys=(),isolate=False):
    print(f"\n  {SEPARATOR[:60]}\n  {name}\n  {SEPARATOR[:60]}")
    if warmup_keys:
        print(f"  pre-warming {len(warmup_keys)} key(s)...",end="",flush=True); warm_keys(client,warmup_keys,mechanism); print(" done")
    waves=[]
    for wave in range(WAVES):
        def one(i):
            item=f"{RUN_ID}-{code}-w{wave}-i{i}"; key=key_for(item,i,wave)
            return classify(client,classifier_input(code,wave,i),cache_key=key if mechanism=="body" else None,session_header=key if mechanism=="header" else None,isolation_key=uuid.uuid4().hex if isolate else None)
        with ThreadPoolExecutor(max_workers=WAVE_SIZE) as ex: rows=list(ex.map(one,range(WAVE_SIZE)))
        s=summary(rows); waves.append(s)
        print(f"  wave {wave+1}: warm {s['warm']}/{s['n']} ({s['warm_pct']:.0f}%) | avg cache {s['avg_cache_pct']:.1f}% | latency p50 {s['p50_ms']:.0f} / p90 {s['p90_ms']:.0f} ms | errors {s['errors']}")
    return {"name":name,"warm_pct":statistics.mean(w["warm_pct"] for w in waves),"avg_cache_pct":statistics.mean(w["avg_cache_pct"] for w in waves),"p50_ms":statistics.mean(w["p50_ms"] for w in waves),"p90_ms":statistics.mean(w["p90_ms"] for w in waves),"errors":sum(w["errors"] for w in waves),"waves":waves}


def main():
    print(f"\n{'═'*72}\n  Stateless classifier sharding — Microsoft Foundry\n{'═'*72}")
    print(f"  model={KIMI_MODEL} | run={RUN_ID} | {WAVES} waves × {WAVE_SIZE} concurrent | shards={NUM_SHARDS}")
    print("  Every measured user input is unique; only the ~6K-token system prefix is reusable.\n")
    c=build_client(); results=[]
    body_single=f"fcls-{RUN_ID}-body-single"; body_shards=[f"fcls-{RUN_ID}-body-{i}" for i in range(NUM_SHARDS)]
    hdr_single=f"fcls-{RUN_ID}-hdr-single"; hdr_shards=[f"fcls-{RUN_ID}-hdr-{i}" for i in range(NUM_SHARDS)]
    results.append(run_strategy(c,"A","A. body single key",lambda item,i,w:body_single,"body",[body_single]))
    results.append(run_strategy(c,"B",f"B. body {NUM_SHARDS} shards",lambda item,i,w:body_shards[stable_shard(item,NUM_SHARDS)],"body",body_shards))
    results.append(run_strategy(c,"C","C. header single value",lambda item,i,w:hdr_single,"header",[hdr_single]))
    results.append(run_strategy(c,"D",f"D. header {NUM_SHARDS} shards",lambda item,i,w:hdr_shards[stable_shard(item,NUM_SHARDS)],"header",hdr_shards))
    results.append(run_strategy(c,"E","E. unique body + isolation key (true cold)",lambda item,i,w:f"fcls-{RUN_ID}-unique-{uuid.uuid4().hex}","body",isolate=True))
    results.append(run_strategy(c,"F","F. no explicit affinity",lambda item,i,w:None,None))
    print(f"\n{SEPARATOR}\n  SUMMARY — Foundry unique-content stateless classifier\n{SEPARATOR}")
    print(f"  {'Strategy':<38} {'warm %':>8} {'avg cache %':>12} {'p50 ms':>9} {'p90 ms':>9} {'errors':>7}")
    print("  "+"─"*88)
    for r in results: print(f"  {r['name']:<38} {r['warm_pct']:>7.0f}% {r['avg_cache_pct']:>11.1f}% {r['p50_ms']:>9.0f} {r['p90_ms']:>9.0f} {r['errors']:>7}")
    print("\n  Compare A↔B (body sharding) and C↔D (header sharding); E is true cold; F is gateway defaults.\n")

if __name__=="__main__": main()
