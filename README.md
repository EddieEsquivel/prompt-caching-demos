# Prompt Caching Demos

End-to-end, runnable demos of **LLM prompt caching on Microsoft Foundry** — built around a production use case: a high-volume **content-safety classifier** that sends the same ~1,280-token system prompt with every one of millions of short user messages.

Written for **Solutions Architects and Solutions Engineers** to run live in front of customers or hand over as a reference.

---

## The problem these demos solve

Any high-volume LLM workload with a **static system prompt** (classifiers, moderation, routing, extraction) pays to re-encode that prompt on every request. Prompt caching eliminates that: the prefix is encoded once and reused, cutting:

- **Cost** — cached input tokens are billed at a discount (default 50% off on Fireworks serverless)
- **Latency** — prefill (input encoding) is skipped, so time-to-first-token drops sharply

The demos prove caching is actually happening with hard numbers — token-level cache-hit stats where the gateway surfaces them, and a TTFT (time-to-first-token) A/B measurement where it doesn't.

---

## What's compared

| Demo | Question it answers |
|---|---|
| [cache_breakpoint_demo.py](cache_breakpoint_demo.py) | How does GPT-5.6's `prompt_cache_breakpoint` compare to Fireworks' `prompt_cache_key`? |
| [kimi_responses_vs_chat_demo.py](kimi_responses_vs_chat_demo.py) | Does `prompt_cache_key` work the same through the Responses API and Chat Completions? |
| [kimi_stored_responses_demo.py](kimi_stored_responses_demo.py) | What does `store=True` persist, and does caching work alongside it? |
| [kimi_cache_key_vs_session_affinity_demo.py](kimi_cache_key_vs_session_affinity_demo.py) | Are body `prompt_cache_key` and header `x-session-affinity` functionally equivalent on direct Fireworks? |
| [kimi_foundry_affinity_pass_through_demo.py](kimi_foundry_affinity_pass_through_demo.py) | Do both affinity mechanisms cache through Foundry in a real multi-turn conversation? |
| [kimi_cache_key_sharding_demo.py](kimi_cache_key_sharding_demo.py) | For a stateless classifier with unique inputs, does sharding improve direct Fireworks cache reuse under load? |
| [kimi_foundry_classifier_sharding_demo.py](kimi_foundry_classifier_sharding_demo.py) | Does the same stateless classifier sharding strategy work through Foundry? |

### The core equivalence

> **`prompt_cache_key` (Fireworks) is the direct equivalent of `prompt_cache_breakpoint` (GPT).** Same outcome — the static system prompt is cached after the first request, and only the short user message is processed on subsequent calls. Different mechanism: GPT marks the cache boundary explicitly in content; Fireworks matches the longest cached prefix automatically, and `prompt_cache_key` routes requests that share it to the same replica(s) where that prefix is cached (it is equivalent to the `x-session-affinity` header, which takes precedence if both are sent).

---

## Repo layout

```
├── cache_breakpoint_demo.py        # GPT-5.6-sol (Responses API, prompt_cache_breakpoint)
│                                    #   vs FW-Kimi-K3-3 (Chat Completions, prompt_cache_key)
│                                    #   + TTFT cache proof via streaming
├── kimi_responses_vs_chat_demo.py  # FW-Kimi-K3-3 through both API shapes, same cache key
│                                    #   + TTFT cache proof for each API style
├── kimi_stored_responses_demo.py   # Responses API store=True lifecycle:
│                                    #   create → GET retrieve → previous_response_id chain → delete
├── kimi_cache_key_vs_session_affinity_demo.py
│                                    # body key vs header on direct Fireworks
├── kimi_foundry_affinity_pass_through_demo.py
│                                    # definitive Foundry multi-turn token proof
├── kimi_cache_key_sharding_demo.py # unique-input stateless classifier, direct
├── kimi_foundry_classifier_sharding_demo.py
│                                    # same classifier sharding matrix via Foundry
├── results/                         # Captured output from live runs (see below)
├── .env.example                     # Required environment variable
└── .venv/                           # (gitignored) Python 3.14 + openai SDK
```

---

## Setup

Requires an Microsoft Foundry project with access to a GPT-5.x Responses-API model and a Fireworks model (e.g. `FW-Kimi-K3-3`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai requests

cp .env.example .env    # then fill in FOUNDRY_KEY, FOUNDRY_ENDPOINT, model names
set -a; source .env; set +a
```

> **Never commit keys.** Foundry demos read `FOUNDRY_KEY`; direct Fireworks demos read `FIREWORKS_KEY`.

Foundry scripts read `FOUNDRY_ENDPOINT`, `KIMI_MODEL`, and `GPT_MODEL`. Direct
Fireworks scripts read `FIREWORKS_BASE` and `FIREWORKS_MODEL`. Concurrency/shard
counts are also environment-configurable; see `.env.example`.

---

## Demo 1 — `cache_breakpoint_demo.py`

Runs the same 5 test messages through two caching strategies:

1. **GPT-5.6-sol** via the Responses API with `prompt_cache_breakpoint: {mode: "explicit"}` on the system-prompt content block
2. **FW-Kimi-K3-3** via Chat Completions with `prompt_cache_key: "safety-classifier-v1"` in the request body

Then runs a **TTFT cache proof**: the same message streamed under two conditions —

- **COLD**: a unique `prompt_cache_isolation_key` per call forces a fresh prefill (no reuse possible)
- **WARM**: a shared `prompt_cache_key` reuses the cached prefix

**Why TTFT?** Total latency = prefill + generation, and generation time is noisy on reasoning models (Kimi-K3-3 thinks before answering). TTFT isolates prefill — the only phase caching affects — so a lower warm TTFT is unambiguous proof the cache is active.

**Sample result** ([full output](results/cache_breakpoint_demo_output.txt)):

```
MODEL: gpt-5.6-sol | prompt_cache_breakpoint
  Call 1: [ALLOW]  2428 ms  [COLD]
  Call 2: [BLOCK]  1583 ms  [WARM (99% cached)]
  ...
  First call (cold): 2428 ms | Avg warm: 1616 ms        ← ~1.5× faster, 98–99% cached
```

---

## Demo 2 — `kimi_responses_vs_chat_demo.py`

Same model (`FW-Kimi-K3-3`), same cache key, **two API shapes**:

| | Responses API | Chat Completions |
|---|---|---|
| Call | `client.responses.create(...)` | `client.chat.completions.create(...)` |
| Cache field | `extra_body={"prompt_cache_key": ...}` | same |
| Text out | `response.output_text` | `choices[0].message.content` |

Per the [Microsoft migration guide](https://learn.microsoft.com/en-us/azure/developer/ai/how-to/azure-openai-to-responses), the Responses API path uses the standard OpenAI SDK pointed at the `/openai/v1/` base URL. **Gotcha found in testing:** this Foundry gateway requires fully typed input items (`{"type": "message", "role": ..., "content": [{"type": "input_text", "text": ...}]}`) — the plain `{"role", "content"}` shortcut from the docs fails validation with `Invalid value: ''`. Also, `prompt_cache_isolation_key` is rejected by the Responses API (Chat Completions only); the demo enforces true-cold there with a unique cache-bucket name per call.

**Sample result** ([full output](results/kimi_responses_vs_chat_demo_output.txt)): both API shapes show 95–100% cached tokens on warm calls — the same `prompt_cache_key` routes both API styles to the same replica(s), so the cached prefix is reused regardless of endpoint.

---

## Demo 3 — `kimi_stored_responses_demo.py`

Walks the full **stateful Responses API lifecycle** with `store=True`:

1. **Create** — 6 stored responses (warmup + 5 messages), each returns a `response_id`
2. **Retrieve** — `GET /responses/{id}` returns the stored completion, usage, status, timestamp
3. **Chain** — a follow-up with `previous_response_id` and *only* a new user message (no system prompt, no history); the server reconstructs full context from storage
4. **Delete** — cleanup via the API

**Sample result** ([full output](results/kimi_stored_responses_demo_output.txt)): all four operations succeed on FW-Kimi-K3-3, with prompt caching active alongside storage (95–96% cached tokens).

---

## Demos 4–7 — affinity and stateless-classifier sharding

- **Direct mechanism equivalence:** `kimi_cache_key_vs_session_affinity_demo.py`
  compares body `prompt_cache_key` and header `x-session-affinity`, including
  cross-mechanism reuse and header-over-body precedence.
- **Foundry multi-turn proof:** `kimi_foundry_affinity_pass_through_demo.py`
  completes all responses so Foundry emits
  `usage.prompt_tokens_details.cached_tokens`. Both mechanisms validated at
  96.8–97.0% cached on later turns.
- **Direct stateless classifier:** `kimi_cache_key_sharding_demo.py` sends a
  byte-identical long policy prefix with unique classifier inputs and compares
  one key, N sharded keys, unique keys, and no explicit affinity.
- **Foundry stateless classifier:** `kimi_foundry_classifier_sharding_demo.py`
  runs the same workload through Foundry and compares both body-key and header
  sharding.

Live stateless-classifier results:

- **Direct Fireworks (12 concurrent):** one shared key and 12 sharded keys both
  averaged 94% warm. Sharding improved mean p50 latency (~6.25s → ~4.85s) but
  did not improve p90 or hit rate at this load. The isolation-key control was
  exactly 0% warm.
- **Foundry (3 concurrent, rate-limit-safe):** body single key 100% warm; body
  shards 83%; header single/shards 83%; isolation-key control 0%; no-affinity
  control 50%. Sharding did not improve hit rate at this low concurrency.

**Interpretation:** sharding is a capacity/load-distribution technique, not a
guaranteed cache-hit improvement. Use the smallest shard count that prevents a
single affinity domain from saturating. Extra shards create additional cache
domains that must each be warmed and can be evicted independently. Foundry's
token rate limit prevented a clean high-concurrency saturation comparison in
this run; the lower-concurrency result proves both mechanisms cache stateless
unique-input traffic, not that four shards are universally optimal.

### Why cache blocks matter for stateless fan-out

KV prefix reuse is performed in token blocks. The variable suffix must begin
*after at least one complete reusable block*; otherwise the cache has no full
block to reuse even when routing lands on the right replica.

This matters more for a stateless classifier than for a growing conversation:

```text
system policy (~1,280 tokens) + unique item
                        ↑ variable content begins before a full Kimi block
                        → 0 cached tokens observed

larger stable policy prefix + unique item
                        ↑ multiple full blocks precede the variable suffix
                        → aligned cached-token counts observed
```

In live standard Kimi K3 validation, varying suffixes did not reuse a partial
prefix until the rendered request crossed roughly 4,096 tokens; reported cached
lengths were then exact 4,096-token multiples (`4,096`, `28,672`, etc.).
Fireworks serving metadata classifies Kimi K3 as a GDN/Mamba-state model with
1,024-token underlying state pages, while the tested production deployment's
effective reusable snapshot/capture boundary is 4,096 tokens. These are
different layers of granularity.

This is **architecture-, engine-, and deployment-specific**, not a universal
API guarantee. Deployment flags and snapshot-capture strategy can override the
underlying defaults, and token-granular models use granularity 1.

Consequences:

1. Put all stable instructions, examples, schemas, and tool definitions first.
2. Put per-request classifier content last.
3. Make the stable prefix large enough to complete one or more cache blocks.
4. `prompt_cache_key`/`x-session-affinity` and sharding control *where* requests
   route; they cannot make an undersized stable prefix cacheable.
5. A unique `prompt_cache_key` is **not** a cold-cache guarantee: it changes
   routing, not cache namespace. Use a unique `prompt_cache_isolation_key` when
   a test must prohibit cross-request prefix reuse.
6. After the prefix is cacheable, shard a bounded set of stable keys to balance
   reuse against multi-replica throughput.

---

## What's stored where (data-retention story)

| Layer | What's stored | Retention / control |
|---|---|---|
| **Microsoft/Foundry** (`store=True` only) | Full request (system prompt + user message) and completion; response ID for retrieval and chaining | Stored until deleted via API; delete works per ID |
| **Fireworks** (prompt cache) | KV-cache of the prompt prefix — **volatile memory only** | Minutes to hours, evicted oldest-first; never persisted to disk. The gateway itself reports `prompt_cache_retention: "in_memory"` |
| **Fireworks logs** | Metadata only (token counts) | Operational, no customer content |

Zero Data Retention is Fireworks' default: prompt and generation data exist only in volatile memory for the duration of a request (plus the cache window if caching is active) and are never written to persistent storage.

**Guidance for a stateless classifier workload:** use `store=False` + `prompt_cache_key` — nothing persists anywhere, and the classification result is never stored by either Microsoft or Fireworks. Use `store=True` only when you need audit trails, chaining, or deferred retrieval — and note this gateway does **not** echo input items back on GET, so log the classified message on your side if you need an audit trail of what was classified.

---

## Reading the results — SA/SE notes

- **`cached_tokens` is the reliable signal.** Some runs show intermittent 0%-cache calls (bucket evicted under load) — cache retention is best-effort, minutes to hours. Expect occasional cold re-writes in production; steady traffic to a stable key maximizes hit rate. At high request rates, shard the key (e.g. `safety-classifier-v1-0` … `-N`): KV cache is replica-local, and a single key concentrates load on a few replicas.
- **Ignore single-run speedup averages** on reasoning models — generation-time variance dominates. Trust the cached-token column and the TTFT A/B.
- **Cache key choice matters:** one stable key per system-prompt version (e.g. `safety-classifier-v1`). Bump the version suffix whenever the prompt changes — a changed prefix invalidates the cache anyway, and a new key keeps buckets clean.
- **Cache boundaries matter:** Kimi K3 has 1,024-token underlying Mamba state
  pages, but the tested standard serverless deployment exposed reusable
  varying-suffix snapshots at 4,096-token boundaries. A ~1,280-token stable
  classifier prefix therefore produced 0 partial-prefix cache hits. Exact full
  prompt repeats can still hit a separate exact-path cache below 4,096. Other
  architectures/engines can be token-granular or use 64, 256, 1,024, 2,048,
  4,096, or deployment-overridden boundaries.
- **Accuracy is unaffected.** Caching reuses encoded input state only; every response is sampled fresh. All demos classify 5/5 correctly warm or cold.
- **Minimum prefix:** GPT-5.x requires ≥1,024 tokens for the cache breakpoint. Fireworks/Kimi behavior is block-aligned; for stateless unique-suffix fan-out, use a stable prefix comfortably larger than one observed cache block.

---

## Key takeaways for customer conversations

1. **`prompt_cache_key` ≡ `prompt_cache_breakpoint`** — same savings, one body field, no header changes (the key itself provides the replica affinity).
2. **~50% off cached input tokens** (Fireworks default discount) + **~1.5× lower warm latency** for this workload shape.
3. **Works through both API shapes** — customer can standardize on the Responses API across all models, or keep Chat Completions; caching is identical.
4. **Caching and storage are independent** — `prompt_cache_key` (RAM-only KV reuse) and `store=True` (server-side persistence) can be used together or separately.
5. **Strong data-retention posture by default** — with `store=False`, nothing persists on either side.
