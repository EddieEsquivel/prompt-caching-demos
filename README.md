# Prompt Caching Demos

End-to-end, runnable demos of **LLM prompt caching on Microsoft Foundry** — built around a production use case: a **Mercado Libre-style content-safety classifier** that sends the same ~1,280-token system prompt with every one of millions of short user messages.

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

### The core equivalence

> **`prompt_cache_key` (Fireworks) is the direct equivalent of `prompt_cache_breakpoint` (GPT).** Same outcome — the static system prompt is cached after the first request, and only the short user message is processed on subsequent calls. Different syntax: a top-level body field naming a cache bucket, vs. a per-content marker.

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

export FOUNDRY_KEY="<your-foundry-key>"     # see .env.example
```

> **Never commit the key.** All three scripts read `FOUNDRY_KEY` from the environment and fail fast if it's unset.

Each script also has `FOUNDRY_ENDPOINT` / model names at the top — edit those if your Foundry project or deployment names differ.

---

## Demo 1 — `cache_breakpoint_demo.py`

Runs the same 5 test messages through two caching strategies:

1. **GPT-5.6-sol** via the Responses API with `prompt_cache_breakpoint: {mode: "explicit"}` on the system-prompt content block
2. **FW-Kimi-K3-3** via Chat Completions with `prompt_cache_key: "meli-safety-classifier-v1"` in the request body

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

**Sample result** ([full output](results/kimi_responses_vs_chat_demo_output.txt)): both API shapes show 95–100% cached tokens on warm calls — the backend cache bucket is shared across API styles, since it's keyed by the `prompt_cache_key` string, not the endpoint.

---

## Demo 3 — `kimi_stored_responses_demo.py`

Walks the full **stateful Responses API lifecycle** with `store=True`:

1. **Create** — 6 stored responses (warmup + 5 messages), each returns a `response_id`
2. **Retrieve** — `GET /responses/{id}` returns the stored completion, usage, status, timestamp
3. **Chain** — a follow-up with `previous_response_id` and *only* a new user message (no system prompt, no history); the server reconstructs full context from storage
4. **Delete** — cleanup via the API

**Sample result** ([full output](results/kimi_stored_responses_demo_output.txt)): all four operations succeed on FW-Kimi-K3-3, with prompt caching active alongside storage (95–96% cached tokens).

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

- **`cached_tokens` is the reliable signal.** Some runs show intermittent 0%-cache calls (bucket evicted under load) — cache retention is best-effort, minutes to hours. Expect occasional cold re-writes in production; steady traffic to one stable key maximizes hit rate.
- **Ignore single-run speedup averages** on reasoning models — generation-time variance dominates. Trust the cached-token column and the TTFT A/B.
- **Cache key choice matters:** one stable key per system-prompt version (e.g. `meli-safety-classifier-v1`). Bump the version suffix whenever the prompt changes — a changed prefix invalidates the cache anyway, and a new key keeps buckets clean.
- **Accuracy is unaffected.** Caching reuses encoded input state only; every response is sampled fresh. All demos classify 5/5 correctly warm or cold.
- **Minimum prefix:** GPT-5.x requires ≥1,024 tokens for the cache breakpoint; Fireworks caching benefits from long stable prefixes similarly. The demo system prompt is ~1,280 tokens to be production-realistic.

---

## Key takeaways for customer conversations

1. **`prompt_cache_key` ≡ `prompt_cache_breakpoint`** — same savings, one body field, no sticky routing or header changes.
2. **~50% off cached input tokens** (Fireworks default discount) + **~1.5× lower warm latency** for this workload shape.
3. **Works through both API shapes** — customer can standardize on the Responses API across all models, or keep Chat Completions; caching is identical.
4. **Caching and storage are independent** — `prompt_cache_key` (RAM-only KV reuse) and `store=True` (server-side persistence) can be used together or separately.
5. **Strong data-retention posture by default** — with `store=False`, nothing persists on either side.
