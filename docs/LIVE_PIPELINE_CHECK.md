# Live pipeline check — what actually runs on "NDL Router + Tools" (24 Sep 2026)

**Short answer: no. Guardian runs on everything; LettuceDetect only runs when the answer came
from a native tool call; Qwen is not wired in at all. The recommended arrangement is not live.**

## What is configured

- Open WebUI (`open-webui-atom`, SARA :3000) has the router as OpenAI connection #4,
  `http://10.0.0.72:8899/v1`, model **`ndl-router-tools`** → base model `vllm-sr/auto`.
- `guardian_source_check` is installed, **`is_active=1` and `is_global=1`** — it runs on every
  model, including the router model. Confirmed live: 8 of the last 40 chats carry a Source check
  note, all on `vllm-sr/auto`.

## What actually fires, probed directly against Envoy :8899

| Request shape | Router decision | Warning header |
|---|---|---|
| Context in **system prompt**, no tools *(Open WebUI knowledge-base RAG)* | `local_privacy` | **`unverified_factual`** |
| Context in system prompt, tools declared | `local_privacy` | – |
| Context pasted into the **user message** | `local_privacy` | **`unverified_factual`** |
| Context as a **`role:tool`** result | **`local_factcheck`** | `hallucination` *(detector ran and fired)* |
| Plain factual question, no context | `local_factcheck` | `unverified_factual` |

Two separate things are going on:

1. **Routing.** `local_privacy` wins for prompts whose context arrives in the system or user
   message, so `local_factcheck` is never reached on that path.
2. **The check itself.** LettuceDetect needs a `role:tool` message. This confirms the earlier
   finding — a system-prompt context string does not count as grounding.

**Consequence:** an ordinary Open WebUI knowledge-base RAG answer is routed to `local_privacy`
and returned with **`unverified_factual`** — the router itself saying it could not verify a
factual answer. LettuceDetect never sees it. That header is not surfaced anywhere in the
Open WebUI interface, so nobody knows.

Answers that *did* use native NDL tools do reach `local_factcheck` and are checked.

## So the live arrangement is

| Component | Runs on knowledge-base RAG? | Runs on native tool answers? |
|---|---|---|
| Guardian (Open WebUI filter) | **yes** — reads `sources`, tool `output`, long pasted text | yes |
| LettuceDetect (router) | **no** | yes |
| Qwen-27B adjudicator | **no** — not wired anywhere | no |

Effective accuracy on the RAG path is therefore **Guardian alone (0.738 measured on the real
filter)**, not the 0.801 of LettuceDetect AND (Guardian OR Qwen), and not even the 0.764 pair.

Evidence of the cost in live chats, from notes written before today's v0.2.0 fix:
*"6 of 25 checked statements are supported"*, *"15 of 25"*, *"8 of 20"* — and the flagged items
include non-claims like *"Here's everything I found, grouped by type"* and *"I did a full
read-through of all three decks."* Those are the over-flagging and own-headings failure modes,
which the question turn and the ensemble both exist to fix.

## Two ways to close the gap

**A. Fix it in the router (right place, shared blast radius).** Make `local_factcheck` reachable
for system-prompt context — either by lowering `local_privacy`'s priority below 150 or tightening
its rule, and by feeding retrieved passages as tool-result context. This changes routing for
everything on the ATOM and should not be done without a maintenance window.

**B. Fix it in the Open WebUI filter (contained, and the code already works).** The filter
already has the passages in hand — that is how Guardian works today. Adding LettuceDetect
(:8077) and Qwen (:8000) as two more calls inside `guardian_source_check.py` puts the full
LettuceDetect AND (Guardian OR Qwen) arrangement on *every* model, independent of routing.

### Latency reality check for option B, at the filter's 25-claim cap

| Stage | Per claim | 25 claims |
|---|---|---|
| Guardian no-think *(today)* | 0.58 s | ~7 s at 2 slots |
| LettuceDetect | 1.03 s | ~3 s at 8-way |
| Guardian think | 3.72 s | ~46 s at 2 slots |
| Qwen-27B | 10.6 s | ~33 s at 8-way |

The full 0.801 ensemble is **~80 s per answer** — far too slow for an interactive chat filter.
Recommended shape instead:

- **LettuceDetect on all claims** (cheap, recall 0.966) → only flagged claims continue.
- **Guardian-think on the flagged subset only.**
- **Leave Qwen out of the interactive path**; keep it for export-time and deck checks, where
  its 0.785 solo and the FinanceBench/DROP caveat both matter more than latency.

That lands near the measured 0.764 pair at a fraction of the cost, because most sentences in a
real answer are grounded and never reach the expensive judge.

## Housekeeping found along the way

Two OpenRouter API keys are sitting in plaintext — one in the shell environment, one in
Open WebUI's `openai.api_keys` config. Both were printed into session transcripts. Worth rotating.
