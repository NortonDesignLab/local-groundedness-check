# Local groundedness checking with LettuceDetect v2 and Granite Guardian

How I check that answers and deliverables built from retrieved documents actually say what the
documents say, without any text leaving my own machines. Two open-weight detectors, measured
against each other on 1,094 labelled items, wired into three places: the inference router, the chat
UI, and the export-time check I run on finished decks and memos.

- **LettuceDetect v2** (`KRLabsOrg/lettucedect-v2-qwen-2b`, Apache-2.0, 2B, served by vLLM) finds
  unsupported *spans* in an answer. Recall 0.97, specificity 0.42. It is the cheap screen.
- **IBM Granite Guardian 4.1-8B** (official Q8_0 GGUF, Apache-2.0, served by llama.cpp) is a
  yes/no *judge* of whether a statement is grounded in the supplied documents. Think mode, with
  the question included, is the adjudicator.
- **Any general local model** (a Qwen3 27B in my case) as a second adjudicator. Flagging when
  *either* heavyweight objects is the single largest measured improvement in the stack.
- **A deterministic orphan-number rule** that fires only on prose-density sources.

**Final configuration, n=1094 HaluBench items, 6 sources, label-balanced:**

| Configuration | Bal. acc | Recall | Specificity | Missed fabrications |
|---|---|---|---|---|
| Guardian think alone | 0.743 | 0.720 | 0.766 | 154 |
| LettuceDetect alone | 0.648 | high | low | screen only |
| LettuceDetect AND Guardian-think | 0.754 | 0.711 | 0.796 | 159 |
| **LettuceDetect AND (Guardian-think OR Qwen-27B) + number rule** | **0.795** | **0.862** | 0.727 | **76** |

Every component was McNemar-tested against the configuration without it; two that looked good
(GBNF grammar, reranker passage selection) were dropped when the measurement said no. Full write-up
in [docs/BENCHMARK.md](docs/BENCHMARK.md). One caveat on the headline: the Qwen arm ran against an
abliterated 27B checkpoint that has since been replaced by stock Qwen3.8-27B, so that row should be
re-measured before being quoted as current.

## Where the checks run

```
                 ┌──────────────────────────── GB10 "ATOM" ─────────────────────────────┐
  Open WebUI ───►│ vLLM Semantic Router :8899 ──► local LLM lane (:8201 / :8000)         │
  (chat, RAG)    │        │  HaluGate plugin: Sentinel → LettuceDetect :8077 → Explainer  │
                 │        └─ only on role:tool-grounded answers; annotates response headers│
                 └───────────────────────────────────────────────────────────────────────┘
        │ outlet filter (openwebui/guardian_source_check.py)
        ▼
   sentences ──► LettuceDetect :8077 (screen) ──► Granite Guardian :8014 think (judge, RTX 5090)
                                                ──► Qwen-27B :8000 (second vote, OR)
                                                ──► orphan-number rule
        │
        ▼  "Source check: 23 of 25 checked statements are supported... Not supported: ..."

  export time:  tools/check_claims.py  claims.json  (same stack, batch, planted controls)
```

1. **Router side (HaluGate in vLLM Semantic Router).** LettuceDetect is the detector backend for the
   router's hallucination-mitigation module. It only fires on answers whose grounding arrived as a
   `role:tool` message, and only because a `local_factcheck` decision references the `fact_check`
   signal; declaring the signal alone did nothing. Config excerpt in
   [deploy/vllm-sr/config.snippet.yaml](deploy/vllm-sr/config.snippet.yaml); what actually fires for
   which request shape in [docs/LIVE_PIPELINE_CHECK.md](docs/LIVE_PIPELINE_CHECK.md).
2. **Chat side (Open WebUI filter).** Ordinary knowledge-base RAG never reaches the router's check
   (context lands in the system prompt), so the filter does the job with the passages it already
   holds. Screen every sentence, judge what the screen flagged, second vote, number rule, append a
   note. Flags only, never blocks. [openwebui/](openwebui/).
3. **Deliverable side (the protocol).** Before a factual deck or memo goes out: verbatim passages
   per claim, one fact per sentence, the question or slide heading attached, planted false
   controls, and a report of what was flagged and why. [docs/PROTOCOL.md](docs/PROTOCOL.md) is the
   procedure; [tools/check_claims.py](tools/check_claims.py) runs it.

## Repo layout

| Path | Contents |
|---|---|
| `deploy/granite-guardian/` | llama.cpp compose for Guardian (GPU-pinned x86 variant and the DGX Spark variant) and a five-case smoke test that also proves the documents reach the prompt |
| `deploy/lettucedetect-v2/` | vLLM compose for LettuceDetect v2 at 7% of a GB10's memory |
| `deploy/vllm-sr/` | the three router config blocks that make the HaluGate detector fire |
| `openwebui/` | the chat filter (v1.3.1), install notes, and the in-container DB installer |
| `tools/check_claims.py` | batch checker: claims + passages JSON in, verdicts and provenance out |
| `examples/claims.json` | a worked input with four true claims and three planted fabrications |
| `bench/` | the full harness: adapters, samplers, runners, analysis, injection test, probes |
| `docs/BENCHMARK.md` | the complete measurement log, n=344 through the n=1094 audit |
| `docs/PROTOCOL.md` | how the judges are used on real deliverables, and what they are strict about |
| `docs/GOTCHAS.md` | everything that silently fails, in the order it cost me time |

## Quick start

**Guardian** (any CUDA box with ~12 GB free):

```bash
cd deploy/granite-guardian && docker compose up -d        # pulls the 8.9 GB GGUF on first start
GUARDIAN_URL=http://localhost:8014 python3 guardian_test.py
```

Expected: `document reaches the prompt: True`, then five cases with faithful ones `ungrounded=no`
and the wrong number, invented detail and invented citation `ungrounded=yes`, about 0.3 to 0.6 s each.

**LettuceDetect** (vLLM, ~8 GB):

```bash
cd deploy/lettucedetect-v2 && docker compose up -d
curl -s localhost:8077/v1/models | head -c 200
```

Send one warm-up request after every restart; the first call JIT-compiles kernels and can exceed a
caller's timeout.

**Check some claims:**

```bash
GUARDIAN_URL=http://host:8014 LETTUCE_URL=http://host:8077 python3 tools/check_claims.py examples/claims.json
```

Exit status 0 clean, 1 something flagged, 2 a planted control was *not* caught (do not trust the
run), 3 nothing reachable. On the example, expect the screen to flag most of the true claims too
(it marked the word "had" and a correct paraphrase as unsupported); that is its measured
specificity, and the judge is what clears them.

## Three things to know before trusting any of this

1. **llama.cpp drops a top-level `documents` field without complaint.** Pass them in
   `chat_template_kwargs`. With no documents Guardian says "grounded" to everything: a silent
   false pass. Every caller here asserts the documents list is non-empty, and the smoke test
   checks the rendered prompt.
2. **Pass the question.** Judging a sentence alone costs 0.06 balanced accuracy and a third more
   false flags (McNemar chi2 7.15). For a slide, the title or section heading is the question.
3. **AND for sources you control, OR for sources you do not.** The two detectors fail to opposite
   injection attacks (Guardian to authority claims in the passage, LettuceDetect to injected
   instructions), so `combine=or` is hard to evade and `combine=and` is accurate. Pick by provenance.

The rest is in [docs/GOTCHAS.md](docs/GOTCHAS.md).

## Hardware this runs on

Two NVIDIA GB10 machines (128 GB unified each) and an x86 workstation with an RTX 5090 and an RTX
3090. LettuceDetect, the router and the local LLM live on the first GB10; Guardian on the workstation's
RTX 5090 beside the embedder (it also ran on the second GB10 for a while, hence the Spark compose);
ChromaDB and a reranker are on the workstation too. None of that is required: Guardian needs about 12 GB
of GPU memory, LettuceDetect about 8 GB, and both speak the OpenAI chat API, so the endpoints in
the valves and environment variables are the only coupling.

## Licences

Code here is MIT. Granite Guardian 4.1-8B and LettuceDetect v2 are Apache-2.0 from their authors.
HaluBench (PatronusAI) is CC-BY-NC-2.0 and is not redistributed; `bench/README.md` says how to
fetch it. The per-item result files are available on request for non-commercial use.

## Status

Live as of 8 October 2026: filter v1.3.1 installed on the Open WebUI instance with its valve on the
5090 copy, LettuceDetect up on the ATOM, Guardian on SARA's RTX 5090 (the Spark copy is stopped to
leave memory for a 1M-context model). Verified that day: smoke test 5/5, example check 3/3 controls.
