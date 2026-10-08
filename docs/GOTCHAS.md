# Gotchas, in the order they cost me time

## Granite Guardian on llama.cpp

**A top-level `documents` field is silently dropped.** The Jinja template only sees
`chat_template_kwargs.documents`. Verified with `/apply-template`: top-level documents render a
183-character prompt with no document in it; under `chat_template_kwargs` the same request renders
811 characters with the document present. With no documents Guardian answers "no" (grounded), so
**the broken transport is a silent false pass**. The smoke suite scores 2/5 broken and 5/5 fixed.
`deploy/granite-guardian/guardian_test.py` asserts the document text appears in the rendered prompt
before it judges anything. Do the same.

**Always assert the documents list is non-empty** for the same reason. Every caller in this repo
does (`assert documents`, or an early return when the conversation holds too little source text).

**Think mode saturates the score token.** The logprob confidence (P(ungrounded) renormalised over
the `yes`/`no` alternatives of the score token) ranks well in no-think mode (AUC 0.815) but in
think mode 146 of 161 flags scored above 0.80, so the "high confidence" mark stopped meaning
anything. The filter suppresses the mark when `guardian_think` is on; the isotonic calibration is
only valid with think off.

**The confidence is a ranking signal, not a probability.** Mean confidence 0.950 against 0.718
accuracy; the 0.9 to 0.99 band is *worse than a coin flip* (0.486). Only >= 0.9999 is trustworthy
(0.966 accuracy, 19% coverage). Same shape other vendors' calibration studies report: gate at the
very top band or not at all.

**Give think mode room.** `max_tokens` 2048 returned parse failures on long passages; 6144 is safe.

**GBNF grammar is not needed.** Measured at n=1094: zero unparseable verdicts in 2188 calls with or
without it, identical accuracy, and it doubles no-think latency. Shipped off.

**This llama.cpp build stores `-hf` downloads under `/root/.cache/huggingface`**, not
`/root/.cache/llama.cpp`. Mount accordingly or the 9 GB download repeats on every recreate.

**Memory:** Q8_0 weights 8.3 GiB + 2.5 GiB KV at 16384 context, f16, 2 slots, about 11.5 GB total.
If the card is shared with a desktop or another model and OOMs, lower `-c` to 16384 or below.

## LettuceDetect v2 on vLLM

**Cold start.** The first inference after a container restart JIT-compiles Triton kernels and can
take long enough to exceed the router's detector timeout ("context deadline exceeded"). Warm calls
take about 1.6 s. Send a warm-up request after every restart or the first real check is lost.

**The prompt must match the training prompt.** The system prompt in `bench/arms.py` is the one that
works; paraphrasing it degrades span detection.

**It is a generative JSON model, so it takes orders.** Instruction-shaped text injected into a
passage suppresses spans 6.5% of the time appended and 12.4% prepended. Authority claims
("pre-approved by compliance") move it 0.0%. Guardian is the mirror image: 6.2% on authority
claims, noise-level on instructions. That asymmetry is why `combine=or` is evasion-resistant.

**It is a screen, never a gate.** Recall 0.966, specificity 0.417 at n=344 (0.648 balanced
accuracy at n=1094). It flags 78% of everything. Its job is to decide what the expensive judge
looks at.

## vLLM Semantic Router (HaluGate)

**A signal is only evaluated when a considered decision references it.** Declaring the
`fact_check` signal was not enough; it returned null, the request-side filter set
`FactCheckNeeded=false`, and the detector plugin never ran. The fix was a `local_factcheck`
decision whose rule references `fact_check: needs_fact_check` (see
`deploy/vllm-sr/config.snippet.yaml`). `evaluate_all_signals` is a per-request API field, not a
config key; putting it in `config.yaml` is a no-op.

**The detector needs a `role:tool` message.** Grounding pasted into the system prompt or the user
message does not count as context (`HasToolsForFactCheck` stays false). Ordinary Open WebUI
knowledge-base RAG injects context into the system prompt, so it is **never checked by the
router**; it returns the header `x-vsr-response-warnings: unverified_factual`, which the UI does not
surface. The Open WebUI filter in this repo exists because of this: it has the passages in hand
regardless of how they arrived.

**Priority shadowing.** A higher-priority decision (`local_privacy`, 300, triggered by `$` and `%`
figures via the PII rule) catches most RAG-shaped prompts before `local_factcheck` (150) is
considered. Detection still runs there if the plugin is attached, but check which lane your
traffic actually takes before trusting the router-side check.

**Lowering the Sentinel threshold had no effect** on whether the detector fired. The gap was signal
evaluation, not the threshold.

## Open WebUI

**Functions and tools are standalone pasted files**; they cannot import a shared module. The
detection core lives once in `bench/core_block.py` and `bench/sync_core.py` writes it into each
target between markers, asserting each still parses. Never hand-edit a copy.

**Tool specs are generated at save time.** Writing a tool's code straight into `webui.db` leaves the
stored spec stale, so new parameters (like `sources`) never reach the model and every check silently
skips. `openwebui/install_into_db.py` regenerates specs with Open WebUI's own `get_tool_specs`.
Restart the container afterwards; compiled functions are cached.

**Sync tool functions run on the event loop** (0.11.x). Anything that calls back into the server
or does slow HTTP must be `async def` and wrap blocking work in `asyncio.to_thread`, or the
single-worker server deadlocks. The filter's `outlet` does exactly that.

**Import JSON ignores `is_active` / `is_global`.** Toggle them in the UI after importing.

**Container networking.** A Docker-network hostname only resolves from containers on that network.
When the detectors live on other machines, use LAN IPs in the valves and verify reachability from
*inside* the Open WebUI container, not from the host.

## Benchmarking

**Underpowered samples hide real effects.** At n=344 think mode (chi2 1.76) and the LettuceDetect
screen (chi2 3.12) looked unproven; at n=1094 both cleared p<0.05 (6.61 and 6.05). Every estimate
also dropped slightly at the larger n. Quote the n=1094 numbers.

**Pairing matters.** LettuceDetect in front of Guardian *no-think* scores worse than think mode
alone (0.729 vs 0.746). The screen only pays in front of think mode. Keep the two settings together.

**Requiring all three judges to agree is a trap**: it collapses to Guardian AND Qwen (0.752) and
trades 21 extra misses for 8 fewer false alarms. OR the heavyweights, AND the screen.

**The number rule must be density-gated.** Ungated, it hurts (0.764 to 0.742) because financial
and numeric-reasoning answers legitimately contain arithmetic derived from the passage (precision
0.33 on DROP). At or below 5 numbers per 1000 source characters it has precision 1.00 across 1438
items.

**Negative results worth knowing.** A reranker for passage selection: grounding recall 92.2% to
98.8%, judge accuracy flat. Atomic decomposition of sentences: recall to 1.000, specificity 0.40 to
0.25, dead heat. Vectara HHEM-2.1-Open as a CPU screen: truncates at 512 tokens, AUC 0.650, drags
the pair down. All measured, all off.

**`pgrep -f <script>` matches the shell that is checking.** Use `ps -eo pid,etimes,cmd | grep [r]egex`.
Checkpoint every run by item id so a host reboot only costs the in-flight batch.
