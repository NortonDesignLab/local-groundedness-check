# Installing the source-check filter in Open WebUI

`guardian_source_check.py` is an Open WebUI **Filter** function (outlet hook). After every
assistant answer it gathers the passages the conversation actually retrieved (knowledge / RAG
`sources`, native tool `function_call_output` items, long pasted user text), splits the answer
into sentences, runs the detection stack, and appends a short **Source check** note. It flags
only; it never edits the answer, never blocks the chat, and stays silent when there are no sources
or every detector is unreachable.

Tested on Open WebUI 0.11.x.

## 1. Install the function

Admin Panel > Functions > **+ New Function**, paste the file, keep the id `guardian_source_check`,
Save. Open WebUI shows a confirmation that functions execute arbitrary code; this is your own code.

Alternative: import `guardian_source_check_import.json`-style files (an array of
`{id, name, type: "filter", content, meta}`) through the hidden file input on the Functions page.
Import does **not** honour `is_active` / `is_global`; set them in the UI afterwards.

## 2. Switch it on and attach it

Either toggle the function **Global** (runs on every model) or tick it under
Workspace > Models > (model) > Filters for the models you use for grounded work.

## 3. Set the valves for your network

The defaults are my LAN. From inside the Open WebUI container the detectors must be reachable by
the address you put in the valve, so test from inside the container, not from the host:

```bash
docker exec open-webui python3 -c "import urllib.request; print(urllib.request.urlopen('http://10.0.0.153:8014/v1/models', timeout=6).status)"
```

| Valve | What it is | My value |
|---|---|---|
| `guardian_url` | llama.cpp server with Granite Guardian | `http://10.0.0.153:8014` (DGX Spark) |
| `lettuce_url` | vLLM server with LettuceDetect v2 | `http://10.0.0.72:8077` |
| `qwen_url` / `qwen_model` | any general OpenAI-compatible model for the second vote | `http://10.0.0.72:8000`, `local` |
| `prefilter` + `guardian_think` | the two evidence-backed settings; **move them together** | both on |
| `use_qwen` | second adjudicator, flag if either objects (largest proven gain) | on |
| `combine` | `and` for sources you control, `or` for untrusted material | `and` |
| `number_check` / `number_density` | orphan-figure rule, prose sources only | on / 5.0 |
| `max_claims` | sentences checked per answer; lower this before turning off think or qwen | 25 |
| `rerank_min_pool` | reranker passage selection; measured no gain | 0 (off) |
| `grammar` | GBNF constraint on Guardian's output; measured no gain | off |
| `calibrate` | isotonic calibration of the confidence; only meaningful with think off | on (inert) |

Every valve description in the file quotes the measurement that set its default.

## 4. Latency to expect

On real answers: 8 claims about 10 s, 15 claims about 9 s when most sentences are grounded (the
screen clears them and the judge sees a handful). When Guardian is reached across the LAN rather
than on the same host, Guardian-think dominates: roughly a minute at the 25-claim cap. Lower
`max_claims` rather than switching off `guardian_think` or `use_qwen`.

## Updating without the UI

`install_into_db.py` runs **inside** the container and writes the function (and, in my setup, two
deck-builder tools) straight into `webui.db`, regenerating tool specs with Open WebUI's own
`get_tool_specs`. Back up the DB first and restart the container afterwards:

```bash
docker cp guardian_source_check.py open-webui:/work/functions/guardian_source_check.py
docker cp install_into_db.py open-webui:/tmp/install_into_db.py
docker exec -w /app/backend open-webui python3 /tmp/install_into_db.py
docker restart open-webui
```

Edit `JOBS` in that script to match the ids you actually have installed.

## Keeping the detection core in sync

The block between `# ===== NDL GROUNDCHECK CORE` and `# ===== END NDL GROUNDCHECK CORE =====`
is generated from `bench/core_block.py` by `bench/sync_core.py`. Edit the source, run the sync,
re-install. Never edit the block in place.
