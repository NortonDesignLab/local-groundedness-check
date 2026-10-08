# Benchmark harness

Everything in `docs/BENCHMARK.md` and `docs/RESULTS_n344.txt` was produced by the scripts here.
The dataset and the raw per-item results are **not** in the repo: HaluBench is CC-BY-NC-2.0, so
you fetch it yourself. The scripts checkpoint by item id and resume, so a crash or reboot only
costs the in-flight batch.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install pandas pyarrow numpy scikit-learn requests
```

Point the adapters at your servers (defaults are my LAN):

```bash
export GUARDIAN_URL=http://10.0.0.121:8014      # llama.cpp, Granite Guardian 4.1-8B
export LETTUCE_URL=http://10.0.0.72:8077        # vLLM, LettuceDetect v2
export QWEN_URL=http://10.0.0.72:8000 QWEN_MODEL=local   # any general model, optional
```

Get the data: download `PatronusAI/HaluBench` from Hugging Face and save it as
`halubench.parquet` in this directory (columns: `id, source_ds, question, passage, answer, label`).

## Reproduce the n=344 study

```bash
python make_sample.py          # 344 items: 6 sources x 2 labels x ~30, seed 20260924, passages <= 6000 chars
python run_bench.py            # Guardian no-think x2, no-think without question, think, LettuceDetect
python analyze.py              # per-arm metrics, repeatability, per-source table
python ensemble.py             # AND / OR combinations, McNemar think vs no-think, GPU cost
```

Then the follow-ups, in the order they were done:

| Script | Question it answers |
|---|---|
| `run_filter_bench.py` | does the real Open WebUI filter improve with the question turn? (yes, chi2 7.15) |
| `run_conf.py` + `conf_analysis.py` | is Guardian's logprob confidence usable? (ranks, AUC 0.815; not calibrated) |
| `arm_qwen.py` | a general 27B as a third judge (0.785 solo; best in OR with Guardian) |
| `arm_jev.py` | Jev (TypeSafe AI) in the judge slot: documented request shape, access pending, never executed; `TYPESAFE_API_KEY` + `python arm_jev.py` fills `big_jev.jsonl` |
| `injection_test.py` + `injection_analysis.py` | can poisoned passages flip the verdict? (1,760 calls, 5 attack styles) |
| `retrieval.py`, `build_pools.py`, `retrieval_quality.py`, `retrieval_downstream.py` | keyword vs reranker vs embeddings for passage selection (better recall, flat accuracy) |
| `numcheck.py` | deterministic orphan-number rule (precision 1.00 when density-gated) |
| `atomize.py` + `decomp_test.py` | judging atomic facts instead of sentences (recall up, specificity down, dead heat) |
| `calibrate.py` | isotonic calibration of the confidence (ECE 0.232 to 0.053) |
| `test_grammar.py` | GBNF grammar on Guardian's output (no measurable benefit) |
| `hhem/run_hhem.py` | Vectara HHEM-2.1-Open as a CPU screen (512-token truncation, rejected) |
| `regress_v1.py` | the shipped filter code end to end over the same items |

## Reproduce the n=1094 audit

```bash
python make_sample_big.py      # triples the sample, carries the original 344 over
python run_big.py              # lettuce, nothink, nothink+grammar, think, qwen
```

The McNemar tests in `docs/BENCHMARK.md` ("v1.3.0: evidence audit") compare each component
against the configuration without it over these results.

## Probes (no dataset needed)

- `tmpl_probe.py`: proves llama.cpp drops a top-level `documents` field (uses `/apply-template`).
- `lp_probe.py`: shows the yes/no logprobs on the score token.
- `probe_router.py`, `probe_router2.py`: which request shapes make the router's LettuceDetect check fire.
- `guardian.py` run directly: a five-case smoke test in think and no-think mode.

## Shared detection core

`core_block.py` is the single source of truth for the prompts, grammar, number rule, atom
splitter and confidence extraction used by the Open WebUI filter. `sync_core.py` writes it into
`../openwebui/guardian_source_check.py` (and any extra standalone files passed as arguments)
between the `NDL GROUNDCHECK CORE` markers, thinning `calibration_map.json` as it goes.

## Qwen arm on the stock checkpoint (v1.3.1)

```bash
QWEN_URL=http://host:8000 QWEN_MODEL=qwen3.8-27b python run_qwen_stock.py   # big_qwen_stock.jsonl
python analyze_qwen_stock.py big_qwen_stock.jsonl                         # final-config table + McNemar
python analyze_qwen_stock.py big_qwen.jsonl                               # same scoring on the abliterated run
```
