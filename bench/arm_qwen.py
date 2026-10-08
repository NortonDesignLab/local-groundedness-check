"""General-purpose Qwen3.8-27B (ATOM :8000) used as a groundedness judge.

Motivated by the LLM-AggreFact leaderboard, where a general Qwen3-32B (77.6 BAcc)
edges out purpose-built detectors like MiniCheck-7B (77.4) and Guardian (76).
Question: is the big model Brad already runs a better adjudicator than Guardian-think?
"""
import json, os, re, time, urllib.request

# Any OpenAI-compatible general model. The n=344 and n=1094 runs used Huihui-Qwen3.8-27B-abliterated (FP8);
# the slot now serves stock Qwen3.8-27B under the alias "local", so re-measure before quoting the Qwen rows as current.
URL = os.environ.get("QWEN_URL", "http://10.0.0.72:8000").rstrip("/") + "/v1/chat/completions"
MODEL = os.environ.get("QWEN_MODEL", "local")

SYSTEM = (
    "You are a strict groundedness judge. You are given a CONTEXT (the only trusted evidence), "
    "the QUESTION that was asked, and an ANSWER. Decide whether every factual element of the "
    "ANSWER is supported by the CONTEXT.\n"
    "An answer is UNGROUNDED if it states anything absent from, or contradicting, the CONTEXT — "
    "including wrong numbers, invented entities, or unsupported extrapolation. Correct "
    "paraphrase and correct arithmetic over stated figures are GROUNDED.\n"
    "Reply with exactly one word: GROUNDED or UNGROUNDED."
)

def qwen_arm(ex, timeout=300, retries=2, think=False):
    user = (f"CONTEXT:\n{ex['passage']}\n\nQUESTION:\n{ex['question']}\n\n"
            f"ANSWER:\n{ex['answer']}\n\nVerdict:")
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 8,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": think},
            "logprobs": True, "top_logprobs": 20}
    last = None
    for a in range(retries + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            dt = time.time() - t0
            ch = r["choices"][0]
            raw = (ch["message"].get("content") or "").strip()
            up = raw.upper()
            if "UNGROUNDED" in up:
                v = True
            elif "GROUNDED" in up:
                v = False
            else:
                v = None
            return {"ungrounded": v, "latency_s": round(dt, 3), "raw": raw[:200],
                    "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": r.get("usage", {}).get("completion_tokens")}
        except Exception as e:
            last = e; time.sleep(1 + 2 * a)
    return {"ungrounded": None, "latency_s": None, "raw": f"ERROR: {last}"}

if __name__ == "__main__":
    import threading
    from concurrent.futures import ThreadPoolExecutor
    SAMPLE = [json.loads(l) for l in open("sample.jsonl")]
    out = "results_qwen27b.jsonl"; done = set()
    try:
        for l in open(out): done.add(json.loads(l)["id"])
    except FileNotFoundError: pass
    todo = [e for e in SAMPLE if e["id"] not in done]
    print(f"{len(todo)} to run", flush=True)
    lock = threading.Lock(); fh = open(out, "a"); t0 = time.time(); n = [0]
    def work(ex):
        r = qwen_arm(ex)
        rec = {"id": ex["id"], "source_ds": ex["source_ds"], "label": ex["label"],
               "pred_ungrounded": r["ungrounded"], "latency_s": r["latency_s"], "raw": r.get("raw")}
        with lock:
            fh.write(json.dumps(rec) + "\n"); fh.flush(); n[0] += 1
            if n[0] % 50 == 0: print(f"{n[0]}/{len(todo)}", flush=True)
    with ThreadPoolExecutor(8) as ex: list(ex.map(work, todo))
    fh.close(); print(f"done in {time.time()-t0:.0f}s")
