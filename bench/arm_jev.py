"""Jev (TypeSafe AI) as a groundedness judge: adapter + runner over the n=1094 sample.

STATUS: stub. No early-access key on this machine, so this has never been executed against the
live API. The request shape follows the public API reference as of October 2026 (one endpoint,
typed questions, a 0-to-1 answer for the boolean type, which the native API calls "noul"). If the
schema has moved by the time you have a key, the only things to touch are `build_body` and
`parse_answer`. Everything else matches the other arms, so `analyze_qwen_stock.py big_jev.jsonl`
scores it in the same table as Guardian, LettuceDetect and Qwen.

Two things this measures that the vendor does not claim. Jev's "can't hallucinate" is a guarantee
that the output fits the schema, not that a sentence is supported by a passage; this arm asks it
the second question. And Jev is hosted, so every passage and answer in the sample leaves the
machine. For HaluBench that is fine. For client documents it is the reason the shipped stack is
local.

Usage:
  export TYPESAFE_API_KEY=...            # early-access key
  export TYPESAFE_MODEL=jev-latest       # or a pinned version
  python arm_jev.py                      # writes big_jev.jsonl, resumes by id
  python analyze_qwen_stock.py big_jev.jsonl   # NOTE: scores it in the Qwen slot of the stack
"""
import json, os, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("TYPESAFE_URL", "https://api.typesafe.ai/v1/systemone")
KEY = os.environ.get("TYPESAFE_API_KEY")
MODEL = os.environ.get("TYPESAFE_MODEL", "jev-latest")
THRESHOLD = float(os.environ.get("JEV_THRESHOLD", "0.5"))   # p(ungrounded) at or above this flags
QUESTION_NAME = "ungrounded"

INSTRUCTIONS = (
    "The CONTEXT is the only trusted evidence. Is any factual element of the ANSWER absent from, "
    "or in contradiction with, the CONTEXT? Wrong numbers, invented entities and unsupported "
    "extrapolation count as unsupported. Correct paraphrase and correct arithmetic over stated "
    "figures do not."
)

def available():
    return bool(KEY)

def build_body(ex):
    """One typed question against a structured state. Keep the state JSON rather than one blob so
    the model sees the three parts as fields."""
    return {
        "model": MODEL,
        "state": {"context": ex["passage"], "question": ex["question"], "answer": ex["answer"]},
        "questions": {
            QUESTION_NAME: {
                "type": "noul",
                "instructions": INSTRUCTIONS,
                "true": "The answer states something the context does not support.",
                "false": "Every factual element of the answer is supported by the context.",
            }
        },
    }

def parse_answer(resp):
    """Return p(ungrounded) in [0,1] or None. Native API: answers.<name>.noul; the Vercel SDK route
    renames it to probability, so accept both."""
    a = (resp.get("answers") or {}).get(QUESTION_NAME) or {}
    for k in ("noul", "probability", "value"):
        if isinstance(a.get(k), (int, float)):
            return float(a[k])
    return None

def jev_arm(ex, timeout=120, retries=2):
    if not KEY:
        return {"ungrounded": None, "latency_s": None, "p_ungrounded": None,
                "raw": "ERROR: no TYPESAFE_API_KEY (access pending)"}
    body = build_body(ex); last = None
    for a in range(retries + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json",
                                                  "authorization": f"Bearer {KEY}"})
            r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            dt = time.time() - t0
            p = parse_answer(r)
            return {"ungrounded": (None if p is None else p >= THRESHOLD),
                    "latency_s": round(dt, 3), "p_ungrounded": p, "raw": json.dumps(r)[:2000]}
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 422):      # schema or auth problem: retrying will not help
                return {"ungrounded": None, "latency_s": None, "p_ungrounded": None,
                        "raw": f"ERROR: HTTP {e.code} {e.read()[:300]!r}"}
            last = e; time.sleep(1 + 2 * a)
        except Exception as e:
            last = e; time.sleep(1 + 2 * a)
    return {"ungrounded": None, "latency_s": None, "p_ungrounded": None, "raw": f"ERROR: {last}"}

if __name__ == "__main__":
    if not available():
        sys.exit("TYPESAFE_API_KEY not set: Jev is early access, this arm is a stub until a key exists.")
    sample_file = sys.argv[1] if len(sys.argv) > 1 else "sample_big.jsonl"
    out = os.environ.get("JEV_OUT", "big_jev.jsonl")
    SAMPLE = [json.loads(l) for l in open(sample_file)]
    done = set()
    try:
        for l in open(out): done.add(json.loads(l)["id"])
    except FileNotFoundError: pass
    todo = [e for e in SAMPLE if e["id"] not in done]
    print(f"model={MODEL} url={URL} todo={len(todo)}", flush=True)
    # smoke one item first: a schema error should stop the run, not write 1094 error rows
    probe = jev_arm(todo[0]) if todo else None
    if probe and probe["ungrounded"] is None:
        sys.exit(f"first call did not yield a verdict, fix build_body/parse_answer: {probe['raw'][:400]}")
    lock = threading.Lock(); fh = open(out, "a"); t0 = time.time(); n = [0]
    def work(e):
        r = jev_arm(e)
        with lock:
            fh.write(json.dumps({"id": e["id"], "source_ds": e["source_ds"], "label": e["label"],
                                 "pred_ungrounded": r["ungrounded"], "latency_s": r["latency_s"],
                                 "p": r["p_ungrounded"], "model": MODEL}) + "\n"); fh.flush(); n[0] += 1
            if n[0] % 100 == 0:
                el = time.time() - t0; print(f"{n[0]}/{len(todo)} {el:.0f}s", flush=True)
    with ThreadPoolExecutor(4) as ex: list(ex.map(work, todo))
    fh.close(); print(f"done {time.time()-t0:.0f}s", flush=True)
