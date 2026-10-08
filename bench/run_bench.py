import json, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ".")
import arms

SAMPLE = [json.loads(l) for l in open("sample.jsonl")]

RUNS = [
    ("guardian_nothink_q_A", lambda ex: arms.guardian_arm(ex, think=False, with_question=True), 2),
    ("guardian_nothink_q_B", lambda ex: arms.guardian_arm(ex, think=False, with_question=True), 2),
    ("guardian_nothink_noq", lambda ex: arms.guardian_arm(ex, think=False, with_question=False), 2),
    ("guardian_think_q",     lambda ex: arms.guardian_arm(ex, think=True,  with_question=True), 2),
    ("lettucedetect",        lambda ex: arms.lettuce_arm(ex), 8),
]

def run(name, fn, conc):
    out = f"results_{name}.jsonl"
    done = set()
    try:
        for l in open(out):
            done.add(json.loads(l)["id"])
    except FileNotFoundError:
        pass
    todo = [e for e in SAMPLE if e["id"] not in done]
    if not todo:
        print(f"[{name}] already complete ({len(done)})", flush=True); return
    print(f"[{name}] {len(todo)} to run, concurrency {conc}", flush=True)
    lock = threading.Lock(); t0 = time.time(); n = [0]
    f = open(out, "a")
    def work(ex):
        r = fn(ex)
        rec = {"id": ex["id"], "source_ds": ex["source_ds"], "label": ex["label"],
               "pred_ungrounded": r.get("ungrounded"), "latency_s": r.get("latency_s"),
               "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("completion_tokens"),
               "n_spans": r.get("n_spans"), "raw": (r.get("raw") or "")[:600]}
        with lock:
            f.write(json.dumps(rec) + "\n"); f.flush(); n[0] += 1
            if n[0] % 25 == 0:
                el = time.time() - t0
                print(f"[{name}] {n[0]}/{len(todo)}  {el:.0f}s  eta {el/n[0]*(len(todo)-n[0]):.0f}s", flush=True)
    with ThreadPoolExecutor(max_workers=conc) as p:
        list(p.map(work, todo))
    f.close()
    print(f"[{name}] DONE {len(todo)} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    only = sys.argv[1:] or None
    for name, fn, conc in RUNS:
        if only and name not in only: continue
        run(name, fn, conc)
    print("ALL RUNS COMPLETE", flush=True)
