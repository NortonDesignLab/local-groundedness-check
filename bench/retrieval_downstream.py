"""Does better passage selection make the judge more accurate?"""
import json, threading, time, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ".")
import guardian as G
from arms import lettuce_arm

BUDGET = 2000
def picks(sel):
    return {json.loads(l)["id"]: json.loads(l)["picked"]
            for l in open(f"retr_{sel}_{BUDGET}.jsonl")}
POOLS = {json.loads(l)["id"]: json.loads(l) for l in open("pools.jsonl")}

def run(sel, arm, fn, conc):
    P = picks(sel)
    out = f"results_retr_{arm}_{sel}.jsonl"
    done = set()
    try:
        for l in open(out): done.add(json.loads(l)["id"])
    except FileNotFoundError: pass
    todo = [i for i in P if i not in done and P[i]]
    if not todo:
        print(f"[{arm}/{sel}] complete", flush=True); return
    print(f"[{arm}/{sel}] {len(todo)}", flush=True)
    lock = threading.Lock(); fh = open(out, "a"); t0 = time.time(); n = [0]
    def work(i):
        e = POOLS[i]
        r = fn(e, P[i])
        rec = {"id": i, "source_ds": e["source_ds"], "label": e["label"],
               "pred_ungrounded": r.get("ungrounded"), "latency_s": r.get("latency_s")}
        with lock:
            fh.write(json.dumps(rec) + "\n"); fh.flush(); n[0] += 1
            if n[0] % 100 == 0:
                el = time.time() - t0
                print(f"[{arm}/{sel}] {n[0]}/{len(todo)} {el:.0f}s eta {el/n[0]*(len(todo)-n[0]):.0f}s", flush=True)
    with ThreadPoolExecutor(conc) as ex: list(ex.map(work, todo))
    fh.close(); print(f"[{arm}/{sel}] done {time.time()-t0:.0f}s", flush=True)

guard = lambda e, docs: G.judge_conv(e["question"], e["answer"], docs, think=True)
lett  = lambda e, docs: lettuce_arm({**e, "passage": "\n\n".join(docs)})

for sel in ("keyword", "rerank"):
    run(sel, "lettuce", lett, 8)
for sel in ("keyword", "rerank"):
    run(sel, "guardian", guard, 2)
print("RETRIEVAL DOWNSTREAM COMPLETE", flush=True)
