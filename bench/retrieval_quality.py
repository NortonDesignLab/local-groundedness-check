"""Does the selected context actually contain the passage the claim came from?"""
import json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import retrieval as R

BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
POOLS = [json.loads(l) for l in open("pools.jsonl")]

def query_for(e):                      # what the judge is asked about
    return f"{e['question']}\n{e['answer']}"

def run(name):
    sel = R.SELECTORS[name]
    out = f"retr_{name}_{BUDGET}.jsonl"
    done = set()
    try:
        for l in open(out): done.add(json.loads(l)["id"])
    except FileNotFoundError: pass
    todo = [e for e in POOLS if e["id"] not in done]
    if not todo:
        print(f"[{name}] done already"); return
    lock = threading.Lock(); fh = open(out, "a"); t0 = time.time(); n = [0]
    def work(e):
        try:
            picked = sel(query_for(e), e["pool"], BUDGET)
        except Exception as ex:
            picked = None
        rec = {"id": e["id"], "source_ds": e["source_ds"], "label": e["label"],
               "hit": None if picked is None else any(g in picked for g in e["gold"]),
               "n_picked": None if picked is None else len(picked),
               "picked": picked}
        with lock:
            fh.write(json.dumps(rec) + "\n"); fh.flush(); n[0] += 1
            if n[0] % 100 == 0: print(f"[{name}] {n[0]}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    with ThreadPoolExecutor(6) as ex: list(ex.map(work, todo))
    fh.close(); print(f"[{name}] {len(todo)} in {time.time()-t0:.0f}s", flush=True)

for s in ("keyword", "rerank", "embed"):
    run(s)

import pandas as pd
rows = []
for s in ("keyword", "rerank", "embed"):
    d = pd.DataFrame([json.loads(l) for l in open(f"retr_{s}_{BUDGET}.jsonl")]).drop_duplicates("id")
    ok = d[d.hit.notna()]
    rows.append({"selector": s, "n": len(ok), "errors": int(d.hit.isna().sum()),
                 "gold in context": f"{ok.hit.mean():.1%}",
                 **{src: f"{g.hit.mean():.0%}" for src, g in ok.groupby("source_ds")}})
print(f"\nGROUNDING RECALL at a {BUDGET}-char context budget, pool of 20 chunks")
print(pd.DataFrame(rows).set_index("selector").to_string())
