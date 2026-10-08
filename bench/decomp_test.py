"""#3 - does judging atoms beat judging whole sentences?"""
import json, re, threading, time, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,".")
import guardian as G

SUB=json.load(open("decomp_subset.json"))
def run(mode):
    out=f"results_decomp_{mode}.jsonl"; done=set()
    try:
        for l in open(out): done.add(json.loads(l)["key"])
    except FileNotFoundError: pass
    jobs=[]
    for e in SUB:
        units = e["sents"] if mode=="sentence" else e["atoms"]
        for i,u in enumerate(units):
            k=f'{e["id"]}::{i}'
            if k not in done: jobs.append((e,k,u))
    if not jobs: print(f"[{mode}] complete",flush=True); return
    print(f"[{mode}] {len(jobs)} units",flush=True)
    lock=threading.Lock(); fh=open(out,"a"); t0=time.time(); n=[0]
    def work(job):
        e,k,u=job
        r=G.judge_conv(e["question"],u,[e["passage"]],think=True)
        with lock:
            fh.write(json.dumps({"key":k,"id":e["id"],"source_ds":e["source_ds"],
                                 "label":e["label"],"unit":u[:200],
                                 "ungrounded":r["ungrounded"]})+"\n"); fh.flush(); n[0]+=1
            if n[0]%50==0: print(f"[{mode}] {n[0]}/{len(jobs)} {time.time()-t0:.0f}s",flush=True)
    with ThreadPoolExecutor(2) as ex: list(ex.map(work,jobs))
    fh.close(); print(f"[{mode}] done {time.time()-t0:.0f}s",flush=True)

run("sentence"); run("atom")
print("DECOMP COMPLETE",flush=True)
