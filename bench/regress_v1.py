"""Regression: run the SHIPPED filter code over the 344 items and confirm the stack's numbers."""
import importlib.util, json, threading, time
from concurrent.futures import ThreadPoolExecutor
import pathlib
sp=importlib.util.spec_from_file_location("gsc", str(pathlib.Path(__file__).resolve().parent.parent/"openwebui"/"guardian_source_check.py"))
g=importlib.util.module_from_spec(sp); sp.loader.exec_module(g)
f=g.Filter(); f.valves.guardian_url="http://10.0.0.121:8014"; f.valves.lettuce_url="http://10.0.0.72:8077"
f.valves.rerank_url="http://localhost:8013"

S=[json.loads(l) for l in open("sample.jsonl")]
out="results_v1_stack.jsonl"; done=set()
try:
    for l in open(out): done.add(json.loads(l)["id"])
except FileNotFoundError: pass
todo=[e for e in S if e["id"] not in done]
print(f"{len(todo)} items", flush=True)
lock=threading.Lock(); fh=open(out,"a"); t0=time.time(); n=[0]
def work(e):
    sc=f._screen(e["answer"],[e["passage"]],e["question"])
    gv,gp=(None,None)
    if sc is not False:
        gv,gp=f._judge(e["answer"],[e["passage"]],e["question"])
    nums=g._orphan_numbers(e["answer"],[e["passage"]],f.valves.number_density)
    with lock:
        fh.write(json.dumps({"id":e["id"],"source_ds":e["source_ds"],"label":e["label"],
                             "screen":sc,"judge":gv,"p":gp,"orphans":len(nums)})+"\n"); fh.flush(); n[0]+=1
        if n[0]%50==0: print(f"{n[0]}/{len(todo)} {time.time()-t0:.0f}s",flush=True)
with ThreadPoolExecutor(2) as ex: list(ex.map(work,todo))
fh.close(); print(f"done {time.time()-t0:.0f}s",flush=True)
