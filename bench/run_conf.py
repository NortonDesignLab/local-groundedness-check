import json, threading, time
from concurrent.futures import ThreadPoolExecutor
import guardian as G

SAMPLE=[json.loads(l) for l in open("sample.jsonl")]
out="results_guardian_conf.jsonl"
done=set()
try:
    for l in open(out): done.add(json.loads(l)["id"])
except FileNotFoundError: pass
todo=[e for e in SAMPLE if e["id"] not in done]
print(f"{len(todo)} to run",flush=True)
lock=threading.Lock(); fh=open(out,"a"); t0=time.time(); n=[0]
def work(ex):
    r=G.judge_conv(ex["question"],ex["answer"],[ex["passage"]],want_confidence=True)
    rec={"id":ex["id"],"source_ds":ex["source_ds"],"label":ex["label"],
         "pred_ungrounded":r["ungrounded"],"p_ungrounded":r["p_ungrounded"],
         "latency_s":r["latency_s"]}
    with lock:
        fh.write(json.dumps(rec)+"\n"); fh.flush(); n[0]+=1
        if n[0]%50==0: print(f"{n[0]}/{len(todo)}",flush=True)
with ThreadPoolExecutor(2) as ex: list(ex.map(work,todo))
fh.close(); print(f"done in {time.time()-t0:.0f}s")
