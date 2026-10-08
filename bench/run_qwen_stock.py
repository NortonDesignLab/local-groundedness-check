"""Re-run the Qwen judge arm of the n=1094 audit against stock Qwen/Qwen3.8-27B (ATOM :8000).
Output: big_qwen_stock.jsonl, same schema as big_qwen.jsonl (which was Huihui abliterated FP8)."""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("QWEN_URL", "http://10.0.0.72:8000")
os.environ.setdefault("QWEN_MODEL", "qwen3.8-27b")
sys.path.insert(0, ".")
from arm_qwen import qwen_arm, MODEL, URL
SAMPLE=[json.loads(l) for l in open("sample_big.jsonl")]
out="big_qwen_stock.jsonl"; done=set()
try:
    for l in open(out): done.add(json.loads(l)["id"])
except FileNotFoundError: pass
todo=[e for e in SAMPLE if e["id"] not in done]
print(f"model={MODEL} url={URL} todo={len(todo)}", flush=True)
lock=threading.Lock(); fh=open(out,"a"); t0=time.time(); n=[0]
def work(e):
    r=qwen_arm(e)
    with lock:
        fh.write(json.dumps({"id":e["id"],"source_ds":e["source_ds"],"label":e["label"],
                             "pred_ungrounded":r.get("ungrounded"),"latency_s":r.get("latency_s"),
                             "raw":r.get("raw"),"model":MODEL})+"\n"); fh.flush(); n[0]+=1
        if n[0]%100==0:
            el=time.time()-t0; print(f"{n[0]}/{len(todo)} {el:.0f}s eta {el/n[0]*(len(todo)-n[0]):.0f}s", flush=True)
with ThreadPoolExecutor(8) as ex: list(ex.map(work,todo))
fh.close(); print(f"done {time.time()-t0:.0f}s", flush=True)
