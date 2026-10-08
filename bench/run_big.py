"""Power-up run: settle the components the 344-item sample left unproven."""
import json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,".")
import guardian as G
from arms import lettuce_arm
from arm_qwen import qwen_arm

SAMPLE=[json.loads(l) for l in open("sample_big.jsonl")]
GRAM_T=G.__dict__.get("_x")  # placeholder
GRAMMAR_NOTHINK = r'''root ::= "<think>\n</think>\n<score> " ("yes" | "no") " </score>"'''

def guardian(ex, think, grammar):
    body_extra = {"grammar": GRAMMAR_NOTHINK} if (grammar and not think) else {}
    return G.judge_conv(ex["question"], ex["answer"], [ex["passage"]], think=think, _extra=body_extra)

ARMS=[("lettuce",   lambda e: lettuce_arm(e), 8),
      ("nothink",   lambda e: guardian(e, False, False), 2),
      ("nothink_gram", lambda e: guardian(e, False, True), 2),
      ("think",     lambda e: guardian(e, True, False), 2),
      ("qwen",      lambda e: qwen_arm(e), 8)]

def run(name, fn, conc):
    out=f"big_{name}.jsonl"; done=set()
    try:
        for l in open(out): done.add(json.loads(l)["id"])
    except FileNotFoundError: pass
    todo=[e for e in SAMPLE if e["id"] not in done]
    if not todo: print(f"[{name}] complete",flush=True); return
    print(f"[{name}] {len(todo)}",flush=True)
    lock=threading.Lock(); fh=open(out,"a"); t0=time.time(); n=[0]
    def work(e):
        r=fn(e)
        with lock:
            fh.write(json.dumps({"id":e["id"],"source_ds":e["source_ds"],"label":e["label"],
                                 "pred_ungrounded":r.get("ungrounded"),"latency_s":r.get("latency_s"),
                                 "p":r.get("p_ungrounded")})+"\n"); fh.flush(); n[0]+=1
            if n[0]%150==0:
                el=time.time()-t0; print(f"[{name}] {n[0]}/{len(todo)} {el:.0f}s eta {el/n[0]*(len(todo)-n[0]):.0f}s",flush=True)
    with ThreadPoolExecutor(conc) as ex: list(ex.map(work,todo))
    fh.close(); print(f"[{name}] done {time.time()-t0:.0f}s",flush=True)

for a in ARMS: run(*a)
print("BIG RUN COMPLETE",flush=True)
