"""Drive the REAL guardian_source_check Filter._judge over the benchmark sample,
with and without the question turn."""
import importlib.util, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

spec = importlib.util.spec_from_file_location(
    "gsc", str(__import__("pathlib").Path(__file__).resolve().parent.parent / "openwebui" / "guardian_source_check.py"))
gsc = importlib.util.module_from_spec(spec); spec.loader.exec_module(gsc)

SAMPLE = [json.loads(l) for l in open("sample.jsonl")]

def run(tag, include_question):
    f = gsc.Filter()
    f.valves.guardian_url = "http://10.0.0.153:8014"
    f.valves.include_question = include_question
    out = f"results_filter_{tag}.jsonl"
    done = set()
    try:
        for l in open(out): done.add(json.loads(l)["id"])
    except FileNotFoundError: pass
    todo = [e for e in SAMPLE if e["id"] not in done]
    print(f"[{tag}] include_question={include_question}  {len(todo)} to run", flush=True)
    lock = threading.Lock(); fh = open(out, "a"); t0 = time.time(); n = [0]
    def work(ex):
        t1 = time.time()
        v = f._judge(ex["answer"], [ex["passage"]], ex["question"])
        rec = {"id": ex["id"], "source_ds": ex["source_ds"], "label": ex["label"],
               "pred_ungrounded": v, "latency_s": round(time.time() - t1, 3)}
        with lock:
            fh.write(json.dumps(rec) + "\n"); fh.flush(); n[0] += 1
            if n[0] % 50 == 0: print(f"[{tag}] {n[0]}/{len(todo)}", flush=True)
    with ThreadPoolExecutor(2) as ex: list(ex.map(work, todo))
    fh.close(); print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)

run("withq", True)
run("noq", False)
