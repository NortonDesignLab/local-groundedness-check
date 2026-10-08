"""#6b - Vectara HHEM-2.1-Open as an encoder screen.

Run inside the vLLM image because the host only has Python 3.14, which has no torch wheels.
An encoder classifier cannot be instruction-injected, which is exactly LettuceDetect's weakness.
"""
import json, os, time
from transformers import AutoModelForSequenceClassification

MODEL = "vectara/hallucination_evaluation_model"
m = AutoModelForSequenceClassification.from_pretrained(MODEL, trust_remote_code=True)
m = m.to("cuda").eval()
print("loaded", flush=True)

sample = [json.loads(l) for l in open("/work/sample.jsonl")]
out = open("/work/results_hhem.jsonl", "w")
t0 = time.time()
B = 16
for i in range(0, len(sample), B):
    batch = sample[i:i + B]
    pairs = [(e["passage"], e["answer"]) for e in batch]
    scores = m.predict(pairs)                      # 1.0 = consistent, 0.0 = hallucinated
    for e, s in zip(batch, [float(x) for x in scores]):
        out.write(json.dumps({"id": e["id"], "source_ds": e["source_ds"], "label": e["label"],
                              "hhem_consistency": s}) + "\n")
    out.flush()
    if (i // B) % 5 == 0:
        print(f"{i+len(batch)}/{len(sample)} {time.time()-t0:.0f}s", flush=True)
out.close()
print(f"done {time.time()-t0:.0f}s", flush=True)
