"""Candidate pool per item: own chunks + distractors from the same source. Gold = own chunks."""
import json, random
from retrieval import chunks

SAMPLE=[json.loads(l) for l in open("sample.jsonl")]
POOL_N=20
by_src={}
for e in SAMPLE: by_src.setdefault(e["source_ds"],[]).append(e)
rng=random.Random(20260924)
out=[]
for e in SAMPLE:
    gold=chunks(e["passage"])
    others=[o for o in by_src[e["source_ds"]] if o["id"]!=e["id"]]
    rng.shuffle(others)
    distract=[]
    for o in others:
        for c in chunks(o["passage"]):
            distract.append(c)
            if len(distract) >= POOL_N-len(gold): break
        if len(distract) >= POOL_N-len(gold): break
    pool=gold+distract
    rng.shuffle(pool)
    out.append({**e,"pool":pool,"gold":gold})
with open("pools.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
import statistics as st
print("items",len(out),"| pool size median",st.median(len(r["pool"]) for r in out),
      "| gold chunks median",st.median(len(r["gold"]) for r in out))
