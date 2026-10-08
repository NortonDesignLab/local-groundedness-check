"""Larger sample so the unproven components can actually be settled.
344 items left every comparison but two underpowered; 1,200 roughly triples the power."""
import pandas as pd
SEED, PER_LABEL, MAXCH = 20260925, 105, 6000
df = pd.read_parquet("halubench.parquet")
for c in ("passage","answer","question"): df[c]=df[c].str.strip()
df = df[(df.passage.str.len()<=MAXCH)&(df.passage.str.len()>50)&(df.answer.str.len()>0)&(df.answer.str.len()<=4000)]
prev = set(pd.read_json("sample.jsonl", lines=True)["id"])
parts=[]
for src,g in df.groupby("source_ds"):
    for lab in ("PASS","FAIL"):
        sub=g[g.label==lab]
        keep=sub[sub.id.isin(prev)]                    # reuse the 344 so old runs still count
        rest=sub[~sub.id.isin(prev)]
        need=max(0, PER_LABEL-len(keep))
        parts.append(pd.concat([keep, rest.sample(min(need,len(rest)), random_state=SEED)]))
s=pd.concat(parts).drop_duplicates("id").sample(frac=1, random_state=SEED).reset_index(drop=True)
print(s.groupby(["source_ds","label"]).size().to_string()); print("total",len(s))
print("carried over from the 344:", s.id.isin(prev).sum())
s.to_json("sample_big.jsonl", orient="records", lines=True)
