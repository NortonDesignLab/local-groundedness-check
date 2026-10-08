import pandas as pd, json
SEED, PER_LABEL, MAXCH = 20260924, 30, 6000
df = pd.read_parquet("halubench.parquet")
df["passage"] = df.passage.str.strip(); df["answer"] = df.answer.str.strip()
df["question"] = df.question.str.strip()
n0 = len(df)
df = df[(df.passage.str.len() <= MAXCH) & (df.passage.str.len() > 50)
        & (df.answer.str.len() > 0) & (df.answer.str.len() <= 4000)]
print(f"length filter: {n0} -> {len(df)} (passage<= {MAXCH} chars)")
parts = []
for src, g in df.groupby("source_ds"):
    for lab in ("PASS", "FAIL"):
        sub = g[g.label == lab]
        take = min(PER_LABEL, len(sub))
        if take < PER_LABEL: print(f"  !! {src}/{lab}: only {take}")
        parts.append(sub.sample(take, random_state=SEED))
s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(s.groupby(["source_ds", "label"]).size().to_string())
print("total", len(s))
print("passage chars:", s.passage.str.len().describe([.5, .9]).round(0).to_dict())
print("answer chars: ", s.answer.str.len().describe([.5, .9]).round(0).to_dict())
s.to_json("sample.jsonl", orient="records", lines=True)
print("wrote sample.jsonl")
