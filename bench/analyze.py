import json, glob, math
import pandas as pd

def load(name):
    rows = [json.loads(l) for l in open(f"results_{name}.jsonl")]
    d = pd.DataFrame(rows).drop_duplicates("id", keep="last")
    d["truth_ungrounded"] = d.label.eq("FAIL")
    return d

def metrics(d, name):
    n = len(d)
    ok = d[d.pred_ungrounded.notna()]
    nulls = n - len(ok)
    y, p = ok.truth_ungrounded, ok.pred_ungrounded.astype(bool)
    tp = int((y & p).sum()); tn = int((~y & ~p).sum())
    fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")      # catch rate on hallucinations
    spec = tn / (tn + fp) if tn + fp else float("nan")     # correct on clean answers
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    lat = ok['latency_s'].dropna() if 'latency_s' in ok.columns else pd.Series(dtype=float)
    return {"arm": name, "n": n, "nulls": nulls,
            "bal_acc": round((rec + spec) / 2, 3), "acc": round((tp + tn) / len(ok), 3),
            "recall_FAIL": round(rec, 3), "specificity_PASS": round(spec, 3),
            "precision_FAIL": round(prec, 3), "f1_FAIL": round(f1, 3),
            "flag_rate": round(p.mean(), 3),
            "lat_p50": round(lat.median(), 2) if len(lat) else None,
            "lat_p95": round(lat.quantile(.95), 2) if len(lat) else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}

ARMS = ["guardian_nothink_q_A", "guardian_nothink_q_B", "guardian_nothink_noq",
        "guardian_think_q", "lettucedetect"]
avail = [a for a in ARMS if glob.glob(f"results_{a}.jsonl")]
data = {a: load(a) for a in avail}
comp = {a: d for a, d in data.items() if len(d) == 344}

print("=" * 100)
print("OVERALL  (n=344 HaluBench, balanced PASS/FAIL, 6 sources)")
print("=" * 100)
rows = [metrics(d, a) for a, d in comp.items()]
if rows:
    print(pd.DataFrame(rows).set_index("arm").to_string())

# Brad's protocol: two no-think passes, flag on either
if "guardian_nothink_q_A" in comp and "guardian_nothink_q_B" in comp:
    A = comp["guardian_nothink_q_A"].set_index("id")
    B = comp["guardian_nothink_q_B"].set_index("id")
    j = A[["label", "source_ds", "pred_ungrounded", "truth_ungrounded"]].join(
        B[["pred_ungrounded"]], rsuffix="_B")
    j = j.dropna(subset=["pred_ungrounded", "pred_ungrounded_B"])
    agree = (j.pred_ungrounded == j.pred_ungrounded_B).mean()
    print(f"\nREPEATABILITY (two identical no-think passes, temp 0, 2 llama.cpp slots)")
    print(f"  pass-to-pass agreement : {agree:.3%}   disagreements: {int((~(j.pred_ungrounded==j.pred_ungrounded_B)).sum())}/{len(j)}")
    either = j.assign(pred_ungrounded=j.pred_ungrounded | j.pred_ungrounded_B)
    both = j.assign(pred_ungrounded=j.pred_ungrounded & j.pred_ungrounded_B)
    print(pd.DataFrame([metrics(either, "flag-on-EITHER pass (Brad's protocol)"),
                        metrics(both, "flag-only-if-BOTH passes")]).set_index("arm").to_string())

print("\n" + "=" * 100)
print("PER-SOURCE balanced accuracy")
print("=" * 100)
per = {}
for a, d in comp.items():
    per[a] = {src: metrics(g, a)["bal_acc"] for src, g in d.groupby("source_ds")}
if per:
    print(pd.DataFrame(per).round(3).to_string())

print("\nNOTE: LettuceDetect v2 was TRAINED on RAGTruth; its RAGTruth column is in-domain.")
