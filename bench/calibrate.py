"""#5 - turn Guardian's logprob score into a usable probability.

Raw P(ungrounded) ranks well (AUC 0.815) but is wildly overconfident: mean 0.950 against
0.718 accuracy, and the 0.9-0.99 band is worse than a coin flip. Isotonic regression fitted
on labelled data fixes the mapping; 5-fold CV keeps the reported numbers honest.
"""
import json
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

d = pd.DataFrame([json.loads(l) for l in open("results_guardian_conf.jsonl")]).drop_duplicates("id", keep="last")
d = d.dropna(subset=["p_ungrounded", "pred_ungrounded"])
p = d.p_ungrounded.values.astype(float)
y = d.label.eq("FAIL").values.astype(int)

def ece(prob, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (prob >= edges[i]) & (prob < edges[i + 1] if i < bins - 1 else prob <= 1.0)
        if m.sum(): e += m.mean() * abs(prob[m].mean() - y[m].mean())
    return e

def bal_acc(pred, y):
    tp=((y==1)&pred).sum(); tn=((y==0)&~pred).sum(); fp=((y==0)&pred).sum(); fn=((y==1)&~pred).sum()
    return ((tp/(tp+fn) if tp+fn else 0)+(tn/(tn+fp) if tn+fp else 0))/2

# out-of-fold calibrated scores
logit = lambda x: np.log(np.clip(x,1e-9,1-1e-9)/(1-np.clip(x,1e-9,1-1e-9)))
oof_iso = np.zeros_like(p); oof_plt = np.zeros_like(p)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=20260924).split(p.reshape(-1,1), y):
    iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr]); oof_iso[te] = iso.predict(p[te])
    lr = LogisticRegression().fit(logit(p[tr]).reshape(-1,1), y[tr])
    oof_plt[te] = lr.predict_proba(logit(p[te]).reshape(-1,1))[:,1]

print(f"n={len(p)}  base rate {y.mean():.3f}")
print(f"{'':22s} {'ECE':>7} {'bal_acc@0.5':>12} {'best thr':>9} {'bal_acc@best':>13}")
for name, s in (("raw logprob", p), ("Platt (5-fold OOF)", oof_plt), ("isotonic (5-fold OOF)", oof_iso)):
    thr = max(np.unique(np.round(s,4)), key=lambda t: bal_acc(s>=t, y))
    print(f"{name:22s} {ece(s,y):7.3f} {bal_acc(s>=0.5,y):12.3f} {thr:9.3f} {bal_acc(s>=thr,y):13.3f}")

print("\nreliability after isotonic (out-of-fold)")
b=pd.DataFrame({"p":oof_iso,"y":y})
g=b.groupby(pd.cut(b.p,[0,.2,.4,.6,.8,1.001],include_lowest=True),observed=True).agg(n=("y","size"),actual=("y","mean"),predicted=("p","mean"))
print(g.round(3).to_string())

# ship the full-data isotonic fit as a lookup table
iso=IsotonicRegression(out_of_bounds="clip").fit(p,y)
xs=np.unique(np.concatenate([np.linspace(0,1,41), p]))
tbl=[[round(float(x),6), round(float(iso.predict([x])[0]),4)] for x in xs]
json.dump({"note":"isotonic map from Guardian raw P(ungrounded) to calibrated probability; fitted on 344 HaluBench items 2026-09-24",
           "points":tbl}, open("calibration_map.json","w"), indent=0)
print(f"\nwrote calibration_map.json ({len(tbl)} points)")
