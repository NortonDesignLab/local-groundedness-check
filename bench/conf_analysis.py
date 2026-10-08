import json, math
import pandas as pd, numpy as np
d=pd.DataFrame([json.loads(l) for l in open("results_guardian_conf.jsonl")]).drop_duplicates("id",keep="last")
d["truth"]=d.label.eq("FAIL"); d=d.dropna(subset=["p_ungrounded","pred_ungrounded"])
p=d.p_ungrounded.values; y=d.truth.values.astype(int)
# AUC (rank-based)
order=np.argsort(p); r=np.empty(len(p)); r[order]=np.arange(1,len(p)+1)
n1,n0=y.sum(),(1-y).sum()
auc=(r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
print(f"n={len(d)}  AUC(P(ungrounded) vs truth) = {auc:.3f}")
# confidence in the MODEL'S OWN answer
conf=np.where(d.pred_ungrounded.values.astype(bool),p,1-p)
correct=(d.pred_ungrounded.values.astype(bool)==y).astype(int)
print(f"mean confidence {conf.mean():.4f} | accuracy {correct.mean():.3f} | ECE-style gap {abs(conf.mean()-correct.mean()):.3f}")
print(f"confidence percentiles: {np.percentile(conf,[1,5,10,25,50,75]).round(5)}")
print("\nACCURACY ABOVE CONFIDENCE THRESHOLD (the 'which band can we automate' view)")
print(f"{'thresh':>10} {'coverage':>9} {'acc>=t':>8} {'n':>5} | {'acc<t':>7} {'n':>5}")
for t in [0.5,0.9,0.99,0.999,0.9999,0.99999,0.999999]:
    m=conf>=t
    if m.sum()==0: continue
    lo=~m
    print(f"{t:>10.6f} {m.mean():>8.1%} {correct[m].mean():>8.3f} {m.sum():>5} | "
          f"{(correct[lo].mean() if lo.sum() else float('nan')):>7.3f} {lo.sum():>5}")
print("\nRELIABILITY (bins of the model's confidence in its own answer)")
bins=[0,.9,.99,.999,.9999,.99999,1.01]
d2=pd.DataFrame({"conf":conf,"correct":correct})
g=d2.groupby(pd.cut(d2.conf,bins),observed=True).agg(n=("correct","size"),acc=("correct","mean"),mean_conf=("conf","mean"))
print(g.round(4).to_string())
# would a confidence gate beat the plain threshold?
print("\nTREATING P(ungrounded) AS A TUNABLE SCORE (vs the model's own yes/no at 0.5)")
best=None
for t in np.unique(np.concatenate([np.linspace(0.001,0.999,60),[0.5,0.9,0.99,0.999]])):
    pr=p>=t
    tp=((y==1)&pr).sum(); tn=((y==0)&~pr).sum(); fp=((y==0)&pr).sum(); fn=((y==1)&~pr).sum()
    rec=tp/(tp+fn) if tp+fn else 0; spec=tn/(tn+fp) if tn+fp else 0
    ba=(rec+spec)/2
    if best is None or ba>best[1]: best=(t,ba,rec,spec,fp,fn)
print(f"  default (score token, == P>=0.5): bal_acc {((y[p>=.5].sum()/n1)+((1-y[p<.5]).sum()/n0))/2:.3f}")
print(f"  best threshold {best[0]:.3f}: bal_acc {best[1]:.3f} (recall {best[2]:.3f}, spec {best[3]:.3f}, fp {best[4]}, fn {best[5]})")
