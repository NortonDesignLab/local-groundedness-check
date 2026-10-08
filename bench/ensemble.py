import json, itertools, math
import pandas as pd
from analyze import load, metrics

arms = ["guardian_nothink_q_A","guardian_think_q","lettucedetect"]
d = {a: load(a).set_index("id") for a in arms}
base = d["guardian_think_q"][["label","source_ds","truth_ungrounded"]].copy()
for a in arms: base[a] = d[a]["pred_ungrounded"]
base = base.dropna()
base["latency_s"] = 0.0

rows=[]
for a in arms:
    rows.append(metrics(base.assign(pred_ungrounded=base[a]), a))
rows.append(metrics(base.assign(pred_ungrounded=base.lettucedetect & base.guardian_think_q),
                    "AND: lettuce prefilter -> guardian(think) adjudicates"))
rows.append(metrics(base.assign(pred_ungrounded=base.lettucedetect | base.guardian_think_q),
                    "OR: flag if either"))
rows.append(metrics(base.assign(pred_ungrounded=base.guardian_think_q & base.guardian_nothink_q_A),
                    "AND: guardian think & no-think"))
print(pd.DataFrame(rows).set_index("arm")[
    ["bal_acc","recall_FAIL","specificity_PASS","precision_FAIL","f1_FAIL","flag_rate","tp","fp","tn","fn"]].to_string())

# McNemar: think vs no-think (same items)
b = int(((base.guardian_think_q == base.truth_ungrounded) & (base.guardian_nothink_q_A != base.truth_ungrounded)).sum())
c = int(((base.guardian_think_q != base.truth_ungrounded) & (base.guardian_nothink_q_A == base.truth_ungrounded)).sum())
chi = (abs(b-c)-1)**2/(b+c) if b+c else 0
print(f"\nMcNemar think vs no-think: think-only-correct={b}, nothink-only-correct={c}, chi2_cc={chi:.2f}")
print("  (chi2 > 3.84 => p<0.05)")

# Cost / throughput
print("\nCOST PER 1000 CLAIMS (local, GPU-seconds only)")
for a,lat in [("guardian_nothink",0.58),("guardian_think",3.72),("lettucedetect",1.03)]:
    print(f"  {a:18s} {lat*1000/60:6.1f} GPU-min serial, {lat*1000/60/2:5.1f} min at 2 slots")
print("  jev (hosted)       $0.042/MTok in; ~700 tok/claim => ~$0.03 per 1000 claims (vendor list price, UNVERIFIED)")
