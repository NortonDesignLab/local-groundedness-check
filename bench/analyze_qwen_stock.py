"""Recompute the n=1094 final-configuration table with a given Qwen arm file.
Usage: python3 analyze_qwen_stock.py big_qwen.jsonl        (old abliterated run; should reproduce BENCHMARK.md)
       python3 analyze_qwen_stock.py big_qwen_stock.jsonl  (stock Qwen3.8-27B)"""
import json, sys
from math import sqrt
import re, math, json as _j
ns={"re":re,"math":math,"json":_j}; exec(open("core_block.py").read().replace("__CALIB_POINTS__","[]"), ns); _orphan_numbers=ns["_orphan_numbers"]

qwen_file = sys.argv[1] if len(sys.argv) > 1 else "big_qwen_stock.jsonl"
def load(f): return {r["id"]: r for r in map(json.loads, open(f))}
S = {e["id"]: e for e in map(json.loads, open("sample_big.jsonl"))}
L, T, Q = load("big_lettuce.jsonl"), load("big_think.jsonl"), load(qwen_file)
ids = [i for i in S if i in L and i in T and i in Q]
print(f"qwen file={qwen_file}  n={len(ids)}  qwen None verdicts={sum(Q[i]['pred_ungrounded'] is None for i in ids)}")

truth = {i: S[i]["label"] == "FAIL" for i in ids}          # True = fabricated
num = {i: bool(_orphan_numbers(S[i]["answer"], [S[i]["passage"]])) for i in ids}
def v(d, i): return bool(d[i]["pred_ungrounded"])            # None counts as not flagged

configs = {
 "Guardian-think alone":            lambda i: v(T,i),
 "Qwen-27B alone":                  lambda i: v(Q,i),
 "LettuceDetect AND think":         lambda i: v(L,i) and v(T,i),
 "LettuceDetect AND (think OR Qwen)": lambda i: v(L,i) and (v(T,i) or v(Q,i)),
 "+ number check":                  lambda i: (v(L,i) and (v(T,i) or v(Q,i))) or num[i],
}
def metrics(pred):
    tp=sum(pred[i] and truth[i] for i in ids); fn=sum((not pred[i]) and truth[i] for i in ids)
    fp=sum(pred[i] and not truth[i] for i in ids); tn=sum((not pred[i]) and not truth[i] for i in ids)
    rec=tp/(tp+fn); spec=tn/(tn+fp); return (tp+fn, rec, spec, (rec+spec)/2, fp, fn)
preds = {name: {i: fn(i) for i in ids} for name, fn in configs.items()}
print(f"{'Configuration':38s} {'BalAcc':>6s} {'Recall':>6s} {'Spec':>6s} {'FP':>4s} {'FN':>4s}")
for name, p in preds.items():
    _, rec, spec, ba, fp, fn = metrics(p)
    print(f"{name:38s} {ba:6.3f} {rec:6.3f} {spec:6.3f} {fp:4d} {fn:4d}")

def mcnemar(a, b):
    """discordant pairs on correctness: b right & a wrong vs a right & b wrong; chi2 with continuity correction"""
    ca = sum((a[i]==truth[i]) and (b[i]!=truth[i]) for i in ids)
    cb = sum((b[i]==truth[i]) and (a[i]!=truth[i]) for i in ids)
    chi2 = (abs(ca-cb)-1)**2/(ca+cb) if ca+cb else 0.0
    return ca, cb, chi2
ca, cb, chi2 = mcnemar(preds["LettuceDetect AND think"], preds["LettuceDetect AND (think OR Qwen)"])
print(f"\nMcNemar, Qwen adjudicator added to LettuceDetect AND think: without-only-right={ca} with-only-right={cb} chi2={chi2:.2f}")

print("\nPer source, final config (+ number check):")
for src in sorted({S[i]["source_ds"] for i in ids}):
    sub=[i for i in ids if S[i]["source_ds"]==src]; p=preds["+ number check"]
    tp=sum(p[i] and truth[i] for i in sub); fn=sum((not p[i]) and truth[i] for i in sub)
    fp=sum(p[i] and not truth[i] for i in sub); tn=sum((not p[i]) and not truth[i] for i in sub)
    print(f"  {src:14s} n={len(sub):4d} balacc={(tp/(tp+fn)+tn/(tn+fp))/2:.3f}")
