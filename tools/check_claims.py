#!/usr/bin/env python3
"""Check a list of factual claims against verbatim source passages with the local stack.

This is the export-time version of the same pipeline that runs inside the Open WebUI
filter (openwebui/guardian_source_check.py):

  stage 1  LettuceDetect v2 screens every claim.  Recall 0.97, specificity 0.42 (n=1094):
           it proposes, it never decides.
  stage 2  Granite Guardian 4.1-8B in think mode adjudicates what the screen flagged.
  stage 3  optional: a general local model as a second adjudicator; a claim is flagged
           when EITHER heavyweight objects (largest proven gain in the stack, chi2 15.67).
  plus     a deterministic orphan-number check that only fires on prose-density sources
           (precision 1.00 across 1438 benchmark items).

Input JSON
  {
    "question": "what the claims answer - optional, but measurably helps the judge",
    "passages": ["verbatim source text", "..."],
    "claims":   ["one fact per sentence", "..."],
    "controls": ["a planted false claim that MUST be flagged", "..."]      # optional
  }

Usage
  python tools/check_claims.py examples/claims.json
  python tools/check_claims.py claims.json --combine or          # untrusted sources
  python tools/check_claims.py claims.json --qwen --json out.json

Environment
  GUARDIAN_URL  llama.cpp server hosting Guardian   (default http://10.0.0.121:8014)
  LETTUCE_URL   vLLM server hosting LettuceDetect   (default http://10.0.0.72:8077)
  QWEN_URL / QWEN_MODEL   any OpenAI-compatible general model (default http://10.0.0.72:8000, "local")

Exit status: 0 all clean, 1 something flagged, 2 a control was NOT caught (do not trust the run),
3 neither detector was reachable.
"""
import argparse, json, math, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

import requests

GUARDIAN_URL = os.environ.get("GUARDIAN_URL", "http://10.0.0.121:8014").rstrip("/")
LETTUCE_URL = os.environ.get("LETTUCE_URL", "http://10.0.0.72:8077").rstrip("/")
LETTUCE_MODEL = os.environ.get("LETTUCE_MODEL", "KRLabsOrg/lettucedect-v2-qwen-2b")
QWEN_URL = os.environ.get("QWEN_URL", "http://10.0.0.72:8000").rstrip("/")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "local")
GUARDIAN_MODEL = "granite-guardian-4.1-8b"

# ---------------------------------------------------------------- prompts (official model-card blocks)
_G_JUDGE_NOTHINK = (
    "<no-think>As a judge agent, assess whether the provided text meets the given judging criteria using all available "
    "information, including conversations, documents, and tools. Provide your score immediately without explanation. "
    "Output empty <think>\n</think> tags followed by your score in <score></score> tags."
)
_G_JUDGE_THINK = (
    "<think>As a judge agent, carefully analyze whether the provided text meets the given judging criteria. Use all "
    "available information, including conversations, documents, and tools. Reason through the evidence step by step "
    "before providing your score. Wrap your reasoning in <think></think> tags, then provide your score in "
    "<score></score> tags."
)
_G_CRITERIA = (
    "A text is considered ungrounded or unfaithful if it includes information lacking support from, or directly "
    "contradicting, the provided document(s)."
)
_G_SCHEMA = "If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'."


def g_block(think):
    return f"{_G_JUDGE_THINK if think else _G_JUDGE_NOTHINK}\n\n### Criteria: {_G_CRITERIA}\n\n### Scoring Schema: {_G_SCHEMA}"


_L_SYSTEM = (
    "You are an expert annotator who identifies hallucinated spans in a generated answer with respect to a given "
    "context (the only trusted evidence). A hallucinated span is a substring of the answer that is not supported by "
    "the context. Spans consistent with the context are not hallucinations.\n\n"
    "Quote each hallucinated span verbatim from the answer and classify it into exactly one category and one "
    "subcategory.\n\n"
    "Categories (the kinds of unsupported span):\n"
    "- contradiction: conflicts with the context (a wrong value, number, date, name, or relationship)\n"
    "- fabricated_reference: an entity, name, identifier, or section that is absent from the context\n"
    "- unsupported_addition: a claim, detail, or behavior the context never states\n\n"
    "Subcategories:\n"
    "- entity: a wrong or invented name, entity, or object\n"
    "- temporal: an incorrect date, time, duration, or ordering\n"
    "- numerical: an incorrect number, quantity, or amount\n"
    "- value: a wrong value, setting, or attribute value\n"
    "- relational: an incorrect relationship or association between things\n"
    "- identifier: an invented identifier or name not found in the context\n"
    "- section: a reference to a section, part, or location that does not exist\n"
    "- attribute: an invented or incorrect attribute or property\n"
    "- claim: an added factual claim the context does not support\n"
    "- behavior: an added or changed action or behavior the context never states\n"
    "- elaboration: extra detail or elaboration beyond what the context supports\n"
    "- subjective: an unsupported subjective or evaluative statement\n"
    "- unspecified: unsupported, with no more specific subtype\n\n"
    'Reply with ONLY a JSON object (no markdown, no code fences): {"hallucinated_spans": [{"text": "...", '
    '"category": "...", "subcategory": "..."}]}. If nothing is unsupported, reply {"hallucinated_spans": []}.'
)

_Q_SYSTEM = (
    "You are a strict groundedness judge. You are given a CONTEXT (the only trusted evidence), the QUESTION that was "
    "asked, and an ANSWER. Decide whether every factual element of the ANSWER is supported by the CONTEXT.\n"
    "An answer is UNGROUNDED if it states anything absent from, or contradicting, the CONTEXT - including wrong "
    "numbers, invented entities, or unsupported extrapolation. Correct paraphrase and correct arithmetic over stated "
    "figures are GROUNDED.\n"
    "Reply with exactly one word: GROUNDED or UNGROUNDED."
)

# ---------------------------------------------------------------- passage selection
def pick_docs(claim, passages, context_chars=12000):
    """Best-matching passages by keyword overlap (a reranker measured no better for the judge)."""
    words = lambda x: set(re.findall(r"[a-z]{4,}|\d[\d.,%]*", x.lower()))
    cw = words(claim)
    order = sorted(range(len(passages)), key=lambda i: -len(cw & words(passages[i])))
    docs, used = [], 0
    for i in order:
        p = passages[i]
        if used + len(p) > context_chars and docs:
            break
        docs.append({"doc_id": str(len(docs)), "text": p})
        used += len(p)
    assert docs, "empty documents = silent false pass"
    return docs


# ---------------------------------------------------------------- detectors
def screen(claim, docs, question):
    """LettuceDetect: True = worth adjudicating, False = clean, None = unreachable."""
    ctx = "\n\n".join(d["text"] for d in docs)
    user = f"User request: {question[:1000]}\n\n{ctx}\n\nAnswer to verify:\n{claim}"
    try:
        r = requests.post(f"{LETTUCE_URL}/v1/chat/completions", timeout=90,
                          json={"model": LETTUCE_MODEL, "temperature": 0, "max_tokens": 512,
                                "messages": [{"role": "system", "content": _L_SYSTEM},
                                             {"role": "user", "content": user}]})
        raw = (r.json()["choices"][0]["message"].get("content") or "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        spans = json.loads(m.group(0) if m else raw).get("hallucinated_spans", [])
        return len(spans) > 0
    except Exception:
        return None


def judge(claim, docs, question, think=True):
    """Granite Guardian: True = ungrounded, False = grounded, None = unreachable/unparseable."""
    q_turn = [{"role": "user", "content": question[:1000]}] if question else []
    body = {"model": GUARDIAN_MODEL, "temperature": 0, "max_tokens": 6144 if think else 32,
            "messages": q_turn + [{"role": "assistant", "content": claim},
                                  {"role": "user", "content": g_block(think)}],
            # llama.cpp silently DROPS a top-level "documents" field; the template only sees this.
            "chat_template_kwargs": {"documents": docs}}
    try:
        r = requests.post(f"{GUARDIAN_URL}/v1/chat/completions", json=body, timeout=600)
        m = r.json()["choices"][0]["message"]
        txt = (m.get("content") or "") + (m.get("reasoning_content") or "")
        s = re.findall(r"<score>\s*(\w+)\s*</score>", txt)
        return (s[-1].lower() == "yes") if s else None
    except Exception:
        return None


def qwen(claim, docs, question):
    """Second adjudicator (any general model). True = ungrounded, None = unavailable."""
    ctx = "\n\n".join(d["text"] for d in docs)
    user = f"CONTEXT:\n{ctx}\n\nQUESTION:\n{question[:1000]}\n\nANSWER:\n{claim}\n\nVerdict:"
    try:
        r = requests.post(f"{QWEN_URL}/v1/chat/completions", timeout=300,
                          json={"model": QWEN_MODEL, "temperature": 0, "max_tokens": 8,
                                "messages": [{"role": "system", "content": _Q_SYSTEM},
                                             {"role": "user", "content": user}],
                                "chat_template_kwargs": {"enable_thinking": False}})
        up = (r.json()["choices"][0]["message"].get("content") or "").strip().upper()
        if "UNGROUNDED" in up:
            return True
        if "GROUNDED" in up:
            return False
        return None
    except Exception:
        return None


# ---------------------------------------------------------------- orphan numbers
_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
              "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
              "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
              "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
              "thousand": 1000, "million": 1000000, "billion": 1000000000}
_NUM_SCALE = {"k": 1e3, "m": 1e6, "bn": 1e9, "b": 1e9, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|percent|k\b|m\b|bn\b|b\b|thousand|million|billion|trillion)?", re.I)
_NUM_TRIVIAL = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 100}


def numbers(text):
    out = set()
    for m in _NUM_RE.finditer(text or ""):
        raw, suf = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        try:
            v = float(raw)
        except ValueError:
            continue
        if suf in ("%", "percent"):
            out.add(("pct", round(v, 6)))
        elif suf in _NUM_SCALE:
            out.add(("num", round(v * _NUM_SCALE[suf], 6)))
            out.add(("num", round(v, 6)))
        else:
            out.add(("num", round(v, 6)))
    low = (text or "").lower()
    for w, v in _NUM_WORDS.items():
        if re.search(r"\b" + w + r"\b", low):
            out.add(("num", float(v)))
    return out


def orphan_numbers(claim, passages, max_density=5.0):
    """Figures in the claim that appear in no passage. Silent when the sources are table-dense."""
    joined = "\n".join(passages)
    density = len(numbers(joined)) / max(len(joined) / 1000.0, 1e-9)
    if density > max_density:
        return []
    src = numbers(joined)
    src_vals = {v for _, v in src}
    bad = []
    for kind, val in numbers(claim):
        if (kind, val) in src or val in src_vals:
            continue
        if kind == "num" and (val in _NUM_TRIVIAL or 1900 <= val <= 2100):
            continue
        bad.append(val)
    return bad


# ---------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="JSON file with passages, claims, optional question and controls")
    ap.add_argument("--combine", choices=["and", "or"], default="and",
                    help="'and' (default) = flag what the screen AND a judge agree on; best accuracy, for sources you "
                         "control. 'or' = flag what either finds; evasion-resistant, for scraped or third-party text.")
    ap.add_argument("--no-think", action="store_true", help="Guardian no-think (6x faster, measurably worse)")
    ap.add_argument("--no-screen", action="store_true", help="skip LettuceDetect, judge everything")
    ap.add_argument("--qwen", action="store_true", help="add the general-model second adjudicator (OR with Guardian)")
    ap.add_argument("--json", help="write per-claim results here")
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    question = (spec.get("question") or "").strip()
    passages = [p.strip() for p in spec["passages"] if p and p.strip()]
    claims = list(spec["claims"])
    controls = list(spec.get("controls") or [])
    items = [(c, False) for c in claims] + [(c, True) for c in controls]
    assert passages, "no passages: the judge answers 'grounded' to anything with no documents"
    think = not a.no_think
    or_mode = a.combine == "or"
    print(f"{len(claims)} claims + {len(controls)} controls against {len(passages)} passages "
          f"({sum(map(len, passages))} chars); combine={a.combine}, think={think}, qwen={a.qwen}", flush=True)

    docs_for = [pick_docs(c, passages) for c, _ in items]
    t0 = time.time()

    screened = None
    if not a.no_screen:
        with ThreadPoolExecutor(8) as ex:
            screened = list(ex.map(lambda i: screen(items[i][0], docs_for[i], question), range(len(items))))
        if all(v is None for v in screened):
            print("!! LettuceDetect unreachable - judging everything with Guardian", flush=True)
            screened = None
    t1 = time.time()

    if screened is None or or_mode:
        to_judge = list(range(len(items)))
    else:
        to_judge = [i for i, v in enumerate(screened) if v is not False]
    verdict = {}
    with ThreadPoolExecutor(2) as ex:                       # the llama.cpp server runs 2 slots
        for i, v in zip(to_judge, ex.map(lambda i: judge(items[i][0], docs_for[i], question, think), to_judge)):
            verdict[i] = v
    if to_judge and all(verdict[i] is None for i in to_judge):
        print("!! Granite Guardian unreachable - nothing was adjudicated", flush=True)
    t2 = time.time()

    qv = {}
    if a.qwen:
        ask = [i for i in to_judge if verdict.get(i) is not True]
        with ThreadPoolExecutor(4) as ex:
            for i, v in zip(ask, ex.map(lambda i: qwen(items[i][0], docs_for[i], question), ask)):
                qv[i] = v
    t3 = time.time()

    rows, flagged, missed_controls, checked = [], 0, 0, 0
    for i, (claim, is_control) in enumerate(items):
        sv = screened[i] if screened is not None else None
        gv = verdict.get(i)
        q = qv.get(i)
        nums = orphan_numbers(claim, passages)
        judged = gv if gv is not None else (q if q is not None else None)
        if gv is True or q is True:
            judged = True
        if or_mode:
            flag = bool(sv) or bool(judged) or bool(nums)
            by = ",".join(n for n, f in (("screen", sv), ("judge", judged), ("numbers", bool(nums))) if f) or "-"
        else:
            flag = bool(judged) or bool(nums)
            by = ",".join(n for n, f in (("judge", judged), ("numbers", bool(nums))) if f) or "-"
            if judged is None and sv:
                by = "UNADJUDICATED (judge unreachable, screen flagged)"
        was_checked = (sv is not None) or (judged is not None)
        checked += was_checked
        flagged += flag and not is_control
        if is_control and not flag:
            missed_controls += 1
        rows.append({"claim": claim, "control": is_control, "screen": sv, "guardian": gv, "qwen": q,
                     "orphan_numbers": nums, "flagged": flag, "by": by})

    def sym(v):
        return "-" if v is None else ("FLAG" if v else "ok")
    print()
    print(f"{'#':>3}  {'screen':6} {'guardian':8} {'qwen':5}  {'result':8} claim")
    for n, r in enumerate(rows, 1):
        tag = " [CONTROL]" if r["control"] else ""
        res = "FLAGGED" if r["flagged"] else "clean"
        print(f"{n:>3}  {sym(r['screen']):6} {sym(r['guardian']):8} {sym(r['qwen']):5}  {res:8} {r['claim'][:90]}{tag}")
        if r["by"] not in ("-", "judge") and r["flagged"] or r["by"].startswith("UNADJ"):
            print(f"{'':36}  via {r['by']}" + (f"; figures not in sources: {r['orphan_numbers'][:3]}" if r["orphan_numbers"] else ""))
    print()
    print(f"checked {checked}/{len(items)} | flagged {flagged} of {len(claims)} claims | "
          f"controls caught {len(controls) - missed_controls}/{len(controls)} | "
          f"screen {t1 - t0:.1f}s, judge {t2 - t1:.1f}s, qwen {t3 - t2:.1f}s")
    if a.json:
        json.dump({"spec": a.spec, "combine": a.combine, "think": think, "qwen": a.qwen, "rows": rows},
                  open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")

    if checked == 0:
        sys.exit(3)
    if missed_controls:
        print("!! a control passed as grounded - do not trust this run (see docs/PROTOCOL.md)")
        sys.exit(2)
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
