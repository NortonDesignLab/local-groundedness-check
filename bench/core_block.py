# ===== NDL GROUNDCHECK CORE (generated - edit core_block.py and run sync_core.py) =====
# Shared detection stack for the Open WebUI filter and the deck builders.
# Every number quoted below is measured on 344 balanced HaluBench items, 2026-09-24,
# and reproduces from bench/ in this repo.

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

def _g_block(think):
    return (f"{_G_JUDGE_THINK if think else _G_JUDGE_NOTHINK}\n\n"
            f"### Criteria: {_G_CRITERIA}\n\n### Scoring Schema: {_G_SCHEMA}")

# GBNF makes the yes/no constraint hard rather than merely trained. Without it 1 call in 344
# returned no parseable <score> and was silently counted as unchecked.
_G_GRAMMAR_THINK = '''root ::= "<think>" [^<]* "</think>" ws "<score>" ws ("yes" | "no") ws "</score>"
ws ::= [ \\t\\n]*'''
_G_GRAMMAR_NOTHINK = '''root ::= "<think>\\n</think>\\n<score> " ("yes" | "no") " </score>"'''

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

# Qwen3-Reranker contract (copied from the production reranker client).
# /v1/score does NOT apply this template; the client must.
_RR_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_RR_PREFIX = ('<|im_start|>system\n'
              'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
              'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
_RR_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Isotonic map from Guardian's raw logprob score to a calibrated probability.
# Fitted on the 344 labelled items; 5-fold out-of-fold ECE 0.232 -> 0.053.
_CALIB = __CALIB_POINTS__

def _calibrate(p):
    """Linear interpolation over the isotonic fit. Returns None if p is None."""
    if p is None:
        return None
    pts = _CALIB
    if p <= pts[0][0]:
        return pts[0][1]
    if p >= pts[-1][0]:
        return pts[-1][1]
    lo, hi = 0, len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pts[mid][0] <= p: lo = mid
        else: hi = mid
    x0, y0 = pts[lo]; x1, y1 = pts[hi]
    return y0 if x1 == x0 else y0 + (y1 - y0) * (p - x0) / (x1 - x0)

# ---------------------------------------------------------------- numbers
_NUM_WORDS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
              "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
              "seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,
              "sixty":60,"seventy":70,"eighty":80,"ninety":90,"hundred":100,"thousand":1000,
              "million":1000000,"billion":1000000000}
_NUM_SCALE = {"k":1e3,"m":1e6,"bn":1e9,"b":1e9,"thousand":1e3,"million":1e6,"billion":1e9,"trillion":1e12}
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|percent|k\b|m\b|bn\b|b\b|thousand|million|billion|trillion)?", re.I)
_NUM_TRIVIAL = {0,1,2,3,4,5,6,7,8,9,10,11,12,100}

def _numbers(text):
    out = set()
    for m in _NUM_RE.finditer(text or ""):
        raw, suf = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        try: v = float(raw)
        except ValueError: continue
        if suf in ("%", "percent"): out.add(("pct", round(v, 6)))
        elif suf in _NUM_SCALE:
            out.add(("num", round(v * _NUM_SCALE[suf], 6))); out.add(("num", round(v, 6)))
        else: out.add(("num", round(v, 6)))
    low = (text or "").lower()
    for w, v in _NUM_WORDS.items():
        if re.search(r"\b" + w + r"\b", low): out.add(("num", float(v)))
    return out

def _orphan_numbers(claim, passages, max_density=5.0):
    """Numbers in the claim that appear in no passage.

    Only fires when the sources are prose rather than tables: measured precision is 1.00 at a
    density of 5 numbers per 1000 characters or below, but falls to 0.55 on FinanceBench and
    0.33 on DROP without the gate, because those answers legitimately contain arithmetic
    derived from the source. Gated, it adds 3 catches for 0 extra false positives.
    """
    joined = "\n".join(passages)
    if not joined: return []
    density = len(_numbers(joined)) / max(len(joined) / 1000.0, 1e-9)
    if density > max_density: return []
    src = _numbers(joined)
    src_nums = {v for k, v in src}
    bad = []
    for kind, val in _numbers(claim):
        if (kind, val) in src or val in src_nums: continue
        if kind == "num" and (val in _NUM_TRIVIAL or 1900 <= val <= 2100): continue
        bad.append(val)
    return bad

# ---------------------------------------------------------------- atoms
_ATOM_SPLIT = re.compile(r"(?:,\s+(?:and|but|while|whereas|although)\s+|\s+;\s*|;\s+|\s+—\s+)", re.I)
_ATOM_VERB = re.compile(r"\b(is|are|was|were|has|have|had|will|would|can|could|did|does|do|grew|rose|fell|"
                        r"reached|reported|found|showed|increased|decreased|includes?|contains?|states?|"
                        r"equals?|remains?|became|\w+ed|\w+s)\b", re.I)

def _atoms(sentence, min_words=4, max_atoms=4):
    s = (sentence or "").strip()
    if not s: return []
    parts = [p.strip(" ,;—") for p in _ATOM_SPLIT.split(s) if p and p.strip(" ,;—")]
    if len(parts) < 2: return [s]
    good = [p for p in parts if len(p.split()) >= min_words and _ATOM_VERB.search(p)]
    return good if 2 <= len(good) <= max_atoms else [s]

# ---------------------------------------------------------------- confidence
def _score_confidence(logprobs):
    """P(ungrounded) renormalised over the yes/no alternatives of the score token."""
    if not logprobs or not logprobs.get("content"): return None
    norm = lambda t: t.strip().strip("<>/").lower()
    text, placed = "", []
    for tok in logprobs["content"]:
        placed.append((len(text), tok)); text += tok.get("token", "")
    cut = text.rfind("<score>")                    # think mode says "no" inside its reasoning too
    stream = [t for off, t in placed if off >= cut] if cut >= 0 else [t for _, t in placed]
    for tok in stream:
        if norm(tok.get("token", "")) not in ("yes", "no"): continue
        lp = {}
        for alt in tok.get("top_logprobs") or []:
            k = norm(alt.get("token", ""))
            if k in ("yes", "no") and k not in lp: lp[k] = alt["logprob"]
        lp.setdefault(norm(tok["token"]), tok.get("logprob", 0.0))
        if not lp: return None
        seen = min(lp.values())
        y, n = lp.get("yes", seen - 20.0), lp.get("no", seen - 20.0)
        mx = max(y, n); ey, en = math.exp(y - mx), math.exp(n - mx)
        return ey / (ey + en)
    return None
# ===== END NDL GROUNDCHECK CORE =====
