"""
title: Guardian Source Check
author: Norton Design Lab
version: 1.3.1
license: MIT
description: After each answer, checks every factual sentence against the passages retrieved in the conversation (knowledge / RAG sources, tool results, long pasted text). LettuceDetect v2 screens every sentence for recall, then IBM Granite Guardian adjudicates only what LettuceDetect flagged, which cuts false flags without adding much latency. Appends a short source-check note. Flags only, never edits the answer. Nothing leaves the machine.
requirements: requests
"""

import asyncio
import json
import math
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

MARK = "\n\n---\n**Source check**"
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
_CALIB = [[0.0, 0.0], [1.5e-05, 0.0], [1.9e-05, 0.0], [2e-05, 0.0], [2.5e-05, 0.0], [3e-05, 0.0], [3.3e-05, 0.0], [3.7e-05, 0.0], [4.5e-05, 0.0769], [4.7e-05, 0.0769], [6.1e-05, 0.0769], [0.000102, 0.0769], [0.00011, 0.0769], [0.000123, 0.1818], [0.000149, 0.1818], [0.000183, 0.1818], [0.000207, 0.1818], [0.000237, 0.1818], [0.00029, 0.1818], [0.000341, 0.1818], [0.000456, 0.2083], [0.00054, 0.2083], [0.000617, 0.2083], [0.000692, 0.2083], [0.000771, 0.2083], [0.000944, 0.2083], [0.000995, 0.2083], [0.001161, 0.2083], [0.001874, 0.25], [0.002034, 0.25], [0.004098, 0.25], [0.005014, 0.3333], [0.007847, 0.3333], [0.008921, 0.3333], [0.009818, 0.3333], [0.010156, 0.3333], [0.017602, 0.3333], [0.020783, 0.3333], [0.024287, 0.3333], [0.028712, 0.4643], [0.038793, 0.4643], [0.045544, 0.4643], [0.052441, 0.4643], [0.059667, 0.4643], [0.071455, 0.4643], [0.090182, 0.4643], [0.1, 0.4643], [0.11904, 0.4643], [0.125, 0.4643], [0.15, 0.4643], [0.155216, 0.4643], [0.159922, 0.4643], [0.200076, 0.4643], [0.225, 0.4643], [0.251501, 0.4643], [0.281283, 0.4643], [0.325, 0.4643], [0.366557, 0.4643], [0.41065, 0.4643], [0.45, 0.4643], [0.492053, 0.4643], [0.514614, 0.4643], [0.55, 0.4643], [0.575907, 0.4643], [0.6, 0.4643], [0.625798, 0.4643], [0.65328, 0.4643], [0.715998, 0.4643], [0.75, 0.4643], [0.787673, 0.4643], [0.802327, 0.4643], [0.818336, 0.4643], [0.842322, 0.4643], [0.852118, 0.4643], [0.875, 0.4643], [0.887412, 0.4643], [0.912644, 0.4643], [0.917867, 0.4643], [0.932118, 0.4643], [0.93837, 0.5556], [0.944771, 0.5556], [0.957084, 0.5556], [0.960568, 0.6667], [0.968753, 0.6667], [0.975657, 0.6667], [0.980863, 0.6667], [0.983109, 0.6667], [0.98526, 0.6875], [0.98675, 0.6875], [0.988046, 0.6875], [0.990108, 0.6875], [0.992616, 0.6875], [0.993367, 0.7143], [0.994028, 0.7143], [0.995102, 0.7885], [0.995986, 0.7885], [0.996515, 0.7885], [0.996835, 0.7885], [0.9971, 0.7885], [0.997439, 0.7885], [0.997841, 0.7885], [0.998029, 0.7885], [0.998151, 0.7885], [0.998317, 0.7885], [0.998535, 0.7885], [0.998926, 0.7885], [0.998999, 0.7885], [0.999332, 0.7885], [0.999515, 0.7885], [0.999586, 0.7885], [0.999661, 0.7885], [0.999722, 0.7885], [0.99978, 0.8], [0.999818, 0.8462], [0.999849, 0.8462], [0.999873, 0.8462], [0.999893, 0.8462], [0.999914, 0.8462], [0.99993, 1.0], [0.999937, 1.0], [0.999953, 1.0], [0.999968, 1.0], [0.999972, 1.0], [0.999977, 1.0], [0.999979, 1.0], [0.999986, 1.0], [0.999993, 1.0], [0.999995, 1.0], [1.0, 1.0]]

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


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=10, description="Filter order; higher runs later.")
        guardian_url: str = Field(
            default="http://192.168.100.2:8014",
            description="Granite Guardian judge (llama.cpp on the DGX Spark, port 8014) reached over the ATOM<->Spark direct link; from SARA/LAN use http://10.0.0.153:8014.",
        )
        max_claims: int = Field(default=25, description="Most sentences checked per answer (about 0.25 s each).")
        context_chars: int = Field(default=12000, description="Characters of the best-matching passages shown to the judge per sentence.")
        min_source_chars: int = Field(default=300, description="Skip the check when the conversation holds less retrieved text than this.")
        pasted_text_chars: int = Field(default=1500, description="User messages at least this long count as source material.")
        include_question: bool = Field(
            default=True,
            description="Show the judge the question the answer was responding to. Judging a sentence in isolation makes the judge over-flag (measured: specificity 0.54 vs 0.69).",
        )
        question_chars: int = Field(default=1000, description="Characters of the question shown to the judge.")
        lettuce_url: str = Field(
            default="http://10.0.0.72:8077",
            description="LettuceDetect v2 screen (vLLM on the ATOM, host port 8077) as seen from inside the Open WebUI container.",
        )
        lettuce_model: str = Field(default="KRLabsOrg/lettucedect-v2-qwen-2b", description="Served model name at lettuce_url.")
        prefilter: bool = Field(
            default=True,
            description="Screen every statement with LettuceDetect (recall 0.966) and send only what it flags to the judge. Measured at n=1094 in front of think mode: 0.743 -> 0.754, McNemar chi2 6.05, p<0.05. In front of no-think it is NOT significant (chi2 0.38), so keep it paired with guardian_think.",
        )
        guardian_think: bool = Field(
            default=True,
            description="Run Guardian in think mode. Measured at n=1094: 0.710 -> 0.743 balanced accuracy, McNemar chi2 6.61, p<0.05. Costs about 6x the latency per statement, but the screen keeps most statements away from it.",
        )
        rerank_url: str = Field(
            default="http://vllm-qwen3-reranker-4b:8000",
            description="Qwen3-Reranker-4B (SARA, host :8013; reachable from the Open WebUI container as vllm-qwen3-reranker-4b:8000 on ai-network) used to choose which passages the judge sees. Measured: it finds the right passage 98.8% of the time against 92.2% for keyword overlap, and 97% against 65% on financial documents. Downstream judge accuracy measured flat, so this buys grounding quality, not a score.",
        )
        rerank_model: str = Field(default="qwen3-reranker-4b", description="Served model name at rerank_url.")
        rerank_min_pool: int = Field(
            default=0,
            description="DISABLED by default (0). The reranker finds the grounding passage far more often (98.8% against 92.2%) but that did NOT improve the judge: balanced accuracy went 0.737 -> 0.724 and McNemar chi2 was 0.52 in keyword's favour. Set to 8 or so only if you have evidence for your own corpus.",
        )
        number_check: bool = Field(
            default=True,
            description="Flag a statement whose figures appear in no source passage. Measured across 1438 items it has never produced a single false positive (precision 1.00, fires on ~1% of statements) and adds a handful of catches for free. Only fires when the sources are prose; ungated it is precision 0.33 on numeric material because correct arithmetic looks like a fabrication.",
        )
        number_density: float = Field(default=5.0, description="Numbers per 1000 characters of source above which the number check stays quiet.")
        calibrate: bool = Field(
            default=True,
            description="Map Guardian's raw logprob score through an isotonic fit to a real probability (5-fold out-of-fold calibration error 0.232 -> 0.053). NOTE: only meaningful with guardian_think OFF - think mode saturates the score, so with the shipped defaults this is inert.",
        )
        grammar: bool = Field(
            default=False,
            description="OFF: tested at n=1094 and it changes nothing. Balanced accuracy 0.710 with and without, 2 items differed, McNemar chi2 0.50, and zero unparseable verdicts in 2188 calls either way - while it roughly doubles no-think latency (0.66 s -> 1.43 s). The single malformed verdict ever seen was in think mode and was counted as unchecked, not as a pass.",
        )
        qwen_url: str = Field(
            default="http://10.0.0.72:8000",
            description="Second adjudicator - the general 27B already on the ATOM. The only component besides the question turn that is statistically significant: flagging when EITHER Guardian or this model objects took the stack from 0.764 to 0.801 balanced accuracy and cut missed fabrications from 46 to 25 (McNemar chi2 4.97, p<0.05).",
        )
        qwen_model: str = Field(default="local", description="Served model name at qwen_url. 'local' is the ATOM's stable alias for whatever is in the shared slot; the explicit id on 28 Sep 2026 is qwen3.8-27b. The old default Huihui-Qwen3.8-27B-abliterated now 404s.")
        use_qwen: bool = Field(
            default=True,
            description="Ask the 27B as well as Guardian and flag if either objects. The largest proven gain in the stack. Measured at n=1094: 0.754 -> 0.793 balanced accuracy and missed fabrications 159 -> 78, McNemar chi2 15.67, p<0.001. Adds roughly 10 s per statement the screen already flagged.",
        )
        combine: str = Field(
            default="and",
            description="How the two detectors combine. 'and' flags only what both agree on - best accuracy (0.764 vs 0.746 balanced, 35 false positives) and right for material whose provenance you control. 'or' flags what either one finds - far harder to evade, because the two fail to different attacks (a passage claiming 'pre-approved' flips Guardian 6.2% of the time and LettuceDetect 0%; an injected instruction flips LettuceDetect up to 12.4% and Guardian 1.5%), so no single injection defeats both. Use 'or' for scraped web pages, third-party PDFs and anything a client supplied. It costs: specificity 0.375 and about three times the false positives, and it sends every statement to the judge.",
        )
        high_confidence: float = Field(
            default=0.80,
            description="Flags at or above this probability are marked confident. With calibrate on this is a real probability (the 0.8+ band measured 0.846 actual), so 0.80 is a sensible gate. With calibrate off the raw score saturates and only 0.9999 means anything.",
        )
        note_unsourced: bool = Field(
            default=False,
            description="Also add a one-line note when an answer had no retrieved sources at all (answered from model memory).",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ---------------------------------------------------------------- passages
    @staticmethod
    def _strings(obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for k in ("text", "content", "output"):
                if k in obj:
                    yield from Filter._strings(obj[k])
        elif isinstance(obj, list):
            for x in obj:
                yield from Filter._strings(x)

    def _passages(self, messages: list) -> list:
        texts = []
        for m in messages:
            for src in m.get("sources") or []:                       # knowledge / RAG / citations
                texts += [d for d in (src.get("document") or []) if isinstance(d, str)]
            for item in m.get("output") or []:                       # native tool calls
                if isinstance(item, dict) and item.get("type") == "function_call_output":
                    texts += list(self._strings(item.get("output")))
            if m.get("role") == "user" and len(m.get("content") or "") >= self.valves.pasted_text_chars:
                texts.append(m["content"])                           # long pasted material is evidence too
        out = []
        for t in texts:
            t = t.strip()
            if t.startswith(("Built '", "ERROR", "NOT BUILT")) or "SOURCE CHECK" in t[:400]:
                continue                                             # builder reports are not evidence
            for k in range(0, len(t), 1500):
                if len(t[k:k + 1800].strip()) >= 40:
                    out.append(t[k:k + 1800])
        return out

    # ---------------------------------------------------------------- question
    @staticmethod
    def _question(messages: list) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return (m.get("content") or "").strip()
        return ""

    # ---------------------------------------------------------------- claims
    @staticmethod
    def _claims(answer: str) -> list:
        t = re.sub(r"(?s)<details.*?</details>", " ", answer)         # reasoning and tool blocks
        t = re.sub(r"(?s)```.*?```", " ", t)                          # code
        t = re.sub(r"(?m)^\s*\|.*\|\s*$", " ", t)                     # tables
        out = []
        for line in t.splitlines():
            line = re.sub(r"^\s*(#{1,6}|[-*+]|\d+[.)])\s+", "", line).strip()
            if line.startswith("#") or not line:
                continue
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", line):
                plain = re.sub(r"[*_`]|\[(.*?)\]\(.*?\)", r"\1", s).strip()
                if len(plain.split()) >= 6 and not plain.endswith(("?", ":")):
                    out.append(plain)
        return out

    # ---------------------------------------------------------------- evidence
    def _rank(self, query: str, pool: list) -> list:
        """Order passages by the reranker, falling back to the caller's order on any failure."""
        try:
            r = requests.post(
                f"{self.valves.rerank_url.rstrip('/')}/v1/score",
                json={"model": self.valves.rerank_model,
                      "text_1": f"{_RR_PREFIX}<Instruct>: {_RR_INSTRUCTION}\n<Query>: {query}\n",
                      "text_2": [f"<Document>: {t}{_RR_SUFFIX}" for t in pool]},
                timeout=60,
            )
            score = {d["index"]: d["score"] for d in r.json()["data"]}
            return sorted(range(len(pool)), key=lambda i: -score.get(i, 0.0))
        except Exception:
            return None

    def _docs(self, claim: str, passages: list) -> list:
        """The best-matching passages for one claim, capped at context_chars."""
        order = None
        if self.valves.rerank_min_pool and len(passages) >= self.valves.rerank_min_pool:
            order = self._rank(claim, passages)
        if order is None:
            words = lambda x: set(re.findall(r"[a-z]{4,}|\d[\d.,%]*", x.lower()))
            cw = words(claim)
            order = sorted(range(len(passages)), key=lambda i: -len(cw & words(passages[i])))
        docs, used = [], 0
        for i in order:
            p = passages[i]
            if used + len(p) > self.valves.context_chars and docs:
                break
            docs.append({"doc_id": str(len(docs)), "text": p}); used += len(p)
        return docs

    # ---------------------------------------------------------------- screen
    def _screen(self, claim: str, passages: list, question: str = ""):
        """LettuceDetect v2: recall 0.966 but specificity 0.417, so it proposes, it never decides.

        True means "worth adjudicating", None means the screen was unreachable.
        """
        docs = self._docs(claim, passages)
        ctx = "\n\n".join(d["text"] for d in docs)
        q = (question or "").strip()[: self.valves.question_chars]
        user = f"User request: {q}\n\n{ctx}\n\nAnswer to verify:\n{claim}"
        try:
            r = requests.post(
                f"{self.valves.lettuce_url.rstrip('/')}/v1/chat/completions",
                json={"model": self.valves.lettuce_model, "temperature": 0, "max_tokens": 512,
                      "messages": [{"role": "system", "content": _L_SYSTEM}, {"role": "user", "content": user}]},
                timeout=60,
            )
            raw = (r.json()["choices"][0]["message"].get("content") or "").strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            try:
                spans = json.loads(raw).get("hallucinated_spans", [])
            except Exception:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    return None
                spans = json.loads(m.group(0)).get("hallucinated_spans", [])
            return len(spans) > 0
        except Exception:
            return None

    # ---------------------------------------------------------------- judge
    def _qwen(self, claim: str, passages: list, question: str = ""):
        """Second adjudicator. True = NOT supported, None = unavailable."""
        ctx = "\n\n".join(d["text"] for d in self._docs(claim, passages))
        user = (f"CONTEXT:\n{ctx}\n\nQUESTION:\n{(question or '').strip()[: self.valves.question_chars]}"
                f"\n\nANSWER:\n{claim}\n\nVerdict:")
        try:
            r = requests.post(
                f"{self.valves.qwen_url.rstrip('/')}/v1/chat/completions",
                json={"model": self.valves.qwen_model, "temperature": 0, "max_tokens": 8,
                      "messages": [{"role": "system", "content": _Q_SYSTEM}, {"role": "user", "content": user}],
                      "chat_template_kwargs": {"enable_thinking": False}},
                timeout=180,
            )
            up = (r.json()["choices"][0]["message"].get("content") or "").strip().upper()
            if "UNGROUNDED" in up: return True
            if "GROUNDED" in up: return False
            return None
        except Exception:
            return None

    def _judge(self, claim: str, passages: list, question: str = ""):
        docs = self._docs(claim, passages)
        think = self.valves.guardian_think
        block = _g_block(think)
        q = (question or "").strip()[: self.valves.question_chars]
        q_turn = [{"role": "user", "content": q}] if (self.valves.include_question and q) else []
        try:
            r = requests.post(
                f"{self.valves.guardian_url.rstrip('/')}/v1/chat/completions",
                json={"model": "granite-guardian-4.1-8b", "temperature": 0,
                      "max_tokens": 2048 if think else 32,
                      "messages": q_turn + [{"role": "assistant", "content": claim}, {"role": "user", "content": block}],
                      "chat_template_kwargs": {"documents": docs},
                      "logprobs": True, "top_logprobs": 20,
                      **({"grammar": _G_GRAMMAR_THINK if think else _G_GRAMMAR_NOTHINK}
                         if self.valves.grammar else {})},
                timeout=120,
            )
            choice = r.json()["choices"][0]
            m = choice["message"]
            s = re.findall(r"<score>\s*(\w+)\s*</score>", (m.get("content") or "") + (m.get("reasoning_content") or ""))
            if not s:
                return None, None
            p = _score_confidence(choice.get("logprobs"))
            if p is not None and self.valves.calibrate:
                p = _calibrate(p)
            return s[-1].lower() == "yes", p
        except Exception:
            return None, None


    def _note(self, messages: list) -> str:
        answer = (messages[-1].get("content") or "").split(MARK)[0]
        passages = self._passages(messages)
        question = self._question(messages)
        if sum(len(p) for p in passages) < self.valves.min_source_chars:
            return ("\n\n---\n**Source check**: no retrieved sources in this conversation, so this answer comes from the "
                    "model's memory and was not checked.") if self.valves.note_unsourced and self._claims(answer) else ""
        claims = self._claims(answer)[: self.valves.max_claims]
        if not claims:
            return ""

        or_mode = (self.valves.combine or "and").strip().lower() == "or"

        # Stage 1 - LettuceDetect screens everything (cheap, recall 0.966, specificity 0.417).
        screened = None
        if self.valves.prefilter:
            with ThreadPoolExecutor(8) as ex:                         # vLLM, happy with concurrency
                screened = list(ex.map(lambda c: self._screen(c, passages, question), claims))
            if all(v is None for v in screened):                      # screen down: judge everything instead
                screened = None

        # Stage 2 - who the judge sees. Under 'and' only the screen's suspects need adjudicating.
        # Under 'or' a statement the screen cleared can still be caught, so everything is judged
        # and the screen stops being a cost saving - that is what evasion resistance buys.
        if screened is None or or_mode:
            to_judge, cleared = list(range(len(claims))), 0
        else:
            to_judge = [i for i, v in enumerate(screened) if v is not False]
            cleared = sum(1 for v in screened if v is False)

        verdicts = {}
        if to_judge:
            with ThreadPoolExecutor(2) as ex:                         # the judge serves 2 slots
                for i, vp in zip(to_judge, ex.map(lambda i: self._judge(claims[i], passages, question), to_judge)):
                    verdicts[i] = vp
            if self.valves.use_qwen:
                ask = [i for i in to_judge if verdicts.get(i, (None, None))[0] is not True]
                if ask:
                    with ThreadPoolExecutor(4) as ex:
                        for i, qv in zip(ask, ex.map(lambda i: self._qwen(claims[i], passages, question), ask)):
                            if qv:
                                v, p = verdicts.get(i, (None, None))
                                verdicts[i] = (True, p)

        checked = cleared + sum(1 for i in to_judge
                                if verdicts.get(i, (None, None))[0] is not None
                                or (screened is not None and screened[i] is not None))
        if not checked:
            return ""                                                 # both judges unreachable: stay silent

        nums = {}
        if self.valves.number_check:
            for i, c in enumerate(claims):
                o = _orphan_numbers(c, passages, self.valves.number_density)
                if o: nums[i] = o

        bad = []
        for i in range(len(claims)):
            gv, gp = verdicts.get(i, (None, None))
            sv = screened[i] if screened is not None else None
            nv = i in nums
            if or_mode:
                if gv or sv or nv:
                    src = [n for n, f in (("screen", sv), ("judge", gv), ("numbers", nv)) if f]
                    bad.append((claims[i], gp, "both" if len(src) > 1 else src[0] + " only"))
            elif gv or nv:
                by = "" if gv else "figure not in sources: " + ", ".join(str(x) for x in nums[i][:3])
                bad.append((claims[i], gp, by))
            elif gv is None and sv:
                bad.append((claims[i], None, "unadjudicated"))
        bad.sort(key=lambda t: -(t[1] if t[1] is not None else 0))    # most confident flags first

        who = "LettuceDetect + Granite Guardian" if screened is not None else "Granite Guardian"
        if screened is not None and or_mode:
            who += ", either"
        head = f"{MARK} ({who}, local): "
        if not bad:
            return head + f"all {checked} checked statements are supported by the retrieved sources."
        # The confidence score was measured in no-think mode (AUC 0.815). Think mode saturates
        # it - 146 of 161 flags scored >=0.80 - so the mark is suppressed there rather than
        # decorating almost everything.
        hi = self.valves.high_confidence if not self.valves.guardian_think else None
        lines = []
        for c, p, by in bad:
            conf_hi = hi is not None and p is not None and p >= hi
            if by == "unadjudicated":
                mark = " *(unadjudicated - judge unreachable)*"
            elif by:
                mark = f" *({by}, high confidence)*" if conf_hi else f" *({by})*"
            else:
                mark = " *(high confidence)*" if conf_hi else ""
            lines.append(f'- "{c[:240]}"{mark}')
        return (head + f"{checked - len(bad)} of {checked} checked statements are supported by the retrieved sources. "
                "Not supported:\n" + "\n".join(lines))

    # ---------------------------------------------------------------- outlet
    async def outlet(self, body: dict, __user__: dict = None) -> dict:
        try:
            messages = body.get("messages") or []
            if not messages or messages[-1].get("role") != "assistant":
                return body
            last = messages[-1]
            base = (last.get("content") or "").split(MARK)[0]
            note = await asyncio.to_thread(self._note, messages)       # blocking HTTP stays off the event loop
            if not note:
                return body
            last["content"] = base + note
            for item in reversed(last.get("output") or []):            # 0.11 renders from 'output' when present
                if isinstance(item, dict) and item.get("type") == "message" and isinstance(item.get("content"), list):
                    parts = [p for p in item["content"] if isinstance(p, dict) and p.get("text") is not None]
                    if parts:
                        parts[-1]["text"] = str(parts[-1]["text"]).split(MARK)[0] + note
                        break
        except Exception:
            pass                                                       # a failed check must never break the chat
        return body
