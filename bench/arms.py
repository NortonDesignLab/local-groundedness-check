"""Detector adapters: all return {'ungrounded': bool|None, 'latency_s', 'raw', ...}."""
import json, os, re, time, urllib.request

# ---------------- Granite Guardian 4.1-8B (SARA :8014, llama.cpp) ----------------
import guardian

def guardian_arm(ex, think=False, with_question=True):
    ans = ex["answer"]
    if with_question:
        # judge the answer in the context of the question it answers
        r = guardian.judge_conv(ex["question"], ans, [ex["passage"]], think=think)
    else:
        r = guardian.judge(ans, [ex["passage"]], think=think)
    return r

# ---------------- LettuceDetect v2 (ATOM 10.0.0.72:8077, vLLM) ----------------
LD_URL = os.environ.get("LETTUCE_URL", "http://10.0.0.72:8077").rstrip("/") + "/v1/chat/completions"
LD_MODEL = "KRLabsOrg/lettucedect-v2-qwen-2b"
LD_SYSTEM = """You are an expert annotator who identifies hallucinated spans in a generated answer with respect to a given context (the only trusted evidence). A hallucinated span is a substring of the answer that is not supported by the context. Spans consistent with the context are not hallucinations.

Quote each hallucinated span verbatim from the answer and classify it into exactly one category and one subcategory.

Categories (the kinds of unsupported span):
- contradiction: conflicts with the context (a wrong value, number, date, name, or relationship)
- fabricated_reference: an entity, name, identifier, or section that is absent from the context
- unsupported_addition: a claim, detail, or behavior the context never states

Subcategories:
- entity: a wrong or invented name, entity, or object
- temporal: an incorrect date, time, duration, or ordering
- numerical: an incorrect number, quantity, or amount
- value: a wrong value, setting, or attribute value
- relational: an incorrect relationship or association between things
- identifier: an invented identifier or name not found in the context
- section: a reference to a section, part, or location that does not exist
- attribute: an invented or incorrect attribute or property
- claim: an added factual claim the context does not support
- behavior: an added or changed action or behavior the context never states
- elaboration: extra detail or elaboration beyond what the context supports
- subjective: an unsupported subjective or evaluative statement
- unspecified: unsupported, with no more specific subtype

Reply with ONLY a JSON object (no markdown, no code fences): {"hallucinated_spans": [{"text": "...", "category": "...", "subcategory": "..."}]}. If nothing is unsupported, reply {"hallucinated_spans": []}."""

def lettuce_arm(ex, timeout=300, retries=2):
    user = f'User request: {ex["question"]}\n\n{ex["passage"]}\n\nAnswer to verify:\n{ex["answer"]}'
    body = {"model": LD_MODEL, "temperature": 0.0, "max_tokens": 1024,
            "messages": [{"role": "system", "content": LD_SYSTEM},
                         {"role": "user", "content": user}]}
    last = None
    for a in range(retries + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(LD_URL, data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            dt = time.time() - t0
            raw = r["choices"][0]["message"]["content"]
            txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
            try:
                spans = json.loads(txt).get("hallucinated_spans", [])
            except Exception:
                m = re.search(r'\{.*\}', txt, re.DOTALL)
                spans = json.loads(m.group(0)).get("hallucinated_spans", []) if m else None
            if spans is None:
                return {"ungrounded": None, "latency_s": round(dt,3), "raw": raw, "n_spans": None}
            return {"ungrounded": len(spans) > 0, "latency_s": round(dt, 3), "raw": raw,
                    "n_spans": len(spans),
                    "cats": sorted({s.get("category") for s in spans}),
                    "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": r.get("usage", {}).get("completion_tokens")}
        except Exception as e:
            last = e; time.sleep(1 + 2 * a)
    return {"ungrounded": None, "latency_s": None, "raw": f"ERROR: {last}", "n_spans": None}

# ---------------- Jev / TypeSafe AI (hosted, needs early-access key) ----------------
JEV_URL = os.environ.get("TYPESAFE_URL", "https://api.typesafe.ai/v1/systemone")
JEV_KEY = os.environ.get("TYPESAFE_API_KEY")

def jev_available():
    return bool(JEV_KEY)

def jev_arm(ex, timeout=120, retries=2):
    """UNVERIFIED request shape - no early-access key on this machine, so this
    adapter has never been executed against the real API. Schema follows the
    launch post: unstructured state in, typed probabilistic decision out."""
    if not JEV_KEY:
        return {"ungrounded": None, "latency_s": None, "raw": "ERROR: no TYPESAFE_API_KEY"}
    prompt = (f"Context (the only trusted evidence):\n{ex['passage']}\n\n"
              f"Question: {ex['question']}\n\nAnswer: {ex['answer']}\n\n"
              "Is the answer fully supported by the context?")
    body = {"model": "jev-latest", "input": prompt,
            "output_type": {"type": "enum", "values": ["grounded", "ungrounded"]}}
    last = None
    for a in range(retries + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(JEV_URL, data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json",
                                                  "authorization": f"Bearer {JEV_KEY}"})
            r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            dt = time.time() - t0
            return {"ungrounded": None, "latency_s": round(dt, 3), "raw": json.dumps(r)[:2000],
                    "_needs_parser": True}
        except Exception as e:
            last = e; time.sleep(1 + 2 * a)
    return {"ungrounded": None, "latency_s": None, "raw": f"ERROR: {last}"}
