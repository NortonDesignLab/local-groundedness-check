"""Passage selection for a claim: keyword overlap (current) vs Qwen3 reranker (:8013) vs embeddings (:8011).

The benchmark ships one passage per item, so selection is a no-op there. To make it a real
retrieval problem we chunk every passage in the sample, then for each item build a candidate
pool of its own chunks plus distractors drawn from the same source, and force a choice with a
small context budget.
"""
import json, re, urllib.request

CHUNK, OVERLAP_CAP = 1500, 1800

# ---- production reranker contract (Qwen3-Reranker-4B served by vLLM) ----
RERANK_URL, RERANK_MODEL = "http://localhost:8013", "qwen3-reranker-4b"
QWEN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_PREFIX = ('<|im_start|>system\n'
           'Judge whether the Document meets the requirements based on the Query and the Instruct '
           'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
           '<|im_start|>user\n')
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def _post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def chunks(text):
    out = []
    for k in range(0, len(text), CHUNK):
        c = text[k:k + OVERLAP_CAP]
        if len(c.strip()) >= 40:
            out.append(c)
    return out

# ---------------------------------------------------------------- selectors
def pick_keyword(query, pool, budget):
    """What the filter and the builders do today."""
    words = lambda x: set(re.findall(r"[a-z]{4,}|\d[\d.,%]*", x.lower()))
    cw = words(query)
    picked, used = [], 0
    for p in sorted(pool, key=lambda p: -len(cw & words(p))):
        if used + len(p) > budget and picked:
            break
        picked.append(p); used += len(p)
    return picked

def pick_rerank(query, pool, budget):
    r = _post(f"{RERANK_URL}/v1/score", {
        "model": RERANK_MODEL,
        "text_1": f"{_PREFIX}<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {query}\n",
        "text_2": [f"<Document>: {t}{_SUFFIX}" for t in pool]})
    score = {d["index"]: d["score"] for d in r["data"]}
    picked, used = [], 0
    for i in sorted(range(len(pool)), key=lambda i: -score.get(i, 0.0)):
        if used + len(pool[i]) > budget and picked:
            break
        picked.append(pool[i]); used += len(pool[i])
    return picked

def pick_embed(query, pool, budget):
    r = _post("http://localhost:8011/v1/embeddings",
              {"model": "nemotron-3-embed-8b", "input": [query] + pool})
    v = [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
    q, docs = v[0], v[1:]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    nrm = lambda a: sum(x * x for x in a) ** 0.5
    sims = [dot(q, d) / (nrm(q) * nrm(d) + 1e-9) for d in docs]
    picked, used = [], 0
    for i in sorted(range(len(pool)), key=lambda i: -sims[i]):
        if used + len(pool[i]) > budget and picked:
            break
        picked.append(pool[i]); used += len(pool[i])
    return picked

SELECTORS = {"keyword": pick_keyword, "rerank": pick_rerank, "embed": pick_embed}
