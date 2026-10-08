"""Granite Guardian 4.1-8B groundedness adapter (official model-card format)."""
import json, math, os, re, time, urllib.request

# llama.cpp server hosting Granite Guardian; override with GUARDIAN_URL=http://host:port
URL = os.environ.get("GUARDIAN_URL", "http://10.0.0.153:8014").rstrip("/") + "/v1/chat/completions"
MODEL = "granite-guardian-4.1-8b"

GUARDIAN_JUDGE_THINK = (
    "<think>As a judge agent, carefully analyze whether the provided text meets the "
    "given judging criteria. Use all available information, including conversations, "
    "documents, and tools. Reason through the evidence step by step before providing "
    "your score. Wrap your reasoning in <think></think> tags, then provide your "
    "score in <score></score> tags."
)
GUARDIAN_JUDGE_NOTHINK = (
    "<no-think>As a judge agent, assess whether the provided text meets the given "
    "judging criteria using all available information, including conversations, "
    "documents, and tools. Provide your score immediately without explanation. "
    "Output empty <think>\\n</think> tags followed by your score in <score></score> tags."
)
SCHEMA = "If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'."
CRITERIA = (
    "A text is considered ungrounded or unfaithful if it includes information "
    "lacking support from, or directly contradicting, the provided document(s). "
    "This risk arises when the text fabricates details, misinterprets the content, "
    "or makes unsupported extrapolations beyond what is explicitly stated in the "
    "document(s)."
)

def block(think):
    j = GUARDIAN_JUDGE_THINK if think else GUARDIAN_JUDGE_NOTHINK
    return f"{j}\n\n### Criteria: {CRITERIA}\n\n### Scoring Schema: {SCHEMA}"

def parse_output(text):
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.findall(r"<score>\s*(.*?)\s*</score>", clean, re.DOTALL)
    return m[0].strip().lower() if m else None

def score_confidence(logprobs):
    """P(ungrounded) from the yes/no token's own distribution.

    The score token is the first `yes`/`no` emitted after the </think> block. Both
    alternatives reliably appear in that token's top_logprobs, so renormalise over the
    two and ignore the rest of the vocabulary. Returns None when the token or its
    counterpart cannot be found.
    """
    if not logprobs or not logprobs.get("content"):
        return None
    norm = lambda t: t.strip().strip("<>/").lower()
    for tok in logprobs["content"]:
        if norm(tok.get("token", "")) not in ("yes", "no"):
            continue
        lp = {}
        for alt in tok.get("top_logprobs") or []:
            k = norm(alt.get("token", ""))
            if k in ("yes", "no") and k not in lp:
                lp[k] = alt["logprob"]
        lp.setdefault(norm(tok["token"]), tok.get("logprob", 0.0))
        if "yes" not in lp and "no" not in lp:
            return None
        # a missing counterpart was outside top_logprobs: floor it well below the seen one
        seen = min(lp.values())
        y = lp.get("yes", seen - 20.0)
        n = lp.get("no", seen - 20.0)
        m = max(y, n)
        ey, en = math.exp(y - m), math.exp(n - m)
        return ey / (ey + en)
    return None


def judge(response_text, documents, think=False, timeout=600, retries=2, _user_prefix=None,
          want_confidence=False, top_logprobs=20, _extra=None):
    """documents: list[str]. Returns dict(score=yes|no|None, ungrounded=bool|None, ...).
    'yes' == UNGROUNDED."""
    assert documents and all(d and d.strip() for d in documents), "empty documents = silent false pass"
    body = {
        "model": MODEL,
        "messages": ([{"role": "user", "content": _user_prefix}] if _user_prefix else []) + [
            {"role": "assistant", "content": response_text},
            {"role": "user", "content": block(think)},
        ],
        # llama.cpp SILENTLY DROPS top-level "documents"; the jinja template only sees
        # variables passed via chat_template_kwargs. Verified with /apply-template.
        "chat_template_kwargs": {
            "documents": [{"doc_id": str(i), "text": d} for i, d in enumerate(documents)]
        },
        "temperature": 0.0,
        "max_tokens": 2048 if think else 64,
    }
    if _extra:
        body.update(_extra)
    if want_confidence:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs
    last = None
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            dt = time.time() - t0
            raw = r["choices"][0]["message"]["content"]
            sc = parse_output(raw)
            conf = score_confidence(r["choices"][0].get("logprobs")) if want_confidence else None
            return {"score": sc, "ungrounded": None if sc is None else sc == "yes",
                    "p_ungrounded": conf,
                    "latency_s": round(dt, 3), "raw": raw,
                    "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": r.get("usage", {}).get("completion_tokens")}
        except Exception as e:
            last = e
            time.sleep(1 + attempt * 2)
    return {"score": None, "ungrounded": None, "p_ungrounded": None, "latency_s": None,
            "raw": f"ERROR: {last}", "prompt_tokens": None, "completion_tokens": None}


def judge_conv(question, response_text, documents, **kw):
    """Groundedness of an answer given the question that prompted it."""
    return judge(response_text, documents, _user_prefix=question, **kw)

if __name__ == "__main__":
    ctx = ('Eat (1964) is a 45-minute underground film created by Andy Warhol and featuring painter '
           "Robert Indiana, filmed on Sunday, February 2, 1964, in Indiana's studio. The film was first "
           "shown by Jonas Mekas on July 16, 1964, at the Washington Square Gallery at 530 West Broadway.\n"
           'Jonas Mekas (December 24, 1922 - January 23, 2019) was a Lithuanian-American filmmaker, poet, '
           'and artist who has been called "the godfather of American avant-garde cinema".')
    cases = [
        ("model-card ungrounded (expect yes)",
         "The film Eat was first shown by Jonas Mekas on December 24, 1922 at the Washington Square Gallery at 530 West Broadway."),
        ("faithful (expect no)",
         "Eat is a 45-minute film by Andy Warhol featuring Robert Indiana."),
        ("paraphrase (expect no)",
         "Warhol's Eat, which runs three quarters of an hour, was shot in Robert Indiana's studio."),
        ("wrong number (expect yes)", "Eat (1964) is a 90-minute underground film by Andy Warhol."),
        ("invented detail (expect yes)", "Eat was filmed in colour on 35mm stock."),
    ]
    for think in (False, True):
        print(f"\n===== think={think} =====")
        for name, txt in cases:
            r = judge(txt, [ctx], think=think)
            print(f"  {name:42s} -> score={r['score']}  {r['latency_s']}s  ctok={r['completion_tokens']}")
