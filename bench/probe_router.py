"""Does the router-side LettuceDetect fact-check actually fire for an Open WebUI RAG chat?"""
import json, urllib.request

URL = "http://10.0.0.72:8899/v1/chat/completions"
CTX = ("The Sears Tower, renamed Willis Tower in 2009, stands 1,451 feet tall to the roof "
       "and was completed in 1973 in Chicago.")
Q = "How tall is Willis Tower and when was it completed?"

def call(label, messages, tools=None):
    body = {"model": "vllm-sr/auto", "temperature": 0, "max_tokens": 120, "messages": messages}
    if tools: body["tools"] = tools
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=180)
        hdrs = {k.lower(): v for k, v in r.headers.items()}
        d = json.loads(r.read())
    except Exception as e:
        print(f"-- {label}: ERROR {e}"); return
    interesting = {k: v for k, v in hdrs.items()
                   if any(s in k for s in ("hallucinat", "fact", "vsr", "router", "decision",
                                           "model", "x-selected", "classif", "reason"))}
    print(f"-- {label}")
    print(f"   headers: {json.dumps(interesting, indent=0)[:600]}")
    print(f"   answer : {d['choices'][0]['message'].get('content','')[:160]!r}")

# 1. Open WebUI RAG shape: retrieved context injected as a system prompt
call("context as SYSTEM prompt (Open WebUI RAG shape)", [
    {"role": "system", "content": "Use only this context:\n" + CTX},
    {"role": "user", "content": Q},
])

# 2. Native tool-call shape: context arrives as a tool result
call("context as role:tool message (native tool call)", [
    {"role": "user", "content": Q},
    {"role": "assistant", "content": None,
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "search", "arguments": '{"q":"willis tower"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": CTX},
], tools=[{"type": "function", "function": {"name": "search", "description": "search docs",
           "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}])

# 3. context as a plain user message (pasted text)
call("context pasted into USER message", [
    {"role": "user", "content": CTX + "\n\n" + Q},
])
