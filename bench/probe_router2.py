import json, urllib.request
URL="http://10.0.0.72:8899/v1/chat/completions"
CTX=("The Sears Tower, renamed Willis Tower in 2009, stands 1,451 feet tall to the roof "
     "and was completed in 1973 in Chicago.")
Q="How tall is Willis Tower and when was it completed?"
TOOLS=[{"type":"function","function":{"name":"search","description":"search docs",
        "parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}]
def call(label, messages, tools=None):
    body={"model":"vllm-sr/auto","temperature":0,"max_tokens":80,"messages":messages}
    if tools: body["tools"]=tools
    try:
        r=urllib.request.urlopen(urllib.request.Request(URL,data=json.dumps(body).encode(),
            headers={"content-type":"application/json"}),timeout=180)
        h={k.lower():v for k,v in r.headers.items()}; r.read()
    except Exception as e:
        print(f"{label:52s} ERROR {e}"); return
    print(f"{label:52s} decision={h.get('x-vsr-selected-decision','?'):16s} warn={h.get('x-vsr-response-warnings','-')}")

call("sys-ctx, NO tools", [{"role":"system","content":"Use only this context:\n"+CTX},{"role":"user","content":Q}])
call("sys-ctx, WITH tools declared", [{"role":"system","content":"Use only this context:\n"+CTX},{"role":"user","content":Q}], TOOLS)
call("role:tool result, WITH tools", [{"role":"user","content":Q},
    {"role":"assistant","content":None,"tool_calls":[{"id":"c1","type":"function","function":{"name":"search","arguments":"{}"}}]},
    {"role":"tool","tool_call_id":"c1","content":CTX}], TOOLS)
call("role:tool result, NO tools declared", [{"role":"user","content":Q},
    {"role":"assistant","content":None,"tool_calls":[{"id":"c1","type":"function","function":{"name":"search","arguments":"{}"}}]},
    {"role":"tool","tool_call_id":"c1","content":CTX}])
call("plain factual, no context at all", [{"role":"user","content":"How tall is Willis Tower?"}])
