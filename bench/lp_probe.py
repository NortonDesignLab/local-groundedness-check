import json, urllib.request
import guardian as G
body_msgs = lambda ans, q: ([{"role":"user","content":q}] if q else []) + [
    {"role":"assistant","content":ans},{"role":"user","content":G.block(False)}]
DOC=[{"doc_id":"0","text":"Study 1 recruited 32 undergraduate students (16 women and 16 men). The mean settlement price was $4,210."}]
for name, ans in [("faithful","Study 1 had 32 students, 16 women and 16 men."),
                  ("wrong number","Study 1 had 48 students, 24 women and 24 men.")]:
    body={"model":G.MODEL,"temperature":0,"max_tokens":32,
          "messages":body_msgs(ans,"How many students were in Study 1?"),
          "chat_template_kwargs":{"documents":DOC},
          "logprobs":True,"top_logprobs":20}
    r=json.loads(urllib.request.urlopen(urllib.request.Request(
        "http://10.0.0.153:8014/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"content-type":"application/json"}),timeout=120).read())
    ch=r["choices"][0]
    print("==",name,"|",repr(ch["message"]["content"]))
    for i,tok in enumerate(ch["logprobs"]["content"]):
        tops={t["token"]:round(t["logprob"],3) for t in tok.get("top_logprobs",[])[:5]}
        print(f"   [{i}] {tok['token']!r:12s} lp={tok['logprob']:.3f} tops={tops}")
