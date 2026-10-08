"""#6a - hard-constrain Guardian's output with a GBNF grammar."""
import json, time, urllib.request
import guardian as G

GRAMMAR = r'''root ::= "<think>" [^<]* "</think>" ws "<score>" ws ("yes" | "no") ws "</score>"
ws ::= [ \t\n]*'''
GRAMMAR_NOTHINK = r'''root ::= "<think>\n</think>\n<score> " ("yes" | "no") " </score>"'''

DOC=["Study 1 recruited 32 undergraduate students (16 women and 16 men). The mean settlement price was $4,210."]
def call(text, think, grammar):
    body={"model":G.MODEL,"temperature":0,"max_tokens":2048 if think else 64,
          "messages":[{"role":"user","content":"How many students were in Study 1?"},
                      {"role":"assistant","content":text},
                      {"role":"user","content":G.block(think)}],
          "chat_template_kwargs":{"documents":[{"doc_id":"0","text":DOC[0]}]}}
    if grammar: body["grammar"]=grammar
    t0=time.time()
    r=json.loads(urllib.request.urlopen(urllib.request.Request(
        "http://10.0.0.153:8014/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"content-type":"application/json"}),timeout=300).read())
    return r["choices"][0]["message"]["content"], round(time.time()-t0,2)

for think,g,label in [(False,None,"no-think, ungrammared"),
                      (False,GRAMMAR_NOTHINK,"no-think, GRAMMAR"),
                      (True,None,"think, ungrammared"),
                      (True,GRAMMAR,"think, GRAMMAR")]:
    try:
        out,dt=call("Study 1 had 48 students.",think,g)
        print(f"{label:24s} {dt:5.2f}s parsed={G.parse_output(out)!r}  raw={out[:70]!r}")
    except Exception as e:
        print(f"{label:24s} ERROR {e}")
