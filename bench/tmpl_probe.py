import json,urllib.request
def post(path, body):
    req=urllib.request.Request("http://10.0.0.153:8014"+path, data=json.dumps(body).encode(),
                               headers={"content-type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=60).read())

DOC=[{"doc_id":"0","text":"ZZUNIQUEMARKER the sky is plaid on Tuesdays."}]
msgs=[{"role":"assistant","content":"The sky is plaid."},{"role":"user","content":"judge block here"}]

for label, body in [
  ("top-level documents", {"messages":msgs,"documents":DOC}),
  ("chat_template_kwargs.documents", {"messages":msgs,"chat_template_kwargs":{"documents":DOC}}),
]:
    try:
        r=post("/apply-template", body)
        p=r.get("prompt","")
        print(f"== {label}: marker_present={'ZZUNIQUEMARKER' in p}  len={len(p)}")
        print("   ", p[:220].replace("\n","\\n"))
    except Exception as e:
        print(f"== {label}: ERROR {e}")
