"""Install v1.3.0 into Open WebUI's database, regenerating tool specs the way the UI does."""
import sys; sys.path.insert(0, "/app/backend")
import json, sqlite3, sys, time, types, importlib.util

DB = "/app/backend/data/webui.db"
JOBS = [
    ("function", "guardian_source_check", "/work/functions/guardian_source_check.py"),
    ("tool",     "ndl_powerpoint_builder", "/work/tools/ndl_pptx_builder.py"),
    ("tool",     "ndl_word_builder",       "/work/tools/ndl_docx_builder.py"),
]

from open_webui.utils.tools import get_tool_specs

def load_module(name, content):
    mod = types.ModuleType(name)
    mod.__dict__["__file__"] = f"{name}.py"
    exec(compile(content, f"{name}.py", "exec"), mod.__dict__)
    return mod

c = sqlite3.connect(DB)
now = int(time.time())
for table, rid, path in JOBS:
    content = open(path).read()
    row = list(c.execute(f"select id from {table} where id=?", (rid,)))
    if not row:
        print(f"!! {table}/{rid} not installed - skipping (create it in the UI first)"); continue
    try:
        mod = load_module(rid, content)
    except Exception as e:
        print(f"!! {rid} failed to compile: {e}"); continue
    if table == "tool":
        try:
            specs = get_tool_specs(mod.Tools())
        except Exception as e:
            print(f"!! {rid} spec generation failed: {e}"); continue
        names = [s["name"] for s in specs]
        params = sorted((specs[0].get("parameters") or {}).get("properties", {}).keys())
        c.execute("update tool set content=?, specs=?, updated_at=? where id=?",
                  (content, json.dumps(specs), now, rid))
        print(f"ok tool/{rid}: {len(content)} chars, specs {names} params={params}")
    else:
        # sanity: the filter must expose the outlet hook
        assert hasattr(mod, "Filter") and hasattr(mod.Filter, "outlet"), "no Filter.outlet"
        c.execute("update function set content=?, updated_at=? where id=?", (content, now, rid))
        print(f"ok function/{rid}: {len(content)} chars")
c.commit()
print("\n--- installed state ---")
for t in ("function", "tool"):
    for r in c.execute(f"select id,length(content) from {t} where id in ('guardian_source_check','ndl_powerpoint_builder','ndl_word_builder')"):
        print(f"  {t}/{r[0]}: {r[1]} chars")
