#!/usr/bin/env python3
"""Inject the canonical detection core into each standalone Open WebUI artifact.

Open WebUI functions and tools are single pasted files, so they cannot import a shared
module. This keeps one source of truth (core_block.py) and writes it into each target
between markers, so the three copies cannot drift.
"""
import json, re, sys, pathlib

HERE = pathlib.Path(__file__).parent
# Targets: the chat filter in this repo, plus any extra standalone files passed on the command line
# (in my setup the two NDL deck builders, which are not published here).
TARGETS = [HERE.parent/"openwebui"/"guardian_source_check.py"] + [pathlib.Path(a) for a in sys.argv[1:]]
START = "# ===== NDL GROUNDCHECK CORE"
END = "# ===== END NDL GROUNDCHECK CORE ====="

def build_core():
    core = (HERE/"core_block.py").read_text()
    pts = json.load(open(HERE/"calibration_map.json"))["points"]
    # keep the table small enough to paste comfortably
    thin = [pts[0]] + [p for i, p in enumerate(pts[1:-1], 1) if i % 3 == 0] + [pts[-1]]
    core = core.replace("__CALIB_POINTS__", json.dumps(thin))
    return core

def indent(block, spaces):
    pad = " " * spaces
    return "\n".join(pad + l if l.strip() else l for l in block.splitlines())

def main():
    core = build_core()
    for t in TARGETS:
        s = t.read_text()
        if START not in s:
            print(f"!! {t.name}: no marker, insert it manually first"); continue
        i = s.index(START); j = s.index(END) + len(END)
        # preserve the indentation the marker sits at (class body vs module level)
        col = i - s.rfind("\n", 0, i) - 1
        body = indent(core.strip(), col) if col else core.strip()
        s = s[:i - col] + body + s[j:] if col else s[:i] + body + s[j:]
        t.write_text(s)
        print(f"ok {t.name}  (core {len(core.splitlines())} lines at col {col})")
    # guard: every target must still parse
    import ast
    for t in TARGETS:
        ast.parse(t.read_text()); print(f"   syntax ok {t.name}")

if __name__ == "__main__":
    main()
