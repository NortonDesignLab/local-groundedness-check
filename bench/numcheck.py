"""#2 - deterministic numeric check: does every number in the claim appear in the sources?

Precision-oriented. An orphan number is strong evidence of fabrication; the absence of one
is no evidence of grounding, so this signal only ever ADDS a flag, never clears one.
"""
import re
from fractions import Fraction

WORDS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,
         "nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
         "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,"thirty":30,
         "forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90,
         "hundred":100,"thousand":1000,"million":1_000_000,"billion":1_000_000_000}
SCALE = {"k":1e3,"m":1e6,"bn":1e9,"b":1e9,"thousand":1e3,"million":1e6,"billion":1e9,"trillion":1e12}
NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|percent|k\b|m\b|bn\b|b\b|thousand|million|billion|trillion)?", re.I)
# ordinals, list indices and years are usually structural, not factual payload
TRIVIAL = {0,1,2,3,4,5,6,7,8,9,10,11,12,100}

def _canon(v: float) -> str:
    f = Fraction(v).limit_denominator(1000)
    return str(f)

def numbers(text: str) -> set:
    out = set()
    for m in NUM_RE.finditer(text or ""):
        raw, suf = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        try: v = float(raw)
        except ValueError: continue
        if suf in ("%", "percent"):
            out.add(("pct", _canon(v)))          # 42% and "42 percent" unify; 0.42 does NOT
        elif suf in SCALE:
            out.add(("num", _canon(v * SCALE[suf])))
            out.add(("num", _canon(v)))          # "3 million" may be written as 3 in a table
        else:
            out.add(("num", _canon(v)))
    for w, v in WORDS.items():
        if re.search(rf"\b{w}\b", (text or "").lower()):
            out.add(("num", _canon(float(v))))
    return out

def _derivable(v: float, src_vals: list, tol=0.005) -> bool:
    """Is v a plain arithmetic consequence of numbers that ARE in the source?

    FinanceBench and DROP answers are full of sums, differences and percentages computed
    from the passage. Without this, the orphan rule fires on correct arithmetic: measured
    precision 0.33 on DROP and 0.55 on FinanceBench before this check was added.
    """
    def close(a, b):
        if b is None: return False
        return abs(a - b) <= max(tol * max(abs(a), abs(b)), 1e-9)
    n = len(src_vals)
    if n > 60:                                   # keep it O(n^2) on a bounded n
        src_vals = src_vals[:60]; n = 60
    for i in range(n):
        a = src_vals[i]
        if close(v, -a) or close(v, abs(a)) or close(v, a * 100) or close(v, a / 100 if a else None):
            return True
        for j in range(n):
            b = src_vals[j]
            cands = [a + b, a - b, a * b]
            if b:
                cands += [a / b, a / b * 100, (a - b) / b * 100]
            if any(close(v, c) for c in cands if c == c and abs(c) != float("inf")):
                return True
    return False


def orphans(claim: str, passages, ignore_trivial=True, check_derived=True) -> list:
    """Numbers in the claim that appear in none of the passages."""
    src = set()
    for p in passages:
        src |= numbers(p)
    src_vals = []
    for _, val in src:
        try: src_vals.append(float(Fraction(val)))
        except Exception: pass
    bad = []
    for kind, val in numbers(claim):
        if (kind, val) in src:
            continue
        if kind == "pct" and ("num", val) in src:     # "42 percent" vs a bare 42 in a table
            continue
        if kind == "num" and ("pct", val) in src:
            continue
        try:
            f = float(Fraction(val))
        except Exception:
            f = None
        if ignore_trivial and f is not None and f in TRIVIAL and kind == "num":
            continue
        if check_derived and f is not None and _derivable(f, src_vals):
            continue                                  # correct arithmetic over sourced figures
        if f is not None and 1900 <= f <= 2100 and kind == "num":
            bad.append((kind, val, "year"))           # years kept but marked
        else:
            bad.append((kind, val, ""))
    return bad

if __name__ == "__main__":
    SRC=["Study 1 recruited 32 undergraduate students (16 women and 16 men). The mean settlement "
         "price was $4,210 (SD = 380). Revenue reached $3.2 million in 2019."]
    for c in ["Study 1 had 32 students.",
              "Study 1 had 48 students.",
              "The mean settlement price was $4,210.",
              "Participants were paid ten dollars.",
              "Revenue reached $3.2 million.",
              "Revenue grew 42 percent.",
              "The study ran in 2019.",
              "Women negotiated 12% higher settlements."]:
        print(f"  {c:52s} -> orphans {orphans(c, SRC)}")
