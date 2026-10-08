"""#3 - split a statement into atomic facts.

Judging one fact at a time is the strongest cross-validated lever in this whole exercise
(Jev 62.6 -> 95.0 split five ways; the question turn +0.072). The risk is the mirror image:
a fragment torn from its sentence loses context and gets over-flagged, which is exactly why
the question turn helped. So each atom is carried with its parent sentence as context.
"""
import re

SPLIT = re.compile(r"(?:,\s+(?:and|but|while|whereas|although)\s+|\s+;\s*|;\s+|\s+—\s+)", re.I)
VERBISH = re.compile(r"\b(is|are|was|were|has|have|had|will|would|can|could|did|does|do|"
                     r"grew|rose|fell|reached|reported|found|showed|increased|decreased|"
                     r"includes?|contains?|states?|equals?|totall?ed|remains?|became|"
                     r"\w+ed|\w+s)\b", re.I)

def atoms(sentence: str, min_words=4, max_atoms=4):
    s = (sentence or "").strip()
    if not s:
        return []
    parts = [p.strip(" ,;—") for p in SPLIT.split(s) if p and p.strip(" ,;—")]
    if len(parts) < 2:
        return [s]
    good = [p for p in parts if len(p.split()) >= min_words and VERBISH.search(p)]
    if len(good) < 2 or len(good) > max_atoms:
        return [s]                                  # not a clean split: judge it whole
    return good

if __name__ == "__main__":
    for s in ["Revenue grew 12 percent, and operating margin improved to 34 percent.",
              "Study 1 had 32 students; the mean settlement price was $4,210.",
              "The plan runs from October 2026 to September 2027.",
              "Targets were confirmed at $12,500 for AMs, and the pay mix moved to 30/70."]:
        print(f"  {s}\n    -> {atoms(s)}")
