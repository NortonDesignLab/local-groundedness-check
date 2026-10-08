# The check protocol: how I use the judges on real deliverables

This is the procedure I run before a factual deliverable (a slide deck, a memo, a chat answer
that will be forwarded) leaves the building. The Open WebUI filter automates a lighter version
of it inside chat; `tools/check_claims.py` is the batch version. Both exist because the manual
protocol kept catching things.

## The procedure

1. **Collect verbatim source passages for every factual claim.** Quote the page, the PDF
   (`pdftotext`), or the retrieved chunk. Search-result summaries are not sources; fetch the page.
   For a library corpus, pass a window around the matching sentence rather than a truncated chunk.
2. **Split the text into one fact per sentence or table row.** A two-fact sentence fails even when
   both facts are in the passages. A multi-sentence cell can be flagged when each sentence passes.
3. **Pass the question (or for a slide, the title / section heading) with each claim.** Judging the
   sentence alone costs 0.06 balanced accuracy and pushes false flags up by a third (measured,
   McNemar chi2 7.15).
4. **Plant at least one false control claim.** Something plausible and wrong (a changed number, an
   invented citation). If a control passes as grounded, the run is invalid: usually the documents
   did not reach the model (see [GOTCHAS.md](GOTCHAS.md)).
5. **Run the stack:** LettuceDetect screens everything, Guardian in think mode adjudicates what
   the screen flagged, a general local model gets a second vote when available, and the orphan-number
   rule runs alongside.
6. **Triage the flags.** Reword or drop anything flagged as a source error. Things flagged because
   they are *ours* (a derived subtraction, a heading, an example) stay, but get disclosed as
   "derived / ours" in the notes rather than silently overridden.
7. **Report what was flagged and why**, including the kept-but-flagged items. The report is part of
   the deliverable's provenance.

## What the judges are strict about (and why that is useful)

- **Inference from context.** A year inferred from an article's date gets flagged. Correct, strictly.
- **Attribution.** "Firm X recommends Y" fails if the passage header shows a different publisher,
  even when the text says Y.
- **Paraphrase of practitioner claims.** Where a paraphrase fails and the verbatim quote passes,
  quote the source.
- **Our own arithmetic.** "47% accessible" derived from "53% restricted" is flagged as unsupported.
  It is a derivation, not an error; disclose it.
- **"Every" and "all".** No judge catches an "all N" overstatement when the recomputation passage
  itself supports it. If you are going to write "every", your recomputation has to include a
  strict count of exceptions first.

## Analytical decks: the recomputation bundle

For decks whose numbers are *our own derived results*, the judges flag nearly everything
(182 of 223 claims on the first such run) because no passage states the derived figure. The fix
that worked:

- an **independent recomputation passage set**: a Python re-derivation of every table, included as
  source text; and
- a small **always-included core-facts bundle** (the inputs: targets, headcounts, dates).

With that, flags fell to 52, every one explainable. Keep the core bundle small: when it grew to
14 summary passages, a true claim started flagging because the verbatim slide passages fell out of
the retrieved top-k and the summary lacked the detail. Always put verbatim source passages ahead of
summaries, and when a true-looking claim flags, re-judge it against the verbatim set before calling
it an error.

## Repeatability

At claim scale (short prompts) two identical no-think passes agreed on 343 of 344 items, and
flag-on-either scored the same as flag-on-both. At deck scale (~4k-token prompts across 2 llama.cpp
slots) the server is **not** deterministic at temperature 0: a control passed on one run and was
caught with identical documents on the next. So for long prompts, judge twice and flag on either;
require all controls caught on both passes. For short claims, spend the time on think mode instead.

## Trusted vs untrusted sources

`combine=and` (screen AND judge agree) is the accurate setting and the right one for material
whose provenance you control: your own documents, your curated library. `combine=or` flags what
either detector finds. It costs about three times the false positives, but the two detectors fail
to *different* injection attacks, so no single poisoned passage flips both. Use `or` for scraped
web pages, third-party PDFs and anything a client supplied. In `or` mode the provenance marks
(`both` / `screen only` / `judge only`) separate the real fabrications from the screen's noise
visually: in testing the real fabrications came back as `both` and the extra false positives as
`screen only`.

## Where the judges are weak

Both are weakest on long documents with derived numbers (FinanceBench 0.65, DROP 0.67 balanced
accuracy, versus 0.95 on clean short-passage QA). Better retrieval did not help: with a reranker
the right passage reached the judge 98.8% of the time instead of 92.2% and accuracy did not move.
The weakness is in the judge's reasoning over numbers, not in finding the passage. The
recomputation bundle above, and the deterministic number rule, exist to compensate for it.
