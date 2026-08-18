# The SweepCF progress board

`sweepcf_board.html` is the page published as the *SweepCF Progress Board* artifact. It carries the
claim ladder, the operator gate table, the selector's control results, and the current paper draft
rendered inline from `docs/paper/sweepcf_draft.md`.

```bash
python3 docs/progress/refresh_board.py     # re-render the draft into the page
```

The refresh **replaces** the draft section rather than inserting one, so it is safe to run
repeatedly. An earlier version inserted, which silently produced a second copy of the draft on
every run — and because the insertion happened at module import time, merely importing the
renderer mutated the page. The rendering helpers now live in `md_render.py`, which does nothing
when imported.

The script refuses to write when the renderer produces an empty body. That guard exists because a
markdown renderer bug once rendered nothing at all, and the only reason it was caught was an
unchanged byte count.
