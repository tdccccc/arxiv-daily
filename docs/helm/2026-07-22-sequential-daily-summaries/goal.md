# Sequential daily paper summaries

status: done
updated: 2026-07-22

## Intent

Generate each selected paper's daily summary in a separate, strictly sequential structured LLM call, then assemble the complete daily Markdown deterministically in code.

## Success criteria

- [x] N selected papers trigger N daily-summary calls in input order with maximum concurrency 1.
- [x] Each response is strictly validated structured data and every selected unique paper appears in the assembled daily report.
- [x] Any final per-paper failure aborts the whole day before writing a daily file.
- [x] Chinese/English labels, topic order, detail links, index summaries, metrics, and cancellation remain compatible.
- [x] Existing section-aware evidence limits remain unchanged and the full verification suite passes.

## Non-goals

- Redesigning HTML or LaTeX section extraction and evidence ranking.
- Changing detail-note generation or adding per-model context-window budgets.
- Writing partial daily reports or placeholder paper summaries after an LLM failure.

## Constraints

- Daily per-paper LLM concurrency is exactly 1.
- Preserve the durable boundary that an existing daily file represents a complete committed report.
- Keep existing `paperCharLimit` and `sectionCharLimit` defaults unchanged.
- Retain `dailyCharLimit` for configuration compatibility even though daily batching will no longer use it.

## Phases

1. P1 — Structured summaries can be assembled into deterministic compatible daily Markdown — status: done
2. P2 — One paper can be summarized through a strictly validated language-specific JSON contract — status: done
3. P3 — The pipeline summarizes all papers sequentially and fails atomically — status: done
4. P4 — Regression migration and full verification complete the initiative — status: done
