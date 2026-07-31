## 2026-07-31 — L1 adjust

- evidence: deleting the `dateWindowNote` option from
  `MarkdownWriter.writeDaily` broke markdown-writer.test.ts:351
  ("writeDaily includes submitted-date fallback notes"); initial grep had
  excluded tests, so the consumer was hidden
- change: reverted markdown-writer.ts to original; T1 now only removes the
  dead `const dateWindowNote = undefined` binding in pipeline.ts and the
  `dateWindowNote` property from the writeDaily call
- disposition: writer option + render hook + test kept (tested dormant
  capability; consider wiring it up or removing it in a later phase)
- next: re-run checks, commit P1
