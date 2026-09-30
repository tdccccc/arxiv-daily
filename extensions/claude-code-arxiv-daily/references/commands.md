# Research command reference

```text
node "${CLAUDE_PLUGIN_ROOT}/dist/arxiv-agent.cjs" COMMAND --workspace PATH [--input REQUEST.json]
```

The input is one JSON object from `--input` or stdin. `status` needs no fields. Missing input means `{}`. stdout is one JSON result; diagnostics use stderr. Use an absolute workspace consistently. The helper has no model key, background daemon, or implicit global configuration.

| Command | Input | Result |
|---|---|---|
| `init` | `library` absolute path; optional `includeMarkdown: true` | Connects once; refuses a different source or scope in an existing workspace |
| `status` | `{}` | Connected library; saved record descriptors; `confirmedDirections`; unreadable and truncated record groups |
| `library` | optional `query` (filename substring), `offset` (default 0), `limit` (1–100, default 30) | Paths, file sizes, filename-derived paper keys, evidence depth, `nextOffset`, `inventoryTruncated` |
| `recent` | `category`; optional `date: YYYY-MM-DD`, `offset`, `limit` | Announcement dates, paginated metadata, `state`, `total`, `nextOffset`; omitted date explicitly selects latest available |
| `paper` | `id` (modern arXiv ID or arxiv.org URL); optional `fullText: true` | Metadata and abstract; optional extracted sections up to 60,000 characters with 12,000 per section; source and failure details |
| `save` | `kind`, `slug`, `title`, `body`, `sources`; optional `expectedSha256` for updates | Saved Markdown, path, hash, status; directions are drafts until confirmed |
| `read` | `kind`, `slug` | Existing Markdown and hash |
| `confirm-direction` | `slug`, `expectedSha256` | Marks the current direction draft confirmed; invoke only after user acceptance |

`kind` is `direction`, `paper`, `reading`, or `daily`. Slugs use lowercase ASCII letters, digits, hyphens, underscores, or dots; no traversal or reserved device names. A modern arXiv ID is a suitable paper/reading slug and an ISO date is a suitable daily slug. Non-arXiv papers can use a short descriptive slug; retain their actual source path in `sources`.

The Markdown body includes its own heading and readable source links. `sources` is an array of at most 100 source strings, separate from the body. This minimal record format does not independently verify scholarly claims or prove human approval. The skill carries those interaction responsibilities.

Example save request:

```json
{
  "kind": "paper",
  "slug": "2606.12345",
  "title": "Paper title from the actual source",
  "body": "# Paper title\n\n## Evidence scope\nAbstract only.\n\n## Summary\nWrite the actual analysis here.\n\n## Sources\nhttps://arxiv.org/abs/2606.12345",
  "sources": ["arxiv:2606.12345"]
}
```

The original library is scanned read-only, excluding symbolic links. `library` does not parse PDF contents; Claude's Read tool handles selected PDFs. The prototype record directory is `<workspace>/arxiv-daily-agent/`, with `directions/`, `papers/`, `reading/`, and `daily/`. `.workspace.json` holds only the source connection; `.cache/` holds optional downloaded source HTML; `.locks/` coordinates writers on this computer.

Do not use this directory as input to the existing Obsidian/CLI data-import tool. There is no automatic profile or index synchronization. Use a trusted local workspace; file validation does not defend against a hostile concurrent process changing paths between checks. File caches and research records can contain private research interests and excerpts.
