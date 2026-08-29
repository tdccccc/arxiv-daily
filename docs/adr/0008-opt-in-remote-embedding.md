# ADR 0008: Opt-in remote embedding with full-text processing consent

Status: Accepted (2026-08-10 Helm P6 follow-up design)

Related: ADR 0001 (one TypeScript core, two hosts); ADR 0004 (personal-library-guided discovery); ADR 0005 (scoped desktop library access); ADR 0006 (unified search entry).

## Context

Local CPU embedding (multilingual-e5-small q8 via transformers.js wasm, single-threaded) makes the first full-library index take hours: measured ~0.3-1s per chunk, and a 134-paper library is ~7000 chunks. The `EmbeddingModel` port is host-neutral, so a remote implementation is architecturally clean, and remote embedding APIs (OpenAI text-embedding-3-small, BGE-family APIs) would cut the first index to minutes at negligible cost (a one-time ~1M tokens ≈ a few cents).

Two facts frame the decision:

- **Embedding models are not chat LLMs.** DeepSeek-class chat APIs (the plugin's LLM provider) do not expose embeddings; "use a cheap LLM for indexing" is not a substitute for the similarity layer. The remote path means a remote *embedding* model.
- **Remote embedding sends full-text chunks off the machine.** The library is primarily public arXiv papers, but the glossary's *Personal literature library* allows explicitly included non-public files (drafts, notes), and the reading set itself is personal research-interest data. The existing consent model ("processing expands from metadata and abstracts to full text") already anticipates this boundary.

## Decision

1. **Keep both, choose at first run.** Local embedding stays a first-class option (offline, private, slow). Remote embedding is an explicit opt-in, offered as a guided choice when a library is first prepared for indexing — before the user hits the hours-long local index. The choice is switchable later in settings; switching embedding models rebuilds the knowledge base (hours), which the guided choice discloses.
2. **Disclosure through the existing consent flow.** Enabling remote embedding requires model-processing authorization at full-text depth; the authorization modal discloses that full-text chunks are sent to the named embedding endpoint to generate similarity vectors. No lighter "just a notice" path: the consent machinery (endpoint change → re-ask, revocation → cancel running operations) already exists and is reused.
3. **What leaves the machine: all full-text chunks.** A partial-depth option (title+abstract remote, full text local) does not solve the bottleneck and is not offered.
4. **Per-paper failure, never mixed models.** Remote failures mark papers `failed` and retry on the next run (existing mechanism). The knowledge base keeps a single `modelId`; vectors from two models are never mixed (cross-model cosine is meaningless).
5. **Separate configuration, multi-endpoint authorization.** Embedding settings (provider/baseUrl/apiKey/model) live in their own settings section, independent of the LLM section. The library authorization record extends to multiple endpoints: the embedding endpoint fingerprint joins the LLM endpoint fingerprint; changing either re-asks consent.
6. **Scope.** Only the remote-embedding feature is implemented now. Local multi-worker parallelism remains a separate, later evaluation.

## Consequences

- Remote users get a minutes-long first index; privacy-sensitive users keep the local path (hours, segmentable across runs via the reuse mechanism).
- The consent model gains a full-text depth for a second named endpoint; revocation and endpoint-change semantics stay unified.
- The non-goal "remote embedding API (optional switch reserved for the future)" is realized as the reserved opt-in — "default fully local" still holds.
