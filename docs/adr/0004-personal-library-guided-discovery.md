# ADR 0004: Personal-library-guided arXiv discovery

Status: Accepted (2026-08-03 grill-with-docs)

Related: ADR 0001 (one TypeScript core, two hosts); ADR 0002 (paper identity and source boundary); ADR 0003 (two products and vault data semantics).

## Context

arXiv Daily currently follows a reliable daily workflow: fetch new arXiv papers, filter them with user-written categories and topics, summarize selected papers, save Markdown, and support later review. A researcher's existing literature library is not yet part of that loop.

A typical personal library is not a curated knowledge graph. It is often a local directory containing mostly PDFs, with incomplete metadata, occasional Markdown, old research directions, and files that do not represent current interests. Requiring the researcher to reorganize or annotate every paper before personalization would make adoption impractical.

We considered three broad directions:

1. Build a general-purpose Research Agent and expose the library as tools or RAG context.
2. Fully parse and embed the whole library before providing personalized results.
3. Extend the existing daily workflow with progressive, researcher-confirmed understanding of the library, adding deeper retrieval only when proven necessary.

The product should first improve a real literature workflow rather than optimize for visible Agent autonomy. The initial value is helping the researcher decide which new papers deserve attention relative to what they already have.

## Decision

### 1. Connect the daily arXiv stream to a personal literature library

The product will support a researcher-chosen **personal literature library** that may live inside or outside the Obsidian Vault. The researcher is not required to migrate it into arXiv Daily's output layout.

The target end-to-end product loop is below. The first implementation initiative delivers steps 1–6; steps 7–9 are later possibilities rather than part of the initial scope.

1. Select a local literature-library directory.
2. Build a paper-level catalog progressively, using metadata and abstracts first.
3. Propose research directions and representative papers.
4. Let the researcher adjust and confirm representative sets and directions.
5. Combine confirmed library-derived directions with manually configured topics for daily discovery.
6. Explain why each selected paper entered the report and what it appears to add relative to named prior works.
7. Let the researcher save valuable discoveries as reading candidates.
8. Periodically review candidates by research direction and make reading decisions.
9. Use lightweight reading dispositions to improve later discovery.

### 2. Make library understanding progressive and inspectable

- Initial cataloging prioritizes title, authors, year, identifiers, abstract, file location, and other inexpensive metadata.
- The product does not require full-text parsing or whole-library chunk embeddings before it becomes useful.
- The product proposes candidate directions and small representative sets; the researcher does not manually select from the entire library.
- A library-derived direction affects daily filtering only after the researcher confirms, corrects, merges, disables, or otherwise accepts it.
- Old or incidentally downloaded papers do not automatically become equally weighted current interests.

### 3. Preserve explicit topics and show discovery provenance

Manual topics and confirmed library-derived directions are complementary discovery sources.

- Daily selection uses their **union**: matching either source may admit a paper.
- Each selected paper shows whether it matched a manual topic, a confirmed library direction, or both.
- A library-derived match names the relevant direction and representative prior papers.
- Personalization must not appear as an unexplained black-box score.

### 4. Rank by relevance and personal novelty

After determining relevance, the product also estimates **personal novelty** relative to the relevant representative set.

- Novelty is expressed as a difference type such as a new task, method, dataset, experiment, efficiency result, or counter-evidence.
- The explanation names its comparison basis and evidence depth.
- Metadata- or abstract-level evidence must not be presented as if it were a full-text finding.
- Unsupported claims such as “challenges” or “supersedes” are not allowed merely because papers are topically similar.

### 5. Treat saving as a reading decision, not endorsement

The default save action creates a **reading candidate**, not confirmed research knowledge.

A reading candidate preserves paper identity, source, discovery reason, related prior works, provisional novelty evidence, direction, and save date. Saving does not imply that the researcher has read the paper or endorsed an automatically generated summary.

Periodic **direction reviews** primarily help the researcher decide which candidates deserve close reading, skimming, or dismissal. After reading or review, lightweight dispositions—relevant or irrelevant, already known or additive, keep or dismiss, with an optional short judgment—provide feedback without requiring a full paper note.

### 6. Do not include autonomous Agent or full RAG in the first implementation initiative

The first implementation initiative uses deterministic workflow orchestration, paper-level retrieval, and bounded LLM generation. It does not include:

- an autonomous reason-act-observe loop;
- full-library chunking and vector indexing;
- automatic full-text comparison of every daily paper;
- a general-purpose chat experience.

A later initiative may evaluate bounded Agent execution and on-demand full-text retrieval when a researcher explicitly asks to deepen a relationship or novelty explanation. These are not current commitments and, if adopted, must remain subordinate to the user workflow and evidence boundary.

### 7. Use informed library-processing consent

- Setup shows the configured model endpoint, selected library scope, eligible content types, and processing depth before library content is sent to that endpoint.
- Confirmation authorizes eligible library-paper content at the disclosed depth; the product does not ask for every file individually.
- Changing the endpoint or expanding from metadata and abstracts to full text requires renewed confirmation, and authorization can be revoked.
- Building a local catalog and sending content to a model endpoint are distinct actions.
- By default, only files identifiable as papers are included.
- Drafts, notes, and other files are ignored unless the researcher explicitly includes them.

### 8. Deliver through the plugin first, share domain logic in core

The Obsidian plugin is the first product host for the initial personalized-daily initiative. If later approved, reading candidates and direction reviews should extend that same product experience.

Host-neutral catalog, profile, discovery, and evidence semantics belong in the shared TypeScript core so later CLI surfaces can reuse them without creating a parallel model. Reading-state semantics move into core only when that later initiative is approved.

### 9. End the first implementation initiative at the personalized daily report

The first implementation initiative delivers:

1. literature-library directory selection and processing consent;
2. paper-level cataloging from metadata and abstracts;
3. proposed directions and representative papers;
4. researcher confirmation of the interest profile;
5. union filtering with manual topics;
6. visible discovery sources and abstract-level personal-novelty explanations in the daily report.

Reading candidates, direction reviews, full-text RAG, autonomous Agent execution, and MCP remain later initiatives.

The primary user acceptance test is whether the library-derived directions surface valuable papers that the researcher's manual topics would have missed, with understandable reasons. Reducing irrelevant papers and topic-maintenance effort are useful secondary outcomes, not the first criterion.

## Consequences

- The immediate product investment is personalized daily discovery, not a generic Agent runtime or vector database.
- Existing manually configured topics remain useful and are not replaced by an inferred profile.
- Recommendation quality is diagnosable: failures can be traced to paper identification, representative-set selection, direction inference, relevance judgment, or novelty evidence.
- The product can provide value to users whose libraries are mostly PDFs and poorly organized.
- Initial setup requires a review step before personalization becomes active, trading a small amount of onboarding effort for control and trust.
- Supporting libraries outside the Vault requires host-specific filesystem capability and explicit permission handling, especially in the plugin.
- Paper-level cataloging and retrieval must be designed before fine-grained RAG. Full-text parsing, chunking, embeddings, and reranking remain optional later projections.
- Reading candidates and dispositions add lifecycle states beyond the current star/index model; their exact storage representation is an implementation decision.
- Generated novelty explanations are provisional evidence, not user-authored knowledge.

## Non-decisions

- The exact PDF parser, metadata providers, embedding model, vector store, or reranker.
- Whether paper-level retrieval initially uses lexical search, embeddings, or a hybrid.
- Exact UI layout for direction confirmation and periodic review.
- Exact cadence or notification mechanism for direction reviews.
- The later Agent runtime, MCP transport, or full-text RAG architecture.
- Whether CLI support ships in the same or a later release.
