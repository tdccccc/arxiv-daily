/**
 * P3 Chunk 1 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/03-obsidian-library-tasks.md):
 * a small, reusable library-fixture start for Obsidian exploration tasks, shared by every personal-
 * library goal task (this chunk's consent/revoke tasks, and later chunks' index-build/search/similarity/
 * review tasks). Built entirely on top of the desktop-acceptance builders (`connectedLibraryFixture`/
 * `readRootIdentity`/`resolveLibraryRoot`, `settings-fixture.mjs`) and the exploration driver's own
 * `applySettingsPatch`/reload path — no parallel implementation of either.
 */
import { connectedLibraryFixture, readRootIdentity, resolveLibraryRoot } from "../../desktop-acceptance/settings-fixture.mjs";

/** fixtures.mjs's `/embeddings` handler returns 3-dimensional vectors; remote embedding settings must declare that dimension, not the exploration default (384). */
export const LIBRARY_EMBEDDING_DIMENSION = 3;

/**
 * P3 Chunk 2: `connectedLibraryFixture`'s own embedding object always carries `apiKey: ""`; Chunk 1's
 * consent/revoke tasks never actually ran the embedding pipeline, so this was never exercised. Chunk 2's
 * tasks do run it (indexing/search), and `validateEmbeddingConfig` (packages/core/src/settings/
 * validation.ts) rejects a remote config with an empty API key before any request is sent — so a real
 * key-shaped fixture value is required, never the model's own claim of one. The embedding API key field
 * is itself restricted from the model (classifyControlRestriction's credentials rule), so this is exactly
 * the kind of value the phase's own assumptions leave as fixture data.
 */
const LIBRARY_EMBEDDING_API_KEY = "fixture-embedding-key";

/**
 * Connects the library to the fixture's own `vault/library` folder (the PDFs `createFixtureEnvironment`
 * already writes there — never a real personal library), sets Embedding mode and, when remote, points
 * its base URL at the fixture server's own origin so fixtures.mjs's `/embeddings` handler serves it
 * directly (the same `${driver.fixture.server.url}/v1` convention `createObsidianSettings` already uses
 * for the LLM endpoint — no HTTP proxy indirection needed), with `dimension` set to
 * `LIBRARY_EMBEDDING_DIMENSION` to match the fixture's actual vectors.
 *
 * `authorized: true` grants library-processing consent through the plugin's own
 * `authorizeLibraryProcessing()` (via `driver.authorizeLibraryProcessing()`, CDP — never a hand-written
 * fingerprint), for a task that needs to start already-granted rather than driving the consent dialog
 * itself. Local embedding needs no consent, so `authorized` only makes sense alongside
 * `embeddingMode: "remote"`.
 *
 * Leaves the dashboard open afterward, the same starting surface every other task's `prepare()` leaves
 * (matching `applySettingsPatch`'s own "leaves no plugin surface open" contract, resolved here rather
 * than left to each call site).
 */
export async function startLibraryFixture(driver, { embeddingMode = "local", authorized = false } = {}) {
  if (!["local", "remote"].includes(embeddingMode)) {
    throw new TypeError(`startLibraryFixture: embeddingMode must be "local" or "remote", received ${JSON.stringify(embeddingMode)}`);
  }
  if (authorized && embeddingMode !== "remote") {
    throw new Error("startLibraryFixture: authorized only makes sense with embeddingMode \"remote\" — local embedding needs no consent to grant");
  }
  const libraryRoot = await resolveLibraryRoot({ vaultPath: driver.fixture.vaultRoot });
  const rootIdentity = await readRootIdentity(libraryRoot);
  const fixtureData = connectedLibraryFixture({ libraryRoot, rootIdentity, embeddingMode });
  await driver.applySettingsPatch(current => {
    current.libraryConnection = fixtureData.libraryConnection;
    current.settings.embedding = {
      ...current.settings.embedding,
      ...fixtureData.settings.embedding,
      baseUrl: `${driver.fixture.server.url}/v1`,
      dimension: LIBRARY_EMBEDDING_DIMENSION,
      // Only set once the fixture actually configures remote embedding (embeddingMode "remote"); a task
      // that starts local (e.g. library-remote-consent) must still start with an empty key, since
      // supplying one itself is part of what the model's own switch-to-remote flow is not asked to prove.
      ...(embeddingMode === "remote" ? { apiKey: LIBRARY_EMBEDDING_API_KEY } : {}),
    };
  });
  if (authorized) await driver.authorizeLibraryProcessing();
  await driver.openDashboard();
  return { libraryRoot, rootIdentity };
}
