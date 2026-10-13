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
    };
  });
  if (authorized) await driver.authorizeLibraryProcessing();
  await driver.openDashboard();
  return { libraryRoot, rootIdentity };
}
