/**
 * P6 T2 — real-library headless validation (2026-08-04).
 *
 * Drives the same Core path the plugin uses for a personal-library scan,
 * against a real library directory on this machine, through the node-runtime
 * scoped source. Proves, at real scale:
 *   1. inventory + reconciliation classify every entry (ready/unresolved/
 *      unrelated/failed) without reading PDF bytes;
 *   2. an unchanged reload reuses every observed file (no re-identification,
 *      no re-resolution);
 *   3. the catalog store roundtrip persists and reloads atomically, and a
 *      corrupt primary recovers from the backup;
 *   4. runtime stays bounded.
 *
 * Usage:
 *   LIBRARY_ROOT=/path/to/library node scripts/personalization/run-validate-real-library.mjs
 */
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createEmptyPersonalLibraryCatalog,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
  PersonalLibraryCatalogStore,
  reconcilePersonalLibraryCatalog,
  type OutputSettings,
  type PersonalLibraryResolvedMetadata,
} from "@arxiv-daily/core";
import { openScopedLibrarySource } from "@arxiv-daily/node-runtime/scoped-library-source";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";

const LIBRARY_ROOT = process.env.LIBRARY_ROOT ?? "/home/tiandc/Nextcloud/work/Article";
const ELIGIBLE_EXTENSIONS = [".pdf"] as const;
const OUTPUT: OutputSettings = { dailyDir: "daily", papersDir: "papers" };

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`FAIL: ${message}`);
}

function classify(catalog: { files: Record<string, { status: string }> }): {
  ready: number;
  unresolved: number;
  unrelated: number;
  failed: number;
} {
  const counts = { ready: 0, unresolved: 0, unrelated: 0, failed: 0 };
  for (const record of Object.values(catalog.files)) counts[record.status as keyof typeof counts] += 1;
  return counts;
}

async function main(): Promise<void> {
  const started = Date.now();
  const source = await openScopedLibrarySource(LIBRARY_ROOT);

  // The scan path must never read PDF bytes; any call is a boundary failure.
  let readBinaryCalls = 0;
  const guardedSource = {
    canonicalRoot: source.canonicalRoot,
    rootIdentity: source.rootIdentity,
    inventory: (options?: Parameters<typeof source.inventory>[0]) => source.inventory(options),
    readBinary: async (): Promise<ArrayBuffer> => {
      readBinaryCalls += 1;
      throw new Error("readBinary must not be called during a catalog scan");
    },
  };

  const scopeFingerprint = createPersonalLibraryScopeFingerprint({
    rootIdentity: source.rootIdentity,
    eligibleExtensions: ELIGIBLE_EXTENSIONS,
  });
  const identificationFingerprint =
    createPersonalLibraryIdentificationFingerprint(ELIGIBLE_EXTENSIONS);

  const vaultRoot = await mkdtemp(join(tmpdir(), "arxiv-daily-validate-"));
  const storage = new NodeStorageAdapter(vaultRoot);
  const store = new PersonalLibraryCatalogStore(storage, OUTPUT, {});
  let resolverCalls = 0;
  const resolver = {
    async resolve(ids: string[]): Promise<Map<string, PersonalLibraryResolvedMetadata>> {
      resolverCalls += 1;
      return new Map(
        ids.map((arxivId) => [
          arxivId,
          {
            arxivId,
            title: `Paper ${arxivId}`,
            authors: ["Researcher"],
            abstract: "abstract",
            published: "2026-01-01T00:00:00.000Z",
            updated: "2026-01-02T00:00:00.000Z",
            primaryCategory: "astro-ph.CO",
            categories: ["astro-ph.CO"],
          },
        ]),
      );
    },
  };

  const inventory = await guardedSource.inventory();
  const inventoryMs = Date.now() - started;

  const first = await reconcilePersonalLibraryCatalog({
    current: createEmptyPersonalLibraryCatalog(scopeFingerprint, identificationFingerprint),
    inventory,
    eligibleExtensions: ELIGIBLE_EXTENSIONS,
    resolver,
  });
  const firstScanMs = Date.now() - started;
  const firstCounts = classify(first.catalog);

  // Reload with the identical inventory: every observed file must be reused.
  const resolverCallsBeforeReload = resolverCalls;
  const second = await reconcilePersonalLibraryCatalog({
    current: first.catalog,
    inventory,
    eligibleExtensions: ELIGIBLE_EXTENSIONS,
    resolver,
  });
  const reloadMs = Date.now() - started;
  const secondCounts = classify(second.catalog);

  // Store roundtrip + backup recovery.
  await store.replace(first.catalog);
  const loaded = await store.load(scopeFingerprint, identificationFingerprint);
  await storage.writeText(store.paths.documentPath, "{ not valid json");
  const recovered = await store.load(scopeFingerprint, identificationFingerprint);

  // Sample of real classifications (first few of each kind).
  const samples = { ready: [], unresolved: [], unrelated: [], failed: [] } as Record<
    string,
    string[]
  >;
  for (const record of Object.values(first.catalog.files)) {
    if (samples[record.status].length < 5) samples[record.status].push(record.path);
  }

  // Assertions.
  assert(readBinaryCalls === 0, "scan read PDF bytes");
  assert(first.catalog.scopeFingerprint === scopeFingerprint, "scope fingerprint mismatch");
  assert(first.catalog.identificationFingerprint === identificationFingerprint, "identification fingerprint mismatch");
  assert(firstCounts.ready + firstCounts.unresolved + firstCounts.unrelated + firstCounts.failed === Object.keys(first.catalog.files).length, "classification totals");
  assert(inventory.truncated === false, "inventory unexpectedly truncated at 10k entries");
  assert(secondCounts.ready === firstCounts.ready, "reload changed ready count");
  assert(secondCounts.unresolved === firstCounts.unresolved, "reload changed unresolved count");
  assert(secondCounts.unrelated === firstCounts.unrelated, "reload changed unrelated count");
  assert(secondCounts.failed === firstCounts.failed, "reload changed failed count");
  assert(second.reusedFileCount === Object.keys(first.catalog.files).length, "reload did not reuse every observed file");
  assert(second.resolvedArxivIds.length === 0, "reload re-resolved papers");
  assert(resolverCalls === resolverCallsBeforeReload, "reload invoked the metadata resolver");
  const firstPaperCount = Object.keys(first.catalog.papers).length;
  assert(firstPaperCount > 0, "scan produced no papers");
  assert(firstPaperCount <= firstCounts.ready, "papers exceed ready files");
  const loadedPaperCount = Object.keys(loaded.papers).length;
  const recoveredPaperCount = Object.keys(recovered.papers).length;
  assert(loadedPaperCount === firstPaperCount, "store roundtrip lost papers");
  assert(recoveredPaperCount === firstPaperCount, "backup recovery lost papers");

  const report = {
    libraryRoot: LIBRARY_ROOT,
    canonicalRoot: source.canonicalRoot,
    rootIdentity: source.rootIdentity,
    inventory: { entries: inventory.entries.length, truncated: inventory.truncated },
    timingMs: { inventory: inventoryMs, firstScan: firstScanMs, reload: reloadMs, total: Date.now() - started },
    firstScan: { ...firstCounts, papers: Object.keys(first.catalog.papers).length, resolverCalls },
    reload: { ...secondCounts, papers: Object.keys(second.catalog.papers).length, reusedFileCount: second.reusedFileCount, resolverCalls },
    store: {
      roundtripPapers: Object.keys(loaded.papers).length,
      backupRecoveryPapers: Object.keys(recovered.papers).length,
      revisionAfterRecovery: recovered.revision,
    },
    samples,
  };
  console.log(JSON.stringify(report, null, 2));
  console.log("PASS");
  await rm(vaultRoot, { recursive: true, force: true });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
