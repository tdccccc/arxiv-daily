import type { StorageAdapter } from "../core/adapters";
import {
  normalizePaperDiscoveryProvenance,
} from "../pipeline/discovery-provenance-marker";
import type { PaperDiscoveryProvenance } from "../pipeline/personalized-paper-filter";
import { normalizePersonalNovelty } from "../pipeline/personalized-novelty";
import type { PersonalNovelty } from "../pipeline/personalized-novelty";
import type { OutputSettings } from "../settings/types";
import {
  portablePathCollisionKey,
  validateVaultRelativeDirectory,
  vaultRelativeDirectoriesCollide,
} from "../settings/validation";
import { modernArxivResources } from "../utils/arxiv";
import {
  formatPaperKey,
  paperKeyFromArxivId,
  parsePaperKey,
  resolvePaperLookupKey,
  tryParsePaperKey,
} from "./paper-key";

/** On-disk schema: map keys are paperKey (`source:externalId`). */
export const PAPER_INBOX_SCHEMA_VERSION = 5;

export type PaperStatus =
  | "inbox"
  | "to_read"
  | "reading"
  | "read"
  | "saved"
  | "ignored";

export type PaperPriority = "low" | "normal" | "high";

export interface PaperInbox {
  schemaVersion: number;
  updatedAt: string;
  papers: Record<string, PaperIndexEntry>;
}

export interface PaperSummary {
  sourceSections?: string;
  coreProblem?: string;
  keyMethod?: string;
  mainResult?: string;
  whyRelevant?: string;
  limitations?: string;
}

export interface PaperIndexEntry {
  /** Stable identity: `source:externalId` (e.g. `arxiv:2606.12345`). */
  paperKey: string;
  /** Source namespace, lowercase `[a-z0-9_]+`. */
  source: string;
  /** Source-local id; also used as short path stem (never embed paperKey in paths). */
  externalId: string;
  /**
   * Compatibility alias for arXiv: same as externalId when source is arXiv.
   * Prefer paperKey / externalId for new code.
   */
  arxivId: string;
  title: string;
  authors: string[];
  published: string;
  updated: string;
  category: string;
  categories?: string[];
  abstract?: string;
  summary?: PaperSummary;
  topics: string[];
  primaryTopic: string;
  detail: boolean;
  status: PaperStatus;
  priority: PaperPriority;
  seenDates: string[];
  dailyReports: string[];
  /** Occurrence projection keyed by normalized committed daily-report path. */
  discoveryProvenanceByReport: Record<string, PaperDiscoveryProvenance>;
  /** Validated personal-novelty occurrence projection keyed by normalized committed daily-report path. */
  noveltyByReport: Record<string, PersonalNovelty>;
  /** Vault-relative note path; stem is externalId, not paperKey. */
  paperPath: string | null;
  arxivUrl: string;
  pdfUrl: string;
  /** Vault-relative PDF path; stem is externalId, not paperKey. */
  pdfPath: string;
  zoteroKey: string;
  zoteroUri: string;
  citationKey: string;
  projects: string[];
}

export interface PaperInboxPaths {
  rootDir: string;
  indexDir: string;
  papersJsonPath: string;
  legacyIndexDir: string;
  legacyPapersJsonPath: string;
}

export interface PaperIndexInspection {
  inbox: PaperInbox;
  /** Validated on-disk document before schema/key normalization. */
  document: unknown | null;
  sourcePath: string | null;
  recoveredFromBackup: boolean;
}

export interface PaperIndexMutationResult<T> {
  result: T;
  changed: boolean;
}

export interface PaperIndexMutation {
  upsertManyFromDailyPapers(
    inputs: PaperIndexUpsert[],
  ): Array<{ entry: PaperIndexEntry; wasNew: boolean }>;
  setSummaries(summaries: Record<string, PaperSummary>): number;
}

export interface PaperIndexUpsert {
  arxivId: string;
  title: string;
  authors: string | string[];
  date: string;
  published?: string;
  updated?: string;
  arxivCategory: string;
  arxivCategories?: string[];
  abstract?: string;
  primaryTopic: string;
  detail: boolean;
  dailyReport?: string;
  paperPath?: string | null;
}

export type PaperDetailsRemovalResult =
  | { kind: "cleared"; entry: PaperIndexEntry }
  | { kind: "removed"; entry: PaperIndexEntry }
  | { kind: "missing" }
  | { kind: "path_mismatch"; actualPath: string | null }
  | {
      kind: "index_failed";
      action: "cleared" | "removed";
      entry: PaperIndexEntry;
      error: unknown;
    };

export type PaperIndexErrorCode =
  | "configuration"
  | "read_failed"
  | "invalid_document"
  | "save_failed"
  | "invalid_input"
  | "operation_failed";

export type PaperIndexDiagnosticFailure =
  | "paper_index_configuration_invalid"
  | "paper_index_unreadable"
  | "paper_index_invalid"
  | "paper_index_save_failed"
  | "paper_index_invalid_input"
  | "paper_index_unavailable";

export class PaperIndexError extends Error {
  constructor(
    message: string,
    readonly cause?: unknown,
    readonly code: PaperIndexErrorCode = "operation_failed",
  ) {
    super(message);
    this.name = "PaperIndexError";
  }
}

/** Return a stable, content-free category suitable for diagnostics and logs. */
export function classifyPaperIndexFailureForDiagnostics(
  error: unknown,
): PaperIndexDiagnosticFailure {
  if (!(error instanceof PaperIndexError)) return "paper_index_unavailable";
  switch (error.code) {
    case "configuration":
      return "paper_index_configuration_invalid";
    case "read_failed":
      return "paper_index_unreadable";
    case "invalid_document":
      return "paper_index_invalid";
    case "save_failed":
      return "paper_index_save_failed";
    case "invalid_input":
      return "paper_index_invalid_input";
    default:
      return "paper_index_unavailable";
  }
}

const paperIndexMutationQueues = new Map<string, Promise<unknown>>();

export function derivePaperInboxPaths(
  output: OutputSettings,
  normalizePath: (path: string) => string = normalizeStoragePath,
): PaperInboxPaths {
  const dailyDir = requireOutputDirectory("dailyDir", output.dailyDir);
  const papersDir = requireOutputDirectory("papersDir", output.papersDir);
  if (vaultRelativeDirectoriesCollide(dailyDir, papersDir)) {
    throw new PaperIndexError(
      "dailyDir and papersDir must be different",
      undefined,
      "configuration",
    );
  }
  const dailyParent = parentDir(dailyDir, normalizePath);
  const papersParent = parentDir(papersDir, normalizePath);
  const root =
    dailyParent &&
    portablePathCollisionKey(dailyParent) === portablePathCollisionKey(papersParent)
      ? dailyParent
      : dailyParent || papersParent || "arxiv-daily";
  const rootDir = normalizePath(root);
  const indexDir = normalizePath(`${rootDir}/.index`);
  const legacyIndexDir = normalizePath(`${rootDir}/index`);
  return {
    rootDir,
    indexDir,
    papersJsonPath: normalizePath(`${indexDir}/papers.json`),
    legacyIndexDir,
    legacyPapersJsonPath: normalizePath(`${legacyIndexDir}/papers.json`),
  };
}

export class PaperIndexStore {
  readonly paths: PaperInboxPaths;

  constructor(
    private storage: StorageAdapter,
    output: OutputSettings,
    private now: () => Date = () => new Date(),
  ) {
    this.paths = derivePaperInboxPaths(output, (path) =>
      this.storage.normalizePath(path),
    );
  }

  async load(): Promise<PaperInbox> {
    return (await this.inspect()).inbox;
  }

  /** Uses the same classified, validated source selection as production reads. */
  async inspect(): Promise<PaperIndexInspection> {
    const selected = await this.readSelectedIndex();
    if (!selected) {
      return {
        inbox: emptyInbox(this.now()),
        document: null,
        sourcePath: null,
        recoveredFromBackup: false,
      };
    }
    return {
      inbox: selected.inbox,
      document: selected.document,
      sourcePath: selected.path,
      recoveredFromBackup: selected.path === `${this.paths.papersJsonPath}.bak`,
    };
  }

  /**
   * Queues a complete read-modify-validate-save transaction for this path.
   * The callback must report whether it changed the normalized inbox.
   */
  mutate<T>(
    mutation: (
      inbox: PaperInbox,
      operations: PaperIndexMutation,
    ) => PaperIndexMutationResult<T>,
  ): Promise<T> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const operations: PaperIndexMutation = {
        upsertManyFromDailyPapers: (inputs) =>
          inputs.map((input) => upsertEntry(inbox, input)),
        setSummaries: (summaries) => setSummariesInInbox(inbox, summaries),
      };
      const { result, changed } = mutation(inbox, operations);
      if (changed) await this.saveUnlocked(inbox);
      return result;
    });
  }

  async upsertFromDailyPaper(input: PaperIndexUpsert): Promise<{
    entry: PaperIndexEntry;
    wasNew: boolean;
  }> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const { entry, wasNew } = upsertEntry(inbox, input);
      await this.saveUnlocked(inbox);
      return { entry, wasNew };
    });
  }

  async upsertManyFromDailyPapers(
    inputs: PaperIndexUpsert[],
  ): Promise<Array<{ entry: PaperIndexEntry; wasNew: boolean }>> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const results = inputs.map((input) => upsertEntry(inbox, input));
      await this.saveUnlocked(inbox);
      return results;
    });
  }

  /**
   * Atomically creates or repairs the complete index projection for a verified
   * manual detail note. Existing user status is preserved; only an entry first
   * created by this mutation receives the requested initial status.
   */
  async reconcileManualDetail(
    input: PaperIndexUpsert,
    paperPath: string,
    intendedStatusForNew: PaperStatus = "saved",
  ): Promise<{ entry: PaperIndexEntry; wasNew: boolean }> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const { entry, wasNew } = upsertEntry(inbox, {
        ...input,
        detail: true,
        paperPath: this.storage.normalizePath(paperPath),
      });
      entry.detail = true;
      entry.paperPath = this.storage.normalizePath(paperPath);
      if (wasNew) entry.status = intendedStatusForNew;
      await this.saveUnlocked(inbox);
      return { entry, wasNew };
    });
  }

  async addDailyReports(ids: string[], dailyReport: string): Promise<void> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      for (const id of ids) {
        const entry = findEntry(inbox, id);
        if (!entry) continue;
        entry.dailyReports = appendUnique(entry.dailyReports, dailyReport);
      }
      await this.saveUnlocked(inbox);
    });
  }

  async reconcileDailyReportOccurrenceProvenance(
    dailyReport: string,
    occurrences: Array<{ arxivId: string; provenance: PaperDiscoveryProvenance }>,
  ): Promise<number> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const report = normalizeStoragePath(dailyReport);
      const next = new Map<string, PaperDiscoveryProvenance>();
      for (const occurrence of occurrences) {
        const provenance = normalizePaperDiscoveryProvenance(occurrence.provenance);
        if (!provenance) throw new PaperIndexError("invalid occurrence discovery provenance");
        next.set(paperKeyFromArxivId(occurrence.arxivId), provenance);
      }
      let changed = 0;
      for (const entry of Object.values(inbox.papers)) {
        const provenance = next.get(entry.paperKey);
        if (provenance) {
          if (JSON.stringify(entry.discoveryProvenanceByReport[report]) !== JSON.stringify(provenance)) {
            entry.discoveryProvenanceByReport[report] = provenance;
            changed += 1;
          }
        } else if (Object.prototype.hasOwnProperty.call(entry.discoveryProvenanceByReport, report)) {
          delete entry.discoveryProvenanceByReport[report];
          changed += 1;
        }
      }
      if (changed > 0) await this.saveUnlocked(inbox);
      return changed;
    });
  }

  /**
   * Sets or clears the per-report personal-novelty occurrence for every indexed
   * entry, mirroring reconcileDailyReportOccurrenceProvenance. Occurrences are
   * strictly normalized; a malformed occurrence rejects the whole mutation so
   * the derived index never persists partial projection state.
   */
  async reconcileDailyReportOccurrenceNovelty(
    dailyReport: string,
    occurrences: Array<{ arxivId: string; novelty: PersonalNovelty }>,
  ): Promise<number> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const report = normalizeStoragePath(dailyReport);
      const next = new Map<string, PersonalNovelty>();
      for (const occurrence of occurrences) {
        const novelty = normalizePersonalNovelty(occurrence.novelty);
        if (!novelty) throw new PaperIndexError("invalid occurrence personal novelty");
        next.set(paperKeyFromArxivId(occurrence.arxivId), novelty);
      }
      let changed = 0;
      for (const entry of Object.values(inbox.papers)) {
        const novelty = next.get(entry.paperKey);
        if (novelty) {
          if (JSON.stringify(entry.noveltyByReport[report]) !== JSON.stringify(novelty)) {
            entry.noveltyByReport[report] = novelty;
            changed += 1;
          }
        } else if (Object.prototype.hasOwnProperty.call(entry.noveltyByReport, report)) {
          delete entry.noveltyByReport[report];
          changed += 1;
        }
      }
      if (changed > 0) await this.saveUnlocked(inbox);
      return changed;
    });
  }

  async setStatus(id: string, status: PaperStatus): Promise<PaperIndexEntry | null> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const entry = findEntry(inbox, id);
      if (!entry) return null;
      entry.status = status;
      await this.saveUnlocked(inbox);
      return entry;
    });
  }

  async setPriority(
    id: string,
    priority: PaperPriority,
  ): Promise<PaperIndexEntry | null> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const entry = findEntry(inbox, id);
      if (!entry) return null;
      entry.priority = priority;
      await this.saveUnlocked(inbox);
      return entry;
    });
  }

  async setSummaries(
    summaries: Record<string, PaperSummary>,
  ): Promise<number> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const changed = setSummariesInInbox(inbox, summaries);
      if (changed > 0) await this.saveUnlocked(inbox);
      return changed;
    });
  }

  async setPaperPath(id: string, paperPath: string): Promise<PaperIndexEntry | null> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const entry = findEntry(inbox, id);
      if (!entry) return null;
      entry.paperPath = this.storage.normalizePath(paperPath);
      entry.detail = true;
      await this.saveUnlocked(inbox);
      return entry;
    });
  }

  async reconcilePaperDetails(
    paperPaths: Record<string, string | null>,
  ): Promise<number> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      let changed = 0;
      for (const [id, paperPath] of Object.entries(paperPaths)) {
        const entry = findEntry(inbox, id);
        if (!entry) continue;
        const normalizedPath = paperPath == null
          ? null
          : this.storage.normalizePath(paperPath);
        const detail = normalizedPath != null;
        if (entry.paperPath === normalizedPath && entry.detail === detail) continue;
        entry.paperPath = normalizedPath;
        entry.detail = detail;
        changed += 1;
      }
      if (changed > 0) await this.saveUnlocked(inbox);
      return changed;
    });
  }

  async clearPaperDetails(ids: string[]): Promise<number> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      let changed = 0;
      for (const id of uniqueStrings(ids)) {
        const entry = findEntry(inbox, id);
        if (!entry || (!entry.detail && entry.paperPath == null)) continue;
        entry.detail = false;
        entry.paperPath = null;
        changed += 1;
      }
      if (changed > 0) await this.saveUnlocked(inbox);
      return changed;
    });
  }

  async removePaperDetailsAtPath(
    id: string,
    expectedPaperPath: string,
    beforeMutation?: (entry: PaperIndexEntry) => Promise<void>,
  ): Promise<PaperDetailsRemovalResult> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const entry = findEntry(inbox, id);
      if (!entry) return { kind: "missing" };

      const expectedPath = this.storage.normalizePath(expectedPaperPath);
      const actualPath = entry.paperPath == null
        ? null
        : this.storage.normalizePath(entry.paperPath);
      if (actualPath !== expectedPath) {
        return { kind: "path_mismatch", actualPath };
      }

      const snapshot = { ...entry };
      const action = entry.dailyReports.length > 0 ? "cleared" : "removed";
      await beforeMutation?.(snapshot);
      if (action === "cleared") {
        entry.detail = false;
        entry.paperPath = null;
      } else {
        delete inbox.papers[entry.paperKey];
      }
      try {
        await this.saveUnlocked(inbox);
      } catch (error) {
        return { kind: "index_failed", action, entry: snapshot, error };
      }
      return { kind: action, entry: snapshot };
    });
  }

  async removePapers(ids: string[]): Promise<number> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      let changed = 0;
      for (const id of uniqueStrings(ids)) {
        const entry = findEntry(inbox, id);
        if (!entry) continue;
        delete inbox.papers[entry.paperKey];
        changed += 1;
      }
      if (changed > 0) await this.saveUnlocked(inbox);
      return changed;
    });
  }

  async setPdfPath(
    id: string,
    pdfPath: string,
  ): Promise<PaperIndexEntry | null> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const entry = findEntry(inbox, id);
      if (!entry) return null;
      entry.pdfPath = this.storage.normalizePath(pdfPath);
      await this.saveUnlocked(inbox);
      return entry;
    });
  }

  async addProject(
    id: string,
    projectPath: string,
  ): Promise<PaperIndexEntry | null> {
    return this.enqueueMutation(async () => {
      const inbox = await this.load();
      const entry = findEntry(inbox, id);
      if (!entry) return null;
      entry.projects = appendUnique(
        entry.projects,
        normalizeStoragePath(projectPath),
      );
      await this.saveUnlocked(inbox);
      return entry;
    });
  }

  /** Lookup by paperKey or bare arXiv id (compat). */
  async get(id: string): Promise<PaperIndexEntry | null> {
    const inbox = await this.load();
    return findEntry(inbox, id);
  }

  async listByStatus(status: PaperStatus): Promise<PaperIndexEntry[]> {
    const inbox = await this.load();
    return Object.values(inbox.papers)
      .filter((entry) => entry.status === status)
      .sort(compareEntries);
  }

  private async ensureDirDeep(dir: string): Promise<void> {
    const parts = this.storage.normalizePath(dir).split("/").filter(Boolean);
    let cur = "";
    for (const part of parts) {
      cur = cur ? `${cur}/${part}` : part;
      if (!(await this.storage.exists(cur))) {
        await this.storage.mkdir(cur);
      }
    }
  }

  private async readSelectedIndex(): Promise<SelectedPaperIndex | null> {
    const candidates = [
      this.paths.papersJsonPath,
      `${this.paths.papersJsonPath}.bak`,
      this.paths.legacyPapersJsonPath,
    ];
    let invalid: unknown;
    let found = false;
    for (const path of candidates) {
      const result = await this.readIndexDocument(path);
      if (result.kind === "missing") continue;
      found = true;
      if (result.kind === "unreadable") {
        throw new PaperIndexError(
          `failed to read paper index: ${path}`,
          result.error,
          "read_failed",
        );
      }
      if (result.kind === "valid") {
        return {
          path,
          raw: result.raw,
          document: result.document,
          inbox: result.inbox,
        };
      }
      invalid ??= result.error;
    }
    if (!found) return null;
    throw new PaperIndexError(
      `failed to load paper index: no valid document found${
        invalid instanceof Error ? `: ${invalid.message}` : ""
      }`,
      invalid,
      "invalid_document",
    );
  }

  private async readIndexDocument(path: string): Promise<PaperIndexReadResult> {
    let exists: boolean;
    try {
      exists = await this.storage.exists(path);
    } catch (error) {
      return { kind: "unreadable", error };
    }
    if (!exists) return { kind: "missing" };

    let raw: string;
    try {
      raw = await this.storage.readText(path);
    } catch (error) {
      return { kind: "unreadable", error };
    }
    try {
      const document: unknown = JSON.parse(raw);
      return {
        kind: "valid",
        raw,
        document,
        inbox: normalizeInbox(document, this.now()),
      };
    } catch (error) {
      return { kind: "corrupt", error };
    }
  }

  private async saveUnlocked(inbox: PaperInbox): Promise<void> {
    const next = normalizeInbox({
      ...inbox,
      schemaVersion: PAPER_INBOX_SCHEMA_VERSION,
      updatedAt: this.now().toISOString(),
      papers: { ...inbox.papers },
    }, this.now());
    const content = `${JSON.stringify(next, null, 2)}\n`;
    await this.ensureDirDeep(this.paths.indexDir);
    try {
      await this.replaceWithBackup(content);
    } catch (error) {
      if (error instanceof PaperIndexError) throw error;
      throw new PaperIndexError(
        `failed to save paper index: ${this.paths.papersJsonPath}${
          error instanceof Error ? `: ${error.message}` : ""
        }`,
        error,
        "save_failed",
      );
    }
    await this.removeLegacyIndexFile();
  }

  private async removeLegacyIndexFile(): Promise<void> {
    if (this.paths.legacyPapersJsonPath === this.paths.papersJsonPath) return;
    try {
      if (await this.storage.exists(this.paths.legacyPapersJsonPath)) {
        await this.storage.remove(this.paths.legacyPapersJsonPath);
      }
    } catch {
      // The hidden index has already been committed. Legacy existence checks
      // and removal are both best effort after that point.
    }
  }

  private enqueueMutation<T>(job: () => Promise<T>): Promise<T> {
    return enqueuePathMutation(
      paperIndexMutationQueues,
      this.paths.papersJsonPath,
      job,
    );
  }

  private async replaceWithBackup(content: string): Promise<void> {
    const primary = this.paths.papersJsonPath;
    const backup = `${primary}.bak`;
    const primaryTmp = `${primary}.tmp`;
    const backupTmp = `${backup}.tmp`;
    await this.removeIfExists(primaryTmp);
    await this.removeIfExists(backupTmp);

    try {
      await this.writePrivateText(primaryTmp, content);

      const primaryRead = await this.readIndexDocument(primary);
      if (primaryRead.kind === "unreadable") throw primaryRead.error;
      let recovery = primaryRead.kind === "valid" ? primaryRead.raw : null;
      if (recovery === null) {
        const backupRead = await this.readIndexDocument(backup);
        if (backupRead.kind === "unreadable") throw backupRead.error;
        if (backupRead.kind === "valid") {
          recovery = backupRead.raw;
        } else {
          const legacyRead = await this.readIndexDocument(this.paths.legacyPapersJsonPath);
          if (legacyRead.kind === "unreadable") throw legacyRead.error;
          if (legacyRead.kind === "valid") recovery = legacyRead.raw;
        }
      }

      if (primaryRead.kind === "valid") {
        await this.publishBackup(backup, backupTmp, primaryRead.raw);
      }

      await this.removeIfExists(primary);
      try {
        await this.storage.rename(primaryTmp, primary);
      } catch (error) {
        await this.restorePrimaryBestEffort(primaryTmp, primary, recovery);
        throw error;
      }
    } finally {
      await this.removeIfExistsBestEffort(primaryTmp);
      await this.removeIfExistsBestEffort(backupTmp);
    }
  }

  private async publishBackup(
    backup: string,
    backupTmp: string,
    content: string,
  ): Promise<void> {
    await this.writePrivateText(backupTmp, content);
    const existing = await this.readIndexDocument(backup);
    if (existing.kind === "unreadable") throw existing.error;
    const recovery = existing.kind === "valid" ? existing.raw : null;
    await this.removeIfExists(backup);
    try {
      await this.storage.rename(backupTmp, backup);
    } catch (error) {
      if (recovery !== null) {
        await this.removeIfExists(backupTmp);
        await this.writePrivateText(backup, recovery);
      }
      throw error;
    }
  }

  private async restorePrimaryBestEffort(
    primaryTmp: string,
    primary: string,
    recovery: string | null,
  ): Promise<void> {
    try {
      await this.removeIfExists(primaryTmp);
      if (recovery === null) return;
      await this.writePrivateText(primaryTmp, recovery);
      await this.storage.rename(primaryTmp, primary);
    } catch {
      // Preserve the primary promotion error. Existing backup/legacy recovery
      // remains available when rollback cleanup or restoration also fails.
    }
  }

  private async writePrivateText(path: string, content: string): Promise<void> {
    if (this.storage.writeTextWithMode) {
      await this.storage.writeTextWithMode(path, content, 0o600);
      return;
    }
    await this.storage.writeText(path, content);
  }

  private async removeIfExists(path: string): Promise<void> {
    if (await this.storage.exists(path)) await this.storage.remove(path);
  }

  private async removeIfExistsBestEffort(path: string): Promise<void> {
    try {
      await this.removeIfExists(path);
    } catch {
      // Temporary cleanup must not turn a committed save into a failure or
      // replace the primary error from an unsuccessful save.
    }
  }
}

type PaperIndexReadResult =
  | { kind: "missing" }
  | { kind: "unreadable"; error: unknown }
  | { kind: "corrupt"; error: unknown }
  | {
      kind: "valid";
      raw: string;
      document: unknown;
      inbox: PaperInbox;
    };

interface SelectedPaperIndex {
  path: string;
  raw: string;
  document: unknown;
  inbox: PaperInbox;
}

function enqueuePathMutation<T>(
  queues: Map<string, Promise<unknown>>,
  key: string,
  job: () => Promise<T>,
): Promise<T> {
  const next = (queues.get(key) ?? Promise.resolve())
    .catch(() => undefined)
    .then(job);
  queues.set(key, next.catch(() => undefined));
  return next;
}

function emptyInbox(now: Date): PaperInbox {
  return {
    schemaVersion: PAPER_INBOX_SCHEMA_VERSION,
    updatedAt: now.toISOString(),
    papers: {},
  };
}

function upsertEntry(
  inbox: PaperInbox,
  input: PaperIndexUpsert,
): { entry: PaperIndexEntry; wasNew: boolean } {
  const resources = modernArxivResources(input.arxivId);
  if (!resources) {
    throw new PaperIndexError(
      `invalid arXiv ID: ${input.arxivId}`,
      undefined,
      "invalid_input",
    );
  }
  const externalId = resources.id;
  const paperKey = formatPaperKey("arxiv", externalId);
  const existing = inbox.papers[paperKey];
  const wasNew = !existing;
  const authors = normalizeAuthors(input.authors);
  const topic = input.primaryTopic.trim();
  const inputCategories = normalizeCategories([
    ...(input.arxivCategories ?? []),
    input.arxivCategory,
  ]);
  const existingCategories = normalizeCategories([
    ...(existing?.categories ?? []),
    existing?.category ?? "",
  ]);
  const categories = appendUniqueMany(existingCategories, inputCategories);
  const published = dateOnly(input.published) || existing?.published || dateOnly(input.date);
  const updated = dateOnly(input.updated) || existing?.updated || dateOnly(input.date);
  const paperPath =
    input.paperPath === undefined
      ? existing?.paperPath ?? null
      : input.paperPath
        ? normalizeStoragePath(input.paperPath)
        : null;

  const entry: PaperIndexEntry = {
    paperKey,
    source: "arxiv",
    externalId,
    arxivId: externalId,
    title: input.title.trim() || existing?.title || externalId,
    authors: authors.length ? authors : existing?.authors ?? [],
    published,
    updated,
    category: inputCategories[0] || existing?.category || categories[0] || "",
    categories,
    abstract: stringOr(input.abstract, existing?.abstract ?? "") || undefined,
    summary: existing?.summary,
    topics: appendUnique(existing?.topics ?? [], topic),
    primaryTopic: topic || existing?.primaryTopic || "",
    detail: Boolean(existing?.detail || input.detail),
    status: existing?.status ?? "inbox",
    priority: existing?.priority ?? "normal",
    seenDates: appendUnique(existing?.seenDates ?? [], dateOnly(input.date)),
    dailyReports: input.dailyReport
      ? appendUnique(existing?.dailyReports ?? [], input.dailyReport)
      : existing?.dailyReports ?? [],
    discoveryProvenanceByReport: existing?.discoveryProvenanceByReport ?? {},
    noveltyByReport: existing?.noveltyByReport ?? {},
    paperPath,
    arxivUrl: resources.absUrl,
    pdfUrl: resources.pdfUrl,
    pdfPath: existing?.pdfPath ?? "",
    zoteroKey: existing?.zoteroKey ?? "",
    zoteroUri: existing?.zoteroUri ?? "",
    citationKey: existing?.citationKey ?? "",
    projects: existing?.projects ?? [],
  };
  inbox.papers[paperKey] = entry;
  return { entry, wasNew };
}

function setSummariesInInbox(
  inbox: PaperInbox,
  summaries: Record<string, PaperSummary>,
): number {
  let changed = 0;
  for (const [id, summary] of Object.entries(summaries)) {
    const entry = findEntry(inbox, id);
    if (!entry) continue;
    const next = mergeSummaries(entry.summary, summary);
    if (sameSummary(entry.summary, next)) continue;
    entry.summary = next;
    changed += 1;
  }
  return changed;
}

function normalizeInbox(raw: unknown, now: Date): PaperInbox {
  if (!isPlainRecord(raw)) throw new Error("paper index must be an object");
  const obj = raw as any;
  if (
    obj.schemaVersion !== 1 &&
    obj.schemaVersion !== 2 &&
    obj.schemaVersion !== 3 &&
    obj.schemaVersion !== 4 &&
    obj.schemaVersion !== PAPER_INBOX_SCHEMA_VERSION
  ) {
    throw new Error(`unsupported schemaVersion: ${obj.schemaVersion}`);
  }
  if (!isPlainRecord(obj.papers)) throw new Error("paper index papers must be an object");
  const papers: Record<string, PaperIndexEntry> = {};
  for (const [id, value] of Object.entries(obj.papers)) {
    const entry = normalizeEntry(id, value);
    papers[entry.paperKey] = entry;
  }
  return {
    schemaVersion: PAPER_INBOX_SCHEMA_VERSION,
    updatedAt:
      typeof obj.updatedAt === "string" && obj.updatedAt
        ? obj.updatedAt
        : now.toISOString(),
    papers,
  };
}

/**
 * Normalize one on-disk entry. Map keys may be:
 * - bare modern arXiv id (schema ≤3) → rekeyed to `arxiv:<id>`
 * - already-normalized paperKey (`arxiv:…` or future sources)
 */
function normalizeEntry(id: string, raw: unknown): PaperIndexEntry {
  const obj = (raw && typeof raw === "object" ? raw : {}) as any;
  const { paperKey, source, externalId, resources } = resolveEntryIdentity(id, obj);
  const status = isPaperStatus(obj.status) ? obj.status : "inbox";
  const priority = isPaperPriority(obj.priority) ? obj.priority : "normal";
  return {
    paperKey,
    source,
    externalId,
    arxivId: source === "arxiv" ? externalId : stringOr(obj.arxivId, externalId),
    title: stringOr(obj.title, externalId),
    authors: normalizeAuthors(obj.authors),
    published: stringOr(obj.published, ""),
    updated: stringOr(obj.updated, ""),
    category: stringOr(obj.category, ""),
    categories: normalizeCategories([
      ...stringArray(obj.categories),
      stringOr(obj.category, ""),
    ]),
    abstract: stringOr(obj.abstract, "") || undefined,
    summary: normalizeSummary(obj.summary),
    topics: stringArray(obj.topics),
    primaryTopic: stringOr(obj.primaryTopic, ""),
    detail: Boolean(obj.detail),
    status,
    priority,
    seenDates: stringArray(obj.seenDates),
    dailyReports: stringArray(obj.dailyReports),
    discoveryProvenanceByReport: normalizeProvenanceByReport(obj.discoveryProvenanceByReport),
    noveltyByReport: normalizeNoveltyByReport(obj.noveltyByReport),
    paperPath: obj.paperPath ? normalizeStoragePath(String(obj.paperPath)) : null,
    arxivUrl: resources?.absUrl ?? stringOr(obj.arxivUrl, ""),
    pdfUrl: resources?.pdfUrl ?? stringOr(obj.pdfUrl, ""),
    pdfPath: stringOr(obj.pdfPath, ""),
    zoteroKey: stringOr(obj.zoteroKey, ""),
    zoteroUri: stringOr(obj.zoteroUri, ""),
    citationKey: stringOr(obj.citationKey, ""),
    projects: stringArray(obj.projects),
  };
}

function resolveEntryIdentity(
  mapKey: string,
  obj: Record<string, unknown>,
): {
  paperKey: string;
  source: string;
  externalId: string;
  resources: ReturnType<typeof modernArxivResources>;
} {
  const parsedKey = tryParsePaperKey(mapKey);
  if (parsedKey) {
    if (parsedKey.source === "arxiv") {
      const keyResources = modernArxivResources(parsedKey.externalId);
      if (!keyResources) {
        throw new Error(`invalid arXiv paper index key: ${mapKey}`);
      }
      const entryIdHint =
        stringOr(obj.externalId, "") ||
        stringOr(obj.arxivId, "") ||
        keyResources.id;
      const entryResources = modernArxivResources(entryIdHint);
      if (!entryResources || entryResources.id !== keyResources.id) {
        throw new Error(
          `arXiv paper index key/entry mismatch: ${mapKey} != ${String(obj.arxivId ?? obj.externalId ?? "")}`,
        );
      }
      return {
        paperKey: formatPaperKey("arxiv", keyResources.id),
        source: "arxiv",
        externalId: keyResources.id,
        resources: keyResources,
      };
    }

    const entryPaperKey = stringOr(obj.paperKey, mapKey);
    const entryParsed = tryParsePaperKey(entryPaperKey);
    if (
      entryParsed &&
      (entryParsed.source !== parsedKey.source ||
        entryParsed.externalId !== parsedKey.externalId)
    ) {
      throw new Error(
        `paper index key/entry mismatch: ${mapKey} != ${entryPaperKey}`,
      );
    }
    const externalId =
      stringOr(obj.externalId, "") || parsedKey.externalId;
    if (externalId !== parsedKey.externalId) {
      throw new Error(
        `paper index key/entry externalId mismatch: ${mapKey} != ${externalId}`,
      );
    }
    return {
      paperKey: formatPaperKey(parsedKey.source, parsedKey.externalId),
      source: parsedKey.source,
      externalId: parsedKey.externalId,
      resources: null,
    };
  }

  // Legacy schema ≤3: bare modern arXiv id as map key.
  const keyResources = modernArxivResources(mapKey);
  const entryResources = modernArxivResources(
    stringOr(obj.arxivId, "") || stringOr(obj.externalId, "") || mapKey,
  );
  if (!keyResources || !entryResources) {
    throw new Error(`invalid arXiv paper index key or entry: ${mapKey}`);
  }
  if (keyResources.id !== entryResources.id) {
    throw new Error(
      `arXiv paper index key/entry mismatch: ${mapKey} != ${String(obj.arxivId)}`,
    );
  }
  return {
    paperKey: formatPaperKey("arxiv", keyResources.id),
    source: "arxiv",
    externalId: keyResources.id,
    resources: keyResources,
  };
}

/** Resolve paperKey or bare arXiv id against an in-memory (normalized) inbox. */
function findEntry(inbox: PaperInbox, id: string): PaperIndexEntry | null {
  try {
    const paperKey = resolvePaperLookupKey(id);
    return inbox.papers[paperKey] ?? null;
  } catch {
    return null;
  }
}

function requireOutputDirectory(name: string, input: unknown): string {
  const result = validateVaultRelativeDirectory(input);
  if (!result.ok || !result.value) {
    throw new PaperIndexError(
      `invalid ${name}: ${result.reason}`,
      undefined,
      "configuration",
    );
  }
  return result.value;
}

function parentDir(
  path: string,
  normalizePath: (path: string) => string = normalizeStoragePath,
): string {
  const parts = normalizePath(path).split("/").filter(Boolean);
  if (parts.length <= 1) return "";
  return parts.slice(0, -1).join("/");
}

function normalizeStoragePath(path: string): string {
  return path
    .replace(/\\/g, "/")
    .replace(/\/+/g, "/")
    .replace(/^\/+|\/+$/g, "");
}

function dateOnly(value: string | undefined): string {
  const trimmed = value?.trim() ?? "";
  const match = /^(\d{4}-\d{2}-\d{2})/.exec(trimmed);
  return match?.[1] ?? trimmed;
}

function appendUnique(items: string[], next: string): string[] {
  const out = [...items];
  const trimmed = next.trim();
  if (trimmed && !out.includes(trimmed)) out.push(trimmed);
  return out;
}

function uniqueStrings(items: string[]): string[] {
  const out: string[] = [];
  for (const item of items) {
    const trimmed = item.trim();
    if (trimmed && !out.includes(trimmed)) out.push(trimmed);
  }
  return out;
}

function appendUniqueMany(items: string[], values: string[]): string[] {
  let out = items;
  for (const value of values) out = appendUnique(out, value);
  return out;
}

function normalizeCategories(values: string[]): string[] {
  const out: string[] = [];
  for (const value of values) {
    const category = value.trim();
    if (category && !out.includes(category)) out.push(category);
  }
  return out;
}

function normalizeProvenanceByReport(value: unknown): Record<string, PaperDiscoveryProvenance> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const out: Record<string, PaperDiscoveryProvenance> = {};
  for (const [path, raw] of Object.entries(value)) {
    const normalizedPath = normalizeStoragePath(path);
    const provenance = normalizePaperDiscoveryProvenance(raw);
    if (normalizedPath && provenance) out[normalizedPath] = provenance;
  }
  return out;
}

/**
 * Strict per-report novelty normalization: structurally invalid novelty records
 * are omitted without failing the whole document (the discovery-provenance
 * precedent); only valid occurrences survive into the derived index.
 */
function normalizeNoveltyByReport(value: unknown): Record<string, PersonalNovelty> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const out: Record<string, PersonalNovelty> = {};
  for (const [path, raw] of Object.entries(value)) {
    const normalizedPath = normalizeStoragePath(path);
    const novelty = normalizePersonalNovelty(raw);
    if (normalizedPath && novelty) out[normalizedPath] = novelty;
  }
  return out;
}

function normalizeSummary(value: unknown): PaperSummary | undefined {
  if (!value || typeof value !== "object") return undefined;
  const obj = value as Record<string, unknown>;
  const summary: PaperSummary = {
    sourceSections: stringOr(obj.sourceSections, ""),
    coreProblem: stringOr(obj.coreProblem, ""),
    keyMethod: stringOr(obj.keyMethod, ""),
    mainResult: stringOr(obj.mainResult, ""),
    whyRelevant: stringOr(obj.whyRelevant, ""),
    limitations: stringOr(obj.limitations, ""),
  };
  for (const key of Object.keys(summary) as Array<keyof PaperSummary>) {
    if (!summary[key]) delete summary[key];
  }
  return Object.keys(summary).length > 0 ? summary : undefined;
}

function mergeSummaries(
  existing: PaperSummary | undefined,
  incoming: PaperSummary,
): PaperSummary | undefined {
  return normalizeSummary({ ...existing, ...normalizeSummary(incoming) });
}

function sameSummary(
  a: PaperSummary | undefined,
  b: PaperSummary | undefined,
): boolean {
  const keys: Array<keyof PaperSummary> = [
    "sourceSections",
    "coreProblem",
    "keyMethod",
    "mainResult",
    "whyRelevant",
    "limitations",
  ];
  return keys.every((key) => a?.[key] === b?.[key]);
}

function normalizeAuthors(value: string | string[] | unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((v) => String(v).trim()).filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(/\s*,\s*/)
      .map((v) => v.trim())
      .filter(Boolean);
  }
  return [];
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((v) => String(v).trim()).filter(Boolean);
}

function stringOr(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

export function isPaperStatus(value: unknown): value is PaperStatus {
  return (
    value === "inbox" ||
    value === "to_read" ||
    value === "reading" ||
    value === "read" ||
    value === "saved" ||
    value === "ignored"
  );
}

export function isPaperPriority(value: unknown): value is PaperPriority {
  return value === "low" || value === "normal" || value === "high";
}

function compareEntries(a: PaperIndexEntry, b: PaperIndexEntry): number {
  if (a.primaryTopic !== b.primaryTopic) {
    return a.primaryTopic.localeCompare(b.primaryTopic);
  }
  const priorityOrder: Record<PaperPriority, number> = {
    high: 0,
    normal: 1,
    low: 2,
  };
  if (priorityOrder[a.priority] !== priorityOrder[b.priority]) {
    return priorityOrder[a.priority] - priorityOrder[b.priority];
  }
  return b.published.localeCompare(a.published);
}
