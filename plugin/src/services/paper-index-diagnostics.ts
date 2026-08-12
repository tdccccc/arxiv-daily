import {
  PAPER_INBOX_SCHEMA_VERSION,
  buildDiagnosticsReport,
  classifyPaperIndexFailureForDiagnostics,
  isPaperPriority,
  isPaperStatus,
  normalizeArxivId,
  type PaperIndexDiagnostics,
} from "@arxiv-daily/core";
import type ArxivDailyPlugin from "../../main";

export function isSupportedPaperIndexSchemaVersion(
  value: unknown,
): value is number {
  return (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 1 &&
    value <= PAPER_INBOX_SCHEMA_VERSION
  );
}

export async function buildSafePluginDiagnosticsReport(
  plugin: ArxivDailyPlugin,
): Promise<string> {
  let paperIndex: PaperIndexDiagnostics;
  let store: ReturnType<ArxivDailyPlugin["buildPaperIndex"]> | undefined;
  try {
    store = plugin.buildPaperIndex();
    paperIndex = await collectPaperIndexDiagnosticsFromStore(plugin, store);
  } catch (error) {
    const failure = classifyPaperIndexFailureForDiagnostics(error);
    plugin.logger.warn(`diagnostics load failed: ${failure}`);
    paperIndex = paperIndexFailureDiagnostics(
      error,
      store?.paths.papersJsonPath,
    );
  }
  return buildDiagnosticsReport({
    settings: plugin.settings,
    runState: plugin.stateStore.snapshot(),
    version: plugin.manifest?.version,
    paperIndex,
  });
}

export function paperIndexFailureDiagnostics(
  error: unknown,
  path = "unavailable",
): PaperIndexDiagnostics {
  return {
    path,
    exists: false,
    error: classifyPaperIndexFailureForDiagnostics(error),
  };
}

export async function collectPaperIndexDiagnostics(
  plugin: ArxivDailyPlugin,
): Promise<PaperIndexDiagnostics> {
  const store = plugin.buildPaperIndex();
  return collectPaperIndexDiagnosticsFromStore(plugin, store);
}

async function collectPaperIndexDiagnosticsFromStore(
  plugin: ArxivDailyPlugin,
  store: ReturnType<ArxivDailyPlugin["buildPaperIndex"]>,
): Promise<PaperIndexDiagnostics> {
  const inspection = await store.inspect();
  const diag: PaperIndexDiagnostics = {
    path: store.paths.papersJsonPath,
    exists: inspection.sourcePath !== null,
    ...(inspection.sourcePath ? { sourcePath: inspection.sourcePath } : {}),
    recoveredFromBackup: inspection.recoveredFromBackup,
  };
  if (inspection.document === null) return diag;

  try {
    const obj = recordOrEmpty(inspection.document);
    const rawSchemaVersion = obj.schemaVersion;
    const schemaVersion =
      typeof rawSchemaVersion === "number" ? rawSchemaVersion : undefined;
    const papers =
      obj.papers && typeof obj.papers === "object"
        ? (obj.papers as Record<string, unknown>)
        : {};
    const statusCounts: Record<string, number> = {};
    const invalidStatuses: string[] = [];
    const invalidPriorities: string[] = [];
    const invalidSeenDates: string[] = [];
    const missingPaperPaths: string[] = [];
    const noteArxivIdMismatches: string[] = [];

    for (const [id, value] of Object.entries(papers)) {
      const entry = recordOrEmpty(value);
      const arxivId = stringOr(entry.arxivId, id);
      const status = stringOr(entry.status, "");
      const priority = stringOr(entry.priority, "");
      if (status) statusCounts[status] = (statusCounts[status] ?? 0) + 1;
      if (!isPaperStatus(status)) {
        invalidStatuses.push(`${arxivId}: ${status || "(missing)"}`);
      }
      if (!isPaperPriority(priority)) {
        invalidPriorities.push(`${arxivId}: ${priority || "(missing)"}`);
      }
      if (!Array.isArray(entry.seenDates)) {
        invalidSeenDates.push(`${arxivId}: seenDates is not an array`);
      } else {
        for (const date of entry.seenDates) {
          if (typeof date !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
            invalidSeenDates.push(`${arxivId}: ${String(date)}`);
          }
        }
      }

      const paperPath = stringOr(entry.paperPath, "");
      if (paperPath && !(await plugin.app.vault.adapter.exists(paperPath))) {
        missingPaperPaths.push(`${arxivId}: ${paperPath}`);
        continue;
      }
      if (paperPath) {
        const noteArxivId = await readNoteArxivId(plugin, paperPath);
        if (noteArxivId && noteArxivId !== arxivId) {
          noteArxivIdMismatches.push(
            `${arxivId}: ${paperPath} has arxiv_id ${noteArxivId}`,
          );
        }
      }
    }
    return {
      ...diag,
      schemaVersion,
      unsupportedSchemaVersion: isSupportedPaperIndexSchemaVersion(
        rawSchemaVersion,
      )
        ? undefined
        : String(rawSchemaVersion),
      total: Object.keys(papers).length,
      statusCounts,
      invalidStatuses,
      invalidPriorities,
      invalidSeenDates,
      missingPaperPaths,
      noteArxivIdMismatches,
    };
  } catch (error) {
    const failure = classifyPaperIndexFailureForDiagnostics(error);
    plugin.logger.warn(`diagnostics probe failed: ${failure}`);
    return {
      ...diag,
      error: failure,
    };
  }
}

async function readNoteArxivId(
  plugin: ArxivDailyPlugin,
  path: string,
): Promise<string | null> {
  const markdown = await plugin.app.vault.adapter.read(path);
  const frontmatter = /^---\s*\n([\s\S]*?)\n---/.exec(markdown)?.[1] ?? "";
  if (!frontmatter) return null;
  const raw = /^arxiv_id:\s*(.+)$/m.exec(frontmatter)?.[1]?.trim() ?? "";
  if (!raw) return null;
  return normalizeArxivId(raw.replace(/^["']|["']$/g, ""));
}

function recordOrEmpty(value: unknown): Record<string, unknown> {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function stringOr(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}
