import type ArxivDailyPlugin from "../../main";
import type { PaperIndexEntry } from "./paper-index";

export async function ensurePaperNote(
  plugin: ArxivDailyPlugin,
  store: ReturnType<ArxivDailyPlugin["buildPaperIndex"]>,
  entry: PaperIndexEntry,
): Promise<string> {
  const writer = plugin.buildMarkdownWriter();
  if (entry.paperPath && (await plugin.app.vault.adapter.exists(entry.paperPath))) {
    plugin.logger.info(
      `paper-note: using existing paper note for ${entry.arxivId} at ${entry.paperPath}`,
    );
    return entry.paperPath;
  }
  const defaultPath = writer.paperDetailPath(entry.arxivId);
  if (await plugin.app.vault.adapter.exists(defaultPath)) {
    plugin.logger.info(
      `paper-note: found paper note for ${entry.arxivId} at ${defaultPath}`,
    );
    await store.setPaperPath(entry.arxivId, defaultPath);
    plugin.logger.info(
      `paper-note: stored paperPath for ${entry.arxivId} -> ${defaultPath}`,
    );
    return defaultPath;
  }
  plugin.logger.info(`paper-note: creating paper note for ${entry.arxivId}`);
  const path = await writer.writePaperNote({ ...entry, paperPath: defaultPath });
  await store.setPaperPath(entry.arxivId, path);
  plugin.logger.info(`paper-note: created paper note for ${entry.arxivId} at ${path}`);
  return path;
}
