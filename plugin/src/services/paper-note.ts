import type ArxivDailyPlugin from "../../main";
import type { PaperIndexEntry } from "./paper-index";

export async function ensurePaperNote(
  plugin: ArxivDailyPlugin,
  store: ReturnType<ArxivDailyPlugin["buildPaperIndex"]>,
  entry: PaperIndexEntry,
): Promise<string> {
  const writer = plugin.buildMarkdownWriter();
  if (entry.paperPath && (await plugin.app.vault.adapter.exists(entry.paperPath))) {
    return entry.paperPath;
  }
  const defaultPath = writer.paperDetailPath(entry.arxivId);
  if (await plugin.app.vault.adapter.exists(defaultPath)) {
    await store.setPaperPath(entry.arxivId, defaultPath);
    return defaultPath;
  }
  const path = await writer.writePaperNote({ ...entry, paperPath: defaultPath });
  await store.setPaperPath(entry.arxivId, path);
  return path;
}
