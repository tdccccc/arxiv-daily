import type { OpenedScopedLibrarySource } from "@arxiv-daily/node-runtime/scoped-library-source";
import { openScopedLibrarySource } from "@arxiv-daily/node-runtime/scoped-library-source";

export async function openObsidianLibrarySource(
  selectedRoot: string,
): Promise<OpenedScopedLibrarySource> {
  return await openScopedLibrarySource(selectedRoot);
}
