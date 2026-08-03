import type { ScopedLibrarySource } from "@arxiv-daily/core";
import { openScopedLibrarySource } from "@arxiv-daily/node-runtime/scoped-library-source";

export async function openObsidianLibrarySource(
  selectedRoot: string,
): Promise<ScopedLibrarySource> {
  return await openScopedLibrarySource(selectedRoot);
}
