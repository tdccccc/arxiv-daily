import type { HttpClient } from "../core/adapters";
import { throwIfCancelled } from "../services/cancellation";
import { normalizeArxivId } from "../services/manual-fetch";
import { searchArxivTitle } from "./arxiv-title-search";
import { extractPdfIdentificationEvidence, PDF_IDENTIFICATION_EVIDENCE_VERSION } from "./pdf-identification-evidence";
import type { PersonalLibraryFileIdentifier } from "./personal-library-reconciliation";
import type { ScopedLibrarySource } from "./scoped-library-source";

export const PDF_IDENTIFICATION_HEAD_BYTES = 4 * 1024 * 1024;
export const PDF_IDENTIFICATION_TAIL_BYTES = 1024 * 1024;

/** Shared scan glue: bounded PDF head/tail evidence, then the existing title search. */
export function createPdfLibraryFileIdentifier(input: {
  source: ScopedLibrarySource;
  http: HttpClient;
}): PersonalLibraryFileIdentifier {
  return {
    version: PDF_IDENTIFICATION_EVIDENCE_VERSION,
    async identify(logicalPath, signal, size) {
      throwIfCancelled(signal);
      try {
        const [head, tail] = await Promise.all([
          input.source.readBinary(logicalPath, { signal, start: 0, end: PDF_IDENTIFICATION_HEAD_BYTES }),
          size && size > PDF_IDENTIFICATION_HEAD_BYTES
            ? input.source.readBinary(logicalPath, { signal, start: size - PDF_IDENTIFICATION_TAIL_BYTES, end: size })
            : Promise.resolve(new ArrayBuffer(0)),
        ]);
        throwIfCancelled(signal);
        const combined = new Uint8Array(head.byteLength + tail.byteLength);
        combined.set(new Uint8Array(head));
        combined.set(new Uint8Array(tail), head.byteLength);
        const evidence = extractPdfIdentificationEvidence(combined);
        const directId = evidence.arxivId ? normalizeArxivId(evidence.arxivId) : null;
        if (evidence.title && (!directId || !/^arxiv:/i.test(evidence.title))) {
          try {
            const found = await searchArxivTitle(input.http, evidence.title, signal);
            const searched = found.arxivId ? normalizeArxivId(found.arxivId) : null;
            if (searched) return searched;
          } catch {
            throwIfCancelled(signal);
            // A failed title lookup must not demote independent direct evidence.
          }
        }
        return directId;
      } catch {
        throwIfCancelled(signal);
        return null;
      }
    },
  };
}
