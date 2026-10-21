import type { HttpClient } from "../core/adapters";
import type { ArxivRequestCoordinator } from "../pipeline/arxiv-fetcher";
import { throwIfCancelled } from "../services/cancellation";
import { normalizeArxivId } from "../services/manual-fetch";
import { searchArxivTitle } from "./arxiv-title-search";
import {
  extractPdfIdentificationEvidence,
  PDF_IDENTIFICATION_EVIDENCE_VERSION,
  type PdfIdentificationEvidence,
} from "./pdf-identification-evidence";
import type { PersonalLibraryFileIdentifier } from "./personal-library-reconciliation";
import type { ScopedLibrarySource } from "./scoped-library-source";

export const PDF_IDENTIFICATION_HEAD_BYTES = 4 * 1024 * 1024;
export const PDF_IDENTIFICATION_TAIL_BYTES = 1024 * 1024;
/**
 * Title lookups that fail this many times in a row end the scan's lookups: it is
 * offline or arXiv is struggling, and waiting out the request spacing for every
 * remaining file would only make the scan crawl.
 */
const MAX_CONSECUTIVE_LOOKUP_FAILURES = 3;

/**
 * Shared scan glue: bounded PDF head/tail evidence, then the existing title search.
 *
 * When the PDF carries an arXiv id and the title search answers too, a title
 * match that is strict and unambiguous wins: it is an arXiv record for the
 * document's own title, while an embedded id is only the file's claim (stream
 * text can cite other papers). A search that finds nothing, is ambiguous or
 * fails never demotes an embedded id, and a title that merely restates an
 * arXiv stamp is not searched.
 *
 * Without an embedded id, a lookup that could not be made (offline, rate
 * limited, an HTTP error) throws instead of returning `null`: the file was not
 * judged, so the scan must not remember it as "not on arXiv".
 */
export function createPdfLibraryFileIdentifier(input: {
  source: ScopedLibrarySource;
  http: HttpClient;
  /** Spacing and cooldowns for the title search; tests pass one with a fake clock. */
  requestCoordinator?: ArxivRequestCoordinator;
}): PersonalLibraryFileIdentifier {
  let consecutiveLookupFailures = 0;
  return {
    version: PDF_IDENTIFICATION_EVIDENCE_VERSION,
    async identify(logicalPath, signal, size) {
      throwIfCancelled(signal);
      let evidence: PdfIdentificationEvidence;
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
        evidence = extractPdfIdentificationEvidence(combined);
      } catch {
        throwIfCancelled(signal);
        return null;
      }
      const directId = evidence.arxivId ? normalizeArxivId(evidence.arxivId) : null;
      if (evidence.title && (!directId || !/^arxiv:/i.test(evidence.title))) {
        try {
          if (consecutiveLookupFailures >= MAX_CONSECUTIVE_LOOKUP_FAILURES) {
            throw new Error("arXiv title lookups are paused after repeated failures");
          }
          const found = await searchArxivTitle(input.http, evidence.title, signal, input.requestCoordinator);
          consecutiveLookupFailures = 0;
          const searched = found.arxivId ? normalizeArxivId(found.arxivId) : null;
          if (searched) return searched;
        } catch (error) {
          throwIfCancelled(signal);
          consecutiveLookupFailures += 1;
          // A failed title lookup must not demote independent direct evidence.
          if (!directId) throw error;
        }
      }
      return directId;
    },
  };
}
