const MODERN_ARXIV_ID_RE = /^(\d{4}\.\d{4,5})(?:v\d+)?$/i;
const ARXIV_URL_PATH_RE = /^\/(abs|pdf|html|e-print)\/(\d{4}\.\d{4,5})(?:v\d+)?(\.pdf)?$/i;
const ALLOWED_ARXIV_HOSTS = new Set(["arxiv.org", "www.arxiv.org"]);

/**
 * Raw path component of an absolute HTTP(S) URL, before any query or fragment.
 * Parsed without a backtracking regex: the equivalent pattern (an optional
 * query group after a greedy path group) is a polynomial ReDoS risk on
 * caller-supplied input.
 */
function rawHttpPath(input: string): string | undefined {
  if (/[\r\n]/.test(input)) return undefined;
  const authority = /^https?:\/\/[^/?#]+/i.exec(input)?.[0];
  if (!authority) return undefined;
  const rest = input.slice(authority.length);
  const end = rest.search(/[?#]/);
  return end === -1 ? rest : rest.slice(0, end);
}

export interface ModernArxivResources {
  id: string;
  absUrl: string;
  pdfUrl: string;
  htmlUrl: string;
  sourceUrl: string;
  atomUrl: string;
}

/** Parse a modern arXiv ID and derive its trusted, canonical resource URLs. */
export function modernArxivResources(input: string): ModernArxivResources | null {
  const candidate = input.normalize("NFC").trim();
  if (!candidate) return null;

  let id: string | undefined;
  const plain = candidate.replace(/^arxiv\s*:\s*/i, "");
  const plainMatch = MODERN_ARXIV_ID_RE.exec(plain);
  if (plainMatch) {
    id = plainMatch[1];
  } else {
    let url: URL;
    try {
      url = new URL(candidate);
    } catch {
      return null;
    }
    const rawPath = rawHttpPath(candidate);
    const pathMatch = rawPath ? ARXIV_URL_PATH_RE.exec(rawPath) : null;
    if (
      (url.protocol !== "https:" && url.protocol !== "http:") ||
      url.username ||
      url.password ||
      url.port ||
      !ALLOWED_ARXIV_HOSTS.has(url.hostname.toLowerCase()) ||
      rawPath !== url.pathname ||
      !pathMatch ||
      (pathMatch[3] && pathMatch[1]?.toLowerCase() !== "pdf")
    ) {
      return null;
    }
    id = pathMatch[2];
  }

  if (!id) return null;
  const canonicalId = id.toLowerCase();
  return {
    id: canonicalId,
    absUrl: `https://arxiv.org/abs/${canonicalId}`,
    pdfUrl: `https://arxiv.org/pdf/${canonicalId}`,
    htmlUrl: `https://arxiv.org/html/${canonicalId}`,
    sourceUrl: `https://arxiv.org/e-print/${canonicalId}`,
    atomUrl: `https://export.arxiv.org/api/query?id_list=${canonicalId}&max_results=1`,
  };
}
