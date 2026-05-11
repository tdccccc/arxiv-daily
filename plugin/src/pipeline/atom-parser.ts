/**
 * Parse an arXiv Atom API response into a Map of base id → abstract.
 *
 * Strips trailing version suffix from <id> URLs ("2605.08080v1" → "2605.08080")
 * so callers can look up by the canonical id form. Entries without a usable
 * id/summary are silently skipped.
 */
export function parseAtomAbstracts(xml: string): Map<string, string> {
  const out = new Map<string, string>();
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  const entries = Array.from(doc.querySelectorAll("entry"));
  for (const entry of entries) {
    const idEl = entry.querySelector("id");
    const summaryEl = entry.querySelector("summary");
    if (!idEl || !summaryEl) continue;
    const fullId = (idEl.textContent ?? "").trim();
    const m = /\/abs\/([^/?#]+?)(v\d+)?$/.exec(fullId);
    if (!m) continue;
    const baseId = m[1];
    const summary = (summaryEl.textContent ?? "").replace(/\s+/g, " ").trim();
    if (summary) out.set(baseId, summary);
  }
  return out;
}
