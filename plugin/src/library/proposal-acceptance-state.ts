import type { ProposalAcceptanceReceipt } from "@arxiv-daily/core";

/** One current proposal per library; switching libraries must not forget edits. */
export function decodeProposalAcceptanceReceipts(raw: unknown): ProposalAcceptanceReceipt[] | null {
  if (raw === undefined) return [];
  if (!Array.isArray(raw)) return null;
  const receipts: ProposalAcceptanceReceipt[] = [];
  const scopes = new Set<string>();
  for (const item of raw) {
    if (!record(item) || Object.keys(item).sort().join(",") !== "processedCandidateIds,proposalId,scopeFingerprint,topicTargets"
      || !id(item.proposalId) || typeof item.scopeFingerprint !== "string"
      || !/^sha256:[a-f0-9]{64}$/u.test(item.scopeFingerprint)
      || scopes.has(item.scopeFingerprint) || !record(item.topicTargets)
      || !Array.isArray(item.processedCandidateIds)
      || item.processedCandidateIds.length > 12
      || !item.processedCandidateIds.every(id)
      || new Set(item.processedCandidateIds).size !== item.processedCandidateIds.length) return null;
    const targets = Object.entries(item.topicTargets);
    if (targets.length > 12 || targets.some(([key, value]) => !id(key) || !id(value))) return null;
    scopes.add(item.scopeFingerprint);
    receipts.push({
      proposalId: item.proposalId, scopeFingerprint: item.scopeFingerprint,
      topicTargets: Object.fromEntries(targets) as Record<string, string>,
      processedCandidateIds: [...item.processedCandidateIds].sort(),
    });
  }
  return receipts;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function id(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= 128 && value.trim() === value;
}
