import * as fs from "node:fs/promises";
import * as path from "node:path";
import { parse, stringify } from "smol-toml";
import { sha256Hex, type LibraryTopicSettingsPort, type ProposalAcceptanceReceipt } from "@arxiv-daily/core";
import { NodeFileLock, NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { CliConfigError, loadCliConfig, type CliRuntimeConfig } from "./config";

/** Settings and acceptance receipts share the config transaction and commit point. */
export function createCliTopicSettings(config: CliRuntimeConfig): LibraryTopicSettingsPort & { assertCurrent(): Promise<void> } {
  let committingRevision: string | undefined;
  const digest = (raw: string) => `sha256:${sha256Hex(raw)}`;
  const check = (raw: string) => {
    if (!config.configRevision || (digest(raw) !== config.configRevision && digest(raw) !== committingRevision)) {
      throw new CliConfigError("CLI configuration changed during library work; reload and retry");
    }
  };
  const decode = async (raw: string) => {
    const loaded = await loadCliConfig({ configPath: config.configPath, readText: async () => raw });
    const acceptances = decodeProposalAcceptanceReceipts(parse(raw).library_proposal_acceptances);
    if (!acceptances) throw new CliConfigError("Invalid library acceptance receipts; restore the configuration before accepting directions");
    return { topics: loaded.settings.arxiv.topics, acceptances };
  };
  return {
    assertCurrent: async () => {
      // A watcher read may start just before our rename and resolve after the
      // new revision is published. Accept its captured revision as well.
      const capturedRevision = config.configRevision;
      const raw = await fs.readFile(config.configPath, "utf8");
      if (capturedRevision && digest(raw) === capturedRevision) return;
      check(raw);
    },
    read: async () => { const raw = await fs.readFile(config.configPath, "utf8"); check(raw); return decode(raw); },
    async change(compute) {
      const target = await fs.realpath(config.configPath);
      const directory = path.dirname(target), name = path.basename(target);
      const locks = new NodeFileLock(directory, { lockRoot: path.join(directory, ".arxiv-daily-config-locks") });
      const lease = await locks.acquire(`cli-config:${name}`, { wait: true });
      if (!lease) throw new CliConfigError("CLI configuration is busy; reload and retry");
      try {
        const raw = await fs.readFile(target, "utf8"); check(raw);
        const next = await compute(await decode(raw));
        const document = parse(raw);
        const content = stringify({ ...document,
          arxiv: { ...(document.arxiv as Record<string, unknown>), topics: next.topics },
          library_proposal_acceptances: next.acceptances,
        });
        await decode(content);
        const loaded = await loadCliConfig({ configPath: config.configPath, readText: async () => content });
        if (await fs.readFile(target, "utf8") !== raw) throw new CliConfigError("CLI configuration changed during library work; reload and retry");
        // Directory watchers can run between rename and resolution of the write.
        // Only our exact next bytes are allowed in that interval.
        committingRevision = digest(content);
        await new NodeStorageAdapter(directory).writeTextAtomic(name, content, 0o600);
        config.settings = loaded.settings;
        config.configRevision = loaded.configRevision;
      } finally { committingRevision = undefined; await lease.release(); }
    },
  };
}


/** One current proposal per library; switching libraries must not forget edits. */
function decodeProposalAcceptanceReceipts(raw: unknown): ProposalAcceptanceReceipt[] | null {
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
