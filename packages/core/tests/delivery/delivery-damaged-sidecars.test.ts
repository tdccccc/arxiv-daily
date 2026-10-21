import { describe, expect, it, vi } from "vitest";
import {
  claimAutomaticDelivery,
  deliveryStatePath,
  finalizeAutomaticDelivery,
  loadDeliveryState,
  markAutomaticDeliveryAttemptStarted,
  releaseAutomaticDeliveryBeforeAttempt,
  shouldSendEmail,
} from "../../src/delivery/delivery-state";
import {
  deliverDailyEmailIfEnabled,
  sampleDailyDigest,
} from "../../src/delivery/deliver-email";
import type { StorageAdapter } from "../../src/core/adapters";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import { EMAIL_DELIVERY_CHANNEL } from "../../src/delivery/types";

const output = DEFAULT_SETTINGS.output;
const claimsDir = `${deliveryStatePath(output)}.claims`;
const RECIPIENT = "reader@example.test";
const email = {
  enabled: true,
  mode: "self" as const,
  to: RECIPIENT,
  fromEmail: "from@example.test",
  apiKey: "re_test",
};

function memoryStorage() {
  const files = new Map<string, string>();
  const storage: StorageAdapter = {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/"),
    async readText(path) {
      const value = files.get(path);
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    },
    async writeText(path, content) { files.set(path, content); },
    async writeTextAtomic(path, content) { files.set(path, content); },
    async createTextExclusive(path, content) {
      if (files.has(path)) return false;
      files.set(path, content);
      return true;
    },
    async guardClaimNamespace() { return { assertCurrent() {}, async release() {} }; },
    async exists(path) { return files.has(path); },
    async mkdir() {},
    async remove(path) { files.delete(path); },
    async rename(from, to) {
      files.set(to, files.get(from)!);
      files.delete(from);
    },
    async list(dir) {
      const prefix = `${dir}/`;
      return [...files.keys()]
        .filter((path) => path.startsWith(prefix))
        .map((path) => ({ path, type: "file" as const }));
    },
  };
  return { files, storage };
}

function harness() {
  const { files, storage } = memoryStorage();
  const request = vi.fn(async () => ({
    status: 200,
    headers: {},
    bodyText: JSON.stringify({ id: "msg_1" }),
  }));
  const logger = { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() };
  const deliver = (date: string) =>
    deliverDailyEmailIfEnabled(sampleDailyDigest({ date }), {
      storage, http: { request }, output, email, logger, sleep: async () => {},
    });
  const claim = async (date: string, owner: string) => {
    const claimed = await claimAutomaticDelivery(storage, output, {
      date, recipient: RECIPIENT, channel: EMAIL_DELIVERY_CHANNEL, owner,
    });
    if (claimed.kind !== "claimed") throw new Error(`expected a claim, got ${claimed.kind}`);
    return claimed;
  };
  const names = () => [...files.keys()].filter((path) => path.startsWith(`${claimsDir}/`)).map((path) => path.slice(claimsDir.length + 1)).sort();
  return { files, storage, request, logger, deliver, claim, names };
}

type Harness = ReturnType<typeof harness>;
type Kind = "claim" | "decision" | "result";

/**
 * Leaves the files a run that died mid-write would: the lifecycle runs up to
 * `kind`, then that one file is replaced by `content` (empty by default).
 */
async function crashDuring(h: Harness, date: string, kind: Kind, content = "") {
  const handle = await h.claim(date, "crashed-owner");
  if (kind !== "claim") {
    await markAutomaticDeliveryAttemptStarted(h.storage, output, handle);
  }
  if (kind === "result") {
    await finalizeAutomaticDelivery(h.storage, output, { ...handle, outcome: "delivered", attempts: 1 });
  }
  const path = { claim: handle.claimPath, decision: handle.decisionPath, result: handle.resultPath }[kind];
  h.files.set(path, content);
  return { handle, path, file: path.slice(claimsDir.length + 1) };
}

const DAMAGE: Array<[string, Kind, string]> = [
  ["an empty claim", "claim", ""],
  ["a whitespace-only claim", "claim", " \n"],
  ["a claim that is not JSON", "claim", "<<<<<<< HEAD\n{}\n=======\n"],
  ["a claim from an unknown schema", "claim", JSON.stringify({ schemaVersion: 2, phase: "claimed" })],
  ["an empty decision", "decision", ""],
  ["a decision that is not JSON", "decision", "{\"schemaVersion\": 1, \"phase\": \"provider_att"],
  ["a decision from an unknown schema", "decision", JSON.stringify({ schemaVersion: 2 })],
  ["an empty result", "result", ""],
  ["a result that is not JSON", "result", "not json"],
  ["a result from an unknown schema", "result", JSON.stringify({ schemaVersion: 2 })],
];

describe("one unreadable delivery sidecar must not stop other days' email", () => {
  it.each(DAMAGE)("still delivers other dates with %s in the claims directory", async (_label, kind, content) => {
    const h = harness();
    const { file } = await crashDuring(h, "2026-07-25", kind, content);

    const [first, second] = [await h.deliver("2026-07-27"), await h.deliver("2026-07-28")];

    expect(first).toEqual({ kind: "delivered", attempts: 1 });
    expect(second).toEqual({ kind: "delivered", attempts: 1 });
    expect(h.request).toHaveBeenCalledTimes(2);
    const state = await loadDeliveryState(h.storage, output);
    expect(shouldSendEmail(state, "2026-07-27", RECIPIENT)).toBe(false);
    expect(shouldSendEmail(state, "2026-07-28", RECIPIENT)).toBe(false);
    // The damaged file is left exactly as found and named in the log.
    expect(h.files.get(`${claimsDir}/${file}`)).toBe(content);
    expect(h.logger.warn).toHaveBeenCalledWith(expect.stringContaining(file));
  });

  it("reports an unreadable file once per delivery, not once per stem generation", async () => {
    const h = harness();
    const { file } = await crashDuring(h, "2026-07-25", "decision");

    await h.deliver("2026-07-27");

    const reports = h.logger.warn.mock.calls.filter(([message]) => String(message).includes(file));
    expect(reports).toHaveLength(1);
  });

  it("keeps the damaged date's earlier record while the other dates are rebuilt", async () => {
    const h = harness();
    await crashDuring(h, "2026-07-25", "decision");
    const before = await loadDeliveryState(h.storage, output);

    await h.deliver("2026-07-27");

    const after = await loadDeliveryState(h.storage, output);
    const key = Object.keys(before.records).find((candidate) => candidate.startsWith("2026-07-25"))!;
    expect(after.records[key]).toEqual(before.records[key]);
    expect(shouldSendEmail(after, "2026-07-25", RECIPIENT)).toBe(false);
  });
});

describe("an unfinished claim of the same date is recovered", () => {
  it.each([
    ["empty", ""],
    ["not JSON", "{\"schemaVersion\": 1, \"phase\": \"clai"],
  ])("sends once from a new generation when the old claim is %s", async (_label, content) => {
    const h = harness();
    const { file } = await crashDuring(h, "2026-07-25", "claim", content);

    const first = await h.deliver("2026-07-25");
    const second = await h.deliver("2026-07-25");

    expect(first).toEqual({ kind: "delivered", attempts: 1 });
    expect(second).toEqual({ kind: "skipped", reason: "already_delivered" });
    expect(h.request).toHaveBeenCalledTimes(1);
    const stem = file.slice(0, file.indexOf("."));
    expect(h.names()).toEqual([
      `${stem}.g0.claim.json`,
      `${stem}.g0.decision.json`,
      `${stem}.g1.claim.json`,
      `${stem}.g1.decision.json`,
      `${stem}.g1.result.json`,
    ]);
    const released = JSON.parse(h.files.get(`${claimsDir}/${stem}.g0.decision.json`)!);
    expect(released).toMatchObject({ phase: "recovered_before_provider_attempt" });
    expect(released.previousOwner).toBeUndefined();
    expect(h.files.get(`${claimsDir}/${file}`)).toBe(content);
  });

  it("never lets the owner of an unfinished claim send once another run has recovered it", async () => {
    const h = harness();
    const owner = await h.claim("2026-07-25", "slow-owner");
    const complete = h.files.get(owner.claimPath)!;
    // Another process reads the claim while its owner is still between creating and writing it.
    h.files.set(owner.claimPath, "");

    const recoverer = await h.deliver("2026-07-25");
    // The owner finishes writing afterwards and then goes for the provider.
    h.files.set(owner.claimPath, complete);
    const late = await markAutomaticDeliveryAttemptStarted(h.storage, output, owner).then(() => "started", (error: Error) => error.message);

    expect(recoverer).toEqual({ kind: "delivered", attempts: 1 });
    expect(late).toMatch(/recovered before provider attempt/);
    expect(h.request).toHaveBeenCalledTimes(1);
  });

  it("does not read older generations once a newer one exists", async () => {
    const h = harness();
    await crashDuring(h, "2026-07-25", "claim");
    await h.deliver("2026-07-25");
    h.logger.warn.mockClear();

    expect(await h.deliver("2026-07-25")).toEqual({ kind: "skipped", reason: "already_delivered" });
    expect(await h.deliver("2026-07-27")).toEqual({ kind: "delivered", attempts: 1 });
    // g0's empty claim is history; only the empty file a stem currently ends in is reported.
    expect(h.logger.warn.mock.calls.filter(([message]) => String(message).includes("unreadable delivery file"))).toEqual([]);
  });
});

describe("a stem that cannot be read stays blocked without touching other dates", () => {
  it.each([
    ["an empty decision", "decision", ""],
    ["a decision from an unknown schema", "decision", JSON.stringify({ schemaVersion: 2 })],
    ["a claim from an unknown schema", "claim", JSON.stringify({ schemaVersion: 2, phase: "claimed" })],
  ] as const)("fails closed for the same date with %s", async (_label, kind, content) => {
    const h = harness();
    const { file } = await crashDuring(h, "2026-07-25", kind, content);
    const before = h.names();

    const result = await h.deliver("2026-07-25");

    expect(result).toEqual({ kind: "failed", reason: "delivery_claim_unreadable", attempts: 0 });
    expect(h.request).not.toHaveBeenCalled();
    expect(h.names()).toEqual(before);
    expect(h.files.get(`${claimsDir}/${file}`)).toBe(content);
    expect(h.logger.error).toHaveBeenCalledWith(expect.stringContaining(`${claimsDir}/${file}`));
    expect(await h.deliver("2026-07-27")).toEqual({ kind: "delivered", attempts: 1 });
  });

  it("names the file in the claim result", async () => {
    const h = harness();
    const { file } = await crashDuring(h, "2026-07-25", "decision");

    const claimed = await claimAutomaticDelivery(h.storage, output, {
      date: "2026-07-25", recipient: RECIPIENT, channel: EMAIL_DELIVERY_CHANNEL,
    });

    expect(claimed).toEqual({ kind: "failed", reason: "delivery_claim_unreadable", file });
  });

  it("keeps blocking when only the result is unreadable after a provider attempt began", async () => {
    const h = harness();
    await crashDuring(h, "2026-07-25", "result");

    expect(await h.deliver("2026-07-25")).toEqual({ kind: "skipped", reason: "provider_attempt_started" });
    expect(h.request).not.toHaveBeenCalled();
    const state = await loadDeliveryState(h.storage, output);
    expect(shouldSendEmail(state, "2026-07-25", RECIPIENT)).toBe(false);
  });

  it("still releases a recovered generation whose result is unreadable", async () => {
    const h = harness();
    const handle = await h.claim("2026-07-25", "gave-up");
    await releaseAutomaticDeliveryBeforeAttempt(h.storage, output, handle, "cancelled");
    h.files.set(handle.resultPath, "");

    expect(await h.deliver("2026-07-25")).toEqual({ kind: "delivered", attempts: 1 });
    expect(h.request).toHaveBeenCalledTimes(1);
  });

  it("fails closed for a result without any decision", async () => {
    const h = harness();
    const handle = await h.claim("2026-07-25", "odd");
    h.files.set(handle.resultPath, "");

    expect(await h.deliver("2026-07-25")).toMatchObject({ kind: "failed", reason: "delivery_claim_unreadable" });
    expect(h.request).not.toHaveBeenCalled();
  });
});
