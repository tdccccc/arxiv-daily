import { describe, expect, it } from "vitest";
import {
  claimAutomaticDelivery,
  deliveryRecordKey,
  deliveryStatePath,
  emptyDeliveryState,
  finalizeAutomaticDelivery,
  markAutomaticDeliveryAttemptStarted,
  markDelivered,
  markFailed,
  readDeliveryState,
  saveDeliveryState,
  shouldSendEmail,
} from "../../src/delivery/delivery-state";
import type { StorageAdapter } from "../../src/core/adapters";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import {
  EMAIL_DELIVERY_CHANNEL,
  EMAIL_HOSTED_CHANNEL,
} from "../../src/delivery/types";

function sharedMemoryStorage(
  initial: Record<string, string> = {},
): StorageAdapter {
  const files = new Map(Object.entries(initial));
  const dirs = new Set<string>();
  return {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/"),
    async readText(path) {
      const value = files.get(path);
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    },
    async writeText(path, content) {
      files.set(path, content);
    },
    async writeTextAtomic(path, content) {
      files.set(path, content);
    },
    async createTextExclusive(path, content) {
      if (files.has(path)) return false;
      files.set(path, content);
      return true;
    },
    async guardClaimNamespace() {
      return { assertCurrent() {}, async release() {} };
    },
    async exists(path) {
      return files.has(path) || dirs.has(path);
    },
    async mkdir(path) {
      dirs.add(path);
    },
    async remove(path) {
      files.delete(path);
      dirs.delete(path);
    },
    async rename(from, to) {
      const value = files.get(from);
      if (value === undefined) throw new Error(`missing ${from}`);
      files.set(to, value);
      files.delete(from);
    },
    async list(dir) {
      const prefix = `${dir}/`;
      return Array.from(files.keys())
        .filter((path) => path.startsWith(prefix))
        .map((path) => ({ path, type: "file" as const }));
    },
  };
}

describe("delivery-state", () => {
  it("builds stable keys with lowercased recipient", () => {
    expect(deliveryRecordKey("2026-07-26", "You@Example.COM")).toBe(
      `2026-07-26|you@example.com|${EMAIL_DELIVERY_CHANNEL}`,
    );
  });

  it("skips when delivered and allows when missing or failed", () => {
    let state = emptyDeliveryState(new Date("2026-07-26T00:00:00.000Z"));
    expect(shouldSendEmail(state, "2026-07-26", "a@b.com")).toBe(true);

    state = markFailed(state, {
      date: "2026-07-26",
      recipient: "a@b.com",
      attempts: 3,
      lastError: "boom",
      now: new Date("2026-07-26T01:00:00.000Z"),
    });
    expect(shouldSendEmail(state, "2026-07-26", "a@b.com")).toBe(true);
    expect(
      state.records[deliveryRecordKey("2026-07-26", "a@b.com")]?.status,
    ).toBe("failed");

    state = markDelivered(state, {
      date: "2026-07-26",
      recipient: "a@b.com",
      attempts: 1,
      now: new Date("2026-07-26T02:00:00.000Z"),
    });
    expect(shouldSendEmail(state, "2026-07-26", "a@b.com")).toBe(false);
    expect(
      state.records[deliveryRecordKey("2026-07-26", "a@b.com")],
    ).toMatchObject({
      status: "delivered",
      attempts: 1,
    });
  });

  it("skips cross-mode once date+recipient is delivered", () => {
    let state = emptyDeliveryState(new Date("2026-07-26T00:00:00.000Z"));
    state = markDelivered(state, {
      date: "2026-07-26",
      recipient: "a@b.com",
      channel: EMAIL_DELIVERY_CHANNEL,
      attempts: 1,
      now: new Date("2026-07-26T02:00:00.000Z"),
    });
    // Same day/to via hosted channel must not auto-send again.
    expect(
      shouldSendEmail(state, "2026-07-26", "a@b.com", EMAIL_HOSTED_CHANNEL),
    ).toBe(false);
    expect(shouldSendEmail(state, "2026-07-27", "a@b.com")).toBe(true);
  });

  it("treats v1 delivered claim records as blocking for old clients", () => {
    const state = {
      schemaVersion: 1 as const,
      updatedAt: "2026-07-26T00:00:00.000Z",
      records: {
        claim: {
          date: "2026-07-26",
          recipient: "a@b.com",
          channel: EMAIL_DELIVERY_CHANNEL,
          status: "delivered" as const,
          updatedAt: "2026-07-26T00:00:00.000Z",
          attempts: 0,
          deliveryPhase: "claimed" as const,
        },
      },
    };

    expect(shouldSendEmail(state, "2026-07-26", "a@b.com")).toBe(false);
  });

  it("keeps raw recipient only in v1 main state and hashes every generation sidecar", async () => {
    const rawRecipient = "Private.Recipient@Example.COM";
    const injectedError = "LOW_LEVEL_ERROR_MARKER_7f19";
    const storage = sharedMemoryStorage();
    const claim = await claimAutomaticDelivery(storage, DEFAULT_SETTINGS.output, {
      date: "2026-08-10",
      recipient: rawRecipient,
      channel: EMAIL_DELIVERY_CHANNEL,
      owner: "owner-safe",
      now: new Date("2026-08-10T00:00:00.000Z"),
    });
    if (claim.kind !== "claimed") throw new Error("expected claim");
    await markAutomaticDeliveryAttemptStarted(
      storage,
      DEFAULT_SETTINGS.output,
      claim,
      new Date("2026-08-10T00:00:01.000Z"),
    );
    await finalizeAutomaticDelivery(storage, DEFAULT_SETTINGS.output, {
      ...claim,
      outcome: "ambiguous",
      attempts: 1,
      errorCode: "provider_outcome_ambiguous",
      lastError: injectedError,
      now: new Date("2026-08-10T00:00:02.000Z"),
    } as Parameters<typeof finalizeAutomaticDelivery>[2]);

    const statePath = deliveryStatePath(DEFAULT_SETTINGS.output);
    const main = await storage.readText(statePath);
    expect(main).toContain(rawRecipient);
    const sidecars = (await storage.list!(`${statePath}.claims`))
      .filter((entry) => entry.type === "file")
      .map((entry) => storage.readText(entry.path));
    const persisted = (await Promise.all(sidecars)).join("\n");
    expect(persisted).not.toContain(rawRecipient);
    expect(persisted).not.toContain(rawRecipient.toLowerCase());
    expect(persisted).not.toContain(injectedError);
    expect(persisted).not.toContain("providerMessageId");
    expect(persisted).toMatch(/"recipientIdentity": "[0-9a-f]{64}"/);
  });

  it("recovers only a stale claim that has no provider-attempt decision", async () => {
    const storage = sharedMemoryStorage();
    const first = await claimAutomaticDelivery(storage, DEFAULT_SETTINGS.output, {
      date: "2026-08-10",
      recipient: "recover@example.com",
      channel: EMAIL_DELIVERY_CHANNEL,
      owner: "owner-one",
      now: new Date("2026-08-10T00:00:00.000Z"),
      recoveryGraceMs: 1_000,
    });
    expect(first.kind).toBe("claimed");

    const recovered = await claimAutomaticDelivery(storage, DEFAULT_SETTINGS.output, {
      date: "2026-08-10",
      recipient: "recover@example.com",
      channel: EMAIL_DELIVERY_CHANNEL,
      owner: "owner-two",
      now: new Date("2026-08-10T00:00:02.000Z"),
      recoveryGraceMs: 1_000,
    });

    expect(recovered).toMatchObject({
      kind: "claimed",
      owner: "owner-two",
      generation: 1,
    });
  });

  it("keeps stale claims blocking after provider attempt starts or becomes ambiguous", async () => {
    const storage = sharedMemoryStorage();
    const first = await claimAutomaticDelivery(storage, DEFAULT_SETTINGS.output, {
      date: "2026-08-10",
      recipient: "blocking@example.com",
      channel: EMAIL_DELIVERY_CHANNEL,
      owner: "owner-one",
      now: new Date("2026-08-10T00:00:00.000Z"),
      recoveryGraceMs: 1_000,
    });
    if (first.kind !== "claimed") throw new Error("expected claim");
    await markAutomaticDeliveryAttemptStarted(
      storage,
      DEFAULT_SETTINGS.output,
      first,
      new Date("2026-08-10T00:00:00.500Z"),
    );
    await finalizeAutomaticDelivery(storage, DEFAULT_SETTINGS.output, {
      ...first,
      outcome: "ambiguous",
      attempts: 1,
      lastError: "response reset",
      now: new Date("2026-08-10T00:00:01.000Z"),
    });

    const retry = await claimAutomaticDelivery(storage, DEFAULT_SETTINGS.output, {
      date: "2026-08-10",
      recipient: "blocking@example.com",
      channel: EMAIL_DELIVERY_CHANNEL,
      owner: "owner-two",
      now: new Date("2026-08-11T00:00:00.000Z"),
      recoveryGraceMs: 1_000,
    });

    expect(retry).toMatchObject({ kind: "blocked" });
  });

  it("fails closed instead of writing private state through a mode-blind fallback", async () => {
    const storage = sharedMemoryStorage();
    storage.writeTextAtomic = undefined;
    storage.writeTextWithMode = undefined;
    let ordinaryWrites = 0;
    const ordinaryWrite = storage.writeText.bind(storage);
    storage.writeText = async (path, content) => {
      ordinaryWrites += 1;
      await ordinaryWrite(path, content);
    };

    await expect(
      saveDeliveryState(
        storage,
        DEFAULT_SETTINGS.output,
        emptyDeliveryState(new Date("2026-08-10T00:00:00.000Z")),
      ),
    ).rejects.toThrow(/private atomic/);
    expect(ordinaryWrites).toBe(0);
  });

  it("fails closed when the host lacks system-wide exclusive create", async () => {
    const storage = sharedMemoryStorage();
    storage.createTextExclusive = undefined;

    await expect(claimAutomaticDelivery(storage, DEFAULT_SETTINGS.output, {
      date: "2026-08-10",
      recipient: "unsupported@example.com",
      channel: EMAIL_DELIVERY_CHANNEL,
    })).resolves.toEqual({
      kind: "failed",
      reason: "delivery_storage_unsupported",
    });
  });

  it("ignores an orphaned legacy global lock instead of permanently stopping delivery", async () => {
    const storage = sharedMemoryStorage({
      [`${deliveryStatePath(DEFAULT_SETTINGS.output)}.lock`]: JSON.stringify({
        owner: "dead-owner",
        createdAt: "2026-08-01T00:00:00.000Z",
      }),
    });

    const claim = await claimAutomaticDelivery(storage, DEFAULT_SETTINGS.output, {
      date: "2026-08-10",
      recipient: "lock@example.com",
      channel: EMAIL_DELIVERY_CHANNEL,
      now: new Date("2026-08-10T00:00:00.000Z"),
    });

    expect(claim.kind).toBe("claimed");
  });

  it("strictly distinguishes missing, valid, corrupt, and unreadable state", async () => {
    const path = deliveryStatePath(DEFAULT_SETTINGS.output);
    const files = new Map<string, string>();
    let unreadable = false;
    const storage: StorageAdapter = {
      normalizePath: (value) => value,
      async readText(value) {
        if (unreadable) throw new Error("permission denied");
        const raw = files.get(value);
        if (raw === undefined) throw new Error("missing");
        return raw;
      },
      async writeText() {},
      async exists(value) {
        return files.has(value);
      },
      async mkdir() {},
      async remove() {},
      async rename() {},
    };

    await expect(readDeliveryState(storage, DEFAULT_SETTINGS.output))
      .resolves.toMatchObject({ kind: "missing" });

    files.set(path, JSON.stringify(emptyDeliveryState()));
    await expect(readDeliveryState(storage, DEFAULT_SETTINGS.output))
      .resolves.toMatchObject({ kind: "valid" });

    files.set(path, "{broken");
    await expect(readDeliveryState(storage, DEFAULT_SETTINGS.output))
      .resolves.toMatchObject({ kind: "corrupt" });

    files.set(path, JSON.stringify({
      ...emptyDeliveryState(),
      records: {
        malformed: {
          date: "2026-07-26",
          recipient: "a@b.com",
          channel: EMAIL_DELIVERY_CHANNEL,
          status: "delivered",
          updatedAt: "2026-07-26T00:00:00.000Z",
          attempts: -1,
        },
      },
    }));
    await expect(readDeliveryState(storage, DEFAULT_SETTINGS.output))
      .resolves.toMatchObject({ kind: "corrupt" });

    unreadable = true;
    await expect(readDeliveryState(storage, DEFAULT_SETTINGS.output))
      .resolves.toMatchObject({ kind: "unreadable" });
  });
});
