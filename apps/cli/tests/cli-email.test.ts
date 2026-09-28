import { describe, expect, it, vi } from "vitest";
import {
  AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE,
  DEFAULT_SETTINGS,
  type HostAdapters,
  type StorageAdapter,
} from "@arxiv-daily/core";
import { DEFAULT_CLI_SCHEDULE, type CliRuntimeConfig } from "../src/config";
import { emailStatus, emailTest } from "../src/email-cmd";

function captureIo() {
  const stdout: string[] = [];
  const stderr: string[] = [];
  return {
    stdout,
    stderr,
    io: {
      stdout: { write: (chunk: string) => stdout.push(String(chunk)) },
      stderr: { write: (chunk: string) => stderr.push(String(chunk)) },
    },
  };
}

function testConfig(): CliRuntimeConfig {
  return {
    settings: {
      ...DEFAULT_SETTINGS,
      email: {
        ...DEFAULT_SETTINGS.email,
        enabled: true,
        mode: "self",
        to: "you@example.com",
        apiKey: "re_key",
      },
    },
    vaultRoot: "/vault",
    cacheDir: "/cache",
    linkStyle: "wikilink",
    configPath: "/home/u/.config/arxiv-daily/config.toml",
    scheduleIntent: { ...DEFAULT_CLI_SCHEDULE },
  };
}

function memoryStorage(exclusive: boolean): StorageAdapter {
  const files = new Map<string, string>();
  return {
    normalizePath: (path) => path,
    readText: async (path) => {
      const value = files.get(path);
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    },
    writeText: async (path, content) => { files.set(path, content); },
    writeTextAtomic: async (path, content) => { files.set(path, content); },
    exists: async (path) => files.has(path),
    mkdir: async () => {},
    remove: async (path) => { files.delete(path); },
    rename: async () => {},
    list: async () => [],
    ...(exclusive
      ? {
          createTextExclusive: async () => true,
          guardClaimNamespace: async () => ({ assertCurrent: () => {}, release: async () => {} }),
        }
      : {}),
  };
}

function testHost(exclusive: boolean): HostAdapters {
  return {
    storage: memoryStorage(exclusive),
    http: {
      request: vi.fn(async () => ({
        status: 200,
        headers: {},
        bodyText: JSON.stringify({ id: "msg" }),
      })),
    },
  } as unknown as HostAdapters;
}

describe("email test", () => {
  it("warns after a successful send when automatic email is unsupported here", async () => {
    const { io, stdout, stderr } = captureIo();
    const code = await emailTest(testConfig(), testHost(false), io, "2026-09-24");
    expect(code).toBe(0);
    expect(stdout.join("")).toMatch(/^email test: delivered/);
    expect(stderr.join("")).toContain(AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE);
  });

  it("stays quiet when automatic email works here", async () => {
    const { io, stderr } = captureIo();
    const code = await emailTest(testConfig(), testHost(true), io, "2026-09-24");
    expect(code).toBe(0);
    expect(stderr.join("")).toBe("");
  });
});

describe("email status", () => {
  it("says auto-send is unsupported on this system", async () => {
    const { io, stdout } = captureIo();
    await emailStatus(testConfig(), io, false);
    expect(stdout.join("")).toContain(
      "auto-send: unsupported on this system (currently Linux only); test emails still send",
    );
  });

  it("reports the usual state when supported", async () => {
    const { io, stdout } = captureIo();
    await emailStatus(testConfig(), io, true);
    expect(stdout.join("")).toContain("auto-send: would run on completed daily");
  });
});
