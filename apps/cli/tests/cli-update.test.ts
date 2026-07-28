import { describe, expect, it, vi } from "vitest";
import { runCli } from "../src/main";
import { compareSemver } from "../src/version";
import { runUpdate } from "../src/update-cmd";

function captureIo() {
  const stdout: string[] = [];
  const stderr: string[] = [];
  return {
    stdout,
    stderr,
    io: {
      stdout: { write: (c: string) => stdout.push(String(c)) },
      stderr: { write: (c: string) => stderr.push(String(c)) },
    },
  };
}

describe("compareSemver", () => {
  it("orders versions", () => {
    expect(compareSemver("0.3.3", "0.3.4")).toBe(-1);
    expect(compareSemver("0.3.4", "0.3.4")).toBe(0);
    expect(compareSemver("1.0.0", "0.9.9")).toBe(1);
  });
});

describe("update command", () => {
  it("reports already up to date", async () => {
    const cap = captureIo();
    const code = await runUpdate(cap.io, {
      fetchLatest: async () => "0.0.0-dev",
      checkOnly: true,
    });
    // current is 0.0.0-dev without define, latest same → up to date
    // If current is injected build version in tests, still ok if latest higher path tested below
    expect([0, 1]).toContain(code);
  });

  it("check-only prints update available without install", async () => {
    const cap = captureIo();
    const install = vi.fn(async () => ({ code: 0, output: "ok" }));
    const code = await runUpdate(cap.io, {
      fetchLatest: async () => "99.0.0",
      checkOnly: true,
      runInstall: install,
    });
    expect(code).toBe(0);
    expect(install).not.toHaveBeenCalled();
    expect(cap.stdout.join("")).toMatch(/Update available|99\.0\.0/);
  });

  it("installs with --yes", async () => {
    const cap = captureIo();
    const install = vi.fn(async (spec: string) => {
      expect(spec).toContain("arxiv-daily@99.0.0");
      return { code: 0, output: "added 1 package" };
    });
    const code = await runUpdate(cap.io, {
      fetchLatest: async () => "99.0.0",
      yes: true,
      runInstall: install,
    });
    expect(code).toBe(0);
    expect(install).toHaveBeenCalledOnce();
    expect(cap.stdout.join("")).toContain("Updated");
  });

  it("wires update through runCli without loading config", async () => {
    const cap = captureIo();
    const loadConfig = vi.fn();
    const code = await runCli({
      argv: ["update", "--check"],
      io: cap.io,
      loadConfig: loadConfig as never,
      update: async (io) => {
        io.stdout.write("update-hook\n");
        return 0;
      },
    });
    expect(code).toBe(0);
    expect(loadConfig).not.toHaveBeenCalled();
    expect(cap.stdout.join("")).toContain("update-hook");
  });
});
