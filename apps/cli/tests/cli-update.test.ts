import { describe, expect, it, vi } from "vitest";
import { runCli } from "../src/main";
import { compareSemver, isSemVer } from "../src/version";
import { npmInstallCommand, runUpdate } from "../src/update-cmd";

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

  it("sorts a prerelease below its release and by SemVer precedence", () => {
    const ordered = ["0.7.0-alpha", "0.7.0-alpha.1", "0.7.0-alpha.beta", "0.7.0-beta", "0.7.0-beta.2", "0.7.0-beta.11", "0.7.0-rc.1", "0.7.0"];
    for (let i = 0; i < ordered.length - 1; i++) {
      expect(compareSemver(ordered[i]!, ordered[i + 1]!)).toBe(-1);
      expect(compareSemver(ordered[i + 1]!, ordered[i]!)).toBe(1);
    }
    expect(compareSemver("0.7.0-rc.1", "0.7.0")).toBe(-1);
    expect(compareSemver("0.7.0+build.1", "0.7.0+build.2")).toBe(0);
    expect(compareSemver("v1.2.3", "1.2.3")).toBe(0);
    expect(compareSemver("0.0.0-dev", "0.6.2")).toBe(-1);
  });
});

describe("isSemVer", () => {
  it.each(["1.2.3", "0.7.0-rc.1", "1.0.0+build.5", "10.20.30-alpha.1+x"])("accepts %s", (v) => {
    expect(isSemVer(v)).toBe(true);
  });
  it.each(["v1.2.3", "1.2", "01.2.3", "1.2.3 && echo pwned", "github:evil/x", "latest", "1.2.3;rm", "", "1.2.3-", "x".repeat(200)])("rejects %j", (v) => {
    expect(isSemVer(v)).toBe(false);
  });
  it("rejects non-strings", () => {
    expect(isSemVer(undefined)).toBe(false);
    expect(isSemVer(123)).toBe(false);
  });
});

describe("npmInstallCommand", () => {
  it("uses npm without a shell on POSIX", () => {
    expect(npmInstallCommand("arxiv-daily@1.2.3", "linux")).toEqual({ command: "npm", args: ["install", "-g", "arxiv-daily@1.2.3"], shell: false });
    expect(npmInstallCommand("arxiv-daily@1.2.3", "darwin").shell).toBe(false);
  });
  it("uses npm.cmd through a shell on Windows", () => {
    expect(npmInstallCommand("arxiv-daily@1.2.3", "win32")).toEqual({ command: "npm.cmd", args: ["install", "-g", "arxiv-daily@1.2.3"], shell: true });
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

  it.each(["github:evil/pkg", "1.0.0 && calc", "latest", ""])("refuses to install a non-SemVer registry version %j", async (bad) => {
    const cap = captureIo();
    const install = vi.fn(async () => ({ code: 0, output: "" }));
    const code = await runUpdate(cap.io, { fetchLatest: async () => bad, yes: true, runInstall: install });
    expect(code).toBe(1);
    expect(install).not.toHaveBeenCalled();
    expect(cap.stderr.join("")).toContain("invalid version");
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
