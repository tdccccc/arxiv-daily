import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { deliveryKeyKind } from "../src/deliver-logic";
import { REQUIRED_SECRET_NAMES } from "../scripts/cutover-preflight.mjs";

const relayFile = (path: string) =>
  readFileSync(fileURLToPath(new URL(`../${path}`, import.meta.url)), "utf8");

const readme = relayFile("README.md");
const runbook = relayFile("docs/runbook-cutover.md");
const wranglerToml = relayFile("wrangler.toml");

/** Lines inside fenced code blocks: the commands a reader is told to run. */
function commandLines(markdown: string): string[] {
  const lines: string[] = [];
  let inBlock = false;
  for (const line of markdown.split("\n")) {
    if (line.startsWith("```")) inBlock = !inBlock;
    else if (inBlock) lines.push(line);
  }
  return lines;
}

describe("how the docs say to deploy", () => {
  it("makes `npm run deploy` the identity-injecting script, not a bare wrangler deploy", () => {
    const scripts = JSON.parse(relayFile("package.json")).scripts as Record<string, string>;
    expect(scripts.deploy).toBe("node scripts/deploy-with-identity.mjs");
  });

  it("never shows a bare `wrangler deploy` command, which would drop BUILD_IDENTITY", () => {
    for (const [name, markdown] of [["README.md", readme], ["docs/runbook-cutover.md", runbook]]) {
      const bare = commandLines(markdown)
        .filter((line) => /wrangler deploy/.test(line))
        .filter((line) => !/--var|--dry-run/.test(line));
      expect(bare, `${name}: ${bare.join(" | ")}`).toEqual([]);
    }
    expect(commandLines(readme)).toContain("npm run deploy                 # checks, then deploys with BUILD_IDENTITY injected");
  });

  it("keeps BUILD_IDENTITY out of wrangler.toml and says why in the README", () => {
    expect(wranglerToml).not.toMatch(/^\s*BUILD_IDENTITY\s*=/m);
    expect(readme).toMatch(/`BUILD_IDENTITY`[^\n]*not\*\* in `wrangler\.toml`/);
  });
});

describe("README matches the relay it documents", () => {
  it("lists exactly the secrets the preflight requires", () => {
    const documented = commandLines(readme)
      .map((line) => /wrangler secret put (\w+)/.exec(line)?.[1])
      .filter((name): name is string => Boolean(name));
    expect([...documented].sort()).toEqual([...REQUIRED_SECRET_NAMES].sort());
  });

  it("documents the Idempotency-Key forms /v1/deliver accepts, and not the old one", () => {
    expect(readme).not.toMatch(/Idempotency-Key:\s*YYYY-MM-DD/);
    const auto = /arxiv-daily:auto:<(\d+) lowercase hex>/.exec(readme);
    const test = /arxiv-daily:test:<(\d+) lowercase hex>/.exec(readme);
    expect(auto && test).toBeTruthy();
    expect(deliveryKeyKind(`arxiv-daily:auto:${"a".repeat(Number(auto![1]))}`)).toBe("auto");
    expect(deliveryKeyKind(`arxiv-daily:test:${"a".repeat(Number(test![1]))}`)).toBe("test");
    expect(readme).toMatch(/`Idempotency-Key` \(\*\*required\*\*\)/);
  });

  it("documents every route the Worker serves and the readiness reasons it reports", () => {
    for (const route of [
      "/health", "/ready", "/v1/verify/start", "/v1/verify?token=", "/v1/deliver",
      "/internal/delivery-v2/cutover", "recipient-v2:", "device-v2:",
    ]) {
      expect(readme, route).toContain(route);
    }
    for (const reason of ["build_identity_missing", "runtime_not_configured", "cutover_locked"]) {
      expect(readme, reason).toContain(reason);
      expect(relayFile("src/index.ts"), reason).toContain(`"${reason}"`);
    }
  });

  it("documents the vars wrangler.toml really sets", () => {
    const documented = [...readme.matchAll(/^\| `([A-Z_]+)` \|/gm)].map((m) => m[1]!);
    expect(documented.sort()).toEqual(["DAILY_QUOTA", "FROM_EMAIL", "FROM_NAME", "PUBLIC_BASE_URL"]);
    for (const name of documented) expect(wranglerToml).toMatch(new RegExp(`^${name}\\s*=`, "m"));
  });
});
