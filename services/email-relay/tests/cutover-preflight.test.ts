import { describe, expect, it, vi } from "vitest";
import {
  checkSourceReadonly,
  createPreflight,
  REQUIRED_SECRET_NAMES,
  REQUIRED_TOML_NAMES,
} from "../scripts/cutover-preflight.mjs";

const FAKE_TOML = `
[[kv_namespaces]]
binding = "STORE"
id = "abc"

[durable_objects]
bindings = [
  { name = "DELIVER_GATE", class_name = "DeliverGate" },
]

[vars]
PUBLIC_BASE_URL = "https://mail.example.test"
FROM_EMAIL = "daily@example.test"
FROM_NAME = "arXiv Daily"
DAILY_QUOTA = "5"
`;

const FAKE_SHA = "a".repeat(40);

function okRun(stdout = "") {
  return { status: 0, stdout, stderr: "" };
}

function failRun(status = 1, stderr = "") {
  return { status, stdout: "", stderr };
}

function fakeDeps(overrides: Record<string, unknown> = {}) {
  const state = {
    whoami: okRun("logged in as x"),
    secrets: okRun(REQUIRED_SECRET_NAMES.join("\n")),
    dryRun: okRun("dry-run ok"),
    health: { status: 200, text: `{"ok":true}` },
    ready: { status: 200, text: `{"automatic":"ready"}` },
  };
  return {
    state,
    deps: {
      runCommand: vi.fn((cmd: string, args: string[]) => {
        const joined = `${cmd} ${args.join(" ")}`;
        if (joined.includes("rev-parse")) return okRun(FAKE_SHA);
        if (joined.includes("whoami")) return state.whoami;
        if (joined.includes("secret list")) return state.secrets;
        if (joined.includes("dry-run")) return state.dryRun;
        return okRun();
      }),
      readTextFile: vi.fn(() => FAKE_TOML),
      httpGet: vi.fn(async (url: string) => {
        if (url.endsWith("/health")) return state.health;
        if (url.endsWith("/ready")) return state.ready;
        return { status: 404, text: "" };
      }),
      ...overrides,
    },
  };
}

function resultMap(results: Array<{ name: string; status: string }>) {
  return Object.fromEntries(results.map((r) => [r.name, r.status]));
}

describe("checkSourceReadonly", () => {
  it("accepts a clean source with dry-run deploy only", () => {
    const source = [
      'const args = ["wrangler", "deploy", "src/index.ts", "--dry-run"];',
      'console.log("nothing else");',
    ].join("\n");
    expect(checkSourceReadonly(source)).toEqual([]);
  });

  it("flags a deploy without --dry-run", () => {
    const source = 'const args = ["wrangler", "deploy", "src/index.ts"];';
    const issues = checkSourceReadonly(source);
    expect(issues).toHaveLength(1);
    expect(issues[0]).toContain("deploy reference(s)");
  });

  it("flags forbidden endpoint and KV mutation tokens", () => {
    const source = [
      'const a = "v1/deliver";',
      'const b = "kv:key put";',
      'const c = "internal/delivery-v2/cutover";',
    ].join("\n");
    const issues = checkSourceReadonly(source);
    expect(issues.length).toBeGreaterThanOrEqual(3);
  });

  it("passes its own real source", async () => {
    // The running script's own source must satisfy the read-only contract.
    const fs = await import("node:fs");
    const { fileURLToPath } = await import("node:url");
    const source = fs.readFileSync(
      fileURLToPath(new URL("../scripts/cutover-preflight.mjs", import.meta.url)),
      "utf8",
    );
    expect(checkSourceReadonly(source)).toEqual([]);
  });
});

describe("createPreflight", () => {
  it("passes when every check passes", async () => {
    const { deps, state } = fakeDeps();
    const preflight = createPreflight(deps as never);
    const { results, ok } = await preflight();
    expect(ok).toBe(true);
    const byName = resultMap(results);
    expect(byName).toEqual({
      gitHead: "pass",
      wranglerToml: "pass",
      wranglerLogin: "pass",
      wranglerSecrets: "pass",
      wranglerDryRun: "pass",
      remoteHealth: "pass",
      remoteReady: "pass",
    });
    expect(state.ready.text).toBeDefined();
  });

  it("fails when a required secret name is missing", async () => {
    const { deps } = fakeDeps();
    deps.runCommand = vi.fn((cmd: string, args: string[]) => {
      const joined = `${cmd} ${args.join(" ")}`;
      if (joined.includes("secret list")) return okRun("RESEND_API_KEY\nTOKEN_SECRET\nIDENTITY_SECRET");
      if (joined.includes("rev-parse")) return okRun(FAKE_SHA);
      if (joined.includes("whoami")) return okRun();
      if (joined.includes("dry-run")) return okRun();
      return okRun();
    });
    const { results, ok } = await createPreflight(deps as never)();
    expect(ok).toBe(false);
    const secrets = results.find((r) => r.name === "wranglerSecrets");
    expect(secrets?.status).toBe("fail");
    expect(secrets?.detail).toContain("DELIVERY_V2_CUTOVER_TOKEN");
  });

  it("fails when wrangler login is missing", async () => {
    const { deps, state } = fakeDeps();
    state.whoami = failRun(1, "not logged in");
    const { ok, results } = await createPreflight(deps as never)();
    expect(ok).toBe(false);
    expect(results.find((r) => r.name === "wranglerLogin")?.status).toBe("fail");
  });

  it("skips login when --skip-login is set", async () => {
    const { deps } = fakeDeps();
    deps.runCommand = vi.fn((cmd: string, args: string[]) => {
      const joined = `${cmd} ${args.join(" ")}`;
      if (joined.includes("whoami")) return failRun(1);
      if (joined.includes("rev-parse")) return okRun(FAKE_SHA);
      if (joined.includes("secret list")) return okRun(REQUIRED_SECRET_NAMES.join("\n"));
      if (joined.includes("dry-run")) return okRun();
      return okRun();
    });
    const { ok, results } = await createPreflight(deps as never)({ skipLogin: true });
    expect(ok).toBe(true);
    expect(results.find((r) => r.name === "wranglerLogin")?.status).toBe("skip");
  });

  it("fails when the dry-run fails", async () => {
    const { deps, state } = fakeDeps();
    state.dryRun = failRun(1, "syntax error");
    const { ok } = await createPreflight(deps as never)();
    expect(ok).toBe(false);
  });

  it("reports 503 /ready as locked without failing", async () => {
    const { deps, state } = fakeDeps();
    state.ready = { status: 503, text: `{"automatic":"locked"}` };
    const { ok, results } = await createPreflight(deps as never)();
    expect(ok).toBe(true);
    const ready = results.find((r) => r.name === "remoteReady");
    expect(ready?.status).toBe("pass");
    expect(ready?.detail).toContain("locked");
  });

  it("fails when remote endpoints are unreachable", async () => {
    const { deps, state } = fakeDeps();
    state.health = { status: 0, text: "" };
    state.ready = { status: 0, text: "" };
    const { ok, results } = await createPreflight(deps as never)();
    expect(ok).toBe(false);
    expect(results.find((r) => r.name === "remoteHealth")?.status).toBe("fail");
    expect(results.find((r) => r.name === "remoteReady")?.status).toBe("fail");
  });

  it("skips remote checks when the base URL is unavailable", async () => {
    const { deps } = fakeDeps();
    deps.readTextFile = vi.fn(() => FAKE_TOML.replace(/PUBLIC_BASE_URL.*\n/, ""));
    const { ok, results } = await createPreflight(deps as never)();
    expect(ok).toBe(false);
    expect(results.find((r) => r.name === "remoteReadonly")?.status).toBe("skip");
    expect(results.find((r) => r.name === "wranglerToml")?.status).toBe("fail");
  });

  it("never leaks secret values into results", async () => {
    const { deps } = fakeDeps();
    deps.runCommand = vi.fn((cmd: string, args: string[]) => {
      const joined = `${cmd} ${args.join(" ")}`;
      if (joined.includes("secret list")) return okRun(`${REQUIRED_SECRET_NAMES.join("\n")}\nsk-supersecretvalue`);
      if (joined.includes("rev-parse")) return okRun(FAKE_SHA);
      if (joined.includes("whoami")) return okRun();
      if (joined.includes("dry-run")) return okRun("supersecretvalue in output");
      return okRun();
    });
    const { results } = await createPreflight(deps as never)();
    const serialized = JSON.stringify(results);
    expect(serialized).not.toContain("supersecretvalue");
  });

  it("fails when wrangler.toml lacks required names", async () => {
    const { deps } = fakeDeps();
    deps.readTextFile = vi.fn(() => FAKE_TOML.replace('PUBLIC_BASE_URL = "https://mail.example.test"\n', ""));
    const { ok, results } = await createPreflight(deps as never)();
    expect(ok).toBe(false);
    const toml = results.find((r) => r.name === "wranglerToml");
    expect(toml?.status).toBe("fail");
    expect(toml?.detail).toContain("PUBLIC_BASE_URL");
  });

  it("rejects a missing or non-40-hex git SHA", async () => {
    const { deps } = fakeDeps();
    deps.runCommand = vi.fn((cmd: string, args: string[]) => {
      const joined = `${cmd} ${args.join(" ")}`;
      if (joined.includes("rev-parse")) return failRun(128, "not a git repo");
      if (joined.includes("secret list")) return okRun(REQUIRED_SECRET_NAMES.join("\n"));
      if (joined.includes("whoami")) return okRun();
      return okRun();
    });
    const { ok, results } = await createPreflight(deps as never)();
    expect(ok).toBe(false);
    expect(results.find((r) => r.name === "gitHead")?.status).toBe("fail");
    expect(results.find((r) => r.name === "wranglerDryRun")?.status).toBe("fail");
  });
});
