import { describe, expect, it, vi } from "vitest";
import {
  createDeploy,
  parseArgs,
} from "../scripts/deploy-with-identity.mjs";

const SHA = "a".repeat(40);
const OTHER_SHA = "b".repeat(40);
const IDENTITY = `email-relay-v2-${SHA}`;
const OTHER_IDENTITY = `email-relay-v2-${OTHER_SHA}`;
const TOML = `[vars]\nPUBLIC_BASE_URL = "https://mail.example.test"\n`;

const ok = (stdout = "") => ({ status: 0, stdout, stderr: "" });
const failed = (status = 1, stderr = "") => ({ status, stdout: "", stderr });

type Fake = {
  status: { status: number; stdout: string; stderr: string };
  head: { status: number; stdout: string; stderr: string };
  deploy: { status: number; stdout: string; stderr: string };
  ready: { status: number; text: string } | Error;
};

function setup(overrides: Partial<Fake> = {}) {
  const state: Fake = {
    status: ok(""),
    head: ok(`${SHA}\n`),
    deploy: ok(""),
    ready: {
      status: 503,
      text: JSON.stringify({ phase: "locked", automatic: "locked", buildIdentity: IDENTITY }),
    },
    ...overrides,
  };
  const runCommand = vi.fn((cmd: string, args: string[]) => {
    if (cmd === "git" && args[0] === "status") return state.status;
    if (cmd === "git" && args[0] === "rev-parse") return state.head;
    if (args.includes("deploy")) return state.deploy;
    return failed(127, `unexpected ${cmd} ${args.join(" ")}`);
  });
  const httpGet = vi.fn(async (url: string) => {
    if (state.ready instanceof Error) throw state.ready;
    return url.endsWith("/ready") ? state.ready : { status: 404, text: "" };
  });
  const deploy = createDeploy({
    runCommand,
    readTextFile: () => TOML,
    httpGet,
  });
  const deployCalls = () =>
    runCommand.mock.calls.filter(([, args]) => (args as string[]).includes("deploy"));
  return { state, runCommand, httpGet, deploy, deployCalls };
}

const names = (steps: Array<{ name: string; status: string }>) =>
  Object.fromEntries(steps.map((step) => [step.name, step.status]));

describe("deploy with identity", () => {
  it("deploys once with the identity of HEAD injected as the only variable", async () => {
    const { deploy, deployCalls, httpGet } = setup();

    const result = await deploy();

    expect(result.ok).toBe(true);
    expect(result.identity).toBe(IDENTITY);
    expect(deployCalls()).toHaveLength(1);
    expect(deployCalls()[0]).toEqual([
      "npx",
      ["--no-install", "wrangler", "deploy", "--var", `BUILD_IDENTITY:${IDENTITY}`],
      { inherit: true },
    ]);
    expect(httpGet).toHaveBeenCalledWith("https://mail.example.test/ready");
    expect(names(result.steps)).toMatchObject({
      cleanTree: "pass",
      gitHead: "pass",
      remoteReady: "pass",
      deploy: "pass",
    });
  });

  it("refuses a tree with uncommitted changes and never calls wrangler", async () => {
    const { deploy, deployCalls, httpGet } = setup({ status: ok(" M src/index.ts\n") });

    const result = await deploy();

    expect(result.ok).toBe(false);
    expect(result.steps.at(-1)).toMatchObject({ name: "cleanTree", status: "fail" });
    expect(deployCalls()).toHaveLength(0);
    expect(httpGet).not.toHaveBeenCalled();
  });

  it("refuses untracked files too, because the identity must name what is deployed", async () => {
    const { deploy, deployCalls } = setup({ status: ok("?? src/scratch.ts\n") });
    expect((await deploy()).ok).toBe(false);
    expect(deployCalls()).toHaveLength(0);
  });

  it.each([
    ["git fails", failed(128, "not a git repository")],
    ["an empty answer", ok("")],
    ["a short SHA", ok("abc123\n")],
    ["an upper-case SHA", ok(`${"A".repeat(40)}\n`)],
    ["a SHA with trailing text", ok(`${SHA} extra\n`)],
  ])("refuses when HEAD is not a 40-hex SHA (%s)", async (_label, head) => {
    const { deploy, deployCalls } = setup({ head });

    const result = await deploy();

    expect(result.ok).toBe(false);
    expect(result.steps.at(-1)).toMatchObject({ name: "gitHead", status: "fail" });
    expect(deployCalls()).toHaveLength(0);
  });

  it.each([
    ["ready", 200],
    ["ready", 503],
    ["sealed", 503],
    ["observing", 503],
    ["inventoried", 503],
  ])("refuses to deploy under a different identity once the relay is bound (%s, HTTP %i)", async (phase, httpStatus) => {
    const { deploy, deployCalls } = setup({
      ready: {
        status: httpStatus,
        text: JSON.stringify({
          protocolGeneration: 2,
          buildIdentity: OTHER_IDENTITY,
          phase,
          automatic: phase === "ready" ? "ready" : "locked",
          readyGeneration: phase === "ready" ? 6 : null,
        }),
      },
    });

    const result = await deploy();

    expect(result.ok).toBe(false);
    const refusal = result.steps.at(-1)!;
    expect(refusal).toMatchObject({ name: "remoteReady", status: "fail" });
    expect(refusal.detail).toContain(OTHER_IDENTITY);
    expect(refusal.detail).toContain(`npm run deploy -- --identity ${OTHER_IDENTITY}`);
    expect(deployCalls()).toHaveLength(0);
  });

  it("deploys when the bound identity is already the one of HEAD", async () => {
    const { deploy, deployCalls } = setup({
      ready: {
        status: 200,
        text: JSON.stringify({ buildIdentity: IDENTITY, phase: "ready", automatic: "ready", readyGeneration: 6 }),
      },
    });
    expect((await deploy()).ok).toBe(true);
    expect(deployCalls()).toHaveLength(1);
  });

  it("releases a newer commit under the bound identity when asked to", async () => {
    const { deploy, deployCalls } = setup({
      ready: {
        status: 200,
        text: JSON.stringify({ buildIdentity: OTHER_IDENTITY, phase: "ready", automatic: "ready", readyGeneration: 6 }),
      },
    });

    const result = await deploy({ identity: OTHER_IDENTITY });

    expect(result.ok).toBe(true);
    expect(result.identity).toBe(OTHER_IDENTITY);
    expect(deployCalls()[0]![1]).toEqual([
      "--no-install", "wrangler", "deploy", "--var", `BUILD_IDENTITY:${OTHER_IDENTITY}`,
    ]);
  });

  it("still refuses a given identity that differs from the bound one", async () => {
    const { deploy, deployCalls } = setup({
      ready: {
        status: 200,
        text: JSON.stringify({ buildIdentity: OTHER_IDENTITY, phase: "ready", automatic: "ready", readyGeneration: 6 }),
      },
    });
    const result = await deploy({ identity: `email-relay-v2-${"c".repeat(40)}` });
    expect(result.ok).toBe(false);
    expect(deployCalls()).toHaveLength(0);
  });

  it.each(["email-relay-v2-abc", "relay-v2-" + "a".repeat(40), `email-relay-v2-${"A".repeat(40)}`, ""])(
    "rejects a malformed given identity %j",
    async (identity) => {
      const { deploy, deployCalls } = setup();
      const result = await deploy({ identity });
      expect(result.ok).toBe(false);
      expect(result.steps.at(-1)).toMatchObject({ name: "identity", status: "fail" });
      expect(deployCalls()).toHaveLength(0);
    },
  );

  it("deploys before the cutover has started, whatever the earlier build was", async () => {
    const { deploy, deployCalls } = setup({
      ready: {
        status: 503,
        text: JSON.stringify({ buildIdentity: OTHER_IDENTITY, phase: "locked", automatic: "locked", readyGeneration: null }),
      },
    });

    const result = await deploy();

    expect(result.ok).toBe(true);
    expect(result.identity).toBe(IDENTITY);
    expect(deployCalls()).toHaveLength(1);
    expect(result.steps.find((step) => step.name === "remoteReady")?.detail).toContain("no identity bound yet");
  });

  it.each([
    ["an unreachable relay", new Error("getaddrinfo ENOTFOUND"), "unreachable"],
    ["a relay that reports a locked runtime", { status: 503, text: JSON.stringify({ automatic: "locked", reason: "build_identity_missing" }) }, "build_identity_missing"],
    ["an older relay that only says locked", { status: 503, text: JSON.stringify({ automatic: "locked" }) }, "could not tell"],
    ["an error page", { status: 502, text: "<html>Bad gateway</html>" }, "could not tell"],
  ] as const)("refuses an implicit identity over %s", async (_label, ready, expected) => {
    const { deploy, deployCalls } = setup({ ready: ready as Fake["ready"] });

    const result = await deploy();

    expect(result.ok).toBe(false);
    const remote = result.steps.find((step) => step.name === "remoteReady")!;
    expect(remote.status).toBe("fail");
    expect(remote.detail).toContain(expected);
    expect(remote.detail).toContain("--identity");
    expect(deployCalls()).toHaveLength(0);
  });

  it("allows recovery with an explicitly recorded identity when readiness is unavailable", async () => {
    const { deploy, deployCalls } = setup({ ready: new Error("offline") });
    const result = await deploy({ identity: OTHER_IDENTITY });
    expect(result.ok).toBe(true);
    expect(result.identity).toBe(OTHER_IDENTITY);
    expect(deployCalls()).toHaveLength(1);
    expect(result.steps.find(step => step.name === "remoteReady")?.status).toBe("warn");
  });

  it("requires an explicit identity when no readiness URL is configured", async () => {
    const runCommand = vi.fn((cmd: string, args: string[]) => ok(args[0] === "rev-parse" ? SHA : ""));
    const httpGet = vi.fn();
    const deploy = createDeploy({ runCommand, readTextFile: () => "[vars]", httpGet });
    expect((await deploy()).ok).toBe(false);
    expect(runCommand.mock.calls.some(([, args]) => args.includes("deploy"))).toBe(false);
    expect(httpGet).not.toHaveBeenCalled();
    expect((await deploy({ identity: IDENTITY })).ok).toBe(true);
  });

  it("does not accept an error response as proof of an unbound relay", async () => {
    const { deploy, deployCalls } = setup({ ready: {
      status: 500, text: JSON.stringify({ phase: "locked", automatic: "locked" }),
    } });
    expect((await deploy()).ok).toBe(false);
    expect(deployCalls()).toHaveLength(0);
  });

  it("checks everything and prints the command on a dry run without deploying", async () => {
    const { deploy, deployCalls } = setup();

    const result = await deploy({ dryRun: true });

    expect(result.ok).toBe(true);
    expect(deployCalls()).toHaveLength(0);
    expect(result.command).toBe(`npx --no-install wrangler deploy --var BUILD_IDENTITY:${IDENTITY}`);
    expect(result.steps.at(-1)).toMatchObject({ name: "deploy", status: "skip" });
  });

  it("reports a failed wrangler deploy", async () => {
    const { deploy } = setup({ deploy: failed(1) });
    const result = await deploy();
    expect(result.ok).toBe(false);
    expect(result.steps.at(-1)).toMatchObject({ name: "deploy", status: "fail" });
  });

  it("uses an explicit remote and never needs a secret", async () => {
    const { deploy, httpGet, runCommand } = setup();
    await deploy({ remoteBaseUrl: "https://other.example.test/" });
    expect(httpGet).toHaveBeenCalledWith("https://other.example.test/ready");
    const everything = JSON.stringify(runCommand.mock.calls);
    expect(everything).not.toMatch(/secret|token/i);
  });

  it("parses its arguments", () => {
    expect(parseArgs([])).toMatchObject({ dryRun: false, errors: [] });
    expect(parseArgs(["--dry-run", "--identity", IDENTITY, "--remote", "https://x.test"]))
      .toEqual({ dryRun: true, identity: IDENTITY, remoteBaseUrl: "https://x.test", errors: [] });
    expect(parseArgs(["--identity"]).errors).toEqual(["--identity needs a value"]);
    expect(parseArgs(["--identity", "--dry-run"]).errors).toEqual(["--identity needs a value"]);
    expect(parseArgs(["--force"]).errors).toEqual(["unknown argument --force"]);
  });
});
