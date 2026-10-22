import { afterEach, expect, it } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { parse } from "smol-toml";
import { loadCliConfig } from "../src/config";
import { patchConfigToml } from "../src/config-write";
import { renderInitToml } from "../src/init";
import { connectCliLibrary } from "../src/library-connection-cmd";
import { createCliTopicSettings } from "../src/library-topic-settings";
import { readWorkbenchSettings, saveWorkbenchSettings } from "../src/workbench/settings";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => fs.rm(root, { recursive: true, force: true }))); });

/** What `init` writes, plus a note the user added beside a value. */
function commentedConfig(root: string, vaultRoot = path.join(root, "vault")): string {
  return renderInitToml({
    vaultRoot, cacheDir: path.join(root, "cache"), apiKey: "sk-fixture-key-123456", baseUrl: "https://api.example.com/v1", model: "model-a", provider: "custom",
    thinkingMode: true, reasoningEffort: "high", categories: ["cs.LG", "cs.AI"], timezone: "UTC", summaryLanguage: "zh",
    topic: { name: "Cosmology", tag: "cosmo", description: "photometric redshift estimation", detail: false },
    email: { enabled: false, mode: "self", to: "", apiKey: "", hostedToken: "" },
    linkStyle: "wikilink", logLevel: "info",
    schedule: { enabled: false, on: "09:30", intervalHours: 0, until: "18:00", weekdaysOnly: true },
  }).replace('model = "model-a"', 'model = "model-a"  # my note: switch to model-b on Fridays');
}

async function fixture(vaultRoot?: string) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-config-write-"));
  roots.push(root);
  const configPath = path.join(root, "config.toml");
  const original = commentedConfig(root, vaultRoot);
  await fs.writeFile(configPath, original, { mode: 0o600 });
  return { root, configPath, original, read: () => fs.readFile(configPath, "utf8") };
}

const commentLines = (text: string) => text.split("\n").filter(line => /^\s*#/.test(line) || /\s#\s/.test(line));

it("changes only the edited value when the workbench saves an unrelated setting", async () => {
  const { configPath, original, read } = await fixture();
  const snapshot = await readWorkbenchSettings(configPath);
  await saveWorkbenchSettings(configPath, { revision: snapshot.revision, values: { ...snapshot.values, maxDailyPapers: 25 } });
  const saved = await read();
  expect(saved).toBe(original.replace(/^max_daily_papers = \d+$/m, "max_daily_papers = 25"));
  expect(saved).toContain("# my note: switch to model-b on Fridays");
  expect((await loadCliConfig({ configPath })).settings.output.maxDailyPapers).toBe(25);
});

it("keeps every comment and the order of the file when several settings change", async () => {
  const { configPath, original, read } = await fixture();
  const snapshot = await readWorkbenchSettings(configPath);
  const topics = snapshot.values.topics.map(topic => ({ ...topic, directions: topic.directions.map(direction => ({ ...direction, text: "photometric redshift estimation, revised" })) }));
  await saveWorkbenchSettings(configPath, { revision: snapshot.revision, values: {
    ...snapshot.values, model: "model-b", categories: ["cs.LG", "cs.AI", "astro-ph.CO"], topics, summaryLanguage: "en", schedule: { ...snapshot.values.schedule, enabled: true },
  } });
  const saved = await read();
  expect(commentLines(saved)).toEqual(expect.arrayContaining(commentLines(original).filter(line => !line.includes("model-a"))));
  expect(saved).toContain("# my note: switch to model-b on Fridays");
  const tables = [...saved.matchAll(/^\[([a-z_.]+)\]$/gm)].map(match => match[1]);
  expect(tables).toEqual(["llm", "arxiv", "output", "email", "schedule", "advanced", "workbench_schedule"]);
  const config = await loadCliConfig({ configPath });
  expect(config.settings.llm.model).toBe("model-b");
  expect(config.settings.llm.apiKey).toBe("sk-fixture-key-123456");
  expect(config.settings.arxiv.categories).toEqual(["cs.LG", "cs.AI", "astro-ph.CO"]);
  expect(config.settings.arxiv.topics[0]?.directions.map(direction => direction.text)).toEqual(["photometric redshift estimation, revised"]);
  expect(config.settings.output.summaryLanguage).toBe("en");
  expect(config.workbenchSchedule?.enabled).toBe(true);
});

it("keeps a hand-written vault_root such as ~/notes unless the user changes it", async () => {
  const { root, configPath, read } = await fixture("~/arxiv-config-write-notes");
  const snapshot = await readWorkbenchSettings(configPath);
  expect(snapshot.values.vaultRoot).toBe(path.join(os.homedir(), "arxiv-config-write-notes"));
  const saved = await saveWorkbenchSettings(configPath, { revision: snapshot.revision, values: { ...snapshot.values, maxDailyPapers: 25 } });
  expect(await read()).toContain('vault_root = "~/arxiv-config-write-notes"');
  const moved = path.join(root, "moved");
  await saveWorkbenchSettings(configPath, { revision: saved.configRevision, values: { ...snapshot.values, vaultRoot: moved } });
  expect(await read()).toContain(`vault_root = ${JSON.stringify(moved)}`);
});

it("appends the library connection to the end of a commented config", async () => {
  const { configPath, original, read } = await fixture();
  const folder = path.join(path.dirname(configPath), "papers");
  await fs.mkdir(folder);
  const next = await connectCliLibrary(await loadCliConfig({ configPath }), folder);
  const saved = await read();
  expect(saved.startsWith(original.trimEnd())).toBe(true);
  expect(saved.slice(original.trimEnd().length)).toMatch(/^\n\n\[library\]\n/);
  expect(parse(saved).library).toEqual(next.libraryConnection);
});

it("keeps comments when accepted directions and their receipts are saved", async () => {
  const { configPath, original, read } = await fixture();
  const config = await loadCliConfig({ configPath });
  const receipt = { scopeFingerprint: `sha256:${"a".repeat(64)}`, proposalId: "proposal", topicTargets: { candidate: config.settings.arxiv.topics[0]!.id }, processedCandidateIds: ["direction"] };
  await createCliTopicSettings(config).change(async current => ({
    topics: current.topics.map(topic => ({ ...topic, directions: [...topic.directions, { id: "accepted", text: "Accepted direction", origin: "library" as const }] })),
    acceptances: [receipt],
  }));
  const saved = await read();
  expect(commentLines(saved)).toEqual(expect.arrayContaining(commentLines(original)));
  expect(saved.slice(saved.lastIndexOf("[[library_proposal_acceptances]]") - 2)).toMatch(/^\n\n\[\[library_proposal_acceptances\]\]\n/);
  const reloaded = await loadCliConfig({ configPath });
  expect(reloaded.settings.arxiv.topics[0]?.directions.map(direction => direction.text)).toEqual(["photometric redshift estimation", "Accepted direction"]);
  expect(parse(saved).library_proposal_acceptances).toEqual([receipt]);
  expect(reloaded.settings.llm.apiKey).toBe("sk-fixture-key-123456");
});

it("writes the whole document out, losing comments but no value, when a change cannot be made in place", () => {
  const raw = [
    "# header", 'vault_root = "/v"', "", "[arxiv]", "# first topic", "[[arxiv.topics]]", 'id = "a"', 'name = "A"', "",
    "[[arxiv.topics.directions]]", 'id = "a1"', 'text = "one"', "", "[[arxiv.topics]]", 'id = "b"', 'name = "B"', "",
    "[[arxiv.topics.directions]]", 'id = "b1"', 'text = "two"', "",
  ].join("\n");
  const before = parse(raw) as { arxiv: { topics: unknown[] } };
  const next = { ...before, arxiv: { ...before.arxiv, topics: [...before.arxiv.topics].reverse() } };
  const text = patchConfigToml(raw, next);
  expect(parse(text)).toEqual(next);
});

it("keeps the line endings of a CRLF file when text is appended", () => {
  const raw = '# header\r\nvault_root = "/v"\r\n\r\n[llm]\r\napi_key = "k"  # note\r\n';
  const next = { ...parse(raw), library: { selected_root: "/lib", state: "connected" }, items: [{ a: 1 }, { a: 2 }] };
  const text = patchConfigToml(raw, next);
  expect(text.replace(/\r\n/g, "")).not.toContain("\n");
  expect(text).toContain("# header\r\n");
  expect(text).toContain("api_key = \"k\"  # note");
  expect(parse(text)).toEqual(next);
});

it("round-trips values that need escaping or are not ASCII", () => {
  const raw = 'vault_root = "/v"\n\n[llm]\n# the key\napi_key = "k"\nmodel = "m"\n';
  const next = { ...parse(raw), llm: { api_key: 'pa$$ "word"\\ \'x\'', model: "模型 — é\nsecond line" } };
  const text = patchConfigToml(raw, next);
  expect(parse(text)).toEqual(next);
  expect(text).toContain("# the key");
});
