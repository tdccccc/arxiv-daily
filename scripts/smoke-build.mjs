import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { noticeBanner, readPakoNotice } from "./release-utils.mjs";

const root = resolve(import.meta.dirname, "..");
const cli = resolve(root, "apps/cli/dist/arxiv-daily-cli.cjs");
const pluginCli = resolve(root, "plugin/arxiv-daily-cli.cjs");
const pythonShim = resolve(root, "arxiv_daily.py");
const pluginBundle = resolve(root, "plugin/main.js");

for (const command of [
  [process.execPath, cli, "--help"],
  [process.execPath, pluginCli, "--help"],
  ["python3", pythonShim, "--help"],
]) {
  const result = spawnSync(command[0], command.slice(1), { cwd: root, encoding: "utf8" });
  if (result.status !== 0 || !result.stdout.includes("Usage:")) {
    fail(`${command.join(" ")} help failed`, result);
  }
}

const temp = await mkdtemp(resolve(tmpdir(), "arxiv-daily-smoke-"));
try {
  const configHome = resolve(temp, "config");
  const badConfig = resolve(configHome, "arxiv-daily/config.toml");
  await mkdir(resolve(configHome, "arxiv-daily"), { recursive: true });
  await writeFile(badConfig, "{not-toml}\n");
  const result = spawnSync(process.execPath, [cli, "run", "--date", "2026-07-15"], {
    cwd: temp,
    encoding: "utf8",
    env: { ...process.env, XDG_CONFIG_HOME: configHome },
  });
  if (result.status !== 2 || !result.stderr.includes("failed to parse CLI config")) {
    fail("CLI did not return exit 2 for a controlled fixed-path TOML config error", result);
  }
} finally {
  await rm(temp, { recursive: true, force: true });
}

const expectedNotice = noticeBanner(await readPakoNotice());
for (const path of [pluginBundle, cli, pluginCli]) {
  const built = await readFile(path, "utf8");
  const count = built.split(expectedNotice).length - 1;
  if (count !== 1) {
    throw new Error(`${path} must contain exactly one complete locked pako notice; found ${count}`);
  }
  for (const required of [
    "Copyright (C) 2014-2017 by Vitaly Puzrin and Andrei Tuputcyn",
    "The above copyright notice and this permission notice shall be included in",
    "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN",
  ]) {
    if (!built.includes(required)) throw new Error(`${path} contains a truncated pako notice`);
  }
}

const bundle = await readFile(pluginBundle, "utf8");
if (bundle.includes("getBuiltinModule")) {
  throw new Error("plugin/main.js must not depend on process.getBuiltinModule");
}
for (const forbidden of ["@arxiv-daily/", "linkedom", "canvas"]) {
  if (bundle.includes(forbidden)) throw new Error(`plugin/main.js contains forbidden text: ${forbidden}`);
}
const runtimeRequires = new Set(
  Array.from(bundle.matchAll(/require\(["']([^"']+)["']\)/g), (match) => match[1]),
);
const allowedRuntimeRequires = new Set([
  "obsidian",
  "electron",
  "node:fs",
  "node:fs/promises",
  "node:path",
]);
for (const specifier of runtimeRequires) {
  if (!allowedRuntimeRequires.has(specifier)) {
    throw new Error(`plugin/main.js contains unexpected runtime require: ${specifier}`);
  }
}
console.log("Build smoke OK");

function fail(message, result) {
  throw new Error(`${message}\nstatus=${result.status}\nstdout=${result.stdout}\nstderr=${result.stderr}`);
}
