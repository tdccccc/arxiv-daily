import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve, join } from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

/** Runs only email status and local native I/O; the child cannot make HTTP calls. */
export async function smokeNativePackage(cli = resolve(import.meta.dirname, "../apps/cli/dist/arxiv-daily-cli.cjs")) {
  const temporary = await mkdtemp(join(tmpdir(), "arxiv-native-package-"));
  const home = join(temporary, "home");
  const config = join(temporary, "config");
  const vault = join(temporary, "vault");
  try {
    await Promise.all([mkdir(home, { mode: 0o700 }), mkdir(vault, { mode: 0o700 }), mkdir(join(config, "arxiv-daily"), { recursive: true })]);
    const offline = join(temporary, "offline.cjs");
    await writeFile(offline, `const deny = () => { throw new Error('HTTP is forbidden in native package smoke'); }; globalThis.fetch = deny; require('node:http').request = deny; require('node:https').request = deny;\n`);
    await writeFile(join(config, "arxiv-daily/config.toml"), `schema_version = 1\nvault_root = ${JSON.stringify(vault)}\ncache_dir = ${JSON.stringify(join(temporary, "cache"))}\n[llm]\napi_key = "fixture-only"\nbase_url = "https://example.invalid/v1"\nmodel = "fixture"\n[email]\nenabled = true\nmode = "self"\nto = "fixture@example.invalid"\nfrom_email = "from@example.invalid"\napi_key = "fixture-only"\n`);
    const result = spawnSync(process.execPath, ["--require", offline, cli, "email", "status"], {
      cwd: temporary, encoding: "utf8",
      env: { ...process.env, PATH: "", HOME: home, USERPROFILE: home, XDG_CONFIG_HOME: config, APPDATA: config },
    });
    if (result.status !== 0 || !result.stdout.includes("auto-send: would run on completed daily")) {
      throw new Error(`Native package status failed: ${result.status}\n${result.stdout}\n${result.stderr}`);
    }
    const cache = join(home, ".arxiv-daily/native");
    const entries = await readdir(cache).catch(() => []);
    const hashes = entries.filter(name => /^[0-9a-f]{64}$/.test(name));
    if (hashes.length !== 1) throw new Error("CLI did not load exactly one bundled native platform component");
    const binary = join(cache, hashes[0], "private-storage.node");
    const probe = `const api = require(process.argv[1]); const dir = api.openDirectory(process.argv[2], 'native-probe', true); try { const file = dir.createFile('private.json'); if (!file) throw new Error('missing native file'); try { file.writeText('local fixture'); if (!file.isPrivate()) throw new Error('native file is not private'); } finally { file.close(); } dir.assertCurrent(); } finally { dir.close(); }`;
    const native = spawnSync(process.execPath, ["-e", probe, binary, vault], { encoding: "utf8" });
    if (native.status !== 0) throw new Error(`Packaged native storage probe failed: ${native.stderr}`);
    if (await readFile(join(vault, "native-probe/private.json"), "utf8") !== "local fixture") throw new Error("Packaged native write did not commit");
    console.log(`Native package smoke OK (${process.platform}-${process.arch}, offline)`);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await smokeNativePackage(process.argv[2] ? resolve(process.argv[2]) : undefined);
}
