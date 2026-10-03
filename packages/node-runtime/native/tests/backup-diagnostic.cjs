// Fixed test data only. No vault, credential or application configuration reads.
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const assert = require("node:assert/strict");

const buildDirectory = path.resolve(__dirname, "../build");
const bindingPath = path.join(buildDirectory, "Release/private_storage.node");
const oldHex = Buffer.from("old").toString("hex");
const newHex = Buffer.from("new").toString("hex");

function snapshot(parent) {
  return Object.fromEntries(["primary.json", "next.tmp", "primary.bak"].map((name) => {
    const filename = path.join(parent, name);
    if (!fs.existsSync(filename)) return [name, null];
    const stat = fs.statSync(filename, { bigint: true });
    // Bound diagnostic output even if a faulty implementation gives a file a huge size.
    const fd = fs.openSync(filename, "r");
    const bytes = Buffer.alloc(64);
    let count;
    try { count = fs.readSync(fd, bytes, 0, bytes.length, 0); } finally { fs.closeSync(fd); }
    return [name, {
      hex: bytes.subarray(0, count).toString("hex"), size: String(stat.size),
      dev: String(stat.dev), ino: String(stat.ino), nlink: String(stat.nlink),
    }];
  }));
}

function nativeMetadata() {
  try { return JSON.parse(fs.readFileSync(path.join(buildDirectory, "native-build.json"), "utf8")); }
  catch { return { available: false }; }
}

function runBackupDiagnostics({ binding, iterations = 100 } = {}) {
  if (!Number.isSafeInteger(iterations) || iterations < 1 || iterations > 100) {
    throw new Error("iterations must be an integer from 1 to 100");
  }
  const api = binding ?? require(bindingPath);
  const names = ["native-final-only", "native-staged"];
  if (process.platform !== "win32") names.push("node-positioned-write", "node-without-truncate");
  const result = {
    ok: true, platform: process.platform, arch: process.arch, node: process.versions.node,
    os: { release: os.release(), version: os.version() },
    filesystem: { type: "unknown" }, nativeBuild: nativeMetadata(), modes: [],
  };
  for (const name of names) {
    const mode = { name, attempted: 0, completed: 0, failure: null };
    result.modes.push(mode);
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "arxiv-backup-diagnostic-"));
      const parent = path.join(root, "state");
      let directory;
      let stage = "create-directory";
      const snapshots = [];
      let oldIdentity;
      mode.attempted += 1;
      const inspect = (nextStage, primary, backup = null, next = null) => {
        if (name === "native-final-only" && nextStage !== "directory-sync") return;
        stage = nextStage;
        const files = snapshot(parent);
        snapshots.push({ stage, files });
        assert.equal(files["primary.json"]?.hex, primary, `${stage}: primary bytes`);
        assert.equal(files["primary.bak"]?.hex ?? null, backup, `${stage}: backup bytes`);
        assert.equal(files["next.tmp"]?.hex ?? null, next, `${stage}: next bytes`);
        const main = files["primary.json"];
        const saved = files["primary.bak"];
        if (nextStage === "write-primary.json") oldIdentity = [main.dev, main.ino];
        if (saved) {
          if (oldIdentity) assert.deepEqual([saved.dev, saved.ino], oldIdentity, `${stage}: backup inode`);
          if (nextStage === "first-link" || nextStage === "occupied-link") {
            assert.deepEqual([saved.dev, saved.ino], [main.dev, main.ino], `${stage}: shared inode`);
            assert.equal(saved.nlink, "2", `${stage}: linked count`);
          } else {
            assert.notDeepEqual([saved.dev, saved.ino], [main.dev, main.ino], `${stage}: replacement inode`);
            assert.equal(saved.nlink, "1", `${stage}: retained link count`);
          }
        }
      };
      try {
        result.filesystem.type = `0x${fs.statfsSync(root, { bigint: true }).type.toString(16)}`;
        if (name.startsWith("native")) directory = api.openDirectory(root, "state", true);
        else fs.mkdirSync(parent, { mode: 0o700 });
        for (const [filename, text] of [["primary.json", "old"], ["next.tmp", "new"]]) {
          stage = `write-${filename}`;
          const next = filename === "next.tmp" ? newHex : null;
          if (directory) {
            const file = directory.createFile(filename);
            assert.ok(file, `${stage}: exclusive create`);
            try {
              file.writeText(text);
              inspect(stage, oldHex, null, next);
            } finally { file.close(); }
          } else {
            const fd = fs.openSync(path.join(parent, filename), "wx+", 0o600);
            try {
              const bytes = Buffer.from(text);
              assert.equal(fs.writeSync(fd, bytes, 0, bytes.length, 0), bytes.length);
              if (name === "node-positioned-write") fs.ftruncateSync(fd, bytes.length);
              fs.fsyncSync(fd);
              inspect(stage, oldHex, null, next);
            } finally { fs.closeSync(fd); }
          }
          inspect(`close-${filename}`, oldHex, null, next);
        }
        stage = "first-link";
        if (directory) assert.equal(directory.link("primary.json", "primary.bak"), true);
        else fs.linkSync(path.join(parent, "primary.json"), path.join(parent, "primary.bak"));
        inspect(stage, oldHex, oldHex, newHex);
        stage = "occupied-link";
        if (directory) assert.equal(directory.link("primary.json", "primary.bak"), false);
        else assert.throws(() => fs.linkSync(path.join(parent, "primary.json"), path.join(parent, "primary.bak")), { code: "EEXIST" });
        inspect(stage, oldHex, oldHex, newHex);
        stage = "rename";
        if (directory) directory.rename("next.tmp", "primary.json");
        else fs.renameSync(path.join(parent, "next.tmp"), path.join(parent, "primary.json"));
        inspect(stage, newHex, oldHex);
        stage = "directory-sync";
        if (directory) directory.sync();
        else {
          const fd = fs.openSync(parent, "r");
          try { fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
        }
        inspect(stage, newHex, oldHex);
        mode.completed += 1;
      } catch (error) {
        mode.failure = { stage, iteration, message: String(error), snapshots };
        // A failing operation may precede its checkpoint; preserve bounded evidence before cleanup.
        if (snapshots.at(-1)?.stage !== stage) {
          try { snapshots.push({ stage, files: snapshot(parent) }); } catch { /* keep the original failure */ }
        }
        result.ok = false;
      } finally {
        directory?.close();
        fs.rmSync(root, { recursive: true, force: true });
      }
      if (mode.failure) break;
    }
  }
  return result;
}

module.exports = { runBackupDiagnostics };
if (require.main === module) {
  try {
    const report = runBackupDiagnostics();
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    if (!report.ok) process.exitCode = 1;
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ ok: false, error: String(error) })}\n`);
    process.exitCode = 1;
  }
}
