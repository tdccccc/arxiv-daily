const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { test, afterEach } = require("node:test");

const bindingPath = path.resolve(__dirname, "../build/Release/private_storage.node");
const roots = [];
function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "arxiv-native-storage-"));
  roots.push(root);
  return root;
}
function binding() { return require(bindingPath); }
afterEach(() => { while (roots.length) fs.rmSync(roots.pop(), { recursive: true, force: true }); });

test("exposes the native directory capability", () => {
  assert.equal(fs.existsSync(bindingPath), true, "build the native storage contract first");
  assert.equal(typeof binding().openDirectory, "function");
});

test("creates private files exclusively and flushes complete content", () => {
  const root = fixture();
  const first = binding().openDirectory(root, "claims/daily", true);
  const second = binding().openDirectory(root, "claims/daily", false);
  try {
    const file = first.createFile("send.json");
    assert.ok(file, "first creator must own a real file");
    try {
      assert.equal(file.isPrivate(), true);
      file.writeText("private recipient: 研究@example.com");
      assert.equal(second.createFile("send.json"), null);
    } finally { file.close(); }
    first.sync();
    assert.equal(fs.readFileSync(path.join(root, "claims/daily/send.json"), "utf8"), "private recipient: 研究@example.com");
    if (process.platform !== "win32") assert.equal(fs.statSync(path.join(root, "claims/daily/send.json")).mode & 0o777, 0o600);
  } finally { first.close(); second.close(); }
});

test("rejects traversal, device names and alternate streams at the native boundary", () => {
  const root = fixture();
  for (const parent of ["../outside", "/absolute", "a/../b", "a//b", "a\\b"]) {
    assert.throws(() => binding().openDirectory(root, parent, true), /path|name/i);
  }
  const dir = binding().openDirectory(root, "claims", true);
  try {
    for (const name of ["../escape", "a/b", "a\\b", "NUL", "con.json", "send:secret", "send.", "send ", "\0"]) {
      assert.throws(() => dir.createFile(name), /path|name/i);
    }
  } finally { dir.close(); }
});

test("treats an existing directory as an occupied exclusive-create name", () => {
  const root = fixture();
  const dir = binding().openDirectory(root, "claims", true);
  try {
    fs.mkdirSync(path.join(root, "claims/occupied"));
    fs.writeFileSync(path.join(root, "claims/occupied/keep"), "untouched");
    assert.equal(dir.createFile("occupied"), null);
    assert.equal(fs.readFileSync(path.join(root, "claims/occupied/keep"), "utf8"), "untouched");
  } finally { dir.close(); }
});

test("does not traverse symlink parents or read a symlink target", () => {
  const root = fixture();
  const outside = fixture();
  fs.symlinkSync(outside, path.join(root, "escape"), process.platform === "win32" ? "junction" : "dir");
  assert.throws(() => binding().openDirectory(root, "escape", true), /unsafe|symlink|reparse/i);
  const dir = binding().openDirectory(root, "claims", true);
  try {
    fs.writeFileSync(path.join(outside, "secret"), "untouched");
    // Windows junctions require no administrator/developer-mode privileges.
    fs.symlinkSync(outside, path.join(root, "claims/link"), process.platform === "win32" ? "junction" : "dir");
    assert.throws(() => dir.openFile("link"), /unsafe|file|reparse/i);
    assert.equal(dir.createFile("link"), null);
    assert.equal(fs.readFileSync(path.join(outside, "secret"), "utf8"), "untouched");
  } finally { dir.close(); }
});

function moveOpenedParent(root, outside) {
  try {
    fs.renameSync(path.join(root, "claims"), path.join(outside, "moved"));
    return true;
  } catch (error) {
    if (process.platform !== "win32" || !["EPERM", "EACCES", "EBUSY"].includes(error.code)) throw error;
    return false;
  }
}

test("rejects a moved-and-linked-back parent or prevents its move while pinned", () => {
  const root = fixture();
  const outside = fixture();
  fs.mkdirSync(path.join(root, "claims/daily"), { recursive: true });
  const dir = binding().openDirectory(root, "claims/daily", true);
  try {
    if (moveOpenedParent(root, outside)) {
      fs.symlinkSync(path.join(outside, "moved"), path.join(root, "claims"), "dir");
      assert.throws(() => dir.assertCurrent(), /replaced|changed|unsafe/i);
    } else {
      dir.assertCurrent();
    }
  } finally { dir.close(); }
});

test("can remove its own uncommitted file through the original directory after a move", () => {
  const root = fixture();
  const outside = fixture();
  const dir = binding().openDirectory(root, "claims/daily", true);
  try {
    const file = dir.createFile("empty.json");
    assert.ok(file);
    file.close();
    const moved = moveOpenedParent(root, outside);
    if (moved) assert.throws(() => dir.assertCurrent(), /replaced|changed|unsafe/i);
    assert.equal(dir.remove("empty.json"), true);
    const parent = moved ? path.join(outside, "moved/daily") : path.join(root, "claims/daily");
    assert.equal(fs.existsSync(path.join(parent, "empty.json")), false);
  } finally { dir.close(); }
});

test("atomically replaces a private primary while retaining a hard-link backup", () => {
  const root = fixture();
  const dir = binding().openDirectory(root, "state", true);
  try {
    for (const [name, value] of [["primary.json", "old"], ["next.tmp", "new"]]) {
      const file = dir.createFile(name);
      assert.ok(file);
      try { file.writeText(value); } finally { file.close(); }
    }
    assert.equal(dir.link("primary.json", "primary.bak"), true);
    assert.equal(dir.link("primary.json", "primary.bak"), false);
    dir.rename("next.tmp", "primary.json");
    dir.sync();
    assert.equal(fs.readFileSync(path.join(root, "state/primary.json"), "utf8"), "new");
    assert.equal(fs.readFileSync(path.join(root, "state/primary.bak"), "utf8"), "old");
    assert.deepEqual(dir.list().sort(), ["primary.bak", "primary.json"]);
  } finally { dir.close(); }
});

test("tightens existing file privacy and rejects closed or wrong receiver handles", () => {
  const root = fixture();
  fs.writeFileSync(path.join(root, "legacy.json"), "legacy", { mode: 0o666 });
  const dir = binding().openDirectory(root, "", false);
  try {
    const file = dir.openFile("legacy.json");
    assert.ok(file);
    try {
      file.restrict();
      assert.equal(file.isPrivate(), true);
      assert.throws(() => file.writeText("not a new file"), /read.only|writable/i);
      assert.throws(() => dir.assertCurrent.call(file), /handle|receiver/i);
    } finally { file.close(); }
    assert.throws(() => file.isPrivate(), /closed/i);
  } finally { dir.close(); }
  assert.throws(() => dir.assertCurrent(), /closed/i);
  assert.throws(() => dir.createFile("after-close"), /closed/i);
});

test("has exactly one winner across independent operating-system processes", async () => {
  const root = fixture();
  const source = `const fs = require('node:fs'); const api = require(process.argv[1]); const dir = api.openDirectory(process.argv[2], 'claims', true); try { const f = dir.createFile('winner'); if (f) { try { f.writeText(String(process.pid)); } finally { f.close(); } process.stdout.write('won'); } else { process.stdout.write('busy'); } } finally { dir.close(); }`;
  const results = await Promise.all(Array.from({ length: 6 }, () => new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["-e", source, bindingPath, root], { stdio: ["ignore", "pipe", "pipe"] });
    let output = "", error = "";
    child.stdout.on("data", chunk => { output += chunk; });
    child.stderr.on("data", chunk => { error += chunk; });
    child.once("error", reject);
    // 'exit' can fire before the child's stdout pipe has finished draining
    // (Node explicitly documents this ordering is not guaranteed); 'close'
    // is the event guaranteed to fire only after stdio has fully ended, so
    // `output` is complete whenever this callback runs.
    child.once("close", code => code === 0 ? resolve(output) : reject(new Error(error)));
  })));
  assert.equal(results.filter(value => value === "won").length, 1);
  assert.equal(results.filter(value => value === "busy").length, 5);
  assert.match(fs.readFileSync(path.join(root, "claims/winner"), "utf8"), /^\d+$/);
});
