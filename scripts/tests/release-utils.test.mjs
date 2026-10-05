import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFile, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import {
  manifestFiles,
  noticeBanner,
  packageFiles,
  readPakoNotice,
  root,
  validateSemVer,
} from "../release-utils.mjs";

const valid = [
  "0.0.0",
  "0.2.1",
  "10.20.30",
  "1.0.0-alpha",
  "1.0.0-alpha.1",
  "1.0.0-0.3.7",
  "1.0.0-x.7.z.92",
  "1.0.0+build.1",
  "1.0.0-beta+exp.sha.5114f85",
];
const invalid = [
  "",
  "v1.2.3",
  "01.2.3",
  "1.02.3",
  "1.2.03",
  "1.2",
  "1.2.3-01",
  "1.2.3-",
  "1.2.3+",
  "1.2.3+bad_thing",
  "1.2.3.4",
  " 1.2.3",
  "1.2.3 ",
];

test("validateSemVer accepts complete SemVer 2.0.0 forms", () => {
  for (const value of valid) assert.equal(validateSemVer(value), value);
});

test("validateSemVer rejects prefixes, partials, whitespace, and leading zeroes", () => {
  for (const value of invalid) assert.throws(() => validateSemVer(value), /Invalid SemVer/);
  assert.throws(() => validateSemVer(undefined), /Invalid SemVer/);
});

test("release tools share the root release package contract", () => {
  assert.deepEqual(packageFiles, [
    "package.json",
    "plugin/package.json",
    "packages/core/package.json",
    "packages/node-runtime/package.json",
    "apps/cli/package.json",
  ]);
  assert.deepEqual(manifestFiles, ["manifest.json", "plugin/manifest.json"]);
});

test("the release workflow runs the release-tool tests during verification", async () => {
  const workflow = await readFile(`${root}/.github/workflows/release.yml`, "utf8");
  const verifyWorkspace = workflow.match(/- name: Verify workspace\n\s+run: \|\n(?<commands>(?:\s{10}.+\n)+)/);
  assert.ok(verifyWorkspace, "release workflow must define the workspace verification step");
  assert.match(verifyWorkspace.groups.commands, /^\s+npm run test:release-tools$/m);
  assert.match(verifyWorkspace.groups.commands, /^\s+npm audit --audit-level=moderate$/m);
  assert.match(verifyWorkspace.groups.commands, /^\s+npm run lint$/m);
  assert.match(verifyWorkspace.groups.commands, /^\s+npm run smoke:install$/m);
  assert.match(
    verifyWorkspace.groups.commands,
    /^\s+NODE_OPTIONS=--max-old-space-size=8192 npm run test:workspaces -- --maxWorkers=1$/m,
  );
});

test("trusted CLI publishing is OIDC-only and constrained to immutable releases", async () => {
  const releaseWorkflow = await readFile(`${root}/.github/workflows/release.yml`, "utf8");
  const publishWorkflow = await readFile(`${root}/.github/workflows/publish-cli.yml`, "utf8");
  assert.doesNotMatch(releaseWorkflow, /npm publish/);
  assert.match(publishWorkflow, /^\s+workflow_run:$/m);
  assert.match(publishWorkflow, /^\s+workflow_dispatch:$/m);
  assert.match(publishWorkflow, /^\s+id-token: write$/m);
  assert.doesNotMatch(publishWorkflow, /NPM_TOKEN|NODE_AUTH_TOKEN/);
  assert.match(publishWorkflow, /^\s+run: npm install --global npm@\^11\.5\.1$/m);
  assert.match(publishWorkflow, /Refusing to overwrite existing npm version/);
  assert.match(publishWorkflow, /^\s+gh release view "\$version" >\/dev\/null$/m);
  assert.match(publishWorkflow, /^\s+NODE_OPTIONS=--max-old-space-size=8192 npm run test:workspaces -- --maxWorkers=1$/m);
  assert.match(publishWorkflow, /^\s+npm audit --audit-level=moderate$/m);
  assert.match(publishWorkflow, /^\s+npm run smoke:install$/m);
  assert.match(publishWorkflow, /^\s+run: npm publish --workspace apps\/cli --access public$/m);
});

test("release-equivalent workflows use the explicit full-workspace test entry", async () => {
  const workflowDir = `${root}/.github/workflows`;
  const workflowFiles = (await readdir(workflowDir))
    .filter((file) => /\.ya?ml$/.test(file));
  const workflows = new Map(
    await Promise.all(workflowFiles.map(async (file) => [
      file,
      await readFile(`${workflowDir}/${file}`, "utf8"),
    ])),
  );
  const fullSuiteCommand =
    "NODE_OPTIONS=--max-old-space-size=8192 npm run test:workspaces -- --maxWorkers=1";

  for (const [file, workflow] of workflows) {
    assert.doesNotMatch(
      workflow,
      /NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1/,
      `${file} must not route a full release suite through the focused root entry`,
    );
  }
  for (const file of ["lint.yml", "release.yml", "publish-cli.yml"]) {
    assert.match(workflows.get(file) ?? "", new RegExp(fullSuiteCommand));
  }
});


test("the bundle banner contains the complete locked pako license exactly once", async () => {
  const notice = await readPakoNotice();
  const lockedLicense = (await readFile(`${root}/node_modules/pako/LICENSE`, "utf8")).trimEnd();
  const banner = noticeBanner(notice);
  assert.equal(notice, lockedLicense);
  assert.match(notice, /^\(The MIT License\)/);
  assert.match(notice, /Copyright \(C\) 2014-2017 by Vitaly Puzrin and Andrei Tuputcyn/);
  assert.match(notice, /OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\nTHE SOFTWARE\.$/);
  assert.equal(banner.split(notice).length - 1, 1);
});

test("the bundle notice survives a Windows CRLF checkout without changing license content", async () => {
  const fixture = await mkdtemp(join(tmpdir(), "arxiv-notice-crlf-"));
  try {
    await mkdir(join(fixture, "scripts"));
    const modulePath = join(fixture, "scripts/release-utils.mjs");
    await copyFile(`${root}/scripts/release-utils.mjs`, modulePath);
    const notices = (await readFile(`${root}/THIRD_PARTY_NOTICES.md`, "utf8")).replace(/\r\n/g, "\n");
    await writeFile(join(fixture, "THIRD_PARTY_NOTICES.md"), notices.replace(/\n/g, "\r\n"));
    const fixtureUtils = await import(pathToFileURL(modulePath).href);
    const notice = await fixtureUtils.readPakoNotice();
    const license = (await readFile(`${root}/node_modules/pako/LICENSE`, "utf8")).replace(/\r\n/g, "\n").trimEnd();
    assert.equal(notice, license);
    assert.equal(fixtureUtils.noticeBanner(notice).split(license).length - 1, 1);
  } finally { await rm(fixture, { recursive: true, force: true }); }
});

test("release checker accepts current metadata and both tools reject malformed versions", async () => {
  const current = JSON.parse(await import("node:fs/promises").then(({ readFile }) => readFile(`${root}/package.json`, "utf8"))).version;
  const check = spawnSync(process.execPath, [`${root}/scripts/check-release-version.mjs`, current], { encoding: "utf8" });
  assert.equal(check.status, 0, check.stderr);
  for (const script of ["check-release-version.mjs", "sync-release-version.mjs"]) {
    const result = spawnSync(process.execPath, [`${root}/scripts/${script}`, "v1.2.3"], { encoding: "utf8" });
    assert.equal(result.status, 2, `${script}: ${result.stdout}${result.stderr}`);
    assert.match(result.stderr, /Invalid SemVer/);
  }
});


test("release checker accepts nested dependencies without accepting unknown workspaces", async () => {
  const fixture = await mkdtemp(join(tmpdir(), "arxiv-release-nested-deps-"));
  try {
    const files = [...packageFiles, ...manifestFiles, "versions.json", "plugin/versions.json",
      "package-lock.json", "THIRD_PARTY_NOTICES.md", "scripts/release-utils.mjs", "scripts/check-release-version.mjs"];
    for (const file of files) {
      await mkdir(dirname(join(fixture, file)), { recursive: true });
      await copyFile(join(root, file), join(fixture, file));
    }
    const lock = JSON.parse(await readFile(join(fixture, "package-lock.json"), "utf8"));
    for (const path of ["apps/cli/node_modules/commander", "apps/cli/node_modules/@fixture/scoped",
      "packages/core/node_modules/outer/node_modules/inner"]) {
      lock.packages[path] = { version: "1.0.0" };
    }
    const check = () => spawnSync(process.execPath,
      [join(fixture, "scripts/check-release-version.mjs"), lock.version], { encoding: "utf8" });
    await writeFile(join(fixture, "package-lock.json"), JSON.stringify(lock));
    const accepted = check();
    assert.equal(accepted.status, 0, accepted.stderr);
    for (const path of ["apps/unexpected", "packages/node_modules-like"]) {
      lock.packages[path] = { name: "unexpected", version: lock.version };
      await writeFile(join(fixture, "package-lock.json"), JSON.stringify(lock));
      const rejected = check();
      assert.equal(rejected.status, 1, rejected.stdout + rejected.stderr);
      assert.ok(rejected.stderr.includes(`unexpected workspace package ${path}`), rejected.stderr);
      delete lock.packages[path];
    }
  } finally { await rm(fixture, { recursive: true, force: true }); }
});
