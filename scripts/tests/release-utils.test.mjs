import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { noticeBanner, readPakoNotice, root, validateSemVer } from "../release-utils.mjs";

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

test("the release workflow runs the release-tool tests during verification", async () => {
  const workflow = await readFile(`${root}/.github/workflows/release.yml`, "utf8");
  const verifyWorkspace = workflow.match(/- name: Verify workspace\n\s+run: \|\n(?<commands>(?:\s{10}.+\n)+)/);
  assert.ok(verifyWorkspace, "release workflow must define the workspace verification step");
  assert.match(verifyWorkspace.groups.commands, /^\s+npm run test:release-tools$/m);
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
