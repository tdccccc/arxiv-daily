import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";
import { RELEASE_ASSETS, releaseAssetRepoPaths } from "../release-assets.mjs";
import {
  ATTEST_LABEL,
  CANONICAL_LABEL,
  DEPLOY_LABEL,
  DOC_LABEL,
  RELEASE_DOC_MARKER,
  RELEASE_DOC_PATH,
  RELEASE_WORKFLOW_PATH,
  UPLOAD_LABEL,
  compareAssetLists,
  parseReleaseDocAssets,
  parseWorkflowReleaseAssets,
  root,
  verifyReleaseAssetSources,
} from "../release-asset-sources.mjs";

const readRepoFile = (relativePath) => readFile(resolve(root, relativePath), "utf8");
const sorted = (assets) => [...assets].sort();

const DOC_FIXTURE = [
  "# Release checklist",
  "",
  "Some prose that must stay readable.",
  "",
  RELEASE_DOC_MARKER,
  "",
  "- `plugin/manifest.json`",
  "- `plugin/main.js`",
  "- `plugin/styles.css`",
  "",
  "More prose.",
].join("\n");

const WORKFLOW_FIXTURE = [
  "jobs:",
  "  release:",
  "    steps:",
  "      - name: Attest release assets",
  "        uses: actions/attest-build-provenance@0f67c3f4856b2e3261c31976d6725780e5e4c373",
  "        with:",
  "          subject-path: |",
  "            plugin/manifest.json",
  "            plugin/main.js",
  "            plugin/styles.css",
  "      - name: Create release",
  "        run: |",
  '          gh release create "$TAG" \\',
  "            plugin/manifest.json plugin/main.js plugin/styles.css \\",
  "            --verify-tag \\",
  '            --notes-file "docs/releases/$TAG.md"',
].join("\n");

function replaceOnce(text, from, to) {
  assert.ok(text.includes(from), `fixture no longer contains ${JSON.stringify(from)}`);
  return text.replace(from, to);
}

test("the canonical release asset list is non-empty and frozen", () => {
  assert.ok(RELEASE_ASSETS.length > 0);
  assert.ok(Object.isFrozen(RELEASE_ASSETS));
  assert.deepEqual(releaseAssetRepoPaths(["a.js"]), ["plugin/a.js"]);
});

// The whole point of the check: every copy in the repository agrees today.
test("every copy of the release asset list agrees with the canonical one", async () => {
  assert.deepEqual(await verifyReleaseAssetSources(), []);
});

test("the release checklist's asset list parses to the canonical assets", async () => {
  assert.deepEqual(
    sorted(parseReleaseDocAssets(await readRepoFile(RELEASE_DOC_PATH))),
    sorted(RELEASE_ASSETS),
  );
});

test("the release workflow's two asset lists parse to the canonical assets", async () => {
  const { attested, uploaded } = parseWorkflowReleaseAssets(
    await readRepoFile(RELEASE_WORKFLOW_PATH),
  );
  assert.deepEqual(sorted(attested), sorted(RELEASE_ASSETS));
  assert.deepEqual(sorted(uploaded), sorted(RELEASE_ASSETS));
});

test("the checklist fixture mirrors the real document's shape", () => {
  assert.deepEqual(sorted(parseReleaseDocAssets(DOC_FIXTURE)), sorted(RELEASE_ASSETS));
  assert.deepEqual(sorted(parseWorkflowReleaseAssets(WORKFLOW_FIXTURE).attested), sorted(RELEASE_ASSETS));
  assert.deepEqual(sorted(parseWorkflowReleaseAssets(WORKFLOW_FIXTURE).uploaded), sorted(RELEASE_ASSETS));
});

// --- the checklist stays prose, so its parser must fail loudly, never emptily ---

test("a checklist without the marker sentence fails instead of parsing nothing", () => {
  const rewritten = replaceOnce(DOC_FIXTURE, RELEASE_DOC_MARKER, "The release ships these files:");
  assert.throws(() => parseReleaseDocAssets(rewritten), (error) => {
    assert.match(error.message, /no line reads exactly/);
    assert.match(error.message, /RELEASE_DOC_MARKER/);
    return true;
  });
});

test("a duplicated marker sentence fails rather than picking one list", () => {
  const rewritten = `${DOC_FIXTURE}\n\n${RELEASE_DOC_MARKER}\n\n- \`plugin/main.js\`\n`;
  assert.throws(() => parseReleaseDocAssets(rewritten), /appears on lines .*exactly once/s);
});

test("a checklist whose bullets became prose fails rather than parsing an empty list", () => {
  const rewritten = replaceOnce(
    DOC_FIXTURE,
    "- `plugin/manifest.json`\n- `plugin/main.js`\n- `plugin/styles.css`",
    "The manifest, the bundle and the stylesheet.",
  );
  assert.throws(() => parseReleaseDocAssets(rewritten), (error) => {
    assert.match(error.message, /no .*bullets under it/);
    assert.match(error.message, /empty list is a parse failure/);
    return true;
  });
});

test("a checklist whose bullets became a table fails rather than parsing an empty list", () => {
  const rewritten = replaceOnce(
    DOC_FIXTURE,
    "- `plugin/manifest.json`\n- `plugin/main.js`\n- `plugin/styles.css`",
    "| asset |\n| --- |\n| plugin/main.js |",
  );
  assert.throws(() => parseReleaseDocAssets(rewritten), /empty list is a parse failure/);
});

test("a bullet that is not a plugin asset path fails and quotes the offending line", () => {
  for (const bullet of ["- see the workflow", "- `plugin/`", "- `dist/main.js`", "- `plugin/a/b.js`"]) {
    const rewritten = replaceOnce(DOC_FIXTURE, "- `plugin/main.js`", bullet);
    assert.throws(() => parseReleaseDocAssets(rewritten), (error) => {
      // Compared as a substring rather than as a pattern: the path was being
      // turned into a regex by escaping dots and nothing else, which is only
      // correct for the paths it happens to hold today.
      assert.ok(error.message.includes(RELEASE_DOC_PATH), error.message);
      assert.ok(error.message.includes("line "), error.message);
      return true;
    });
  }
});

test("a checklist listing the same asset twice fails", () => {
  const rewritten = replaceOnce(DOC_FIXTURE, "- `plugin/main.js`", "- `plugin/styles.css`");
  assert.throws(() => parseReleaseDocAssets(rewritten), /more than once/);
});

test("non-text input fails rather than parsing as empty", () => {
  for (const value of [undefined, null, 0, {}]) {
    assert.throws(() => parseReleaseDocAssets(value), /expected markdown text/);
    assert.throws(() => parseWorkflowReleaseAssets(value), /expected YAML text/);
  }
});

// --- the workflow parser must fail loudly too ---

test("an unparsable workflow fails instead of yielding no assets", () => {
  assert.throws(() => parseWorkflowReleaseAssets("jobs: [\n  unclosed"), /not valid YAML/);
});

test("a workflow without jobs or steps fails", () => {
  assert.throws(() => parseWorkflowReleaseAssets("name: Release\n"), /no `jobs:` mapping/);
  assert.throws(() => parseWorkflowReleaseAssets("jobs:\n  release:\n    runs-on: x\n"), /no workflow steps/);
});

test("a workflow whose attestation step disappeared fails", () => {
  const rewritten = replaceOnce(
    WORKFLOW_FIXTURE,
    "uses: actions/attest-build-provenance@0f67c3f4856b2e3261c31976d6725780e5e4c373",
    "uses: actions/checkout@v5",
  );
  assert.throws(() => parseWorkflowReleaseAssets(rewritten), /exactly one .*attest-build-provenance.* found 0/s);
});

test("an attestation step without a subject-path list fails", () => {
  const rewritten = replaceOnce(
    WORKFLOW_FIXTURE,
    "        with:\n          subject-path: |\n            plugin/manifest.json\n            plugin/main.js\n            plugin/styles.css",
    "        with:\n          subject-path: ''",
  );
  assert.throws(() => parseWorkflowReleaseAssets(rewritten), /no `subject-path` list/);
});

test("an attested path outside plugin/ fails", () => {
  const rewritten = replaceOnce(WORKFLOW_FIXTURE, "            plugin/main.js\n", "            dist/main.js\n");
  assert.throws(() => parseWorkflowReleaseAssets(rewritten), /subject-path.*not a/s);
});

test("a workflow that no longer creates the release fails", () => {
  const rewritten = replaceOnce(WORKFLOW_FIXTURE, 'gh release create "$TAG" \\', 'echo skip \\');
  assert.throws(() => parseWorkflowReleaseAssets(rewritten), /exactly one step running .*found 0/s);
});

test("a release created without uploading assets fails", () => {
  const rewritten = replaceOnce(
    WORKFLOW_FIXTURE,
    "            plugin/manifest.json plugin/main.js plugin/styles.css \\\n",
    "",
  );
  assert.throws(() => parseWorkflowReleaseAssets(rewritten), /uploads no assets/);
});

test("an uploaded path outside plugin/ fails", () => {
  const rewritten = replaceOnce(WORKFLOW_FIXTURE, "plugin/styles.css \\", "README.md \\");
  assert.throws(() => parseWorkflowReleaseAssets(rewritten), /gh release create.*not a/s);
});

// --- the difference report has to name both sides and the offending file ---

test("compareAssetLists names both sides, both lists and each one-sided file", () => {
  const message = compareAssetLists("A", ["main.js", "styles.css"], "B", ["main.js", "extra.js"]);
  assert.match(message, /A has \[main\.js, styles\.css\]/);
  assert.match(message, /B has \[main\.js, extra\.js\]/);
  assert.match(message, /B is missing styles\.css/);
  assert.match(message, /B additionally lists extra\.js/);
});

test("compareAssetLists ignores ordering, which is not part of the contract", () => {
  assert.equal(compareAssetLists("A", ["a", "b"], "B", ["b", "a"]), null);
});

// --- one-sided edits, in every direction ---

async function issuesWithDoc(mutate) {
  return verifyReleaseAssetSources({
    read: async (relativePath) => {
      const text = await readRepoFile(relativePath);
      return relativePath === RELEASE_DOC_PATH ? mutate(text) : text;
    },
  });
}

async function issuesWithWorkflow(mutate) {
  return verifyReleaseAssetSources({
    read: async (relativePath) => {
      const text = await readRepoFile(relativePath);
      return relativePath === RELEASE_WORKFLOW_PATH ? mutate(text) : text;
    },
  });
}

test("adding an asset to the checklist alone is reported against the canonical list", async () => {
  const issues = await issuesWithDoc((text) =>
    replaceOnce(text, "- `plugin/styles.css`", "- `plugin/styles.css`\n- `plugin/extra.js`"));
  assert.equal(issues.length, 1);
  assert.match(issues[0], new RegExp(CANONICAL_LABEL.replace(/[.\\/()]/g, "\\$&")));
  assert.ok(issues[0].includes(DOC_LABEL), issues[0]);
  assert.match(issues[0], /additionally lists extra\.js/);
});

test("dropping an asset from the checklist alone is reported by name", async () => {
  const issues = await issuesWithDoc((text) => replaceOnce(text, "- `plugin/styles.css`\n", ""));
  assert.equal(issues.length, 1);
  assert.match(issues[0], /is missing styles\.css/);
});

test("adding an asset to the workflow alone is reported for both workflow lists", async () => {
  const issues = await issuesWithWorkflow((text) => {
    const attested = replaceOnce(
      text,
      "            plugin/styles.css\n",
      "            plugin/styles.css\n            plugin/extra.js\n",
    );
    return replaceOnce(
      attested,
      "plugin/manifest.json plugin/main.js plugin/styles.css \\",
      "plugin/manifest.json plugin/main.js plugin/styles.css plugin/extra.js \\",
    );
  });
  assert.equal(issues.length, 2);
  assert.ok(issues.some((issue) => issue.includes(ATTEST_LABEL)), issues.join("\n"));
  assert.ok(issues.some((issue) => issue.includes(UPLOAD_LABEL)), issues.join("\n"));
  for (const issue of issues) assert.match(issue, /additionally lists extra\.js/);
});

test("attesting a different set than it uploads is reported", async () => {
  const issues = await issuesWithWorkflow((text) =>
    replaceOnce(text, "            plugin/styles.css\n", ""));
  assert.equal(issues.length, 1);
  assert.ok(issues[0].includes(ATTEST_LABEL), issues[0]);
  assert.match(issues[0], /is missing styles\.css/);
});

// Changing the shared list alone must be reported against every other copy,
// including the acceptance harness's deployment, which is what silently
// measured a stale stylesheet before these lists were joined.
test("changing the canonical list alone is reported against all four copies", async () => {
  const issues = await verifyReleaseAssetSources({
    canonical: [...RELEASE_ASSETS, "extra.js"],
  });
  assert.equal(issues.length, 4);
  for (const label of [DOC_LABEL, ATTEST_LABEL, UPLOAD_LABEL, DEPLOY_LABEL]) {
    assert.ok(issues.some((issue) => issue.includes(label)), `${label} not reported:\n${issues.join("\n")}`);
  }
  for (const issue of issues) assert.match(issue, /is missing extra\.js/);
});

// --- a parse failure, or an empty list, must never read as agreement ---

test("an unreadable source is reported rather than skipped", async () => {
  const issues = await verifyReleaseAssetSources({
    read: async () => {
      throw new Error("ENOENT");
    },
  });
  assert.ok(issues.length >= 2, issues.join("\n"));
});

test("sources that parse to nothing fail instead of agreeing", async () => {
  const issues = await verifyReleaseAssetSources({ read: async () => "" });
  assert.ok(issues.length >= 2, issues.join("\n"));
  assert.ok(issues.some((issue) => issue.includes(RELEASE_DOC_PATH)), issues.join("\n"));
  assert.ok(issues.some((issue) => issue.includes(RELEASE_WORKFLOW_PATH)), issues.join("\n"));
});

test("a checklist reformatted away from bullets fails the check", async () => {
  const issues = await issuesWithDoc((text) =>
    replaceOnce(
      text,
      "- `plugin/manifest.json`\n- `plugin/main.js`\n- `plugin/styles.css`",
      "the manifest, the bundle and the stylesheet.",
    ));
  assert.equal(issues.length, 1);
  assert.match(issues[0], /empty list is a parse failure/);
});

test("an empty canonical list is a defect, not something to agree with", async () => {
  const issues = await verifyReleaseAssetSources({ canonical: [] });
  assert.ok(issues.length > 0);
  assert.match(issues[0], /is empty/);
  assert.match(issues[0], /defect/);
});

test("a non-array canonical list is a defect", async () => {
  for (const canonical of [null, "main.js", {}, 3]) {
    const issues = await verifyReleaseAssetSources({ canonical });
    assert.ok(issues.length > 0, `${JSON.stringify(canonical)} passed`);
  }
});
