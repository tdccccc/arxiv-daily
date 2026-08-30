import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import {
  DEFAULT_SCREENSHOT_DIR,
  createScreenshotWriter,
  elementRectExpression,
  padClip,
  resolveScreenshotDir,
  screenshotFileName,
} from "../desktop-acceptance/screenshots.mjs";

test("screenshots default to the git-ignored output directory", () => {
  const dir = resolveScreenshotDir({ repoRoot: "/repo", env: {} });
  assert.equal(dir, path.join("/repo", DEFAULT_SCREENSHOT_DIR));
  assert.ok(DEFAULT_SCREENSHOT_DIR.startsWith("."), "the output directory must be a dot directory");
});

test("an explicit output directory is honoured, relative to the repository", () => {
  assert.equal(
    resolveScreenshotDir({ repoRoot: "/repo", env: { OBSIDIAN_ACCEPTANCE_SCREENSHOT_DIR: "out/shots" } }),
    "/repo/out/shots",
  );
  assert.equal(
    resolveScreenshotDir({ repoRoot: "/repo", env: { OBSIDIAN_ACCEPTANCE_SCREENSHOT_DIR: "/tmp/shots" } }),
    "/tmp/shots",
  );
});

test("a screenshot name has to be a slug, so it cannot escape the output directory", () => {
  assert.equal(screenshotFileName("library-row-narrow-panel"), "library-row-narrow-panel.png");
  for (const bad of ["../escape", "Has Spaces", "trailing-", "", "a/b"]) {
    assert.throws(() => screenshotFileName(bad), TypeError, `expected ${JSON.stringify(bad)} to be refused`);
  }
});

test("a clip is padded and rounded outwards, and never starts off the page", () => {
  assert.deepEqual(padClip({ x: 10.4, y: 20.6, width: 100.2, height: 50.1 }, { padding: 8 }), {
    x: 2,
    y: 12,
    width: 117,
    height: 67,
    scale: 1,
  });
  assert.deepEqual(padClip({ x: 1, y: 1, width: 10, height: 10 }, { padding: 8 }), {
    x: 0,
    y: 0,
    width: 19,
    height: 19,
    scale: 1,
  });
});

test("an element that never laid out is refused rather than captured as an empty image", () => {
  assert.throws(() => padClip({ x: 0, y: 0, width: 0, height: 12 }), /not laid out/);
  assert.throws(() => padClip(null), TypeError);
});

test("the rectangle expression spans from the first element to the last", () => {
  const expression = elementRectExpression({ selector: ".a", throughSelector: ".b" });
  assert.match(expression, /querySelector\(".a"\)/);
  assert.match(expression, /querySelector\(".b"\)/);
  assert.match(expression, /window.scrollX/);
});

test("capturing writes the decoded PNG and records what it captured", async () => {
  const writes = [];
  const sent = [];
  const client = {
    send: async (method, params) => {
      sent.push([method, params]);
      if (method === "Page.captureScreenshot") return { data: Buffer.from("png-bytes").toString("base64") };
      return {};
    },
  };
  const writer = await createScreenshotWriter({
    client,
    evaluate: async () => JSON.stringify({ x: 4, y: 6, width: 40, height: 20 }),
    outputDir: "/out",
    fs: {
      mkdir: async () => undefined,
      writeFile: async (file, content) => writes.push([file, content]),
    },
  });
  const entry = await writer.capture("library-row-wide-panel", { selector: ".setting-item" });

  assert.equal(writes.length, 1);
  assert.equal(writes[0][0], path.join("/out", "library-row-wide-panel.png"));
  assert.equal(writes[0][1].toString(), "png-bytes");
  assert.deepEqual(entry.clip, { x: 0, y: 0, width: 52, height: 34, scale: 1 });
  assert.deepEqual(writer.written().map((item) => item.name), ["library-row-wide-panel"]);
  // Expanding the viewport for a capture would move the element that was just
  // measured, so the clip stays inside the viewport the caller set up.
  const capture = sent.find(([method]) => method === "Page.captureScreenshot");
  assert.equal(capture[1].captureBeyondViewport, false);
});

test("a selector that matches nothing fails loudly instead of capturing the wrong region", async () => {
  const writer = await createScreenshotWriter({
    client: { send: async () => ({ data: "" }) },
    evaluate: async () => null,
    outputDir: "/out",
    fs: { mkdir: async () => undefined, writeFile: async () => undefined },
  });
  await assert.rejects(
    () => writer.capture("missing-element", { selector: ".nope" }),
    /nothing matched/,
  );
});
