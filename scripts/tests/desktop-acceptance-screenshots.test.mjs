import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import zlib from "node:zlib";
import {
  DEFAULT_SCREENSHOT_DIR,
  captureTargetExpression,
  createScreenshotWriter,
  decodePng,
  judgeCapturedImage,
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
  const expression = captureTargetExpression({ selector: ".a", throughSelector: ".b" });
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
    evaluate: async () =>
      JSON.stringify({
        found: true,
        visible: true,
        rect: { x: 4, y: 6, width: 40, height: 20 },
        viewport: { width: 1100, height: 1400, scrollX: 0, scrollY: 0 },
      }),
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

// ── capture preconditions ───────────────────────────────────────────────────
//
// A run whose renderer had collapsed once wrote ten PNGs of an error page and a
// blank canvas, and every one of them landed silently. These pin the checks
// that now stand between "measured something" and "wrote a file": the target
// has to exist, be visible, have a size, and lie in the region the camera
// actually photographs — and the bytes that come back have to carry more than
// one colour.

/** A minimal non-interlaced 8-bit PNG, so the integrity check has real bytes. */
function png({ width, height, pixel, channels = 4, filter = 0 }) {
  const stride = width * channels;
  const flat = Buffer.alloc(height * stride);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const rgba = pixel(x, y);
      for (let c = 0; c < channels; c += 1) flat[y * stride + x * channels + c] = rgba[c];
    }
  }
  const paeth = (a, b, c) => {
    const p = a + b - c;
    const pa = Math.abs(p - a);
    const pb = Math.abs(p - b);
    const pc = Math.abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
  };
  const raw = Buffer.alloc(height * (1 + stride));
  for (let y = 0; y < height; y += 1) {
    const row = y * (1 + stride);
    raw[row] = filter;
    for (let i = 0; i < stride; i += 1) {
      const value = flat[y * stride + i];
      const a = i >= channels ? flat[y * stride + i - channels] : 0;
      const b = y > 0 ? flat[(y - 1) * stride + i] : 0;
      const c = i >= channels && y > 0 ? flat[(y - 1) * stride + i - channels] : 0;
      const encoded =
        filter === 0 ? value
        : filter === 1 ? value - a
        : filter === 2 ? value - b
        : filter === 3 ? value - ((a + b) >> 1)
        : value - paeth(a, b, c);
      raw[row + 1 + i] = encoded & 0xff;
    }
  }
  const chunk = (type, data) => {
    const out = Buffer.alloc(12 + data.length);
    out.writeUInt32BE(data.length, 0);
    out.write(type, 4, "latin1");
    data.copy(out, 8);
    return out; // the CRC is left zero: the decoder reads content, not integrity
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = channels === 4 ? 6 : channels === 3 ? 2 : 0;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", zlib.deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const solid = png({ width: 8, height: 4, pixel: () => [250, 250, 250, 255] });
const varied = png({ width: 8, height: 4, pixel: (x) => [x * 20, 10, 10, 255] });

/**
 * A writer whose renderer answers with `state` and whose camera returns `image`,
 * recording every write so a refusal can be checked to have written nothing.
 */
async function writerFor({ state, image = varied }) {
  const writes = [];
  const writer = await createScreenshotWriter({
    client: {
      send: async (method) =>
        method === "Page.captureScreenshot" ? { data: image.toString("base64") } : {},
    },
    evaluate: async () => JSON.stringify(state),
    outputDir: "/out",
    fs: {
      mkdir: async () => undefined,
      writeFile: async (file, content) => writes.push([file, content]),
    },
  });
  return { writer, writes };
}

const visibleTarget = {
  found: true,
  visible: true,
  rect: { x: 10, y: 10, width: 200, height: 60 },
  viewport: { width: 1100, height: 1400, scrollX: 0, scrollY: 0 },
};

test("a target that is present, visible, sized and on screen is captured", async () => {
  const { writer, writes } = await writerFor({ state: visibleTarget });
  await writer.capture("personal-library-section-local-embedding", { selector: ".setting-item" });
  assert.equal(writes.length, 1);
});

test("a target that is not in the document writes nothing", async () => {
  const { writer, writes } = await writerFor({
    state: { found: false, selector: ".setting-item", viewport: visibleTarget.viewport },
  });
  await assert.rejects(
    () => writer.capture("personal-library-section-local-embedding", { selector: ".setting-item" }),
    /nothing matched/,
  );
  assert.deepEqual(writes, []);
});

test("a target laid out at zero size writes nothing", async () => {
  const { writer, writes } = await writerFor({
    state: { ...visibleTarget, rect: { x: 10, y: 10, width: 0, height: 60 } },
  });
  await assert.rejects(
    () => writer.capture("library-row-wide-panel", { selector: ".setting-item" }),
    /0x60|not laid out|no size/,
  );
  assert.deepEqual(writes, []);
});

test("a target that is present but not visible writes nothing", async () => {
  const { writer, writes } = await writerFor({
    state: { ...visibleTarget, visible: false, invisible: "display: none" },
  });
  await assert.rejects(
    () => writer.capture("library-row-wide-panel", { selector: ".setting-item" }),
    /not visible|display: none/,
  );
  assert.deepEqual(writes, []);
});

test("a target scrolled out of the photographed viewport writes nothing", async () => {
  const { writer, writes } = await writerFor({
    state: { ...visibleTarget, rect: { x: 10, y: 3000, width: 200, height: 60 } },
  });
  await assert.rejects(
    () => writer.capture("library-row-wide-panel", { selector: ".setting-item" }),
    /viewport/,
  );
  assert.deepEqual(writes, []);
});

test("a caller-measured rectangle is held to the same viewport rule", async () => {
  const { writer, writes } = await writerFor({ state: visibleTarget.viewport });
  await assert.rejects(
    () => writer.capture("library-row-wide-panel", { rect: { x: 10, y: 3000, width: 200, height: 60 } }),
    /viewport/,
  );
  assert.deepEqual(writes, []);
  await writer.capture("library-row-wide-panel", { rect: { x: 10, y: 10, width: 200, height: 60 } });
  assert.equal(writes.length, 1);
});

test("an image that is one flat colour end to end writes nothing", async () => {
  const { writer, writes } = await writerFor({ state: visibleTarget, image: solid });
  await assert.rejects(
    () => writer.capture("personal-library-section-authorized", { selector: ".setting-item" }),
    /one colour|single colour|blank/i,
  );
  assert.deepEqual(writes, []);
});

test("an empty capture writes nothing", async () => {
  const { writer, writes } = await writerFor({ state: visibleTarget, image: Buffer.alloc(0) });
  await assert.rejects(
    () => writer.capture("personal-library-section-authorized", { selector: ".setting-item" }),
    /no image bytes|empty/i,
  );
  assert.deepEqual(writes, []);
});

test("the uniform-colour judge answers about content, and abstains on a format it cannot read", () => {
  assert.equal(judgeCapturedImage(varied).ok, true);
  assert.equal(judgeCapturedImage(varied).measured, true);
  assert.equal(judgeCapturedImage(solid).ok, false);
  // Legitimate uniformity is a real thing, so the rule is the whole image being
  // one colour — never a fraction, never a comparison with a stored baseline.
  assert.match(judgeCapturedImage(solid).reason, /every one of its 32 pixels/);
  // Not a PNG this decoder handles: reported as unmeasured, never as a failure.
  const unmeasured = judgeCapturedImage(Buffer.from("not a png at all"));
  assert.equal(unmeasured.ok, true);
  assert.equal(unmeasured.measured, false);
});

test("the capture-target expression asks the renderer for visibility, size and viewport", () => {
  const expression = captureTargetExpression({ selector: ".setting-item" });
  assert.match(expression, /getComputedStyle/);
  assert.match(expression, /getClientRects|getBoundingClientRect/);
  assert.match(expression, /innerWidth/);
});

test("the decoder handles what the renderer actually emits: RGB, and every scanline filter", () => {
  // Real captures come back as 8-bit RGB, filtered per scanline. A decoder that
  // only handled unfiltered RGBA would abstain on every genuine image and the
  // blank-capture check would quietly stop existing.
  const pixel = (x, y) => [(x * 31 + y * 7) % 256, (x * 5) % 256, (y * 11) % 256, 255];
  for (const channels of [3, 4]) {
    for (const filter of [0, 1, 2, 3, 4]) {
      const image = decodePng(png({ width: 6, height: 5, pixel, channels, filter }));
      assert.ok(image, `filter ${filter} at ${channels} channels was not decoded`);
      assert.equal(image.channels, channels);
      const got = Array.from(image.pixels.subarray(3 * channels, 4 * channels));
      assert.deepEqual(got, pixel(3, 0).slice(0, channels), `filter ${filter}, ${channels} channels`);
    }
  }
});
