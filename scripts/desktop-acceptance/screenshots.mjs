import fsPromises from "node:fs/promises";
import path from "node:path";
import zlib from "node:zlib";

/**
 * Screenshots of the real renderer, written to a git-ignored directory so a
 * person can look at the states an assertion only describes in words.
 *
 * These are evidence for human judgement, not a comparison baseline: nothing
 * here diffs pixels, and no assertion depends on a stored image. Wording,
 * timing and visual polish still need a person, and this is what they look at.
 *
 * Which is exactly why a written file has to be worth looking at. A run whose
 * renderer had collapsed once wrote ten PNGs of Obsidian's error page and a
 * blank rectangle, all of them silently, and their mere existence read as
 * evidence that the states they were named after had been reached. So a
 * capture is now refused unless its subject is in the document, visible, laid
 * out with a size, and inside the region the camera actually photographs — and
 * unless the bytes that come back carry more than a single colour.
 */

/** Relative to the repository root; `.acceptance-out` is already git-ignored. */
export const DEFAULT_SCREENSHOT_DIR = path.join(".acceptance-out", "desktop-acceptance");

export function resolveScreenshotDir({ repoRoot, env = process.env }) {
  const override = env.OBSIDIAN_ACCEPTANCE_SCREENSHOT_DIR;
  if (typeof override === "string" && override.trim().length > 0) {
    return path.isAbsolute(override) ? override : path.resolve(repoRoot, override);
  }
  return path.resolve(repoRoot, DEFAULT_SCREENSHOT_DIR);
}

/**
 * A file name has to say which state the image shows without opening it, and
 * must not be able to escape the output directory.
 */
export function screenshotFileName(name) {
  if (typeof name !== "string" || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(name)) {
    throw new TypeError(
      `screenshot name must be a lowercase, hyphen-separated slug, received ${JSON.stringify(name)}`,
    );
  }
  return `${name}.png`;
}

/**
 * Round a rectangle out to whole pixels and pad it, so a clip never shaves the
 * border off the element it is meant to show. Negative origins are clamped:
 * CDP refuses a clip that starts outside the page.
 */
export function padClip(rect, { padding = 8 } = {}) {
  if (!rect || !Number.isFinite(rect.width) || !Number.isFinite(rect.height)) {
    throw new TypeError(`screenshot clip needs a measured rectangle, received ${JSON.stringify(rect)}`);
  }
  if (rect.width <= 0 || rect.height <= 0) {
    throw new Error(
      `refusing to capture a ${rect.width}x${rect.height} region: the element is not laid out`,
    );
  }
  const x = Math.max(0, Math.floor(rect.x - padding));
  const y = Math.max(0, Math.floor(rect.y - padding));
  return {
    x,
    y,
    width: Math.ceil(rect.x + rect.width + padding) - x,
    height: Math.ceil(rect.y + rect.height + padding) - y,
    scale: 1,
  };
}

/**
 * How much of the subject has to lie inside the photographed viewport.
 *
 * `captureBeyondViewport` is off (see the capture itself), so whatever falls
 * outside comes back as empty canvas. Half is a deliberately forgiving floor:
 * it passes a section slightly clipped at the bottom, and refuses the case this
 * exists for — a rectangle measured somewhere the camera is not pointing.
 */
export const MIN_VISIBLE_FRACTION = 0.5;

/** The photographed region, for a caller that measured its own rectangle. */
export const VIEWPORT_EXPRESSION =
  `JSON.stringify({ width: window.innerWidth, height: window.innerHeight, scrollX: window.scrollX, scrollY: window.scrollY })`;

/**
 * Measure one element, or the span from one element to another, in page
 * coordinates, and report alongside it everything needed to decide whether
 * photographing it would mean anything: whether it was found at all, whether it
 * is visible, and where the viewport currently sits.
 *
 * The facts come back as data rather than as a verdict, so the judgement is a
 * pure function that can be tested without a renderer.
 */
export function captureTargetExpression({ selector, throughSelector }) {
  return `(() => {
    const viewport = { width: window.innerWidth, height: window.innerHeight, scrollX: window.scrollX, scrollY: window.scrollY };
    const first = document.querySelector(${JSON.stringify(selector)});
    if (!first) return JSON.stringify({ found: false, selector: ${JSON.stringify(selector)}, viewport });
    const last = ${throughSelector ? `document.querySelector(${JSON.stringify(throughSelector)})` : "first"};
    if (!last) return JSON.stringify({ found: false, selector: ${JSON.stringify(throughSelector ?? selector)}, viewport });
    first.scrollIntoView({ block: "start" });
    const invisible = (el) => {
      const style = window.getComputedStyle(el);
      if (style.display === "none") return "display: none";
      if (style.visibility === "hidden" || style.visibility === "collapse") return "visibility: " + style.visibility;
      if (Number(style.opacity) === 0) return "opacity: 0";
      if (el.getClientRects().length === 0) return "it produced no client rects, so it was never laid out";
      return null;
    };
    const reason = invisible(first) ?? invisible(last);
    const a = first.getBoundingClientRect();
    const b = last.getBoundingClientRect();
    const left = Math.min(a.left, b.left) + window.scrollX;
    const top = Math.min(a.top, b.top) + window.scrollY;
    const right = Math.max(a.right, b.right) + window.scrollX;
    const bottom = Math.max(a.bottom, b.bottom) + window.scrollY;
    return JSON.stringify({
      found: true,
      selector: ${JSON.stringify(selector)},
      visible: reason === null,
      invisible: reason,
      rect: { x: left, y: top, width: right - left, height: bottom - top },
      viewport,
    });
  })()`;
}

const round = (value) => Math.round(value * 10) / 10;

/**
 * Is this rectangle worth photographing? It needs a size, and it needs to be
 * where the camera is pointing.
 *
 * A viewport that could not be read is reported as unmeasured rather than as a
 * failure: an unreadable renderer is a different problem, and the application
 * state guard is the one that owns it.
 */
export function judgeCaptureRegion({ rect, viewport }, { minVisibleFraction = MIN_VISIBLE_FRACTION } = {}) {
  if (!rect || !Number.isFinite(rect.width) || !Number.isFinite(rect.height)) {
    return { ok: false, measured: true, reason: `no rectangle was measured, received ${JSON.stringify(rect)}` };
  }
  if (rect.width <= 0 || rect.height <= 0) {
    return {
      ok: false,
      measured: true,
      reason: `the subject is a ${round(rect.width)}x${round(rect.height)} region: it has no size to photograph`,
    };
  }
  if (!viewport || !Number.isFinite(viewport.width) || !Number.isFinite(viewport.height)) {
    return { ok: true, measured: false, reason: "the viewport could not be read, so containment was not judged" };
  }
  const left = Math.max(rect.x, viewport.scrollX ?? 0);
  const top = Math.max(rect.y, viewport.scrollY ?? 0);
  const right = Math.min(rect.x + rect.width, (viewport.scrollX ?? 0) + viewport.width);
  const bottom = Math.min(rect.y + rect.height, (viewport.scrollY ?? 0) + viewport.height);
  const visible = Math.max(0, right - left) * Math.max(0, bottom - top);
  const fraction = visible / (rect.width * rect.height);
  if (fraction < minVisibleFraction) {
    return {
      ok: false,
      measured: true,
      reason:
        `only ${Math.round(fraction * 100)}% of the ${round(rect.width)}x${round(rect.height)} subject at `
        + `(${round(rect.x)}, ${round(rect.y)}) lies inside the ${viewport.width}x${viewport.height} viewport being `
        + `photographed — the rest of the image would be empty canvas`,
    };
  }
  return { ok: true, measured: true, reason: `${Math.round(fraction * 100)}% of the subject is inside the viewport` };
}

/**
 * The same question for a target located by selector, with the two failures
 * that only a selector can have — nothing matched, and matched but hidden —
 * answered first, because each names a different thing that went wrong.
 */
export function judgeCaptureTarget(state, options = {}) {
  if (!state || typeof state !== "object") {
    return { ok: false, measured: false, reason: `the renderer described no target, received ${JSON.stringify(state)}` };
  }
  if (!state.found) {
    return {
      ok: false,
      measured: true,
      reason: `nothing matched ${JSON.stringify(state.selector ?? options.selector)}, so the capture would have `
        + `photographed whatever happens to be at that spot`,
    };
  }
  if (state.visible === false) {
    return {
      ok: false,
      measured: true,
      reason: `${JSON.stringify(state.selector ?? options.selector)} is in the document but not visible `
        + `(${state.invisible ?? "no reason reported"})`,
    };
  }
  return judgeCaptureRegion(state, options);
}

// ── what came back ──────────────────────────────────────────────────────────

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
/** Samples per pixel, by PNG colour type. */
const CHANNELS = { 0: 1, 2: 3, 4: 2, 6: 4 };

const paeth = (a, b, c) => {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  return pb <= pc ? b : c;
};

/**
 * Decode the 8-bit, non-interlaced PNG the debugging protocol produces.
 *
 * `null` means "this decoder cannot read these bytes", never "these bytes are
 * bad" — the caller reports that as unmeasured. Chunk CRCs are not verified:
 * the bytes arrive over a loopback socket from the renderer this process
 * launched, so the question is what the image shows, not whether it survived a
 * transfer.
 */
export function decodePng(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 8 || !buffer.subarray(0, 8).equals(PNG_SIGNATURE)) return null;
  let offset = 8;
  let header = null;
  const parts = [];
  while (offset + 8 <= buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString("latin1", offset + 4, offset + 8);
    const start = offset + 8;
    const end = start + length;
    if (end > buffer.length) return null;
    if (type === "IHDR") {
      header = {
        width: buffer.readUInt32BE(start),
        height: buffer.readUInt32BE(start + 4),
        depth: buffer[start + 8],
        colorType: buffer[start + 9],
        interlace: buffer[start + 12],
      };
    } else if (type === "IDAT") {
      parts.push(buffer.subarray(start, end));
    } else if (type === "IEND") {
      break;
    }
    offset = end + 4;
  }
  const channels = header ? CHANNELS[header.colorType] : undefined;
  if (!header || header.depth !== 8 || header.interlace !== 0 || !channels) return null;
  if (header.width <= 0 || header.height <= 0 || parts.length === 0) return null;

  let raw;
  try {
    raw = zlib.inflateSync(Buffer.concat(parts));
  } catch {
    return null;
  }
  const stride = header.width * channels;
  if (raw.length < header.height * (stride + 1)) return null;

  const pixels = Buffer.alloc(header.height * stride);
  for (let y = 0; y < header.height; y += 1) {
    const filter = raw[y * (stride + 1)];
    const src = y * (stride + 1) + 1;
    const dst = y * stride;
    const prev = dst - stride;
    for (let i = 0; i < stride; i += 1) {
      const x = raw[src + i];
      const a = i >= channels ? pixels[dst + i - channels] : 0;
      const b = y > 0 ? pixels[prev + i] : 0;
      const c = i >= channels && y > 0 ? pixels[prev + i - channels] : 0;
      let value;
      if (filter === 0) value = x;
      else if (filter === 1) value = x + a;
      else if (filter === 2) value = x + b;
      else if (filter === 3) value = x + ((a + b) >> 1);
      else if (filter === 4) value = x + paeth(a, b, c);
      else return null;
      pixels[dst + i] = value & 0xff;
    }
  }
  return { width: header.width, height: header.height, channels, pixels };
}

/**
 * The weakest honest statement about a captured image: it is not one flat
 * colour from corner to corner.
 *
 * That is deliberately the whole rule. A percentage threshold, a comparison
 * against a stored baseline, or anything that reasons about *which* colours
 * appear would be a pixel-diff by another name — brittle, unexplainable, and
 * the thing this harness has said from the start it does not do. A uniform
 * rectangle needs none of that: every state these captures are named after
 * contains text, a border or a control, so a single-colour image cannot be the
 * state it claims to show, whatever colour it is. Legitimate flat colour stays
 * legitimate — a flat *region* inside a varied image is untouched, because the
 * rule only fires when there is nothing else in the frame at all.
 *
 * A format this decoder cannot read is reported as unmeasured, never failed.
 */
export function judgeCapturedImage(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length === 0) {
    return { ok: false, measured: true, reason: "the renderer returned no image bytes" };
  }
  const image = decodePng(buffer);
  if (!image) {
    return { ok: true, measured: false, reason: "not an 8-bit non-interlaced PNG, so its content was not judged" };
  }
  const { pixels, channels, width, height } = image;
  const first = pixels.subarray(0, channels);
  for (let offset = channels; offset < pixels.length; offset += channels) {
    for (let c = 0; c < channels; c += 1) {
      if (pixels[offset + c] !== first[c]) {
        return { ok: true, measured: true, reason: `${width}x${height}, more than one colour` };
      }
    }
  }
  return {
    ok: false,
    measured: true,
    reason:
      `every one of its ${width * height} pixels is ${JSON.stringify(Array.from(first))} — the capture is a blank `
      + `rectangle, not the state it is named after`,
  };
}

/**
 * Capture PNGs through the debugging protocol the session already owns, so the
 * harness still needs no browser-automation dependency.
 */
export async function createScreenshotWriter({
  client,
  evaluate,
  outputDir,
  fs = fsPromises,
}) {
  await fs.mkdir(outputDir, { recursive: true });
  // Screenshots are served by the Page domain; enabling it is idempotent.
  await client.send("Page.enable");
  const written = [];

  /**
   * @param name  slug that names the state, not the element
   * @param where `{ rect }`, or `{ selector, throughSelector }` measured in the
   *              renderer just before the capture
   */
  async function capture(name, where) {
    const file = path.join(outputDir, screenshotFileName(name));
    // Nothing is written until the subject has been shown to be worth
    // photographing and the returned bytes have been shown to hold an image.
    // A refusal therefore leaves no file behind at all, rather than a file
    // whose existence claims a state was reached.
    const refuse = (reason) => {
      throw new Error(`refusing to write ${name}.png: ${reason}`);
    };

    let rect = where.rect;
    if (!rect) {
      const raw = await evaluate(captureTargetExpression(where));
      if (typeof raw !== "string") {
        throw new Error(
          `nothing matched ${JSON.stringify(where.selector)}, so ${name} would have captured the wrong thing`,
        );
      }
      const state = JSON.parse(raw);
      const verdict = judgeCaptureTarget(state, where);
      if (!verdict.ok) refuse(verdict.reason);
      rect = state.rect;
    } else {
      const rawViewport = await evaluate(VIEWPORT_EXPRESSION);
      const viewport = typeof rawViewport === "string" ? JSON.parse(rawViewport) : null;
      const verdict = judgeCaptureRegion({ rect, viewport }, where);
      if (!verdict.ok) refuse(verdict.reason);
    }
    const clip = padClip(rect, where);
    // The clip is deliberately taken from inside the current viewport rather
    // than with `captureBeyondViewport`: expanding the viewport for a capture
    // re-runs layout, which moves the very element that was just measured.
    // Callers make the region visible by sizing the viewport instead.
    const { data } = await client.send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: false,
      clip,
    });
    const image = Buffer.from(data, "base64");
    const integrity = judgeCapturedImage(image);
    if (!integrity.ok) refuse(integrity.reason);
    await fs.writeFile(file, image);
    const entry = { name, file, clip };
    written.push(entry);
    return entry;
  }

  return { capture, written: () => [...written], outputDir };
}
