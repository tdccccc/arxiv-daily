/**
 * Checks for C0 control characters (U+0000-U+001F) and DEL (U+007F) by char
 * code instead of a regular expression. A regex literal or a `RegExp`
 * constructed from a pattern string containing these code points trips
 * eslint's `no-control-regex` (Obsidian's hosted review treats it as an
 * error-prone pattern); scanning char codes is an honest equivalent rather
 * than a workaround, and reads at least as clearly at each call site.
 *
 * `allowedCodes` lets callers carve out specific control characters (for
 * example tab/LF/CR in a multiline field) while still rejecting everything
 * else in the control range.
 */
export function hasControlCharacter(value: string, allowedCodes?: ReadonlySet<number>): boolean {
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    const isControl = code <= 0x1f || code === 0x7f;
    if (isControl && !allowedCodes?.has(code)) return true;
  }
  return false;
}
