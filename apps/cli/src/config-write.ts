import { patch } from "@decimalturn/toml-patch";
import { parse, stringify } from "smol-toml";

type Table = Record<string, unknown>;

/**
 * The config text with `next` applied, touching only what differs from `raw`: comments, key order and
 * layout of everything else stay as the user left them. The result is read back with the parser the CLI
 * loads its config with; if that is not exactly `next` (the editor cannot express the change), the whole
 * document is written out instead, which loses comments but never a value.
 */
export function patchConfigToml(raw: string, next: Table): string {
  try {
    const before = parse(raw);
    // The editor cannot add a table or an array of tables at the root, but text after the last table
    // is scoped by its own [header], so those go on the end.
    const appended = Object.keys(next).filter(key => !Object.hasOwn(before, key) && hasHeader(next[key]));
    const eol = raw.includes("\r\n") ? "\r\n" : "\n";
    let text = patch(raw, Object.fromEntries(Object.entries(next).filter(([key]) => !appended.includes(key))));
    for (const key of appended) text = `${text.trimEnd()}${eol}${eol}${stringify({ [key]: next[key] }).replace(/\n/g, eol)}`;
    if (sameToml(parse(text), next)) return text;
  } catch { /* Not editable in place: write the document out. */ }
  return stringify(next);
}

/** What `stringify` writes under its own `[key]` or `[[key]]` header. */
function hasHeader(value: unknown): boolean {
  return isTable(value) || (Array.isArray(value) && value.length > 0 && value.every(isTable));
}

function isTable(value: unknown): value is Table {
  return typeof value === "object" && value !== null && !Array.isArray(value) && !(value instanceof Date);
}

/** Equal as TOML documents: key order is free, and `undefined`, which TOML cannot hold, counts as absent. */
function sameToml(a: unknown, b: unknown): boolean {
  if (a instanceof Date || b instanceof Date) return a instanceof Date && b instanceof Date && a.getTime() === b.getTime();
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((item, index) => sameToml(item, b[index]));
  }
  if (isTable(a) && isTable(b)) {
    const keys = (table: Table) => Object.keys(table).filter(key => table[key] !== undefined);
    const left = keys(a);
    return left.length === keys(b).length && left.every(key => Object.hasOwn(b, key) && sameToml(a[key], b[key]));
  }
  return Object.is(a, b);
}
