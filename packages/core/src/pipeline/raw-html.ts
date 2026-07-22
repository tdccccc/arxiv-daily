const HTML_TAG_NAME = String.raw`[A-Za-z][A-Za-z0-9-]*`;
const HTML_ATTRIBUTE_NAME = String.raw`[A-Za-z_:][A-Za-z0-9_.:-]*`;
const HTML_ATTRIBUTE_VALUE = "(?:[^\\s\"'=<>`]+|'[^']*'|\"[^\"]*\")";
const HTML_ATTRIBUTE = String.raw`(?:\s+${HTML_ATTRIBUTE_NAME}(?:\s*=\s*${HTML_ATTRIBUTE_VALUE})?)`;
const HTML_OPEN_TAG_RE = new RegExp(
  String.raw`^<${HTML_TAG_NAME}(?:${HTML_ATTRIBUTE})*\s*\/?>$`,
);
const HTML_CLOSE_TAG_RE = new RegExp(String.raw`^<\/${HTML_TAG_NAME}\s*>$`);

export function neutralizeRawHtml(value: string): string {
  let out = "";
  for (let index = 0; index < value.length;) {
    const start = value.indexOf("<", index);
    if (start < 0) return out + value.slice(index);
    out += value.slice(index, start);
    const construct = readRawHtmlConstruct(value, start);
    if (!construct) {
      out += "<";
      index = start + 1;
      continue;
    }
    out += `&lt;${construct.slice(1)}`;
    index = start + construct.length;
  }
  return out;
}

export function containsRawHtmlConstruct(value: string): boolean {
  return neutralizeRawHtml(value) !== value;
}

function readRawHtmlConstruct(value: string, start: number): string | null {
  const rest = value.slice(start);
  for (const [prefix, suffix] of [
    ["<!--", "-->"],
    ["<![CDATA[", "]]>"] ,
    ["<?", "?>"],
  ] as const) {
    if (!rest.startsWith(prefix)) continue;
    const end = value.indexOf(suffix, start + prefix.length);
    return end < 0 ? null : value.slice(start, end + suffix.length);
  }
  if (/^<![A-Z]/.test(rest)) {
    const end = value.indexOf(">", start + 3);
    return end < 0 ? null : value.slice(start, end + 1);
  }
  if (!/^<\/?[A-Za-z]/.test(rest)) return null;

  let quote = "";
  for (let index = start + 1; index < value.length; index += 1) {
    const char = value[index]!;
    if (quote) {
      if (char === quote) quote = "";
      continue;
    }
    if (char === "\"" || char === "'") {
      quote = char;
      continue;
    }
    if (char === "<") return null;
    if (char !== ">") continue;
    const candidate = value.slice(start, index + 1);
    return HTML_OPEN_TAG_RE.test(candidate) || HTML_CLOSE_TAG_RE.test(candidate)
      ? candidate
      : null;
  }
  return null;
}
