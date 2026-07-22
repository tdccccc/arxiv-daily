const ARXIV_ID = String.raw`(\d{4}\.\d{4,5})`;
const COMMENT_MARKER = String.raw`<!--[ \t]*arxiv-daily:${ARXIV_ID}:(?:selection:)?(watch|highlight)[ \t]*-->`;
const CHECKBOX_LINE = String.raw`^[ \t]*[-*][ \t]+\[([ xX])\][^\r\n]*?${COMMENT_MARKER}[ \t]*\r?$`;

/** Historical and current daily checkbox controls, anchored to one physical line. */
export function dailySelectionMarkerRegExp(flags = "gm"): RegExp {
  return new RegExp(CHECKBOX_LINE, flags);
}
