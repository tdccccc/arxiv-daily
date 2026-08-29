const PAPER_DATA_CLOSE_TAG = /<\/\s*paper_data\s*>/gi;

export function escapePaperDataFence(value: string): string {
  return value.replace(PAPER_DATA_CLOSE_TAG, escapeXmlTagChars);
}

function escapeXmlTagChars(value: string): string {
  return value.replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}
