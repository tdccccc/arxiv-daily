const PAPER_DATA_CLOSE_TAG = /<\/\s*paper_data\s*>/gi;

export function escapePaperDataFence(value: string): string {
  return value.replace(PAPER_DATA_CLOSE_TAG, (match) =>
    match.replace("<", "&lt;").replace(">", "&gt;"),
  );
}
