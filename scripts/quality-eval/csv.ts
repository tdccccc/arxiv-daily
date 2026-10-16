/**
 * RFC 4180 CSV for the label sheet. The writer prefixes a UTF-8 BOM so
 * spreadsheets open Chinese text correctly; the reader accepts either line
 * ending and a leading BOM, so a sheet saved by a spreadsheet reads back.
 */
const BOM = "﻿";

export function encodeCsv(rows: readonly (readonly string[])[]): string {
  return BOM + rows.map((row) => row.map(encodeCell).join(",")).join("\r\n") + "\r\n";
}

function encodeCell(cell: string): string {
  return /[",\r\n]/.test(cell) ? `"${cell.replace(/"/g, '""')}"` : cell;
}

export function parseCsv(text: string): string[][] {
  const source = text.startsWith(BOM) ? text.slice(1) : text;
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  let started = false;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index]!;
    if (quoted) {
      if (char === "\r" && source[index + 1] === "\n") { cell += "\n"; index += 1; }
      else if (char !== '"') cell += char;
      else if (source[index + 1] === '"') { cell += '"'; index += 1; }
      else quoted = false;
    } else if (char === '"' && cell === "") {
      quoted = true;
      started = true;
    } else if (char === ",") {
      row.push(cell);
      cell = "";
      started = true;
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && source[index + 1] === "\n") index += 1;
      if (!started) continue;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
      started = false;
    } else {
      cell += char;
      started = true;
    }
  }
  if (quoted) throw new Error("CSV has an unterminated quoted field");
  if (started || cell !== "") {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}
