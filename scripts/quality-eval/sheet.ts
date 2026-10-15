import { SHEET_COLUMNS } from "./export";
import { parseCsv } from "./csv";

export interface SheetRow {
  /** Spreadsheet row number: the header is row 1. */
  row: number;
  date: string;
  id: string;
  title: string;
  hadAbstractAtFilter: boolean;
  modelCategory: string;
  modelDirections: number[];
  modelScore: number;
  modelOutcome: string;
  /** Null while the researcher has not judged the row. */
  label: "y" | "n" | null;
  labelDirections: number[];
  note: string;
}

export class SheetError extends Error {
  constructor(readonly problems: readonly string[]) {
    super(problems.join("\n"));
    this.name = "SheetError";
  }
}

const MAX_REPORTED_PROBLEMS = 30;

/**
 * Read a filled label sheet. Every bad cell is reported with its spreadsheet row
 * number, so the researcher can fix the file rather than guess what was dropped.
 */
export function parseSheet(text: string, legendNumbers: ReadonlySet<number>): SheetRow[] {
  const table = parseCsv(text);
  const header = (table[0] ?? []).map((cell) => cell.trim());
  const missing = SHEET_COLUMNS.filter((column) => !header.includes(column));
  if (missing.length > 0) throw new SheetError([`the sheet is missing column(s): ${missing.join(", ")}`]);
  const at = Object.fromEntries(SHEET_COLUMNS.map((column) => [column, header.indexOf(column)])) as Record<(typeof SHEET_COLUMNS)[number], number>;
  const legendText = legendNumbers.size > 0 ? `1..${Math.max(...legendNumbers)}` : "none";

  const problems: string[] = [];
  const rows: SheetRow[] = [];
  table.slice(1).forEach((cells, index) => {
    if (cells.every((cell) => cell.trim() === "")) return;
    const row = index + 2;
    const cell = (column: (typeof SHEET_COLUMNS)[number]): string => (cells[at[column]] ?? "").trim();
    const problem = (message: string): void => { problems.push(`row ${row}: ${message}`); };

    const id = cell("id");
    if (!id) problem("id is empty");
    const flag = cell("had_abstract_at_filter").toLowerCase();
    if (flag !== "true" && flag !== "false") problem(`had_abstract_at_filter must be true or false, got "${cell("had_abstract_at_filter")}"`);
    const score = Number(cell("model_score"));
    if (cell("model_score") === "" || !Number.isFinite(score)) problem(`model_score must be a number, got "${cell("model_score")}"`);

    const labelText = cell("label").toLowerCase();
    if (labelText !== "" && labelText !== "y" && labelText !== "n") {
      problem(`unknown label "${cell("label")}" (use y, n or leave it empty)`);
    }
    const modelDirections = parseNumbers(cell("model_directions"), "model_directions", legendNumbers, legendText, problem);
    const labelDirections = parseNumbers(cell("label_direction"), "label_direction", legendNumbers, legendText, problem);
    if (labelDirections.length > 0 && labelText !== "y") {
      problem(`label_direction is only for rows labelled y, but label is "${cell("label")}"`);
    }

    rows.push({
      row, date: cell("date"), id, title: cells[at.title] ?? "",
      hadAbstractAtFilter: flag === "true", modelCategory: cell("model_category"), modelDirections,
      modelScore: score, modelOutcome: cell("model_outcome"),
      label: labelText === "y" || labelText === "n" ? labelText : null,
      labelDirections, note: cells[at.note] ?? "",
    });
  });
  if (problems.length > 0) {
    const shown = problems.slice(0, MAX_REPORTED_PROBLEMS);
    if (problems.length > shown.length) shown.push(`... and ${problems.length - shown.length} more`);
    throw new SheetError(shown);
  }
  return rows;
}

/** Legend numbers separated by ; , or whitespace, as a spreadsheet user may type them. */
function parseNumbers(
  text: string,
  column: string,
  legend: ReadonlySet<number>,
  legendText: string,
  problem: (message: string) => void,
): number[] {
  const numbers: number[] = [];
  for (const part of text.split(/[;,，；、\s]+/).filter(Boolean)) {
    if (!/^\d+$/.test(part)) { problem(`${column} must be legend numbers like 1;2, got "${part}"`); continue; }
    const value = Number(part);
    if (!legend.has(value)) { problem(`${column} has unknown direction number ${value} (the legend has ${legendText})`); continue; }
    if (!numbers.includes(value)) numbers.push(value);
  }
  return numbers;
}
