import { Notice } from "obsidian";

export type LogLevel = "debug" | "info" | "warn" | "error";
const ORDER: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };

export class Logger {
  constructor(private level: LogLevel = "info") {}

  setLevel(level: LogLevel) {
    this.level = level;
  }

  private allowed(l: LogLevel) {
    return ORDER[l] >= ORDER[this.level];
  }

  debug(msg: string, ...rest: unknown[]) {
    if (this.allowed("debug")) console.debug("[arxiv-daily]", msg, ...rest);
  }
  info(msg: string, ...rest: unknown[]) {
    if (this.allowed("info")) console.log("[arxiv-daily]", msg, ...rest);
  }
  warn(msg: string, ...rest: unknown[]) {
    if (this.allowed("warn")) console.warn("[arxiv-daily]", msg, ...rest);
  }
  error(msg: string, ...rest: unknown[]) {
    if (this.allowed("error")) console.error("[arxiv-daily]", msg, ...rest);
  }

  notice(msg: string, timeoutMs = 5000) {
    new Notice(msg, timeoutMs);
  }
}
