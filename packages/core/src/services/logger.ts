import { redactText, sanitizeValue } from "../utils/redaction";

export type LogLevel = "debug" | "info" | "warn" | "error";
export type NoticeSink = (message: string, timeoutMs: number) => void;

const ORDER: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };

const MAX_BUFFER_SIZE = 5000;

export class Logger {
  private buffer: string[] = [];
  private timezone: string | undefined;
  private secrets: string[] = [];

  constructor(
    private level: LogLevel = "info",
    private noticeSink?: NoticeSink,
    tz?: string,
  ) {
    this.timezone = tz;
  }

  setLevel(level: LogLevel) {
    this.level = level;
  }

  setNoticeSink(noticeSink: NoticeSink | undefined) {
    this.noticeSink = noticeSink;
  }

  setTimezone(tz: string | undefined) {
    this.timezone = tz;
  }

  setSensitiveValues(values: readonly string[]): void {
    this.secrets = [...values];
    this.buffer = this.buffer.map((entry) => this.sanitizeText(entry));
  }

  private sanitizeText(value: unknown): string {
    return redactText(value, { secrets: this.secrets });
  }

  private sanitizeRest(values: unknown[]): unknown[] {
    return values.map((value) => sanitizeValue(value, { secrets: this.secrets }));
  }

  private allowed(l: LogLevel) {
    return ORDER[l] >= ORDER[this.level];
  }

  private formatTimestamp(): string {
    const now = new Date();
    if (this.timezone) {
      const fmt = new Intl.DateTimeFormat("en-GB", {
        timeZone: this.timezone,
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hourCycle: "h23",
      });
      const ms = String(now.getMilliseconds()).padStart(3, "0");
      return fmt.format(now) + "." + ms;
    }
    return now.toISOString().slice(11, 23);
  }

  private push(level: string, msg: string, ...rest: unknown[]) {
    const ts = this.formatTimestamp();
    const safeMessage = this.sanitizeText(msg);
    const safeRest = this.sanitizeRest(rest);
    const restStr = safeRest.length ? " " + safeRest.map((r) => String(r)).join(" ") : "";
    this.buffer.push(`${ts} [${level.toUpperCase()}] ${safeMessage}${restStr}`);
    if (this.buffer.length > MAX_BUFFER_SIZE) {
      this.buffer.splice(0, this.buffer.length - MAX_BUFFER_SIZE);
    }
  }

  getBuffer(): string[] {
    return [...this.buffer];
  }

  clearBuffer(): void {
    this.buffer = [];
  }

  debug(msg: string, ...rest: unknown[]) {
    if (this.allowed("debug")) {
      console.debug("[arxiv-daily]", this.sanitizeText(msg), ...this.sanitizeRest(rest));
      this.push("debug", msg, ...rest);
    }
  }
  info(msg: string, ...rest: unknown[]) {
    if (this.allowed("info")) {
      console.log("[arxiv-daily]", this.sanitizeText(msg), ...this.sanitizeRest(rest));
      this.push("info", msg, ...rest);
    }
  }
  warn(msg: string, ...rest: unknown[]) {
    if (this.allowed("warn")) {
      console.warn("[arxiv-daily]", this.sanitizeText(msg), ...this.sanitizeRest(rest));
      this.push("warn", msg, ...rest);
    }
  }
  error(msg: string, ...rest: unknown[]) {
    if (this.allowed("error")) {
      console.error("[arxiv-daily]", this.sanitizeText(msg), ...this.sanitizeRest(rest));
      this.push("error", msg, ...rest);
    }
  }

  notice(msg: string, timeoutMs = 5000) {
    if (this.noticeSink) {
      this.noticeSink(this.sanitizeText(msg), timeoutMs);
      return;
    }
    this.info(msg);
  }
}
