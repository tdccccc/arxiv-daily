const REDACTED = "[REDACTED]";
const SENSITIVE_KEY = /^(?:authorization|api[-_]?key|access[-_]?token|token|secret|password)$/i;
const BEARER_RE = /\bBearer\s+[^\s,;]+/gi;
const KEY_VALUE_RE = /\b(api[-_]?key|access[-_]?token|token|secret|password)(\s*[=:]\s*)([^\s,;&]+)/gi;
const JSON_SECRET_RE = /(["'](?:api[-_]?key|access[-_]?token|token|secret|password)["']\s*:\s*["'])([^"']*)(["'])/gi;
const COMMON_API_KEY_RE = /\b(?:sk|rk|pk|api)[-_][A-Za-z0-9_-]{8,}\b/g;

export interface RedactionOptions {
  secrets?: readonly string[];
}

export function redactText(value: unknown, options: RedactionOptions = {}): string {
  let text = stringify(value);
  for (const secret of normalizedSecrets(options.secrets)) {
    text = text.split(secret).join(REDACTED);
  }
  return text
    .replace(BEARER_RE, `Bearer ${REDACTED}`)
    .replace(JSON_SECRET_RE, `$1${REDACTED}$3`)
    .replace(KEY_VALUE_RE, `$1$2${REDACTED}`)
    .replace(COMMON_API_KEY_RE, REDACTED);
}

export function redactUrl(value: unknown, options: RedactionOptions = {}): string {
  const text = redactText(value, options);
  try {
    const url = new URL(text);
    if (url.username) url.username = REDACTED;
    if (url.password) url.password = REDACTED;
    const queryKeys: string[] = [];
    url.searchParams.forEach((_value, key) => queryKeys.push(key));
    for (const key of queryKeys) {
      if (SENSITIVE_KEY.test(key)) url.searchParams.set(key, REDACTED);
    }
    return url.toString();
  } catch {
    return text.replace(
      /([?&](?:api[-_]?key|access[-_]?token|token|secret|password)=)[^&#\s]*/gi,
      `$1${REDACTED}`,
    );
  }
}

export function redactError(error: unknown, options: RedactionOptions = {}): Error {
  const source = error instanceof Error ? error : new Error(stringify(error));
  const safe = new Error(redactText(source.message, options));
  safe.name = source.name;
  if (source.stack) safe.stack = redactText(source.stack, options);
  const status = (source as Error & { status?: unknown }).status;
  if (typeof status === "number") (safe as Error & { status?: number }).status = status;
  return safe;
}

export function sanitizeValue(value: unknown, options: RedactionOptions = {}): unknown {
  if (value instanceof Error) return redactError(value, options);
  if (typeof value === "string") return redactText(value, options);
  if (value == null || typeof value === "number" || typeof value === "boolean") return value;
  return redactText(value, options);
}

function normalizedSecrets(secrets: readonly string[] | undefined): string[] {
  return [...new Set((secrets ?? []).map((value) => value.trim()).filter((value) => value.length >= 4))]
    .sort((a, b) => b.length - a.length);
}

function stringify(value: unknown): string {
  if (typeof value === "string") return value;
  if (value instanceof Error) return value.message;
  try {
    return typeof value === "object" && value !== null ? JSON.stringify(value) : String(value);
  } catch {
    return String(value);
  }
}
