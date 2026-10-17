export interface OptionSpec {
  /** Options that stand alone, such as `--yes`. */
  flags?: readonly string[];
  /** Options followed by exactly one value, such as `--date 2026-06-13`. */
  values?: readonly string[];
  /** Maximum number of positional arguments. */
  positionals?: number;
}

export interface ParsedOptions {
  flags: ReadonlySet<string>;
  values: Readonly<Record<string, string>>;
  positionals: readonly string[];
}

/**
 * Strict argument walk for one subcommand: anything not named in `spec` (unknown
 * option, repeated option, missing value, extra positional) throws, so a typo
 * never silently turns into a different, real operation.
 */
export function parseOptions(command: string, args: readonly string[], spec: OptionSpec): ParsedOptions {
  const flags = new Set<string>();
  const values: Record<string, string> = {};
  const positionals: string[] = [];
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]!;
    if (spec.flags?.includes(arg)) {
      if (flags.has(arg)) throw new Error(`${command}: ${arg} given more than once`);
      flags.add(arg);
    } else if (spec.values?.includes(arg)) {
      if (Object.hasOwn(values, arg)) throw new Error(`${command}: ${arg} given more than once`);
      const value = args[index + 1];
      if (!value || value.startsWith("--")) throw new Error(`${arg} requires a value`);
      values[arg] = value;
      index += 1;
    } else if (arg.startsWith("-") && arg !== "-") {
      throw new Error(`${command}: unknown option ${arg}`);
    } else if (positionals.length < (spec.positionals ?? 0)) {
      positionals.push(arg);
    } else {
      throw new Error(`${command}: unexpected argument ${arg}`);
    }
  }
  return { flags, values, positionals };
}

/** A real calendar day written as YYYY-MM-DD (rejects 2026-02-30). */
export function isCalendarDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(value);
  return Number.isFinite(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}
