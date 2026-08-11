import {
  sanitizeDetailSelection,
  validateScheduleConfig,
  validateSchedulerConfig,
  validateVaultRelativeDirectory,
  vaultRelativeDirectoriesCollide,
  type LogLevel,
  type PluginSettings,
  type RunHistoryStore,
  type StateStore,
} from "@arxiv-daily/core";

export interface SettingsValueChange {
  key: string;
  value: unknown;
}

export interface PreparedOutputStores {
  stateStore: StateStore;
  runHistoryStore: RunHistoryStore;
}

export interface ExplicitSettingsChange<TPrepared = unknown> {
  changes: readonly SettingsValueChange[];
  validateCandidate?: (candidate: PluginSettings) => void;
  prepare?: (candidate: PluginSettings) => Promise<TPrepared> | TPrepared;
  install?: (prepared: TPrepared, candidate: PluginSettings) => void;
}

export interface SettingsChangeDependencies {
  settings: PluginSettings;
  persistSettings: (candidate: PluginSettings) => Promise<void>;
  prepareOutputStores?: (
    candidate: PluginSettings,
  ) => Promise<PreparedOutputStores>;
  installOutputStores?: (prepared: PreparedOutputStores) => void;
  hasActiveOutputWork?: () => boolean;
  beginOutputTransition?: () => () => void;
  reportPostCommitError?: (action: string, error: unknown) => void;
  setLoggerLevel?: (level: LogLevel) => void;
  setLoggerTimezone?: (timezone: string) => void;
  restartScheduler?: () => void;
  setScheduleEnabled?: (enabled: boolean) => void;
  refreshSensitiveValues?: () => void;
}

/** A rejected change includes the pre-change values a renderer can restore. */
export class SettingsChangeError extends Error {
  constructor(
    message: string,
    private readonly previousSettings: PluginSettings,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SettingsChangeError";
  }

  restoreValue(key: string): unknown {
    return readPath(this.previousSettings, key);
  }
}

/**
 * Plugin-local serialized transaction boundary for scalar and runtime-coupled
 * settings changes. Candidates remain private until persistence succeeds.
 */
export class SettingsChangeService {
  private queue: Promise<void> = Promise.resolve();

  constructor(private readonly deps: SettingsChangeDependencies) {}

  changeValue(key: string, value: unknown): Promise<void> {
    return this.change({ changes: [{ key, value }] });
  }

  /** Serialize an existing complex-editor mutation with candidate transactions. */
  persistCurrent(): Promise<void> {
    return this.enqueue(async () => {
      const previous = cloneSettings(this.deps.settings);
      const candidate = cloneSettings(this.deps.settings);
      candidate.detailSelection = sanitizeDetailSelection(
        candidate.detailSelection,
      );
      const commitPaths = changedLeafPaths(previous, candidate);
      try {
        assertLiveCommitEligible(this.deps.settings, candidate, commitPaths);
        await this.deps.persistSettings(candidate);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new SettingsChangeError(message, previous, { cause: error });
      }
      if (!this.commitLiveSettings(candidate, commitPaths)) return;
      this.runPostCommit("refresh sensitive values", () =>
        this.deps.refreshSensitiveValues?.());
    });
  }

  change<TPrepared = unknown>(
    request: ExplicitSettingsChange<TPrepared>,
  ): Promise<void> {
    return this.enqueue(() => this.applyChange(request));
  }

  private enqueue(operation: () => Promise<void>): Promise<void> {
    const queued = this.queue.then(operation);
    this.queue = queued.catch(() => undefined);
    return queued;
  }

  private async applyChange<TPrepared>(
    request: ExplicitSettingsChange<TPrepared>,
  ): Promise<void> {
    const previous = cloneSettings(this.deps.settings);
    const candidate = cloneSettings(this.deps.settings);
    let changedKeys: string[] = [];
    let commitPaths: string[] = [];
    let preparedOutput: PreparedOutputStores | undefined;
    let prepared: TPrepared | undefined;
    let releaseOutputTransition: (() => void) | undefined;
    try {
      try {
        const requestedKeys = Array.from(new Set(request.changes.map(({ key }) => key)));
        for (const change of request.changes) {
          writePath(candidate, change.key, cloneValue(change.value));
        }
        if (requestedKeys.every(
          (key) => valuesEqual(readPath(previous, key), readPath(candidate, key)),
        )) return;
        candidate.detailSelection = sanitizeDetailSelection(
          candidate.detailSelection,
        );

        this.validateCandidate(candidate, requestedKeys);
        request.validateCandidate?.(candidate);
        changedKeys = requestedKeys.filter(
          (key) => !valuesEqual(readPath(previous, key), readPath(candidate, key)),
        );
        commitPaths = changedLeafPaths(previous, candidate);
        if (commitPaths.length === 0) return;
        assertLiveCommitEligible(this.deps.settings, candidate, commitPaths);

        const outputDirectoryChanged = changedKeys.some(isOutputDirectoryKey);
        if (outputDirectoryChanged) {
          if (this.deps.hasActiveOutputWork?.()) {
            throw new Error("Output directories cannot change while operations or runs are active");
          }
          if (!this.deps.prepareOutputStores || !this.deps.installOutputStores) {
            throw new Error("Output store preparation is unavailable");
          }
          releaseOutputTransition = this.deps.beginOutputTransition?.();
          if (this.deps.hasActiveOutputWork?.()) {
            throw new Error("Output directories cannot change while operations or runs are active");
          }
          preparedOutput = await this.deps.prepareOutputStores(candidate);
        }
        prepared = await request.prepare?.(candidate);
        if (outputDirectoryChanged && this.deps.hasActiveOutputWork?.()) {
          throw new Error("Output directories cannot change while operations or runs are active");
        }
        await this.deps.persistSettings(candidate);
      } catch (error) {
        if (error instanceof SettingsChangeError) throw error;
        const message = error instanceof Error ? error.message : String(error);
        throw new SettingsChangeError(message, previous, { cause: error });
      }

      // Persistence is durable from here on. The live commit is a gate: no
      // candidate stores or runtime effects may install unless it completes.
      if (!this.commitLiveSettings(candidate, commitPaths)) return;
      this.runPostCommit("refresh sensitive values", () =>
        this.deps.refreshSensitiveValues?.());
      if (preparedOutput) {
        const outputStores = preparedOutput;
        this.runPostCommit("install output stores", () =>
          this.deps.installOutputStores?.(outputStores));
      }
      if (request.install) {
        this.runPostCommit("install prepared settings", () =>
          request.install?.(prepared as TPrepared, candidate));
      }
      this.installRuntimeEffects(previous, candidate, changedKeys);
    } finally {
      this.releaseOutputTransition(releaseOutputTransition);
    }
  }

  private validateCandidate(
    candidate: PluginSettings,
    changedKeys: readonly string[],
  ): void {
    if (changedKeys.some(isOutputDirectoryKey)) {
      for (const field of ["dailyDir", "papersDir"] as const) {
        const validation = validateVaultRelativeDirectory(candidate.output[field]);
        if (!validation.ok || !validation.value) {
          throw new Error(`Invalid output.${field}: ${validation.reason ?? "Invalid path"}`);
        }
        candidate.output[field] = validation.value;
      }
      if (
        vaultRelativeDirectoriesCollide(
          candidate.output.dailyDir,
          candidate.output.papersDir,
        )
      ) {
        throw new Error("Daily and papers directories must be different");
      }
    }

    if (changedKeys.includes("arxiv.timezone")) {
      const timezone = candidate.arxiv.timezone.trim();
      if (!isValidTimezone(timezone)) {
        throw new Error(`Invalid timezone: ${candidate.arxiv.timezone}`);
      }
      candidate.arxiv.timezone = timezone;
    }

    if (
      changedKeys.includes("advanced.logLevel") &&
      !isLogLevel(candidate.advanced.logLevel)
    ) {
      throw new Error(`Invalid log level: ${String(candidate.advanced.logLevel)}`);
    }

    if (changedKeys.includes("schedule.tickIntervalMin")) {
      const interval = candidate.schedule.tickIntervalMin;
      if (!Number.isFinite(interval) || interval < 1) {
        throw new Error(`Invalid scheduler tick interval: ${String(interval)}`);
      }
    }

    if (changedKeys.some((key) => key.startsWith("schedule."))) {
      const validation = candidate.schedule.enabled
        ? validateSchedulerConfig(candidate)
        : validateScheduleConfig(candidate);
      if (!validation.ok) {
        throw new Error(validation.reasons.join("; "));
      }
    }
  }

  private commitLiveSettings(
    candidate: PluginSettings,
    commitPaths: readonly string[],
  ): boolean {
    try {
      assertLiveCommitEligible(this.deps.settings, candidate, commitPaths);
      commitChangedPaths(this.deps.settings, candidate, commitPaths);
      return true;
    } catch (error) {
      try {
        this.deps.reportPostCommitError?.("commit live settings", error);
      } catch {
        // Durable post-commit reporting must not become rollback UI.
      }
      return false;
    }
  }

  private installRuntimeEffects(
    previous: PluginSettings,
    candidate: PluginSettings,
    changedKeys: readonly string[],
  ): void {
    if (changedKeys.includes("advanced.logLevel")) {
      this.runPostCommit("apply logger level", () =>
        this.deps.setLoggerLevel?.(candidate.advanced.logLevel));
    }
    if (changedKeys.includes("arxiv.timezone")) {
      this.runPostCommit("apply logger timezone", () =>
        this.deps.setLoggerTimezone?.(candidate.arxiv.timezone));
    }
    if (changedKeys.includes("schedule.tickIntervalMin")) {
      this.runPostCommit("restart scheduler", () =>
        this.deps.restartScheduler?.());
    }
    if (
      changedKeys.includes("schedule.enabled") &&
      previous.schedule.enabled !== candidate.schedule.enabled
    ) {
      this.runPostCommit("apply schedule enabled", () =>
        this.deps.setScheduleEnabled?.(candidate.schedule.enabled));
    }
  }

  private runPostCommit(action: string, operation: () => void): void {
    try {
      operation();
    } catch (error) {
      try {
        this.deps.reportPostCommitError?.(action, error);
      } catch {
        // Post-commit reporting must not turn a durable change into rollback UI.
      }
    }
  }

  private releaseOutputTransition(release: (() => void) | undefined): void {
    if (!release) return;
    this.runPostCommit("release output transition", release);
  }
}

export function isValidTimezone(timezone: string): boolean {
  if (!timezone.trim()) return false;
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: timezone }).format();
    return true;
  } catch (error) {
    if (error instanceof RangeError) return false;
    throw error;
  }
}

function cloneSettings(settings: PluginSettings): PluginSettings {
  return {
    llm: { ...settings.llm },
    arxiv: {
      ...settings.arxiv,
      categories: [...settings.arxiv.categories],
      topics: settings.arxiv.topics.map((topic) => ({ ...topic })),
    },
    detailSelection: { ...settings.detailSelection },
    output: { ...settings.output },
    schedule: { ...settings.schedule },
    advanced: { ...settings.advanced },
    email: { ...settings.email },
  };
}

function isOutputDirectoryKey(key: string): boolean {
  return key === "output.dailyDir" || key === "output.papersDir";
}

function isLogLevel(value: unknown): value is LogLevel {
  return value === "debug" || value === "info" || value === "warn" || value === "error";
}

function readPath(settings: PluginSettings, key: string): unknown {
  let value: unknown = settings;
  for (const part of key.split(".")) {
    if (!isRecord(value) || !(part in value)) return undefined;
    value = value[part];
  }
  return value;
}

function writePath(settings: PluginSettings, key: string, value: unknown): void {
  const parts = key.split(".");
  const final = parts.pop();
  if (!final || parts.length === 0) throw new Error(`Invalid settings key: ${key}`);
  let target: Record<string, unknown> = settings as unknown as Record<string, unknown>;
  for (const part of parts) {
    const next = target[part];
    if (!isRecord(next)) throw new Error(`Unknown settings key: ${key}`);
    target = next;
  }
  if (!(final in target)) throw new Error(`Unknown settings key: ${key}`);
  target[final] = value;
}

function cloneValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    const items: unknown[] = value;
    return items.map((item) => cloneValue(item));
  }
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, cloneValue(item)]),
    );
  }
  return value;
}

function valuesEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) && Array.isArray(right)) {
    return left.length === right.length && left.every(
      (item, index) => valuesEqual(item, right[index]),
    );
  }
  if (!isRecord(left) || !isRecord(right)) return false;
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);
  return leftKeys.length === rightKeys.length && leftKeys.every(
    (key) => key in right && valuesEqual(left[key], right[key]),
  );
}

function changedLeafPaths(
  previous: unknown,
  candidate: unknown,
  prefix = "",
): string[] {
  if (valuesEqual(previous, candidate)) return [];
  if (isRecord(previous) && isRecord(candidate)) {
    return Array.from(new Set([
      ...Object.keys(previous),
      ...Object.keys(candidate),
    ])).flatMap((key) => changedLeafPaths(
      previous[key],
      candidate[key],
      prefix ? `${prefix}.${key}` : key,
    ));
  }
  return prefix ? [prefix] : [];
}

function assertLiveCommitEligible(
  target: PluginSettings,
  source: PluginSettings,
  paths: readonly string[],
): void {
  for (const path of paths) {
    const parts = path.split(".");
    const final = parts.pop();
    if (!final) throw new Error(`Invalid live settings path: ${path}`);
    let targetParent: unknown = target;
    let sourceParent: unknown = source;
    for (const part of parts) {
      if (!isRecord(targetParent) || !isRecord(sourceParent)) {
        throw new Error(`Live settings path is unavailable: ${path}`);
      }
      targetParent = targetParent[part];
      sourceParent = sourceParent[part];
    }
    if (!isRecord(targetParent) || !isRecord(sourceParent)) {
      throw new Error(`Live settings path is unavailable: ${path}`);
    }
    assertCommitValueEligible(
      targetParent,
      final,
      targetParent[final],
      sourceParent[final],
      path,
    );
  }
}

function assertCommitValueEligible(
  parent: Record<string, unknown>,
  key: string,
  current: unknown,
  next: unknown,
  path: string,
): void {
  const descriptor = Object.getOwnPropertyDescriptor(parent, key);
  if (!descriptor) throw new Error(`Live settings property is unavailable: ${path}`);
  if (descriptor.get || descriptor.set) {
    throw new Error(`Live settings accessor cannot be committed atomically: ${path}`);
  }
  if (!descriptor.writable && !canCommitInPlace(current, next)) {
    throw new Error(`Live settings property is not writable: ${path}`);
  }
  if (Array.isArray(current) && Array.isArray(next)) {
    const lengthDescriptor = Object.getOwnPropertyDescriptor(current, "length");
    if (!lengthDescriptor?.writable && current.length !== next.length) {
      throw new Error(`Live settings array length is not writable: ${path}`);
    }
    const sharedLength = Math.min(current.length, next.length);
    for (let index = 0; index < sharedLength; index += 1) {
      assertCommitValueEligible(
        current as unknown as Record<string, unknown>,
        String(index),
        current[index],
        next[index],
        `${path}.${index}`,
      );
    }
    return;
  }
  if (!isRecord(current) || !isRecord(next)) return;
  for (const currentKey of Object.keys(current)) {
    if (!(currentKey in next)) {
      const currentDescriptor = Object.getOwnPropertyDescriptor(current, currentKey);
      if (!currentDescriptor?.configurable) {
        throw new Error(`Live settings property cannot be removed: ${path}.${currentKey}`);
      }
    }
  }
  for (const [childKey, childNext] of Object.entries(next)) {
    if (!(childKey in current)) {
      if (!Object.isExtensible(current)) {
        throw new Error(`Live settings object is not extensible: ${path}`);
      }
      continue;
    }
    assertCommitValueEligible(
      current,
      childKey,
      current[childKey],
      childNext,
      `${path}.${childKey}`,
    );
  }
}

function commitChangedPaths(
  target: PluginSettings,
  source: PluginSettings,
  paths: readonly string[],
): void {
  for (const path of paths) {
    const current = readPath(target, path);
    const next = readPath(source, path);
    if (canCommitInPlace(current, next)) commitInPlace(current, next);
    else writePath(target, path, cloneValue(next));
  }
}

function commitInPlace(target: unknown, source: unknown): void {
  if (Array.isArray(target) && Array.isArray(source)) {
    const targetItems: unknown[] = target;
    const sourceItems: unknown[] = source;
    const sharedLength = Math.min(targetItems.length, sourceItems.length);
    for (let index = 0; index < sharedLength; index += 1) {
      const current: unknown = targetItems[index];
      const next: unknown = sourceItems[index];
      if (canCommitInPlace(current, next)) commitInPlace(current, next);
      else targetItems[index] = next;
    }
    if (targetItems.length > sourceItems.length) {
      targetItems.splice(sourceItems.length);
    } else if (sourceItems.length > targetItems.length) {
      targetItems.push(...sourceItems.slice(targetItems.length));
    }
    return;
  }
  if (!isRecord(target) || !isRecord(source)) return;
  for (const key of Object.keys(target)) {
    if (!(key in source)) delete target[key];
  }
  for (const [key, next] of Object.entries(source)) {
    const current = target[key];
    if (canCommitInPlace(current, next)) commitInPlace(current, next);
    else target[key] = next;
  }
}

function canCommitInPlace(target: unknown, source: unknown): boolean {
  return (
    (Array.isArray(target) && Array.isArray(source)) ||
    (isRecord(target) && isRecord(source))
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
