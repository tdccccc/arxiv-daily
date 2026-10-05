// Core and node-runtime also run under Node (the CLI), where there is no
// `window`, so they cannot follow Obsidian's `window.setTimeout` advice. These
// wrappers keep that platform decision in one place. They read the global
// object on every call so test fake timers that patch it still apply.
interface TimerHost {
  setTimeout: typeof setTimeout;
  clearTimeout: typeof clearTimeout;
  setInterval: typeof setInterval;
  clearInterval: typeof clearInterval;
}
const host = (): TimerHost => globalThis;

export function setTimer(handler: () => void, ms?: number): ReturnType<typeof setTimeout> {
  return host().setTimeout(handler, ms);
}

export function clearTimer(id: ReturnType<typeof setTimeout> | undefined): void {
  host().clearTimeout(id);
}

export function setRepeatingTimer(handler: () => void, ms?: number): ReturnType<typeof setInterval> {
  return host().setInterval(handler, ms);
}

export function clearRepeatingTimer(id: ReturnType<typeof setInterval> | undefined): void {
  host().clearInterval(id);
}
