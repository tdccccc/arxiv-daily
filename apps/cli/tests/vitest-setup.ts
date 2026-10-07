// happy-dom (the DOM environment the `@vitest-environment happy-dom` test
// files use) does not implement `Document#compatMode`, so it reads back as
// `undefined`. KaTeX's `render()` (building math directly into a DOM
// element — see `apps/cli/src/workbench/web/markdown-dom.ts`) treats
// anything other than `"CSS1Compat"` as "this page is in quirks mode" and
// permanently disables itself with a loud console warning. The real
// workbench page always ships `<!doctype html>` (see
// `apps/cli/src/workbench/web/index.html`), so it is never in quirks mode;
// this only patches the test environment to match that reality instead of
// the default implementation gap.
if (typeof document !== "undefined" && document.compatMode !== "CSS1Compat") {
  Object.defineProperty(document, "compatMode", { value: "CSS1Compat", configurable: true, writable: true });
}
