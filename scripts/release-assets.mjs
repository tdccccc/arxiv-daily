/**
 * The single list of files that ship as an Obsidian release, and the single
 * list the desktop acceptance harness deploys into the test vault.
 *
 * These two used to be independent literals. When `styles.css` was missing from
 * the acceptance list, every settings-page geometry measurement described the
 * stylesheet the test vault happened to hold rather than the one the branch
 * builds — the run stayed green and the numbers were simply about the wrong
 * file. Adding `styles.css` closed that one hole; keeping one list closes the
 * category, because a fourth asset cannot reach the release without reaching
 * the harness too.
 *
 * The prose copy in `docs/release.md` and the two copies in
 * `.github/workflows/release.yml` are checked against this list by
 * `scripts/release-asset-sources.mjs` (see `scripts/tests/release-assets.test.mjs`),
 * so a one-sided edit to any of them fails `npm run test:release-tools`.
 *
 * This module deliberately has no imports: the acceptance harness loads it.
 */
export const RELEASE_ASSETS = Object.freeze(["main.js", "manifest.json", "styles.css"]);

/** Where each asset lives in the repository, as `docs/release.md` names them. */
export function releaseAssetRepoPaths(assets = RELEASE_ASSETS) {
  return assets.map((name) => `plugin/${name}`);
}
