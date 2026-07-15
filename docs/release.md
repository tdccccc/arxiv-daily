# Release checklist

Pushing a plain SemVer tag such as `0.2.2` runs
`.github/workflows/release.yml`. Obsidian release assets remain exactly:

- `plugin/manifest.json`
- `plugin/main.js`
- `plugin/styles.css`

## Update versions

From a clean `main` checkout:

```bash
export VERSION=0.2.2
export MIN_APP_VERSION=1.4.0
npm version "$VERSION" --workspaces --include-workspace-root --no-git-tag-version
```

Update `manifest.json` and `plugin/manifest.json` to `VERSION` and
`MIN_APP_VERSION`. Add the same entry to `versions.json` and
`plugin/versions.json`. There is one authoritative root `package-lock.json`;
there must not be a workspace-local lockfile.

## Verify locally

```bash
npm ci
npm run check:boundaries
npm run typecheck
npm test
npm run build
node apps/cli/dist/arxiv-daily-cli.cjs --help
node plugin/arxiv-daily-cli.cjs --help
python3 arxiv_daily.py --help
```

The CLI build's canonical output is
`apps/cli/dist/arxiv-daily-cli.cjs`; it also refreshes the compatibility bundle
at `plugin/arxiv-daily-cli.cjs`. The Python file remains a deprecated delegating
shim.

Review the metadata and generated plugin assets, commit the version bump, then
tag and push:

```bash
git tag "$VERSION"
git push origin main
git push origin "$VERSION"
```

Tags must not have a leading `v`. After CI succeeds, verify that the GitHub release
contains the three Obsidian assets listed above. No `obsidian-releases` PR is
needed for an already-listed plugin.
