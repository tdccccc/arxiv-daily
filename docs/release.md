# Release checklist

Pushing a stable SemVer tag such as `0.3.0` runs
`.github/workflows/release.yml`. Tags must not have a `v` prefix, prerelease
suffix, or build metadata. The workflow is serialized per tag, refuses to
replace an existing GitHub release, uploads without `--clobber`, and uses the
curated `docs/releases/<version>.md` rather than generated notes.

The Obsidian release assets remain exactly:

- `plugin/manifest.json`
- `plugin/main.js`
- `plugin/styles.css`

Production plugin and CLI bundles contain the complete locked pako notice from
`THIRD_PARTY_NOTICES.md`.

## Prepare metadata

Use Node.js 20.11.0 or newer from a clean checkout. Create the curated notes
before tagging, then synchronize metadata with the repository tool:

```bash
export VERSION=0.3.0
npm ci
npm run sync:release-version -- "$VERSION"
npm run check:release-version -- "$VERSION"
```

The sync command updates workspace package versions and internal dependency
specifiers, both Obsidian manifests and version maps, and the authoritative
root `package-lock.json`. It does not publish, commit, tag, or push. It does not
create release notes. Do not use workspace-local lockfiles.

Review every changed metadata file and ensure `docs/releases/$VERSION.md` is
curated for that release. In particular, confirm both `versions.json` files map
the release to the `minAppVersion` in the manifests.

## Verify locally

```bash
npm ci
npm run test:release-tools
npm run check:release-version -- "$VERSION"
npm run check:boundaries
npm run lint
npm run typecheck
npm test
npm run build
npm run smoke:build
```

`npm run lint` applies `eslint-plugin-obsidianmd`'s recommended flat config to
the production Obsidian plugin TypeScript and the root `manifest.json` and
`LICENSE`; it excludes tests, generated bundles, and non-plugin workspaces. The
public ESLint rules approximate Obsidian's source scanner, but they are not the
complete hosted review and do not replace its CSS or release asset checks.

`smoke:build` exercises the canonical CLI, the copied plugin CLI, the Python
compatibility shim, and the production plugin bundle. It also requires the
complete locked pako license notice exactly once in every JavaScript bundle.

The canonical CLI output is `apps/cli/dist/arxiv-daily-cli.cjs`; the build also
refreshes `plugin/arxiv-daily-cli.cjs`. The Python file remains a deprecated
delegating shim.

## Tag and publish

Commit the reviewed preparation, then create an annotated stable tag at that
exact commit:

```bash
git tag -a "$VERSION" -m "arXiv Daily $VERSION"
git push origin main
git push origin "$VERSION"
```

Never move or reuse a published tag. If the workflow fails before creating a
release, fix the underlying commit and publish a new version; do not replace
assets or mutate an existing release. The workflow verifies the checked-out tag
target, version metadata, curated-note presence, tests, build, smoke checks, and
provenance before creating the GitHub release.

After CI succeeds, verify that the release notes are the curated file, the
three assets are present, and their provenance attestations are available.

arXiv Daily is listed through Obsidian's Community directory. The current new
plugin flow is to sign in at `community.obsidian.md`, link the repository owner's
GitHub account, choose **Plugins → New plugin**, provide the repository URL,
accept the policies and support commitment, and submit. The directory reads the
root `manifest.json` from the default branch and reviews the matching GitHub
release; automated feedback is resolved by committing fixes and publishing a
new, incremented release. This replaces the former `obsidian-releases` pull
request submission flow. An already-listed plugin needs no resubmission or
`obsidian-releases` pull request: Obsidian obtains later versions from matching
GitHub releases.
