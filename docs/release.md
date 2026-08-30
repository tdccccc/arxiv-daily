# Release checklist

Pushing a stable SemVer tag such as `0.3.1` runs
`.github/workflows/release.yml`. Tags must not have a `v` prefix, prerelease
suffix, or build metadata. The workflow is serialized per tag, refuses to
replace an existing GitHub release, uploads without `--clobber`, and uses the
curated `docs/releases/<version>.md` rather than generated notes.

After the GitHub release succeeds, `.github/workflows/publish-cli.yml`
publishes the CLI (`arxiv-daily`) through npm Trusted Publishing with build
provenance. npm binds that workflow's GitHub OIDC identity; no long-lived npm
publish token or one-time password is passed to Actions.

The Obsidian release assets remain exactly:

- `plugin/manifest.json`
- `plugin/main.js`
- `plugin/styles.css`

That list is not decorative. `npm run test:release-tools` reads it back and holds
it to `RELEASE_ASSETS` in `scripts/release-assets.mjs`, which is also what
`.github/workflows/release.yml` attests and uploads and what the desktop
acceptance harness deploys into the test vault. Changing the assets means
changing all of them in the same commit; changing only one of them fails the
check by name.

Production plugin and CLI bundles contain the complete locked pako notice from
`THIRD_PARTY_NOTICES.md`.

## Prepare metadata

Use Node.js 20.19.0 or newer from a clean checkout. Create the curated notes
before tagging, then synchronize metadata with the repository tool:

```bash
export VERSION=0.3.1
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
NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1
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
Also verify the npm package: `npm view arxiv-daily version` should report the
new version with the expected `dist-tag latest`, and the published package
should carry a build provenance attestation (visible in the registry metadata
for that version).

Configure npm Trusted Publishing for package `arxiv-daily` with GitHub owner
`tdccccc`, repository `arxiv-daily`, and workflow `publish-cli.yml`. Leave the
environment blank unless the workflow is changed to use a named GitHub
environment. The workflow uses a GitHub-hosted runner, Node 22.17.0, npm 11.5.1
or newer, and `id-token: write` as required by npm.

If CLI publication fails after the immutable GitHub release already exists, do
not re-run or move the tag. Manually dispatch **Publish CLI to npm** with that
existing stable version. The same trusted workflow checks out the immutable
tag, requires the matching GitHub release, refuses an already-published npm
version, reruns the complete release gate, and publishes through OIDC. If the
package was already published, fix forward by bumping to a new version.

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
