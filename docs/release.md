# Release checklist

This document records the maintainer flow for publishing a new Obsidian
plugin release of arXiv Daily.

The plugin is already listed in the Obsidian Community directory. After the
initial listing, future versions are discovered from GitHub releases
automatically; do not open a PR against `obsidianmd/obsidian-releases`.

## What the release workflow does

Pushing a SemVer tag such as `0.2.2` triggers
`.github/workflows/release.yml`.

The workflow:

1. Installs dependencies in `plugin/` with `npm ci`.
2. Verifies the tag matches:
   - `manifest.json`
   - `plugin/manifest.json`
   - `plugin/package.json`
3. Verifies both version maps contain the tag:
   - `versions.json`
   - `plugin/versions.json`
4. Runs `npm test`.
5. Runs `npm run build`.
6. Creates or updates the GitHub release for that tag.
7. Uploads `plugin/manifest.json`, `plugin/main.js`, and `plugin/styles.css`.

Obsidian installs updates from the GitHub release whose tag exactly matches
`manifest.json.version`.

## Before releasing

Start from a clean working tree on `main`:

```bash
git status --short
git pull --ff-only
```

Choose the next version. Use plain `x.y.z` tags, without a leading `v`.

```bash
export VERSION=0.2.2
export MIN_APP_VERSION=1.4.0
```

## Update version files

Update the package version first. This keeps `plugin/package.json` and
`plugin/package-lock.json` in sync.

```bash
npm --prefix plugin version "$VERSION" --no-git-tag-version
```

Then update both manifests and both Obsidian version maps:

```bash
node - <<'NODE'
const fs = require("fs");

const version = process.env.VERSION;
const minAppVersion = process.env.MIN_APP_VERSION;

if (!version) throw new Error("VERSION is required");
if (!minAppVersion) throw new Error("MIN_APP_VERSION is required");

for (const file of ["manifest.json", "plugin/manifest.json"]) {
  const json = JSON.parse(fs.readFileSync(file, "utf8"));
  json.version = version;
  json.minAppVersion = minAppVersion;
  fs.writeFileSync(file, `${JSON.stringify(json, null, 2)}\n`);
}

for (const file of ["versions.json", "plugin/versions.json"]) {
  const json = JSON.parse(fs.readFileSync(file, "utf8"));
  json[version] = minAppVersion;
  fs.writeFileSync(file, `${JSON.stringify(json, null, 2)}\n`);
}
NODE
```

If the release changes the minimum supported Obsidian version, update
`MIN_APP_VERSION` before running the script.

## Verify locally

Run the same core checks as CI before tagging:

```bash
node -e '
const version = process.env.VERSION;
for (const file of ["manifest.json", "plugin/manifest.json", "plugin/package.json"]) {
  const json = require(`./${file}`);
  if (json.version !== version) throw new Error(`${file} has ${json.version}, expected ${version}`);
}
for (const file of ["versions.json", "plugin/versions.json"]) {
  const json = require(`./${file}`);
  if (!json[version]) throw new Error(`${file} is missing ${version}`);
}
console.log(`release metadata OK for ${version}`);
'

cd plugin
npm test
npm run build
cd ..
```

Review the diff:

```bash
git diff -- manifest.json versions.json plugin/manifest.json plugin/versions.json plugin/package.json plugin/package-lock.json
```

Expected release metadata changes:

- `manifest.json`
- `versions.json`
- `plugin/manifest.json`
- `plugin/versions.json`
- `plugin/package.json`
- `plugin/package-lock.json`

## Commit and tag

Commit the version bump:

```bash
git add manifest.json versions.json plugin/manifest.json plugin/versions.json plugin/package.json plugin/package-lock.json
git commit -m "chore(release): bump to $VERSION"
```

Create and push the tag:

```bash
git tag "$VERSION"
git push origin main
git push origin "$VERSION"
```

The tag push starts the release workflow.

## After pushing the tag

Watch the workflow:

```bash
gh run list --workflow release.yml --limit 5
```

Check the release assets after the workflow finishes:

```bash
gh release view "$VERSION" --json tagName,isDraft,isPrerelease,assets
```

The release must contain:

- `manifest.json`
- `main.js`
- `styles.css`

## Marketplace update behavior

The Obsidian Community directory already has this plugin ID:

```text
arxiv-daily
```

For future versions, Obsidian reads the latest committed root
`manifest.json`, then downloads assets from the GitHub release with the same
tag. No `obsidian-releases` PR is needed.

There may be cache delay before the new version appears in Obsidian's in-app
community plugin browser.

## Failure notes

- If CI says a version does not match the tag, fix the relevant file, commit,
  delete the local and remote tag, then create the tag again.
- If CI says `versions.json` is missing the tag, add the new version to both
  root and plugin `versions.json`.
- If the release exists but assets are stale, rerun the workflow or push the
  tag again after fixing the commit; the workflow uploads assets with
  `--clobber`.
- Do not use tags like `v0.2.2`; Obsidian expects versions in `x.y.z` format.
