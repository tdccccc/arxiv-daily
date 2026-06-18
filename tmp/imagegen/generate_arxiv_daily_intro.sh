#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IMAGE_GEN="${CODEX_HOME:-$HOME/.codex}/skills/.system/imagegen/scripts/image_gen_codex_config.py"
PROMPT_FILE="tmp/imagegen/arxiv-daily-intro.prompt.txt"
OUT_FILE="output/imagegen/arxiv-daily-intro.png"
AGENT_CONFIG_ROOT="${AGENT_CONFIG_ROOT:-/home/tiandc/Documents/agent-configs/codex}"

if [[ -n "${CODEX_IMAGE_PROFILE:-}" ]]; then
  export CODEX_CONFIG="$AGENT_CONFIG_ROOT/$CODEX_IMAGE_PROFILE/config.toml"
fi

if [[ -n "${CODEX_IMAGE_CONFIG:-}" ]]; then
  export CODEX_CONFIG="$CODEX_IMAGE_CONFIG"
fi

if [[ -z "${CODEX_IMAGE_PROVIDER:-}" && -n "${CODEX_CONFIG:-}" ]]; then
  export CODEX_IMAGE_PROVIDER="custom"
fi

mkdir -p tmp/imagegen output/imagegen

cat > "$PROMPT_FILE" <<'PROMPT'
Use case: productivity-visual
Asset type: widescreen product introduction poster for an Obsidian plugin

Create a polished 16:9 hero image for an Obsidian plugin called "arXiv Daily".

The image should visually explain a research workflow:
daily arXiv papers flow into Obsidian, the plugin filters papers by research topics, generates Markdown daily reports, shows a dashboard with search/filter/calendar/starred papers, and lets users open paper notes, arXiv pages, PDFs, and detailed summaries.

Scene/backdrop:
A modern dark-mode Obsidian-like knowledge workspace on a desktop screen. On the right side, show a clean dashboard interface with paper rows, a small calendar, topic filter chips, star markers, and progress/status elements. Around it, show floating paper cards, markdown note pages, arXiv-style paper icons, PDF icons, and subtle workflow arrows from "arXiv feed" to "filter" to "daily report" to "dashboard review".

Composition:
16:9 landscape, premium product poster layout.
Leave generous clean negative space on the left and lower-left area for overlaid Chinese text.
Main dashboard visual should occupy the right half.
Keep the visual hierarchy clear and not cluttered.
Use realistic spacing and readable UI proportions, but do not render actual readable text.

Style:
Professional SaaS/product illustration, crisp semi-realistic UI mockup, subtle 3D depth, clean research productivity aesthetic.
Dark graphite Obsidian-inspired background, restrained purple accent, arXiv red accent, white paper surfaces, small green success/progress accents.
Calm, focused, technical, trustworthy.

Important constraints:
No readable text.
No Chinese characters.
No fake words.
No letters or numbers.
No logos or trademarks.
No watermark.
Do not use the official Obsidian logo or arXiv logo; only use abstract visual references.
Avoid busy clutter.
Avoid cartoon mascots.
Avoid decorative blobs or generic gradient orbs.
Make it look like a serious academic research workflow tool, not a marketing landing-page cliche.
PROMPT

ARGS=(
  generate
  --model gpt-image-2
  --quality high
  --size 2048x1152
  --prompt-file "$PROMPT_FILE"
  --out "$OUT_FILE"
  --force
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

python3 "$IMAGE_GEN" "${ARGS[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run only; no image was generated."
else
  echo "Wrote $OUT_FILE"
fi
