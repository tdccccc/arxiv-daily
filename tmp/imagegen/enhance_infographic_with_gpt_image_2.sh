#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IMAGE_GEN="${CODEX_HOME:-$HOME/.codex}/skills/.system/imagegen/scripts/image_gen_codex_config.py"
INPUT_IMAGE="${INPUT_IMAGE:-output/imagegen/arxiv-daily-infographic.png}"
PROMPT_FILE="tmp/imagegen/arxiv-daily-infographic-enhance.prompt.txt"
OUT_FILE="${OUT_FILE:-output/imagegen/arxiv-daily-infographic-gpt-image-2.png}"
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
Asset type: polished product infographic for an Obsidian plugin

Input image role:
The input image is the current infographic layout and content reference. Preserve its core information, Chinese wording, and Dashboard structure, but redesign it to look more visually attractive, refined, and memorable.

Primary request:
Create a more beautiful, polished, modern flat product infographic for "arXiv Daily", an Obsidian plugin for daily arXiv paper discovery and review.

Keep the same main content:
- Title: "arXiv Daily"
- Chinese subtitle about turning daily arXiv papers into an Obsidian filtering, summarizing, and review workflow
- Left-side sections explaining what the plugin does and how to install/use it
- Right-side Dashboard mockup with tabs, filters, stats, daily reports calendar, Run Today / Run Pending / More buttons, and a paper table

Design direction:
Make the result less stiff than the reference. Use a high-end flat SaaS/product illustration style with better visual hierarchy, more elegant spacing, subtle depth, and a stronger hero composition.
Keep it clean, academic, and professional. It should feel like a polished plugin showcase graphic for a README, release page, or Community Plugins listing.

Composition:
16:9 landscape.
Keep the title and explanatory sections on the left.
Keep the Dashboard mockup as the main visual on the right.
Use tasteful decorative paper cards, thin workflow lines, subtle background shapes, and soft shadows if helpful, but do not clutter the image.

Dashboard fidelity:
The Dashboard should remain recognizably similar to the input:
tabs, toolbar buttons, filters, stats, daily reports calendar, batch controls, and table columns.
Do not turn it into a generic app mockup.

Text constraints:
Preserve all Chinese text from the input as accurately as possible.
Do not invent extra paragraphs.
Do not add specific user config, personal paths, or concrete topic names.
Avoid fake Chinese characters, garbled text, misspellings, or random labels.
If any small UI text is difficult, simplify it rather than inventing nonsense.

Visual constraints:
No official Obsidian logo.
No official arXiv logo.
No watermarks.
No mascots.
No excessive gradients, blob decorations, or busy marketing clutter.
Keep the image flat, crisp, and readable.
PROMPT

ARGS=(
  edit
  --model gpt-image-2
  --quality high
  --size 2048x1152
  --image "$INPUT_IMAGE"
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
