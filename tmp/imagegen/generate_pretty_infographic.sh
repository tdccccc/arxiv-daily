#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="output/imagegen"
SVG="$OUT_DIR/arxiv-daily-infographic-v3.svg"
PNG="$OUT_DIR/arxiv-daily-infographic-v3.png"
mkdir -p "$OUT_DIR"

cat > "$SVG" <<'SVG'
<svg xmlns="http://www.w3.org/2000/svg" width="2048" height="1152" viewBox="0 0 2048 1152">
  <defs>
    <style>
      .font { font-family: "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif; }
      .ink { fill: #202432; }
      .muted { fill: #657187; }
      .faint { fill: #8b95a9; }
      .white { fill: #ffffff; }
      .paper { fill: #ffffff; stroke: #dce2ee; stroke-width: 2; }
      .panel { fill: #f8f9fc; stroke: #d7deeb; stroke-width: 2; }
      .soft { fill: #eef1f7; }
      .soft2 { fill: #f7f8fb; }
      .purple { fill: #7864e8; }
      .purple-text { fill: #6d58df; }
      .purple-soft { fill: #f0edff; }
      .green { fill: #2aa876; }
      .green-soft { fill: #e9f8f2; }
      .red { fill: #c7433f; }
      .red-soft { fill: #fff0ee; }
      .gold { fill: #f4b24b; }
      .line { stroke: #c8cfdd; stroke-width: 2.4; fill: none; }
      .title { font-size: 82px; font-weight: 900; letter-spacing: 0; }
      .subtitle { font-size: 29px; font-weight: 520; }
      .h2 { font-size: 31px; font-weight: 830; }
      .h3 { font-size: 23px; font-weight: 780; }
      .body { font-size: 21px; font-weight: 520; }
      .small { font-size: 18px; font-weight: 560; }
      .tiny { font-size: 15px; font-weight: 520; }
      .ui { font-size: 16px; font-weight: 560; }
      .ui-bold { font-size: 17px; font-weight: 760; }
    </style>
    <filter id="posterShadow" x="-15%" y="-15%" width="130%" height="145%">
      <feDropShadow dx="0" dy="22" stdDeviation="24" flood-color="#5a6070" flood-opacity="0.18"/>
    </filter>
    <filter id="cardShadow" x="-20%" y="-20%" width="140%" height="150%">
      <feDropShadow dx="0" dy="10" stdDeviation="13" flood-color="#5a6070" flood-opacity="0.12"/>
    </filter>
    <linearGradient id="leftPanel" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#252637"/>
      <stop offset="1" stop-color="#343052"/>
    </linearGradient>
    <linearGradient id="pageBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#eef1f8"/>
      <stop offset="1" stop-color="#f8f4ef"/>
    </linearGradient>
  </defs>

  <rect width="2048" height="1152" fill="url(#pageBg)"/>
  <path d="M92 126C386 44 666 73 948 142C1256 218 1510 196 1956 86V1098H92Z" fill="#ffffff" opacity="0.72"/>

  <g class="font">
    <rect x="84" y="64" width="1880" height="1024" rx="40" fill="#ffffff" stroke="#dbe1ec" stroke-width="2" filter="url(#posterShadow)"/>

    <g transform="translate(114,94)">
      <rect x="0" y="0" width="560" height="964" rx="32" fill="url(#leftPanel)"/>
      <text x="52" y="86" class="title" fill="#ffffff">arXiv Daily</text>
      <text x="56" y="132" class="subtitle" fill="#cdd3e3">每日论文发现，直接进入 Obsidian</text>

      <g transform="translate(56,184)">
        <rect x="0" y="0" width="448" height="66" rx="22" fill="#ffffff" opacity="0.08"/>
        <text x="26" y="42" class="small" fill="#ffffff">arXiv feed → topic filter → Markdown → review</text>
      </g>

      <g transform="translate(56,300)">
        <text x="0" y="0" class="h2" fill="#ffffff">它解决什么？</text>
        <g class="body" fill="#d8deec">
          <text x="0" y="62">每天论文太多，先由插件初筛</text>
          <text x="0" y="112">按研究主题归类，减少无效浏览</text>
          <text x="0" y="162">自动生成 Markdown 日报和论文笔记</text>
          <text x="0" y="212">Dashboard 里搜索、星标和回看</text>
        </g>
      </g>

      <g transform="translate(56,600)">
        <rect x="0" y="0" width="448" height="250" rx="24" fill="#ffffff"/>
        <text x="28" y="50" class="h2 ink">安装和第一次运行</text>
        <g class="body muted">
          <text x="30" y="106">1. Community plugins 中搜索安装</text>
          <text x="30" y="150">2. Enable arXiv Daily</text>
          <text x="30" y="194">3. 设置 API Key、分类和研究主题</text>
          <text x="30" y="238">4. 打开 Dashboard，点击 Run Today</text>
        </g>
      </g>

      <g transform="translate(56,884)">
        <rect x="0" y="0" width="448" height="48" rx="24" fill="#7864e8"/>
        <text x="224" y="32" text-anchor="middle" class="small" fill="#ffffff">适合每日发现、初筛和重点追踪</text>
      </g>
    </g>

    <g transform="translate(630,128)">
      <g transform="translate(76,0)">
        <rect x="0" y="0" width="1078" height="74" rx="26" fill="#f8f9fc" stroke="#dce2ee" stroke-width="2"/>
        <rect x="36" y="16" width="172" height="42" rx="21" class="red-soft"/>
        <text x="80" y="43" class="small red">arXiv 新论文</text>
        <path d="M226 37H294" class="line"/>
        <rect x="312" y="16" width="158" height="42" rx="21" class="purple-soft"/>
        <text x="352" y="43" class="small purple-text">主题筛选</text>
        <path d="M488 37H556" class="line"/>
        <rect x="574" y="16" width="190" height="42" rx="21" class="green-soft"/>
        <text x="607" y="43" class="small green">Markdown 日报</text>
        <path d="M782 37H850" class="line"/>
        <rect x="868" y="16" width="174" height="42" rx="21" class="soft"/>
        <text x="906" y="43" class="small muted">回看与星标</text>
      </g>

      <g transform="translate(0,130)" filter="url(#cardShadow)">
        <rect x="0" y="0" width="1248" height="744" rx="32" class="panel"/>
        <g transform="translate(32,30)">
          <text x="0" y="34" class="h3 ink">arXiv Daily Dashboard</text>
          <rect x="1070" y="4" width="92" height="32" rx="7" class="soft"/>
          <text x="1116" y="26" text-anchor="middle" class="ui muted">Refresh</text>
        </g>

        <g transform="translate(32,88)">
          <rect x="0" y="0" width="128" height="34" rx="7" fill="#ffffff" stroke="#7864e8" stroke-width="2"/>
          <text x="22" y="24" class="ui ink">Starred</text>
          <rect x="90" y="7" width="24" height="20" rx="10" fill="#dfe3ec"/>
          <text x="98" y="22" class="tiny muted">8</text>
          <rect x="138" y="0" width="104" height="34" rx="7" class="soft" stroke="#dfe3ec"/>
          <text x="168" y="24" class="ui muted">All</text>
          <rect x="204" y="7" width="28" height="20" rx="10" fill="#dfe3ec"/>
          <text x="212" y="22" class="tiny muted">42</text>
          <rect x="252" y="0" width="164" height="34" rx="7" class="soft" stroke="#dfe3ec"/>
          <text x="276" y="24" class="ui muted">Detail summary</text>

          <rect x="690" y="0" width="88" height="34" rx="7" class="soft" stroke="#dfe3ec"/>
          <text x="734" y="23" text-anchor="middle" class="ui muted">Refresh</text>
          <rect x="788" y="0" width="112" height="34" rx="7" class="green"/>
          <text x="844" y="23" text-anchor="middle" class="ui" fill="#ffffff">Run Today</text>
          <rect x="910" y="0" width="126" height="34" rx="7" class="soft" stroke="#dfe3ec"/>
          <text x="973" y="23" text-anchor="middle" class="ui muted">Run Pending</text>
          <rect x="1046" y="0" width="84" height="34" rx="7" class="soft" stroke="#dfe3ec"/>
          <text x="1088" y="23" text-anchor="middle" class="ui muted">More</text>
        </g>

        <g transform="translate(32,152)">
          <rect x="0" y="0" width="646" height="176" rx="12" fill="#ffffff" stroke="#e0e5ef"/>
          <text x="20" y="32" class="ui-bold ink">Filters</text>
          <text x="20" y="72" class="tiny muted">Search</text>
          <rect x="20" y="84" width="590" height="32" rx="6" fill="#f2f4f8" stroke="#dfe3ec"/>
          <text x="36" y="106" class="tiny faint">title, author, arXiv ID, summary...</text>
          <text x="20" y="146" class="tiny muted">Topic</text>
          <rect x="76" y="124" width="188" height="32" rx="6" fill="#f2f4f8" stroke="#dfe3ec"/>
          <text x="96" y="146" class="tiny faint">Any topic</text>
          <text x="294" y="146" class="tiny muted">First seen</text>
          <rect x="376" y="124" width="148" height="32" rx="6" fill="#f2f4f8" stroke="#dfe3ec"/>
          <rect x="536" y="124" width="34" height="32" rx="6" class="soft" stroke="#dfe3ec"/>
        </g>

        <g transform="translate(712,152)">
          <rect x="0" y="0" width="462" height="260" rx="12" fill="#ffffff" stroke="#e0e5ef"/>
          <text x="20" y="32" class="ui-bold ink">Daily reports</text>
          <rect x="230" y="10" width="58" height="28" rx="6" class="soft" stroke="#dfe3ec"/>
          <text x="259" y="29" text-anchor="middle" class="tiny muted">Today</text>
          <text x="322" y="30" class="tiny muted">2026-06</text>
          <g transform="translate(20,58)">
            <g class="tiny muted">
              <text x="14" y="18">Sun</text><text x="72" y="18">Mon</text><text x="132" y="18">Tue</text><text x="190" y="18">Wed</text><text x="252" y="18">Thu</text><text x="312" y="18">Fri</text><text x="374" y="18">Sat</text>
            </g>
            <g transform="translate(0,30)">
              <rect x="0" y="0" width="48" height="34" rx="5" class="soft2" stroke="#e0e5ef"/>
              <rect x="58" y="0" width="48" height="34" rx="5" class="soft2" stroke="#e0e5ef"/>
              <rect x="116" y="0" width="48" height="34" rx="5" class="soft2" stroke="#7864e8" stroke-width="2"/>
              <rect x="174" y="0" width="48" height="34" rx="5" class="soft2" stroke="#e0e5ef"/>
              <rect x="232" y="0" width="48" height="34" rx="5" class="soft2" stroke="#7864e8" stroke-width="2"/>
              <rect x="290" y="0" width="48" height="34" rx="5" class="soft2" stroke="#7864e8" stroke-width="2"/>
              <rect x="348" y="0" width="48" height="34" rx="5" class="soft2" stroke="#e0e5ef"/>
              <rect x="0" y="44" width="48" height="34" rx="5" class="soft2" stroke="#e0e5ef"/>
              <rect x="58" y="44" width="48" height="34" rx="5" class="soft2" stroke="#7864e8" stroke-width="2"/>
              <rect x="116" y="44" width="48" height="34" rx="5" fill="#f0edff" stroke="#7864e8" stroke-width="2"/>
              <rect x="174" y="44" width="48" height="34" rx="5" class="soft2" stroke="#e0e5ef"/>
              <rect x="232" y="44" width="48" height="34" rx="5" class="soft2" stroke="#e0e5ef"/>
              <rect x="290" y="44" width="48" height="34" rx="5" class="soft2" stroke="#e0e5ef"/>
              <rect x="348" y="44" width="48" height="34" rx="5" class="soft2" stroke="#e0e5ef"/>
            </g>
          </g>
        </g>

        <g transform="translate(32,352)">
          <rect x="0" y="0" width="646" height="68" rx="10" fill="#ffffff" stroke="#e0e5ef"/>
          <text x="22" y="42" class="tiny muted">Total</text><text x="80" y="42" class="ui-bold ink">42</text>
          <text x="166" y="42" class="tiny muted">Starred</text><text x="238" y="42" class="ui-bold ink">8</text>
          <text x="336" y="42" class="tiny muted">Details</text><text x="404" y="42" class="ui-bold ink">5</text>
          <text x="494" y="42" class="tiny muted">Daily reports</text><text x="600" y="42" class="ui-bold ink">12</text>
        </g>

        <g transform="translate(32,444)">
          <rect x="0" y="0" width="1142" height="44" rx="8" fill="#ffffff"/>
          <text x="18" y="29" class="tiny muted">0 selected</text>
          <rect x="116" y="8" width="84" height="28" rx="6" class="soft" stroke="#dfe3ec"/>
          <text x="158" y="28" text-anchor="middle" class="tiny muted">Save</text>
          <rect x="208" y="8" width="98" height="28" rx="6" class="soft" stroke="#dfe3ec"/>
          <text x="257" y="28" text-anchor="middle" class="tiny muted">Ignore</text>
          <text x="748" y="29" class="tiny muted">Showing 20</text>
          <rect x="836" y="8" width="102" height="28" rx="6" class="soft" stroke="#dfe3ec"/>
          <text x="887" y="28" text-anchor="middle" class="tiny muted">20 rows</text>
          <text x="960" y="29" class="tiny muted">Sort</text>
          <rect x="996" y="8" width="124" height="28" rx="6" class="soft" stroke="#dfe3ec"/>
          <text x="1058" y="28" text-anchor="middle" class="tiny muted">Starred first</text>
        </g>

        <g transform="translate(32,504)">
          <rect x="0" y="0" width="1142" height="164" rx="8" fill="#ffffff" stroke="#d9deea"/>
          <rect x="0" y="0" width="1142" height="38" rx="8" fill="#eef1f7"/>
          <text x="26" y="25" class="tiny muted">☐</text>
          <text x="72" y="25" class="tiny muted">Star</text>
          <text x="150" y="25" class="tiny muted">Title</text>
          <text x="752" y="25" class="tiny muted">Topic</text>
          <text x="860" y="25" class="tiny muted">Published</text>
          <text x="1006" y="25" class="tiny muted">Actions</text>
          <g transform="translate(0,38)">
            <text x="26" y="31" class="tiny muted">☐</text>
            <text x="78" y="31" class="ui" fill="#7864e8">★</text>
            <text x="150" y="22" class="ui-bold ink">A relevant paper title</text>
            <text x="150" y="42" class="tiny muted">YYMM.NNNNN · Authors</text>
            <text x="752" y="31" class="tiny muted">主题</text>
            <text x="860" y="31" class="tiny muted">YYYY-MM-DD</text>
            <text x="1008" y="31" class="tiny muted">□ □ □ □</text>
          </g>
          <line x1="0" y1="90" x2="1142" y2="90" stroke="#e1e5ee"/>
          <g transform="translate(0,90)">
            <text x="26" y="31" class="tiny muted">☐</text>
            <text x="78" y="31" class="ui faint">☆</text>
            <text x="150" y="22" class="ui-bold ink">Paper with detail summary available</text>
            <text x="150" y="42" class="tiny muted">YYMM.NNNNN · Authors</text>
            <text x="752" y="31" class="tiny muted">主题</text>
            <text x="860" y="31" class="tiny muted">YYYY-MM-DD</text>
            <text x="1008" y="31" class="tiny muted">□ □ □ □</text>
          </g>
        </g>
      </g>

      <g transform="translate(46,904)">
        <rect x="0" y="0" width="1090" height="48" rx="24" fill="#ffffff" stroke="#dce2ee"/>
        <text x="30" y="31" class="small muted">输出保持 Markdown 原生：日报、论文笔记和 PDF 都可在 Obsidian 中长期保存、链接和回看。</text>
      </g>
    </g>
  </g>
</svg>
SVG

google-chrome --headless --disable-gpu --no-sandbox \
  --screenshot="$PNG" \
  --window-size=2048,1152 \
  "file://$ROOT_DIR/$SVG"

echo "Wrote $PNG"
echo "Wrote $SVG"
