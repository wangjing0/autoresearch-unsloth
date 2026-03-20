#!/usr/bin/env python3
"""
Autoresearch Dashboard -- Live visualization of Pareto frontier prompt optimization.

Reads results.jsonl and frontier.jsonl, serves a live-updating dashboard.

Usage:
    uv run python autoresearch_skills/dashboard.py
    uv run python autoresearch_skills/dashboard.py --port 8501
"""

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent / "data"
RESULTS_FILE = BASE_DIR / "results.jsonl"
STATE_FILE = BASE_DIR / "state.json"
PROMPT_FILE = BASE_DIR / "prompt.txt"
BEST_PROMPT_FILE = BASE_DIR / "best_prompt.txt"
INITIAL_PROMPT_FILE = BASE_DIR / "initial_prompt.txt"
FRONTIER_FILE = BASE_DIR / "frontier.jsonl"
DIAGRAMS_DIR = BASE_DIR / "diagrams"

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Skills Autoresearch -- Pareto Frontier</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #faf9f7; color: #2d2a26; padding: 32px; max-width: 1400px; margin: 0 auto; }

  .header { display: flex; align-items: center; gap: 16px; margin-bottom: 32px; }
  .header h1 { font-size: 28px; font-weight: 700; color: #2d2a26; }
  .badge { font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 4px; letter-spacing: 1px; color: white; }
  .badge-live { background: #c0392b; }
  .badge-mode { background: #8e44ad; margin-left: 4px; }
  .subtitle { color: #8a8580; font-size: 14px; margin-top: 4px; }

  .stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 32px; }
  .stat-card { background: white; border-radius: 12px; padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .stat-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #8a8580; margin-bottom: 6px; }
  .stat-value { font-size: 28px; font-weight: 700; }
  .stat-value.green { color: #27ae60; }
  .stat-value.orange { color: #c0784a; }
  .stat-value.purple { color: #8e44ad; }
  .stat-value.blue { color: #2980b9; }
  .stat-value.neutral { color: #2d2a26; }
  .stat-sub { font-size: 11px; color: #8a8580; margin-top: 2px; }

  .chart-container { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 32px; }
  .chart-container h3 { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #8a8580; margin-bottom: 12px; }
  .chart-container canvas { width: 100% !important; height: 300px !important; }
  .chart-legend { display: flex; gap: 16px; margin-top: 8px; font-size: 12px; color: #8a8580; }
  .chart-legend span { display: inline-flex; align-items: center; gap: 4px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

  .criteria-bar { background: white; border-radius: 12px; padding: 20px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 32px; }
  .criteria-bar h3 { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #8a8580; margin-bottom: 14px; }
  .criteria-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }
  .crit-item { text-align: center; }
  .crit-item .crit-avg { font-size: 28px; font-weight: 700; }
  .crit-item .crit-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: #8a8580; margin-top: 2px; }
  .crit-item .crit-bar { height: 6px; border-radius: 3px; background: #eee; margin-top: 6px; overflow: hidden; }
  .crit-item .crit-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }

  .section { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 32px; }
  .section h3 { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #8a8580; margin-bottom: 16px; }

  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #8a8580; padding: 8px 6px; border-bottom: 1px solid #eee; }
  td { padding: 7px 6px; border-bottom: 1px solid #f5f4f2; font-size: 12px; }
  .status-keep { color: #27ae60; font-weight: 600; }
  .status-discard { color: #8a8580; }
  .mode-refine { color: #2980b9; font-weight: 600; }
  .mode-explore { color: #d35400; font-weight: 600; }
  .weakest-tag { background: #fef3e7; color: #c0784a; font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: 600; }

  .frontier-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
  .frontier-card { background: #faf9f7; border-radius: 8px; padding: 14px; border: 1px solid #eee; }
  .frontier-card .fc-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .frontier-card .fc-score { font-size: 20px; font-weight: 700; color: #2d2a26; }
  .frontier-card .fc-pct { font-size: 13px; color: #8a8580; font-weight: 600; }
  .frontier-card .fc-run { font-size: 11px; color: #8a8580; }
  .frontier-card .fc-criteria { display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; margin-bottom: 8px; }
  .frontier-card .fc-crit { text-align: center; font-size: 11px; }
  .frontier-card .fc-crit-val { font-size: 14px; font-weight: 700; }
  .frontier-card .fc-crit-label { color: #8a8580; font-size: 8px; text-transform: uppercase; }
  .frontier-card .fc-prompt { font-family: 'SF Mono', monospace; font-size: 11px; color: #6a6560; line-height: 1.4; max-height: 60px; overflow: hidden; text-overflow: ellipsis; }

  .prompts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 32px; }
  .prompt-container { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .prompt-container h3 { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #8a8580; margin-bottom: 12px; }
  .prompt-text { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; line-height: 1.6; color: #4a4540; white-space: pre-wrap; word-break: break-word; background: #faf9f7; padding: 16px; border-radius: 8px; max-height: 300px; overflow-y: auto; }

  @media (max-width: 768px) {
    .stats { grid-template-columns: repeat(3, 1fr); }
    .criteria-grid { grid-template-columns: repeat(3, 1fr); }
    .prompts-grid { grid-template-columns: 1fr; }
    body { padding: 16px; }
  }
</style>
</head>
<body>

<div class="header">
  <div>
    <div style="display:flex;align-items:center;gap:12px;">
      <h1>Skills Autoresearch</h1>
      <span class="badge badge-live" id="live-badge">LIVE</span>
      <span class="badge badge-mode" id="mode-badge">--</span>
    </div>
    <div class="subtitle" id="subtitle">Pareto frontier prompt optimization -- refreshes every 15s</div>
  </div>
</div>

<div class="stats">
  <div class="stat-card">
    <div class="stat-label">Best Score</div>
    <div class="stat-value orange" id="stat-best">--</div>
    <div class="stat-sub" id="stat-best-sub"></div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Baseline</div>
    <div class="stat-value neutral" id="stat-baseline">--</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Improvement</div>
    <div class="stat-value green" id="stat-improvement">--</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Frontier</div>
    <div class="stat-value purple" id="stat-frontier">--</div>
    <div class="stat-sub" id="stat-frontier-sub"></div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Weakest</div>
    <div class="stat-value blue" id="stat-weakest">--</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Runs</div>
    <div class="stat-value neutral" id="stat-runs">--</div>
    <div class="stat-sub" id="stat-runs-sub"></div>
  </div>
</div>

<div class="criteria-bar" id="criteria-bar" style="display:none;">
  <h3>Latest Criteria Scores (0-10 scale)</h3>
  <div class="criteria-grid" id="criteria-grid"></div>
</div>

<div class="chart-container">
  <h3>Score Over Time (max 10.00)</h3>
  <canvas id="mainChart"></canvas>
  <div class="chart-legend">
    <span><span class="legend-dot" style="background:#2980b9"></span> REFINE</span>
    <span><span class="legend-dot" style="background:#d35400"></span> EXPLORE</span>
    <span><span class="legend-dot" style="background:#c0784a"></span> New best</span>
  </div>
</div>

<div class="section" id="frontier-section" style="display:none;">
  <h3>Pareto Frontier</h3>
  <div class="frontier-grid" id="frontier-grid"></div>
</div>

<div class="section">
  <h3>Run History</h3>
  <div style="overflow-x:auto;">
  <table>
    <thead>
      <tr><th>#</th><th>Score</th><th>Text</th><th>Color</th><th>Layout</th><th>Labels</th><th>Clarity</th><th>Icons</th><th>Mode</th><th>Weakest</th><th>Time</th></tr>
    </thead>
    <tbody id="run-table"></tbody>
  </table>
  </div>
</div>

<div class="prompts-grid">
  <div class="prompt-container">
    <h3>Initial Prompt</h3>
    <div class="prompt-text" id="initial-prompt">Loading...</div>
  </div>
  <div class="prompt-container">
    <h3>Current Best Prompt</h3>
    <div class="prompt-text" id="best-prompt">Loading...</div>
  </div>
</div>

<script>
const ORANGE = '#c0784a';
const BLUE = '#2980b9';
const RED_ORANGE = '#d35400';
const GREEN = '#27ae60';
const PURPLE = '#8e44ad';
const MAX_SCORE = 10.0;

const CRIT_KEYS = ['text_quality', 'color_palette', 'layout', 'label_discipline', 'visual_clarity', 'icon_quality'];
const CRIT_LABELS = { text_quality: 'Text', color_palette: 'Color', layout: 'Layout', label_discipline: 'Labels', visual_clarity: 'Clarity', icon_quality: 'Icons' };
const CRIT_COLORS = { text_quality: '#8e44ad', color_palette: '#2980b9', layout: '#27ae60', label_discipline: '#c0784a', visual_clarity: '#2c3e50', icon_quality: '#d35400' };

const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { grid: { display: false }, ticks: { font: { size: 11 }, color: '#8a8580' } },
    y: { grid: { color: '#f0efed' }, ticks: { font: { size: 11 }, color: '#8a8580' } }
  }
};

let mainChart;

function initCharts() {
  const ctx = document.getElementById('mainChart').getContext('2d');
  mainChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Score',
        data: [],
        borderColor: ORANGE,
        backgroundColor: 'rgba(192,120,74,0.08)',
        fill: true,
        tension: 0.3,
        pointRadius: 5,
        pointBackgroundColor: [],
        pointBorderColor: [],
        pointBorderWidth: 2,
      }]
    },
    options: {
      ...chartDefaults,
      scales: {
        ...chartDefaults.scales,
        y: { ...chartDefaults.scales.y, min: 0, ticks: { ...chartDefaults.scales.y.ticks } }
      }
    }
  });
}

function updateMainChart(runs) {
  const labels = runs.map(r => r.run);
  const data = runs.map(r => r.score);
  mainChart.data.labels = labels;
  mainChart.data.datasets[0].data = data;

  let runningBest = -1;
  const bgColors = [];
  const borderColors = [];
  runs.forEach((r, idx) => {
    const isExplore = (r.mode || '').toUpperCase() === 'EXPLORE';
    const isNewBest = data[idx] > runningBest;
    if (isNewBest) runningBest = data[idx];
    if (isNewBest) { bgColors.push(ORANGE); borderColors.push(ORANGE); }
    else if (isExplore) { bgColors.push(RED_ORANGE); borderColors.push(RED_ORANGE); }
    else { bgColors.push(BLUE); borderColors.push(BLUE); }
  });
  mainChart.data.datasets[0].pointBackgroundColor = bgColors;
  mainChart.data.datasets[0].pointBorderColor = borderColors;
  mainChart.update('none');
}

function renderCriteriaBar(lastRun) {
  const bar = document.getElementById('criteria-bar');
  const grid = document.getElementById('criteria-grid');
  const s10 = lastRun.scores;
  if (!s10) { bar.style.display = 'none'; return; }
  bar.style.display = '';

  const scoreColor = v => v >= 8 ? GREEN : v >= 6 ? BLUE : v >= 4 ? ORANGE : '#c0392b';
  grid.innerHTML = CRIT_KEYS.map(c => {
    const val = s10[c] ?? 0;
    const pct = (val / 10 * 100).toFixed(0);
    const color = CRIT_COLORS[c] || '#8a8580';
    return `<div class="crit-item">
      <div class="crit-avg" style="color:${scoreColor(val)}">${val.toFixed(1)}</div>
      <div class="crit-label">${CRIT_LABELS[c] || c}</div>
      <div class="crit-bar"><div class="crit-bar-fill" style="width:${pct}%;background:${color}"></div></div>
    </div>`;
  }).join('');
}

async function fetchData() {
  try {
    const res = await fetch('/api/data');
    return await res.json();
  } catch (e) {
    console.error('Fetch error:', e);
    return null;
  }
}

function formatTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function renderFrontier(frontier) {
  const section = document.getElementById('frontier-section');
  const grid = document.getElementById('frontier-grid');
  if (!frontier || frontier.length === 0) { section.style.display = 'none'; return; }
  section.style.display = '';

  const critColor = (val) => val >= 8 ? GREEN : val >= 6 ? BLUE : val >= 4 ? ORANGE : '#c0392b';
  grid.innerHTML = frontier.map(m => {
    const overall = m.overall ?? 0;
    const s10 = m.scores || m;
    return `<div class="frontier-card">
      <div class="fc-header">
        <div><span class="fc-score">${typeof overall === 'number' ? overall.toFixed(2) : overall}</span><span style="color:#8a8580;font-size:13px">/10</span></div>
        <div class="fc-run">Run ${m.run ?? '?'}</div>
      </div>
      <div class="fc-criteria">
        ${CRIT_KEYS.map(c => { const v = s10[c] ?? 0; return `<div class="fc-crit"><div class="fc-crit-val" style="color:${critColor(v)}">${typeof v === 'number' ? v.toFixed(1) : v}</div><div class="fc-crit-label">${CRIT_LABELS[c] || c}</div></div>`; }).join('')}
      </div>
      <div class="fc-prompt">${(m.prompt || '').substring(0, 150).replace(/</g, '&lt;')}...</div>
    </div>`;
  }).join('');
}

async function refresh() {
  const data = await fetchData();
  if (!data || !data.runs || data.runs.length === 0) return;

  const runs = data.runs;
  const scores = runs.map(r => r.score);
  const baseline = scores[0];
  const best = Math.max(...scores);
  const lastRun = runs[runs.length - 1];
  const frontier = data.frontier || [];
  // Stats
  document.getElementById('stat-best').textContent = (typeof best === 'number' ? best.toFixed(2) : best) + '/10';
  document.getElementById('stat-best-sub').textContent = '';
  document.getElementById('stat-baseline').textContent = (typeof baseline === 'number' ? baseline.toFixed(2) : baseline) + '/10';
  const improvement = baseline > 0 ? ((best - baseline) / baseline * 100).toFixed(1) : '--';
  const improvEl = document.getElementById('stat-improvement');
  improvEl.textContent = improvement === '--' ? '--' : (improvement > 0 ? '+' : '') + improvement + '%';
  improvEl.className = 'stat-value ' + (improvement > 0 ? 'green' : improvement < 0 ? 'orange' : 'neutral');

  document.getElementById('stat-frontier').textContent = frontier.length;
  document.getElementById('stat-frontier-sub').textContent = frontier.length === 1 ? 'member' : 'members';

  document.getElementById('stat-weakest').textContent = lastRun.weakest || '--';

  let kept = 0, runningBest = -1;
  scores.forEach(s => { if (s > runningBest) { kept++; runningBest = s; } });
  document.getElementById('stat-runs').textContent = runs.length;
  document.getElementById('stat-runs-sub').textContent = kept + ' kept';

  const lastMode = (lastRun.mode || 'REFINE').toUpperCase();
  const modeBadge = document.getElementById('mode-badge');
  modeBadge.textContent = lastMode;
  modeBadge.style.background = lastMode === 'EXPLORE' ? '#d35400' : '#2980b9';

  // Criteria averages bar
  renderCriteriaBar(lastRun);

  // Main chart
  updateMainChart(runs);

  // Frontier
  renderFrontier(frontier);

  // Table
  const tbody = document.getElementById('run-table');
  const rows = runs.map((r, idx) => {
    const mode = (r.mode || '').toUpperCase();
    const modeClass = mode === 'EXPLORE' ? 'mode-explore' : 'mode-refine';
    const s10 = r.scores || {};
    const fmt = v => typeof v === 'number' ? v.toFixed(1) : '?';
    return `<tr>
      <td>${r.run}</td>
      <td><strong>${typeof r.score === 'number' ? r.score.toFixed(2) : r.score}</strong></td>
      <td>${fmt(s10.text_quality)}</td>
      <td>${fmt(s10.color_palette)}</td>
      <td>${fmt(s10.layout)}</td>
      <td>${fmt(s10.label_discipline)}</td>
      <td>${fmt(s10.visual_clarity)}</td>
      <td>${fmt(s10.icon_quality)}</td>
      <td class="${modeClass}">${mode || '--'}</td>
      <td>${r.weakest ? '<span class="weakest-tag">' + r.weakest + '</span>' : '--'}</td>
      <td>${formatTime(r.timestamp)}</td>
    </tr>`;
  }).reverse();
  tbody.innerHTML = rows.join('');

  // Prompts
  document.getElementById('initial-prompt').textContent = data.initial_prompt || '(not saved yet)';
  document.getElementById('best-prompt').textContent = data.best_prompt || 'Loading...';

  // Subtitle
  document.getElementById('subtitle').textContent =
    `Pareto frontier -- ${runs.length} runs -- frontier: ${frontier.length} -- best: ${typeof best === 'number' ? best.toFixed(2) : best}/10 -- last: ${formatTime(lastRun?.timestamp)}`;
}

initCharts();
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif parsed.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            runs = []
            if RESULTS_FILE.exists():
                for line in RESULTS_FILE.read_text().strip().split("\n"):
                    if line.strip():
                        try:
                            runs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

            frontier = []
            if FRONTIER_FILE.exists():
                for line in FRONTIER_FILE.read_text().strip().split("\n"):
                    if line.strip():
                        try:
                            frontier.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

            initial_prompt = ""
            if INITIAL_PROMPT_FILE.exists():
                initial_prompt = INITIAL_PROMPT_FILE.read_text().strip()

            best_prompt = ""
            if BEST_PROMPT_FILE.exists():
                best_prompt = BEST_PROMPT_FILE.read_text().strip()

            data = {
                "runs": runs,
                "frontier": frontier,
                "initial_prompt": initial_prompt,
                "best_prompt": best_prompt,
            }
            self.wfile.write(json.dumps(data).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Skills Autoresearch Dashboard")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{args.port}")
    print(f"Reading from: {RESULTS_FILE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")


if __name__ == "__main__":
    main()
