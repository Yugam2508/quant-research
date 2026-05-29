#!/usr/bin/env python3
"""
dashboard.py — local web dashboard for quant-research reports

Serves a clean UI showing your daily pulse reports.
Run with: python3 dashboard.py
Then open: http://localhost:5000
"""

import sys
import json
import subprocess
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

REPORTS_DIR = Path(__file__).parent / "reports" / "daily"
PORT = 5000

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quant Research Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0a0b;
    --bg2: #111113;
    --bg3: #1a1a1f;
    --border: rgba(255,255,255,0.07);
    --border2: rgba(255,255,255,0.12);
    --text: #e8e8ea;
    --muted: #6b6b75;
    --accent: #7fff7f;
    --accent2: #4db8ff;
    --red: #ff6b6b;
    --amber: #ffc04d;
    --mono: 'DM Mono', monospace;
    --sans: 'DM Sans', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
    display: grid;
    grid-template-columns: 240px 1fr;
    grid-template-rows: 48px 1fr;
  }

  /* topbar */
  header {
    grid-column: 1 / -1;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 24px;
    gap: 12px;
    background: var(--bg);
  }
  .logo {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.05em;
  }
  .logo span { color: var(--muted); }
  .run-btn {
    margin-left: auto;
    background: transparent;
    border: 1px solid var(--border2);
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .run-btn:hover { border-color: var(--accent); color: var(--accent); }
  .run-btn.running { color: var(--amber); border-color: var(--amber); }
  .pulse-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

  /* sidebar */
  nav {
    border-right: 1px solid var(--border);
    padding: 16px 0;
    overflow-y: auto;
    background: var(--bg);
  }
  .nav-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    color: var(--muted);
    padding: 0 16px 8px;
    text-transform: uppercase;
  }
  .report-item {
    display: flex;
    flex-direction: column;
    padding: 10px 16px;
    cursor: pointer;
    border-left: 2px solid transparent;
    transition: all 0.1s;
    gap: 2px;
  }
  .report-item:hover { background: var(--bg3); }
  .report-item.active {
    border-left-color: var(--accent);
    background: var(--bg2);
  }
  .report-date {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
  }
  .report-tag {
    font-size: 11px;
    color: var(--muted);
  }
  .report-tag.sample { color: var(--amber); }
  .no-reports {
    padding: 16px;
    font-size: 12px;
    color: var(--muted);
    font-family: var(--mono);
  }

  /* main content */
  main {
    padding: 32px 40px;
    overflow-y: auto;
    max-width: 860px;
  }

  /* empty state */
  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 60vh;
    gap: 12px;
    color: var(--muted);
    text-align: center;
  }
  .empty-title {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text);
  }
  .empty-sub { font-size: 12px; max-width: 280px; line-height: 1.7; }
  .empty-cmd {
    font-family: var(--mono);
    font-size: 12px;
    background: var(--bg3);
    border: 1px solid var(--border2);
    padding: 8px 16px;
    border-radius: 4px;
    color: var(--accent);
    margin-top: 8px;
  }

  /* toast */
  .toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: var(--bg3);
    border: 1px solid var(--border2);
    padding: 12px 18px;
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 12px;
    opacity: 0;
    transform: translateY(8px);
    transition: all 0.2s;
    pointer-events: none;
    z-index: 100;
  }
  .toast.show { opacity: 1; transform: translateY(0); }
  .toast.success { border-color: var(--accent); color: var(--accent); }
  .toast.error { border-color: var(--red); color: var(--red); }
  .toast.info { border-color: var(--accent2); color: var(--accent2); }

  /* markdown report styles */
  .report-content h1 {
    font-family: var(--mono);
    font-size: 18px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }
  .report-content h2 {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 32px 0 12px;
  }
  .report-content p {
    color: var(--text);
    line-height: 1.75;
    margin-bottom: 12px;
    font-size: 14px;
  }
  .report-content ul {
    list-style: none;
    margin-bottom: 12px;
  }
  .report-content ul li {
    padding: 6px 0 6px 16px;
    border-left: 2px solid var(--border2);
    margin-bottom: 6px;
    color: var(--text);
    font-size: 14px;
    line-height: 1.65;
  }
  .report-content ul li strong {
    color: var(--accent2);
    font-weight: 500;
  }
  .report-content table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 12px;
    margin: 12px 0 24px;
  }
  .report-content th {
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border2);
    color: var(--muted);
    font-weight: 400;
    letter-spacing: 0.05em;
  }
  .report-content td {
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }
  .report-content tr:hover td { background: var(--bg3); }
  .report-content blockquote {
    border-left: 2px solid var(--border2);
    padding: 4px 16px;
    color: var(--muted);
    font-style: italic;
    margin: 12px 0;
    font-size: 13px;
  }
  .report-content code {
    font-family: var(--mono);
    font-size: 12px;
    background: var(--bg3);
    padding: 2px 6px;
    border-radius: 3px;
    color: var(--accent2);
  }
  .report-content hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
  }

  /* loading spinner */
  .spinner {
    width: 16px; height: 16px;
    border: 2px solid var(--border2);
    border-top-color: var(--amber);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: inline-block;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<header>
  <div class="pulse-dot"></div>
  <div class="logo">quant<span>/</span>research</div>
  <button class="run-btn" id="runBtn" onclick="triggerRun()">
    ▶ run daily pulse
  </button>
</header>

<nav id="sidebar">
  <div class="nav-label">reports</div>
  <div id="reportList"></div>
</nav>

<main id="mainContent">
  <div class="empty" id="emptyState">
    <div class="empty-title">no report loaded</div>
    <div class="empty-sub">select a report from the sidebar, or run the daily pulse to generate one</div>
    <div class="empty-cmd">python3 runs/daily_pulse.py</div>
  </div>
  <div class="report-content" id="reportContent" style="display:none"></div>
</main>

<div class="toast" id="toast"></div>

<script>
let reports = [];

function showToast(msg, type='info') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show ' + type;
  setTimeout(() => t.className = 'toast', 3000);
}

async function loadReports() {
  try {
    const r = await fetch('/api/reports');
    reports = await r.json();
    const list = document.getElementById('reportList');
    if (reports.length === 0) {
      list.innerHTML = '<div class="no-reports">no reports yet</div>';
      return;
    }
    list.innerHTML = reports.map((r, i) =>
      `<div class="report-item ${i===0?'active':''}" onclick="loadReport('${r.filename}', this)">
        <div class="report-date">${r.date}</div>
        <div class="report-tag ${r.sample?'sample':''}">${r.sample ? '⚠ sample' : '● live'}</div>
      </div>`
    ).join('');
    if (reports.length > 0) loadReport(reports[0].filename, list.querySelector('.report-item'));
  } catch(e) {
    showToast('failed to load reports', 'error');
  }
}

async function loadReport(filename, el) {
  document.querySelectorAll('.report-item').forEach(x => x.classList.remove('active'));
  if (el) el.classList.add('active');
  try {
    const r = await fetch('/api/report/' + filename);
    const data = await r.json();
    document.getElementById('emptyState').style.display = 'none';
    const content = document.getElementById('reportContent');
    content.style.display = 'block';
    content.innerHTML = marked.parse(data.content);
  } catch(e) {
    showToast('failed to load report', 'error');
  }
}

async function triggerRun() {
  const btn = document.getElementById('runBtn');
  btn.innerHTML = '<span class="spinner"></span> running...';
  btn.classList.add('running');
  btn.disabled = true;
  showToast('starting daily pulse run...', 'info');
  try {
    const r = await fetch('/api/run', { method: 'POST' });
    const data = await r.json();
    if (data.success) {
      showToast('report generated!', 'success');
      await loadReports();
    } else {
      showToast('run failed: ' + (data.error || 'unknown error'), 'error');
    }
  } catch(e) {
    showToast('run failed', 'error');
  } finally {
    btn.innerHTML = '▶ run daily pulse';
    btn.classList.remove('running');
    btn.disabled = false;
  }
}

loadReports();
</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence default logging

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif parsed.path == '/api/reports':
            reports = []
            if REPORTS_DIR.exists():
                for f in sorted(REPORTS_DIR.glob('*.md'), reverse=True):
                    reports.append({
                        'filename': f.name,
                        'date': f.stem.replace('-sample', ''),
                        'sample': 'sample' in f.stem,
                    })
            self.send_json(reports)

        elif parsed.path.startswith('/api/report/'):
            filename = parsed.path.split('/')[-1]
            filepath = REPORTS_DIR / filename
            if filepath.exists() and filepath.suffix == '.md':
                self.send_json({'content': filepath.read_text()})
            else:
                self.send_json({'error': 'not found'}, 404)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/run':
            try:
                result = subprocess.run(
                    [sys.executable, 'runs/daily_pulse.py'],
                    capture_output=True, text=True, timeout=120,
                    cwd=Path(__file__).parent
                )
                if result.returncode == 0:
                    self.send_json({'success': True})
                else:
                    self.send_json({'success': False, 'error': result.stderr[-300:]})
            except subprocess.TimeoutExpired:
                self.send_json({'success': False, 'error': 'timed out after 120s'})
            except Exception as e:
                self.send_json({'success': False, 'error': str(e)})
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    server = HTTPServer(('localhost', PORT), Handler)
    print(f"dashboard running at http://localhost:{PORT}")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
