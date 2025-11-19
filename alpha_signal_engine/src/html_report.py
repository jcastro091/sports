import os, yaml
from jinja2 import Template
from datetime import datetime

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Alpha Signal Backtest Report</title>
  <style>
    body { font-family: system-ui, Arial, sans-serif; padding: 24px; }
    .card { border: 1px solid #eee; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    h1 { margin-top: 0; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #eee; padding: 8px; text-align: left; }
  </style>
</head>
<body>
  <h1>Alpha Signal Backtest Report</h1>
  <div class="card">
    <strong>Generated:</strong> {{ ts }}
  </div>
  <div class="card">
    <h2>Summary</h2>
    <table>
      {% for k, v in summary.items() %}
      <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
      {% endfor %}
    </table>
  </div>
</body>
</html>
"""

def write_report(summary: dict, out_path: str):
    html = Template(TEMPLATE).render(ts=datetime.now().isoformat(timespec="seconds"), summary=summary)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[html_report] wrote {out_path}")
