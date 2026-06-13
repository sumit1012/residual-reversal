"""Self-contained HTML visual report generator."""

from __future__ import annotations

import json
import logging
import os
from math import comb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from residrev.backtest import BacktestResult
    from residrev.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _rolling_mean(values: list, window: int = 21) -> list:
    out = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        chunk = [x for x in values[start : i + 1] if x is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def _fmt(val, fmt=".2f", suffix=""):
    if val is None:
        return "—"
    try:
        return f"{val:{fmt}}{suffix}"
    except (ValueError, TypeError):
        return "—"


def _color(val, good_positive: bool = True) -> str:
    """CSS colour class: pos-val / neg-val / neutral."""
    if val is None:
        return "neutral"
    try:
        v = float(val)
    except (ValueError, TypeError):
        return "neutral"
    if v > 0:
        return "pos-val" if good_positive else "neg-val"
    if v < 0:
        return "neg-val" if good_positive else "pos-val"
    return "neutral"


# ---------------------------------------------------------------------------
# CSS + JS template
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #f5f6fa; color: #1a1d2e; font-size: 14px; }
.page { max-width: 1200px; margin: 0 auto; padding: 24px 20px 48px; }

/* ── header ── */
.header { background: #1a1d2e; color: #fff; border-radius: 12px;
          padding: 28px 32px; margin-bottom: 24px; }
.header h1 { font-size: 22px; font-weight: 700; letter-spacing: -.3px; }
.header .sub { color: #8892a4; font-size: 13px; margin-top: 6px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
         font-size: 11px; font-weight: 600; margin-left: 10px;
         vertical-align: middle; }
.badge-pass { background: #16a34a22; color: #16a34a; border:1px solid #16a34a55; }
.badge-warn { background: #ca8a0422; color: #ca8a04; border:1px solid #ca8a0455; }
.badge-fail { background: #dc262622; color: #dc2626; border:1px solid #dc262655; }
.badge-skip { background: #64748b22; color: #64748b; border:1px solid #64748b55; }

/* ── scorecard row ── */
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
         margin-bottom: 24px; }
.card { background: #fff; border-radius: 10px; padding: 18px 20px;
        border: 1px solid #e2e8f0; box-shadow: 0 1px 3px #0001; }
.card .label { font-size: 11px; text-transform: uppercase; letter-spacing: .6px;
               color: #64748b; margin-bottom: 8px; }
.card .value { font-size: 26px; font-weight: 700; line-height: 1; }
.card .sub    { font-size: 12px; color: #94a3b8; margin-top: 4px; }
.pos-val { color: #16a34a; }
.neg-val { color: #dc2626; }
.neutral { color: #334155; }

/* ── section ── */
.section { background: #fff; border-radius: 10px; border: 1px solid #e2e8f0;
           box-shadow: 0 1px 3px #0001; padding: 24px; margin-bottom: 20px; }
.section h2 { font-size: 15px; font-weight: 600; margin-bottom: 18px;
              padding-bottom: 10px; border-bottom: 1px solid #f1f5f9;
              color: #0f172a; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.chart-wrap { position: relative; height: 220px; }
.chart-wrap-tall { position: relative; height: 280px; }

/* ── table ── */
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: #f8fafc; text-align: left; padding: 8px 12px;
     font-weight: 600; color: #475569; font-size: 11px;
     text-transform: uppercase; letter-spacing: .5px;
     border-bottom: 1px solid #e2e8f0; }
td { padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8fafc; }

/* ── checklist ── */
.check-row { display: flex; align-items: center; padding: 10px 0;
             border-bottom: 1px solid #f1f5f9; }
.check-row:last-child { border-bottom: none; }
.check-name { width: 220px; font-weight: 600; font-size: 13px; }
.check-detail { flex: 1; font-size: 12px; color: #64748b; }

/* ── footer ── */
.footer { text-align: center; color: #94a3b8; font-size: 12px;
          margin-top: 32px; }
"""

_CHART_DEFAULTS = """
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.color = "#64748b";
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding = 16;
"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _scorecards(summary: dict, config) -> str:
    ns = summary.get("net_sharpe")
    gs = summary.get("gross_sharpe")
    ic = summary.get("mean_daily_ic")
    tstat = summary.get("ic_tstat")
    dd = summary.get("max_drawdown", 0) or 0
    ann_ret = (summary.get("net_annual_return") or 0) * 100
    to = (summary.get("annual_turnover") or 0) * 100
    cost = summary.get("mean_cost_bps") or 0

    def card(label, value_html, sub=""):
        return (
            f'<div class="card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value_html}</div>'
            f'<div class="sub">{sub}</div>'
            f"</div>"
        )

    def sv(val, fmt=".2f", good_positive=True, suffix=""):
        c = _color(val, good_positive)
        return f'<span class="{c}">{_fmt(val, fmt, suffix)}</span>'

    html = '<div class="cards">'
    html += card("Net Sharpe", sv(ns), "annualized")
    html += card("Gross Sharpe", sv(gs), "before costs")
    html += card("IC t-stat", sv(tstat), f"mean IC {_fmt(ic, '.4f')}")
    html += card("Net Ann. Return", sv(ann_ret, ".1f", suffix="%"), "daily PnL × 252")
    html += card("Max Drawdown", sv(dd * 100, ".1f", False, "%"), "cumulative PnL trough")
    html += card("Ann. Turnover", f'<span class="neutral">{_fmt(to, ".0f", suffix="%")}</span>', "one-way")
    html += card("Mean Cost", f'<span class="neutral">{_fmt(cost, ".2f", suffix=" bps")}</span>', "per day")
    html += card("Universe", f'<span class="neutral">{config.universe_size}</span>', f"{summary.get('n_trading_days',0)} trading days")
    html += "</div>"
    return html


def _pnl_chart(pnl_dates: list, cum_net: list, cum_gross: list) -> str:
    data = {
        "dates": pnl_dates,
        "net": [round(v, 8) for v in cum_net],
        "gross": [round(v, 8) for v in cum_gross],
    }
    return f"""
<div class="section">
  <h2>Cumulative PnL — Net vs Gross</h2>
  <div class="chart-wrap-tall">
    <canvas id="pnlChart"></canvas>
  </div>
</div>
<script>
(function(){{
  var d = {json.dumps(data)};
  new Chart(document.getElementById('pnlChart'), {{
    type: 'line',
    data: {{
      labels: d.dates,
      datasets: [
        {{ label: 'Net PnL', data: d.net, borderColor: '#2563eb',
           backgroundColor: '#2563eb18', fill: true,
           borderWidth: 2, pointRadius: 0, tension: 0.2 }},
        {{ label: 'Gross PnL', data: d.gross, borderColor: '#94a3b8',
           borderDash: [4,3], fill: false,
           borderWidth: 1.5, pointRadius: 0, tension: 0.2 }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{ tooltip: {{ callbacks: {{
        label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toExponential(3)
      }} }} }},
      scales: {{
        x: {{ ticks: {{ maxTicksLimit: 8, maxRotation: 0 }},
               grid: {{ display: false }} }},
        y: {{ grid: {{ color: '#f1f5f9' }} }}
      }}
    }}
  }});
}})();
</script>"""


def _year_sharpe_chart(years: list, year_sharpes: list) -> str:
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in year_sharpes]
    data = {"years": [str(y) for y in years], "sharpes": [round(v, 4) for v in year_sharpes], "colors": colors}
    return f"""
<div>
  <h2 style="font-size:14px;font-weight:600;margin-bottom:14px;color:#0f172a;">Per-Year Net Sharpe</h2>
  <div class="chart-wrap">
    <canvas id="yearChart"></canvas>
  </div>
</div>
<script>
(function(){{
  var d = {json.dumps(data)};
  new Chart(document.getElementById('yearChart'), {{
    type: 'bar',
    data: {{
      labels: d.years,
      datasets: [{{
        label: 'Annual Sharpe',
        data: d.sharpes,
        backgroundColor: d.colors,
        borderRadius: 5
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ display: false }} }},
        y: {{ grid: {{ color: '#f1f5f9' }},
               ticks: {{ callback: v => v.toFixed(1) }} }}
      }}
    }}
  }});
}})();
</script>"""


def _ic_chart(ic_dates: list, ic_rolling: list) -> str:
    data = {"dates": ic_dates, "ic": [round(v, 6) if v is not None else None for v in ic_rolling]}
    return f"""
<div>
  <h2 style="font-size:14px;font-weight:600;margin-bottom:14px;color:#0f172a;">IC — 21-Day Rolling Mean</h2>
  <div class="chart-wrap">
    <canvas id="icChart"></canvas>
  </div>
</div>
<script>
(function(){{
  var d = {json.dumps(data)};
  new Chart(document.getElementById('icChart'), {{
    type: 'line',
    data: {{
      labels: d.dates,
      datasets: [{{
        label: 'Rolling IC',
        data: d.ic,
        borderColor: '#7c3aed',
        backgroundColor: '#7c3aed12',
        fill: true,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        spanGaps: true
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ maxTicksLimit: 6, maxRotation: 0 }},
               grid: {{ display: false }} }},
        y: {{ grid: {{ color: '#f1f5f9' }},
               ticks: {{ callback: v => v.toFixed(3) }} }}
      }}
    }}
  }});
}})();
</script>"""


def _factor_exposure_table(fe_means: dict) -> str:
    rows = ""
    for factor, val in fe_means.items():
        c = _color(val)
        rows += (
            f"<tr><td>{factor}</td>"
            f'<td class="{c}">{val:.2e}</td>'
            f"<td>{'✓ neutral' if abs(val) < 1e-6 else '⚠ check'}</td></tr>"
        )
    return f"""
<div class="section">
  <h2>Factor Exposures (time-averaged)</h2>
  <table>
    <tr><th>Factor</th><th>Mean Exposure</th><th>Status</th></tr>
    {rows}
  </table>
</div>"""


def _factor_bar_chart(fe_means: dict) -> str:
    labels = list(fe_means.keys())
    values = [round(v, 4) for v in fe_means.values()]
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in values]
    data = {"labels": labels, "values": values, "colors": colors}
    return f"""
<div>
  <h2 style="font-size:14px;font-weight:600;margin-bottom:14px;color:#0f172a;">Factor Exposures</h2>
  <div class="chart-wrap">
    <canvas id="feChart"></canvas>
  </div>
</div>
<script>
(function(){{
  var d = {json.dumps(data)};
  new Chart(document.getElementById('feChart'), {{
    type: 'bar',
    data: {{
      labels: d.labels,
      datasets: [{{
        label: 'Mean Exposure',
        data: d.values,
        backgroundColor: d.colors,
        borderRadius: 4
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ color: '#f1f5f9' }},
               ticks: {{ callback: v => v.toExponential(1) }} }},
        y: {{ grid: {{ display: false }} }}
      }}
    }}
  }});
}})();
</script>"""


def _cost_sensitivity_section(cs_data: list[dict]) -> str:
    if not cs_data:
        return ""
    rows = ""
    for row in cs_data:
        mult = row["mult"]
        ns = row["net_sharpe"]
        ret = row["ann_ret"]
        dd = row["max_dd"]
        c = _color(ns)
        if mult == "breakeven":
            rows += f"<tr><td><strong>Breakeven</strong></td><td class='{c}'>{_fmt(ns, '.2f')}×</td><td>—</td><td>—</td></tr>"
        else:
            rows += (
                f"<tr><td>{mult}×</td>"
                f"<td class='{c}'>{_fmt(ns, '.2f')}</td>"
                f"<td>{_fmt(ret * 100, '.1f', suffix='%')}</td>"
                f"<td class='neg-val'>{_fmt(dd * 100, '.1f', suffix='%')}</td></tr>"
            )
    return f"""
<div>
  <h2 style="font-size:14px;font-weight:600;margin-bottom:14px;color:#0f172a;">Cost Sensitivity</h2>
  <table>
    <tr><th>Cost mult.</th><th>Net Sharpe</th><th>Ann. Return</th><th>Max DD</th></tr>
    {rows}
  </table>
</div>"""


def _cpcv_section(checklist: dict | None, config) -> str:
    if not checklist or "checks" not in checklist:
        return ""
    cpcv = checklist["checks"].get("cpcv_oos_sharpe", {})
    if not cpcv or "oos_sharpes" not in cpcv:
        return ""

    n_paths = comb(config.cpcv_n_groups, config.cpcv_k_test)
    sharpes = [round(v, 4) for v in cpcv["oos_sharpes"]]
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in sharpes]
    labels = [f"P{i+1}" for i in range(len(sharpes))]

    pct_pos = cpcv.get("pct_positive", 0) * 100
    data = {"labels": labels, "sharpes": sharpes, "colors": colors}

    stats_rows = (
        f"<tr><td>Paths</td><td>{n_paths}</td></tr>"
        f"<tr><td>Mean OOS Sharpe</td><td class='{_color(cpcv.get('mean'))}'>{_fmt(cpcv.get('mean'), '.2f')}</td></tr>"
        f"<tr><td>Median</td><td class='{_color(cpcv.get('median'))}'>{_fmt(cpcv.get('median'), '.2f')}</td></tr>"
        f"<tr><td>Std</td><td>{_fmt(cpcv.get('std'), '.2f')}</td></tr>"
        f"<tr><td>Range</td><td>[{_fmt(cpcv.get('min'), '.2f')}, {_fmt(cpcv.get('max'), '.2f')}]</td></tr>"
        f"<tr><td>% Positive</td><td class='{_color(pct_pos - 50)}'>{pct_pos:.0f}%</td></tr>"
    )

    return f"""
<div class="section">
  <h2>CPCV — Out-of-Sample Paths</h2>
  <div class="two-col">
    <div>
      <div class="chart-wrap">
        <canvas id="cpcvChart"></canvas>
      </div>
    </div>
    <div>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        {stats_rows}
      </table>
    </div>
  </div>
</div>
<script>
(function(){{
  var d = {json.dumps(data)};
  new Chart(document.getElementById('cpcvChart'), {{
    type: 'bar',
    data: {{
      labels: d.labels,
      datasets: [{{
        label: 'OOS Sharpe',
        data: d.sharpes,
        backgroundColor: d.colors,
        borderRadius: 4
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ display: false }} }},
        y: {{ grid: {{ color: '#f1f5f9' }},
               ticks: {{ callback: v => v.toFixed(1) }} }}
      }}
    }}
  }});
}})();
</script>"""


def _checklist_section(checklist: dict | None) -> str:
    if not checklist or "checks" not in checklist:
        return ""

    checks = checklist["checks"]
    overall = checklist.get("overall", "")
    n_pass = checklist.get("n_pass", 0)
    n_warn = checklist.get("n_warn", 0)
    n_fail = checklist.get("n_fail", 0)

    def badge(status):
        s = (status or "skip").lower()
        cls = {"pass": "badge-pass", "warn": "badge-warn", "fail": "badge-fail"}.get(s, "badge-skip")
        return f'<span class="badge {cls}">{s.upper()}</span>'

    def note_for(name, chk):
        s = chk.get("status", "")
        if name == "skip_day_test":
            return f"Overall Sharpe {_fmt(chk.get('overall_sharpe'), '.2f')}; max weekday deviation {_fmt(chk.get('note', '').replace('Max weekday Sharpe deviation from overall: ', ''), '.2f')}"
        if name == "cost_stress_test":
            return f"Sharpe 1×: {_fmt(chk.get('sharpe_1x'), '.2f')} | 2×: {_fmt(chk.get('sharpe_2x'), '.2f')}"
        if name == "factor_crash_stress":
            parts = []
            for p, label in [("covid", "COVID"), ("rate_hike", "Rate hike")]:
                info = chk.get(p, {})
                if "sharpe" in info:
                    parts.append(f"{label}: Sharpe {_fmt(info['sharpe'], '.2f')}")
            return " | ".join(parts) if parts else chk.get("note", "")
        if name == "deflated_sharpe":
            if "dsr" in chk:
                return f"DSR: {chk['dsr']:.3f}  ({chk.get('n_trials', '?')} trials)"
            return chk.get("note", "")
        if name == "cpcv_oos_sharpe":
            return f"Mean OOS Sharpe {_fmt(chk.get('mean'), '.2f')} over {chk.get('n_paths', '?')} paths"
        return chk.get("note", "")

    labels = {
        "skip_day_test": "Skip-Day Bias",
        "cost_stress_test": "Cost Stress (2×)",
        "factor_crash_stress": "Factor Crash Stress",
        "deflated_sharpe": "Deflated Sharpe (DSR)",
        "cpcv_oos_sharpe": "CPCV OOS Sharpe",
    }

    rows = ""
    for name, chk in checks.items():
        label = labels.get(name, name.replace("_", " ").title())
        rows += (
            f'<div class="check-row">'
            f'<div class="check-name">{label}{badge(chk.get("status"))}</div>'
            f'<div class="check-detail">{note_for(name, chk)}</div>'
            f"</div>"
        )

    overall_badge = badge(overall)
    return f"""
<div class="section">
  <h2>Pre-Trust Checklist {overall_badge}
    <span style="font-size:12px;font-weight:400;color:#64748b;margin-left:12px;">
      {n_pass} pass · {n_warn} warn · {n_fail} fail
    </span>
  </h2>
  {rows}
</div>"""


# ---------------------------------------------------------------------------
# Top-level assembler
# ---------------------------------------------------------------------------

def generate_html_report(
    result: "BacktestResult",
    summary: dict,
    checklist: dict | None,
    config: "Config",
    output_path: str,
    run_id: str = "",
) -> None:
    """Write a self-contained HTML visual report to output_path."""
    from residrev.analysis import cost_sensitivity

    # PnL series
    pnl_dates = [str(d.date()) for d in result.pnl.index]
    cum_net = result.pnl.cumsum().tolist()
    cum_gross = result.gross_pnl.cumsum().tolist()

    # IC series + rolling mean
    ic_dates = [str(d.date()) for d in result.ic_series.index]
    ic_raw = [v if v == v else None for v in result.ic_series.tolist()]  # NaN → None
    ic_rolling = _rolling_mean(ic_raw, 21)

    # Per-year Sharpe
    per_year = summary.get("per_year_sharpe", {})
    years = sorted(per_year.keys())
    year_sharpes = [per_year[y] for y in years]

    # Factor exposures
    fe_means = summary.get("factor_exposures", {})

    # Cost sensitivity
    cs_data = []
    try:
        cs_df = cost_sensitivity(result)
        for mult, row in cs_df.iterrows():
            if mult == "breakeven_multiplier":
                cs_data.append({"mult": "breakeven", "net_sharpe": row["net_sharpe"],
                                 "ann_ret": float("nan"), "max_dd": float("nan")})
            else:
                cs_data.append({"mult": mult, "net_sharpe": row["net_sharpe"],
                                 "ann_ret": row["annualized_return"], "max_dd": row["max_drawdown"]})
    except Exception:
        pass

    start = summary.get("start_date", config.start_date)
    end = summary.get("end_date", config.end_date)
    run_label = f"Run {run_id}" if run_id else "Run"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Residual Reversal — {run_label}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>{_CSS}</style>
</head>
<body>
<div class="page">

  <!-- header -->
  <div class="header">
    <h1>Residual Short-Horizon Reversal
      <span class="badge badge-{'pass' if (checklist or {}).get('overall') == 'pass' else 'fail' if (checklist or {}).get('overall') == 'fail' else 'warn'}">
        {((checklist or {}).get('overall') or 'no checklist').upper()}
      </span>
    </h1>
    <div class="sub">{run_label} &nbsp;·&nbsp; {start} → {end}
      &nbsp;·&nbsp; Liquid-{config.universe_size} universe
      &nbsp;·&nbsp; FF5 + UMD factor model</div>
  </div>

  <!-- scorecards -->
  {_scorecards(summary, config)}

  <!-- cumulative PnL chart -->
  {_pnl_chart(pnl_dates, cum_net, cum_gross)}

  <!-- per-year + IC side by side -->
  <div class="section">
    <h2>Annual Sharpe &amp; Information Coefficient</h2>
    <div class="two-col">
      {_year_sharpe_chart(years, year_sharpes)}
      {_ic_chart(ic_dates, ic_rolling)}
    </div>
  </div>

  <!-- factor exposures + cost sensitivity -->
  <div class="section">
    <h2>Factor Exposures &amp; Cost Sensitivity</h2>
    <div class="two-col">
      {_factor_bar_chart(fe_means)}
      {_cost_sensitivity_section(cs_data)}
    </div>
  </div>

  <!-- CPCV -->
  {_cpcv_section(checklist, config)}

  <!-- checklist -->
  {_checklist_section(checklist)}

  <div class="footer">
    Generated by <strong>residrev</strong> · {run_label} · {end}
  </div>
</div>
<script>{_CHART_DEFAULTS}</script>
</body>
</html>"""

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML report written to %s", output_path)
