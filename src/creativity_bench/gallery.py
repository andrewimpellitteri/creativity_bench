"""Standalone, escaped HTML review gallery for saved narrative diversity runs."""

from __future__ import annotations

import html
import json
from pathlib import Path


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _curve(values: list) -> str:
    # Numeric-only coordinates: saved run content never becomes SVG markup.
    values = [int(value) for value in values]
    if not values:
        return "<p>No acceptance curve recorded.</p>"
    count = len(values)
    ceiling = max(count, max(values), 1)
    points = " ".join(
        f"{40 + i * 520 / count:.1f},{170 - v * 140 / ceiling:.1f}"
        for i, v in enumerate([0, *values])
    )
    return (
        f'<svg viewBox="0 0 600 210" role="img" '
        f'aria-label="Cumulative accepted stories over {count} attempts">'
        '<path d="M40 20 V170 H565" fill="none" stroke="currentColor"/>'
        f'<polyline points="{points}" fill="none" stroke="#236c51" stroke-width="3"/>'
        '<text x="40" y="195">0</text>'
        f'<text x="540" y="195">{count}</text>'
        '<text x="250" y="205">Attempts</text>'
        f'<text x="10" y="35">{ceiling}</text>'
        '<text x="10" y="170">0</text></svg>'
        f"<p>Cumulative accepted stories: {_escape(', '.join(map(str, values)))}</p>"
    )


def write_gallery(run_path: str | Path, out_path: str | Path) -> Path:
    """Render a saved run as HTML and return the output path."""
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    task = (run.get("tasks") or {}).get("same_but_different")
    if not isinstance(task, dict):
        raise ValueError(
            "Run has no same_but_different task; run that task before creating a gallery"
        )
    details = task.get("details") or {}
    metadata = run.get("metadata") or {}
    sections = []
    for number, premise in enumerate(details.get("premises", []), 1):
        attempts = []
        for attempt in premise.get("transcript", []):
            verdict = attempt.get("verdict") or {}
            accepted = attempt.get("accepted") is True
            status = "Accepted" if accepted else "Rejected"
            reasons = ", ".join(attempt.get("rejection_reasons") or []) or "None"
            judgment_status = attempt.get("status") or (
                "ok" if verdict else "Not judged (see rejection reason)"
            )
            fields = [
                ("Rejection reasons", reasons),
                ("Judgment status", judgment_status),
                ("Evidence", verdict.get("evidence", "Not recorded")),
                ("Plot summary", verdict.get("summary", "Not recorded")),
            ]
            field_html = "".join(
                f"<dt>{label}</dt><dd>{_escape(value)}</dd>" for label, value in fields
            )
            verdict_html = _escape(json.dumps(verdict, ensure_ascii=False, indent=2))
            raw_html = _escape(
                json.dumps(attempt.get("judge_responses", []), ensure_ascii=False, indent=2)
            )
            attempts.append(
                f'<details class="attempt"><summary>Attempt {_escape(attempt.get("attempt", "?"))}'
                f" · {status}</summary><h3>Story</h3>"
                f"<pre>{_escape(attempt.get('story') or 'No story generated')}</pre>"
                f"<dl>{field_html}</dl><details><summary>Verdict and raw judge responses</summary>"
                f"<pre>{verdict_html}</pre><pre>{raw_html}</pre></details></details>"
            )
        sections.append(
            f"<section><h2>Premise {number}</h2><p>{_escape(premise.get('premise', ''))}</p>"
            "<h3>Cumulative accepted stories</h3>"
            + _curve(premise.get("acceptance_curve", []))
            + "".join(attempts)
            + "</section>"
        )
    model = _escape(run.get("model", "Unknown"))
    judge = _escape(details.get("judge_model") or metadata.get("judge_model", "Unknown"))
    protocol = _escape(details.get("protocol") or metadata.get("protocol_version", "Unknown"))
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Same But Different — story gallery</title><style>
body{font:17px/1.55 system-ui,sans-serif;max-width:900px;margin:40px auto;
padding:0 20px;color:#172d26;background:#fafbf9}
h1,h2,h3{line-height:1.2}section{border-top:2px solid #c7d4ca;margin-top:36px;padding-top:20px}
.attempt{background:white;border:1px solid #c7d4ca;border-radius:8px;margin:12px 0;padding:16px}
summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}
dt{font-weight:650}dd{margin:0 0 12px}svg{width:100%;max-width:600px}svg text{font-size:12px}
.notice{padding:16px;background:#eaf1ec;border-radius:8px}
</style></head><body><h1>Same But Different</h1>"""
    document += (
        f"<p><strong>Model:</strong> {model}<br><strong>Judge:</strong> {judge}"
        f"<br><strong>Protocol:</strong> {protocol}</p>"
        '<p class="notice"><strong>Exploratory judgments.</strong> Acceptance reflects '
        "an automated assessment of premise adherence, comprehensibility, and plot "
        "distinctness. Review the stories and evidence; this is not a validated "
        "measure of creativity.</p>" + "".join(sections) + "</body></html>"
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
