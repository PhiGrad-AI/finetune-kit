"""Self-contained HTML report: table of runs and an inline-SVG dot plot of the objective."""
from html import escape
from pathlib import Path

from finetune_kit import commands, experiment as ex, store

COLORS = {"keep": "#1a7f37", "keep-simpler": "#0969da", "discard": "#8c959f", "baseline": "#111111"}
CRASH = "#cf222e"
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 720, 280, 56, 16, 16, 40

CSS = """
body{font:14px/1.45 system-ui,sans-serif;margin:2rem auto;max-width:960px;padding:0 1rem;color:#1f2328}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #d0d7de;padding:4px 8px;text-align:left}
td.n{text-align:right;font-variant-numeric:tabular-nums}.v{font-weight:600}
svg{max-width:100%;height:auto;background:#fff;border:1px solid #d0d7de}
"""


def _color(verdict: str) -> str:
    return COLORS.get(verdict, CRASH if verdict in ("crash", "timeout") else "#57606a")


def _fmt(v) -> str:
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _plot(runs: list[dict], best_mean: float | None, bar: float | None, direction: str) -> str:
    pts = [r for r in runs if isinstance(r.get("objective_mean"), (int, float))]
    if not pts:
        return "<p>No finished runs yet.</p>"
    keep_line = None
    if best_mean is not None and bar is not None:
        keep_line = best_mean + bar if direction == "max" else best_mean - bar
    ys = [r["objective_mean"] + s * (r.get("objective_std") or 0) for r in pts for s in (-1, 1)]
    ys += [keep_line] if keep_line is not None else []
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.05 * span, hi + 0.05 * span

    def y(v: float) -> float:
        return PAD_T + (hi - v) / (hi - lo) * (H - PAD_T - PAD_B)

    step = (W - PAD_L - PAD_R) / len(pts)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="objective per run">']
    for v in (lo, (lo + hi) / 2, hi):
        out.append(
            f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#eaeef2"/>'
            f'<text x="{PAD_L - 6}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end">{v:.3f}</text>'
        )
    if keep_line is not None:
        out.append(
            f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(keep_line):.1f}" y2="{y(keep_line):.1f}" '
            f'stroke="#1a7f37" stroke-dasharray="5 4"/><text x="{W - PAD_R}" y="{y(keep_line) - 4:.1f}" '
            f'font-size="11" text-anchor="end" fill="#1a7f37">keep bar {keep_line:.4f}</text>'
        )
    for i, r in enumerate(pts):
        x = PAD_L + step * (i + 0.5)
        m, s = r["objective_mean"], r.get("objective_std") or 0.0
        c = _color(r.get("verdict", ""))
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y(m - s):.1f}" y2="{y(m + s):.1f}" stroke="{c}"/>'
            f'<circle cx="{x:.1f}" cy="{y(m):.1f}" r="5" fill="{c}"><title>{escape(str(r["run_id"]))}: '
            f'{m:.4f} ± {s:.4f}</title></circle>'
            f'<text x="{x:.1f}" y="{H - PAD_B + 16}" font-size="11" text-anchor="middle">{escape(str(r["run_id"]))}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def render(exp_dir: Path) -> str:
    exp_dir = Path(exp_dir)
    cfg, _, _ = ex.load(exp_dir)
    summary = commands.compare(exp_dir)
    runs = summary["runs"]
    direction, _ = store.metric_info(cfg, cfg.objective)
    best_row = next((r for r in runs if r["run_id"] == summary["best"]), None)
    best_mean = best_row["objective_mean"] if best_row else None
    rows = "".join(
        f'<tr><td>{escape(str(r["run_id"]))}</td><td>{escape(str(r["stage"]))}</td>'
        f'<td class="n">{_fmt(r.get("objective_mean"))}</td><td class="n">{_fmt(r.get("objective_std"))}</td>'
        f'<td class="n">{_fmt(r.get("guardrail_mean"))}</td><td class="n">{_fmt(r.get("minutes"))}</td>'
        f'<td class="v" style="color:{_color(r.get("verdict", ""))}">{escape(str(r.get("verdict", "")))}</td>'
        f'<td>{escape(str(r.get("description", "")))}</td></tr>'
        for r in runs
    )
    b = summary["budget_left"]
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{escape(cfg.name)}</title>'
        f"<style>{CSS}</style></head><body><h1>{escape(cfg.name)}</h1>"
        f"<p>{escape(cfg.task)} &middot; {escape(cfg.model)} &middot; objective {escape(cfg.objective.metric)} "
        f"({direction}) &middot; best {escape(str(summary['best']))} &middot; streak {summary['streak']} "
        f"&middot; budget left {b['runs']} runs, {b['hours']} h &middot; stop: {str(summary['stop']).lower()}</p>"
        f"{_plot(runs, best_mean, summary['noise']['keep_bar'], direction)}"
        "<table><thead><tr><th>run</th><th>stage</th><th>objective</th><th>&plusmn;std</th>"
        f"<th>guardrail</th><th>min</th><th>verdict</th><th>description</th></tr></thead><tbody>{rows}</tbody></table>"
        "</body></html>"
    )


def write_report(exp_dir: Path) -> Path:
    path = Path(exp_dir) / "report.html"
    path.write_text(render(exp_dir))
    return path
