"""
Visual building blocks for the Streamlit app (CSS + small HTML snippets).

Kept separate from `streamlit_app.py` so the app file reads like a story:
inputs -> prediction -> display.
"""

from __future__ import annotations

import html
from typing import List, Optional, Tuple

import plotly.graph_objects as go

# Palette ------------------------------------------------------------------
INK = "#1E2B4A"        # deep navy: primary text, top prediction
SLATE = "#5B6B86"      # secondary text, other predictions
LINE = "#DCE1EA"       # hairlines and tracks
PAPER = "#F6F7F9"      # page background
TAG = "#F2B630"        # merchandise-tag yellow: the one accent
WHITE = "#FFFFFF"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Public+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, label, li {{
    font-family: 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
h1, h2, h3, .display {{
    font-family: 'Archivo', 'Public Sans', system-ui, sans-serif !important;
    color: {INK};
    letter-spacing: -0.01em;
}}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}

/* Hero ------------------------------------------------------------------ */
.hero h1 {{ font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin: 0 0 .35rem 0; }}
.hero p  {{ color: {SLATE}; font-size: 1.05rem; max-width: 62ch; margin: 0; }}

/* The merchandise tag: the one bold element of the page ------------------ */
.tag-wrap {{ padding: 18px 6px 6px 22px; }}
.tag {{
    position: relative;
    background: {INK};
    color: {WHITE};
    padding: 26px 30px 22px 64px;
    clip-path: polygon(34px 0, 100% 0, 100% 100%, 34px 100%, 0 50%);
    transform: rotate(-1.5deg);
    transform-origin: left center;
    animation: swing 0.9s ease-out 1;
}}
.tag::before {{               /* punched hole */
    content: ""; position: absolute; left: 22px; top: 50%;
    width: 16px; height: 16px; margin-top: -8px; border-radius: 50%;
    background: {PAPER}; box-shadow: inset 0 0 0 3px {TAG};
}}
.tag-label {{ color: #B9C3D6; font-size: .9rem; }}
.tag-category {{
    font-family: 'Archivo', sans-serif; font-weight: 800;
    font-size: clamp(2rem, 4.2vw, 3.1rem); line-height: 1.05; margin: 2px 0 12px 0;
    overflow-wrap: anywhere;
}}
.tag-conf {{ font-size: 1rem; color: #DCE3F0; }}
.tag-pct  {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.6rem; color: {TAG}; }}
.tag-meter {{ height: 6px; background: rgba(255,255,255,.18); margin: 10px 0 12px 0; }}
.tag-meter > div {{ height: 100%; background: {TAG}; }}
.tag-foot {{ font-size: .85rem; color: #B9C3D6; }}
@keyframes swing {{ 0% {{ transform: rotate(-6deg); }} 60% {{ transform: rotate(0.5deg); }} 100% {{ transform: rotate(-1.5deg); }} }}
@media (prefers-reduced-motion: reduce) {{ .tag {{ animation: none; }} }}

/* Text-vs-photo reliance bar ------------------------------------------- */
.split {{ display: flex; height: 30px; margin: 6px 0 4px 0; border: 1px solid {LINE}; }}
.split > div {{ display: flex; align-items: center; padding: 0 10px; font-size: .85rem; white-space: nowrap; }}
.split .t {{ background: {INK}; color: {WHITE}; }}
.split .i {{ background: {TAG}; color: {INK}; justify-content: flex-end; }}
.muted {{ color: {SLATE}; font-size: .9rem; }}

/* Empty state ------------------------------------------------------------ */
.empty {{ border: 1.5px dashed {LINE}; padding: 34px 28px; color: {SLATE}; background: {WHITE}; }}
.empty strong {{ color: {INK}; }}

/* Metric strip on the performance page -------------------------------- */
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0;
        border-top: 2px solid {INK}; border-bottom: 1px solid {LINE}; margin: 6px 0 18px 0; }}
.kpi {{ padding: 14px 16px 12px 0; }}
.kpi .v {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.9rem; color: {INK}; }}
.kpi .k {{ color: {SLATE}; font-size: .88rem; }}

button:focus-visible, [role="tab"]:focus-visible {{ outline: 3px solid {TAG} !important; outline-offset: 2px; }}
</style>
"""


def hero(title: str, subtitle: str) -> str:
    return f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>'


def prediction_tag(label: str, confidence: float, inputs_used: List[str]) -> str:
    source = {("text",): "product name", ("image",): "photo",
              ("text", "image"): "product name and photo"}[tuple(inputs_used)]
    pct = 100 * confidence
    return f"""
    <div class="tag-wrap"><div class="tag" role="status" aria-live="polite">
        <div class="tag-label">Predicted category</div>
        <div class="tag-category">{html.escape(label)}</div>
        <div class="tag-conf"><span class="tag-pct">{pct:.1f}%</span>&nbsp; confidence</div>
        <div class="tag-meter"><div style="width:{pct:.1f}%"></div></div>
        <div class="tag-foot">Read from the {source}</div>
    </div></div>"""


def reliance_bar(text_weight: Optional[float]) -> str:
    if text_weight is None:
        return ""
    t = max(0.0, min(1.0, text_weight))
    return f"""
    <div class="muted">How much this decision leaned on each input</div>
    <div class="split" aria-label="Text {t:.0%}, photo {1 - t:.0%}">
        <div class="t" style="width:{100 * t:.1f}%">Name {t:.0%}</div>
        <div class="i" style="width:{100 * (1 - t):.1f}%">Photo {1 - t:.0%}</div>
    </div>"""


def empty_state(title: str, body_html: str) -> str:
    return f'<div class="empty"><strong>{html.escape(title)}</strong><br>{body_html}</div>'


def kpi_strip(items: List[Tuple[str, str]]) -> str:
    cells = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="k">{html.escape(k)}</div></div>'
                    for k, v in items)
    return f'<div class="kpis">{cells}</div>'


def topk_chart(top_k: List[Tuple[str, float]]) -> go.Figure:
    labels = [l for l, _ in top_k][::-1]
    probs = [p for _, p in top_k][::-1]
    colors = [SLATE] * (len(labels) - 1) + [INK]
    fig = go.Figure(go.Bar(
        x=probs, y=labels, orientation="h", marker_color=colors,
        text=[f"{p:.1%}" for p in probs], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
    fig.update_layout(
        height=56 + 38 * len(labels), margin=dict(l=8, r=48, t=8, b=8),
        xaxis=dict(range=[0, 1.08], visible=False), yaxis=dict(tickfont=dict(size=14, color=INK)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.35,
        font=dict(family="Public Sans, sans-serif"))
    return fig


def per_class_f1_chart(per_class: dict, class_names: List[str]) -> go.Figure:
    rows = sorted(((c, per_class[c]["f1-score"], per_class[c]["support"]) for c in class_names),
                  key=lambda r: r[1])
    fig = go.Figure(go.Bar(
        x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h",
        marker_color=[TAG if r[1] < 0.8 else INK for r in rows],
        customdata=[r[2] for r in rows],
        hovertemplate="%{y}: F1 %{x:.3f} (%{customdata} test items)<extra></extra>"))
    fig.update_layout(height=40 + 24 * len(rows), margin=dict(l=8, r=20, t=8, b=30),
                      xaxis=dict(range=[0, 1], title="F1-score (yellow = below 0.80)", gridcolor=LINE),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Public Sans, sans-serif", color=INK))
    return fig
