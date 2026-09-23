"""Resumen ejecutivo — ¿cuánta cartera está en riesgo, dónde se concentra y qué hacer primero?

Vista para Dirección Financiera y Gestión de Cartera (DB-03: 100 % de los KPI exigidos
presentes): volumen, riesgo Alto predicho, mora real según Datacrédito, monto financiado
total y por categoría de riesgo, tendencias contra un periodo comparable, concentración por
segmentos, hallazgos automáticos, concordancia con el riesgo observado y un reporte
ejecutivo descargable (ME-03).
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.components import (badge, download_bar, empty_state, esc, filter_chips, footer, insight, kpi_card,
                             page_header, section)
from core.config import (ACTION_ROUTES, APP_NAME, PROGRAMA_ACADEMICO, RECALL_ALTO_TARGET, RISK_COLORS, RISK_ORDER,
                         UNIVERSIDAD)
from core.data import get_active_df, get_active_meta, segment_summary
from core.filters import active_chips, apply_filters
from core.metrics import confusion, load_artifacts
from core.nav import PAGES, can_access
from core.theme import YELLOW, fmt_cop, fmt_delta_pp, fmt_int, fmt_num, fmt_pct, fmt_period, pal, show_fig, theme_mode

P = "resumen"
NAN = float("nan")

_WINDOWS = {"3 m": 3, "6 m": 6, "12 m": 12}
_GRAN = {"Mes": "periodo", "Trimestre": "trimestre", "Semestre": "semestre"}
_DIMS = {
    "Segmento de programa": "programa_cluster",
    "Facultad": "facultad",
    "Programa": "programa",
    "Tipo de interés": "tipo_interes",
    "Scoring externo": "score_corto",
    "Cohorte": "cohorte",
    "Nivel": "nivel",
}
_DIM_SINGULAR = {v: k.lower() for k, v in _DIMS.items()}
_QUICK_LINKS = [
    ("cola", "Priorizar casos", "Lista ordenada por índice de prioridad, con rutas R1–R4 y exportación."),
    ("geografia", "Ubicar el riesgo", "Dónde se concentran los créditos en alerta por departamento y ciudad."),
    ("modelo", "Auditar el modelo", "Métricas por clase, curvas ROC/PR y simulador de umbral para Alto."),
    ("segmentos", "Perfilar segmentos", "Perfiles de riesgo por programa, cohorte, scoring y tipo de interés."),
]

# ======================================================================================
# Estilos propios de la página (usan los tokens --sat-* definidos por core.theme)
# ======================================================================================
_CSS = """
<style>
.rs-spark{width:100%;height:30px;display:block;margin-top:10px;overflow:visible}
.rs-spark-cap{font-size:10.5px;color:var(--sat-subtle);margin-top:3px;letter-spacing:.02em}
.rs-window{font-size:12.5px;color:var(--sat-muted);line-height:1.45;padding-top:2px}
.rs-window b{color:var(--sat-text);font-weight:700}
.rs-risk{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:var(--sat-radius);
  padding:16px 18px 14px 20px;box-shadow:var(--sat-shadow);position:relative;overflow:hidden;height:100%}
.rs-risk:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--rc)}
.rs-risk-top{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--sat-muted);font-weight:600}
.rs-dot{width:10px;height:10px;border-radius:50%;background:var(--rc);flex:none;
  box-shadow:0 0 0 3px color-mix(in srgb,var(--rc) 22%,transparent)}
.rs-risk-name{color:var(--sat-text);font-weight:700}
.rs-pill{margin-left:auto;font-size:12px;font-weight:700;padding:2px 9px;border-radius:999px;
  background:color-mix(in srgb,var(--rc) 13%,transparent);color:var(--sat-text);white-space:nowrap}
.rs-risk-val{font-family:'Space Grotesk','Inter',sans-serif;font-size:30px;font-weight:700;color:var(--sat-text);
  margin-top:8px;line-height:1.1;letter-spacing:-.02em}
.rs-risk-val small{font-family:'Inter',sans-serif;font-size:13px;color:var(--sat-muted);font-weight:500;letter-spacing:0}
.rs-risk-money{font-size:13px;color:var(--sat-muted);margin-top:4px}
.rs-risk-money b{color:var(--sat-text)}
.rs-meter{height:8px;border-radius:8px;background:color-mix(in srgb,var(--rc) 14%,transparent);margin-top:10px;overflow:hidden}
.rs-meter i{display:block;height:100%;background:var(--rc);border-radius:8px}
.rs-risk-foot{display:flex;justify-content:space-between;font-size:12px;color:var(--sat-muted);margin-top:9px;flex-wrap:wrap;gap:4px 10px}
.rs-legend{display:flex;flex-direction:column;gap:6px;margin-top:2px}
.rs-legend-row{display:grid;grid-template-columns:14px 1fr auto auto;gap:10px;align-items:center;font-size:13px;
  color:var(--sat-muted);padding:6px 10px;border-radius:10px;background:var(--sat-surface-2);border:1px solid var(--sat-border)}
.rs-legend-row .sw{width:10px;height:10px;border-radius:3px}
.rs-legend-row .nm{color:var(--sat-text);font-weight:600}
.rs-legend-row .v{color:var(--sat-text);font-weight:700;font-variant-numeric:tabular-nums;text-align:right}
.rs-legend-row .s{font-variant-numeric:tabular-nums;min-width:52px;text-align:right}
.rs-note{font-size:12.5px;color:var(--sat-muted);margin:-4px 0 6px 0;line-height:1.45}
.rs-seg{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:var(--sat-radius);
  padding:16px 18px;box-shadow:var(--sat-shadow)}
.rs-seg-k{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--sat-muted)}
.rs-seg-name{font-family:'Space Grotesk','Inter',sans-serif;font-size:20px;font-weight:700;color:var(--sat-text);
  margin:4px 0 8px 0;line-height:1.2;word-break:break-word}
.rs-seg-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0 12px 0}
.rs-seg-grid div{background:var(--sat-surface-2);border:1px solid var(--sat-border);border-radius:12px;padding:8px 10px}
.rs-seg-grid span{display:block;font-size:11.5px;color:var(--sat-muted);font-weight:600}
.rs-seg-grid b{display:block;font-size:17px;color:var(--sat-text);margin-top:2px}
.rs-stack{display:flex;height:12px;border-radius:7px;overflow:hidden;gap:2px;background:var(--sat-surface-2)}
.rs-stack i{display:block;height:100%}
.rs-stack-cap{display:flex;gap:12px;flex-wrap:wrap;font-size:11.5px;color:var(--sat-muted);margin-top:5px}
.rs-stack-cap span:before{content:"";display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;background:var(--c)}
.rs-sub{margin-top:12px;border-top:1px solid var(--sat-border);padding-top:10px}
.rs-sub-t{font-size:12px;font-weight:700;color:var(--sat-text);margin-bottom:6px}
.rs-sub-row{display:grid;grid-template-columns:1fr 92px 54px;gap:8px;align-items:center;font-size:12.5px;color:var(--sat-muted);margin:5px 0}
.rs-sub-row .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--sat-text)}
.rs-sub-row .br{height:7px;border-radius:7px;background:var(--sat-surface-2);overflow:hidden}
.rs-sub-row .br i{display:block;height:100%;background:var(--sat-alto);border-radius:7px}
.rs-sub-row .pc{text-align:right;font-variant-numeric:tabular-nums;font-weight:700;color:var(--sat-text)}
.rs-mini{display:flex;flex-direction:column;gap:10px}
.rs-mini-t{background:var(--sat-surface-2);border:1px solid var(--sat-border);border-radius:14px;padding:11px 14px}
.rs-mini-t .l{font-size:12px;color:var(--sat-muted);font-weight:600}
.rs-mini-t .v{font-family:'Space Grotesk','Inter',sans-serif;font-size:24px;font-weight:700;color:var(--sat-text);line-height:1.15}
.rs-mini-t .s{font-size:12px;color:var(--sat-muted)}
.rs-target{position:relative;height:8px;border-radius:8px;background:color-mix(in srgb,var(--sat-alto) 14%,transparent);margin:8px 0 4px 0}
.rs-target i{position:absolute;left:0;top:0;bottom:0;border-radius:8px;background:var(--sat-alto)}
.rs-target b{position:absolute;top:-4px;width:2px;height:16px;background:var(--sat-text);border-radius:2px}
.rs-route{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:var(--sat-radius);
  padding:14px 16px;box-shadow:var(--sat-shadow);height:100%;border-top:4px solid var(--rc)}
.rs-route-h{display:flex;align-items:center;justify-content:space-between;gap:8px}
.rs-route-code{font-weight:800;font-size:12px;letter-spacing:.08em;color:var(--sat-muted)}
.rs-route-sla{font-size:11.5px;font-weight:700;padding:2px 8px;border-radius:999px;background:var(--sat-surface-2);
  border:1px solid var(--sat-border);color:var(--sat-muted);white-space:nowrap}
.rs-route-n{font-family:'Space Grotesk','Inter',sans-serif;font-size:26px;font-weight:700;color:var(--sat-text);margin-top:6px;line-height:1.1}
.rs-route-n small{font-family:'Inter',sans-serif;font-size:12.5px;font-weight:500;color:var(--sat-muted)}
.rs-route-name{font-weight:700;color:var(--sat-text);font-size:14px;margin-top:2px}
.rs-route-a{font-size:12.5px;color:var(--sat-muted);margin-top:4px;line-height:1.4}
.rs-link-t{font-weight:700;color:var(--sat-text);font-size:15px}
.rs-link-d{font-size:12.8px;color:var(--sat-muted);line-height:1.4;min-height:36px}
</style>
"""


# ======================================================================================
# Utilidades de cálculo
# ======================================================================================
def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _div(a, b) -> float:
    try:
        return float(a) / float(b) if b else NAN
    except (TypeError, ValueError):
        return NAN


def _stats(d: pd.DataFrame) -> dict:
    """KPI de un subconjunto de cartera."""
    yp = d["y_pred"].astype(str)
    monto = d["valor_financiacion"].fillna(0)
    expo = d["exposicion_riesgo"].fillna(0)
    mora = d["mora_flag"].fillna(0)
    n = len(d)
    out = {"n": n, "est": int(d["id_estudiante"].nunique()), "monto": float(monto.sum()), "expo": float(expo.sum()),
           "n_mora": int(mora.sum())}
    for r in RISK_ORDER:
        m = (yp == r).to_numpy()
        out[f"n_{r}"] = int(m.sum())
        out[f"monto_{r}"] = float(monto[m].sum())
        out[f"expo_{r}"] = float(expo[m].sum())
        out[f"mora_{r}"] = float(mora[m].mean()) if m.any() else NAN
        out[f"ticket_{r}"] = float(monto[m].mean()) if m.any() else NAN
    out["pct_alto"] = _div(out["n_Alto"], n)
    out["pct_mora"] = _div(out["n_mora"], n)
    out["pct_monto_alto"] = _div(out["monto_Alto"], out["monto"])
    out["ticket"] = _div(out["monto"], n)
    return out


def _shift(p: str, k: int) -> str:
    return str(pd.Period(p, "M") + k)


def _window_label(a: str, b: str) -> str:
    return fmt_period(a) if a == b else f"{fmt_period(a)} – {fmt_period(b)}"


def _pct_change(a, b) -> float:
    return (a - b) / b if b else NAN


def _delta_pp(cur, prev, suffix: str, higher_is_bad: bool = True):
    """(texto, dirección) para una diferencia en puntos porcentuales."""
    if _isnan(cur) or _isnan(prev):
        return None, "flat"
    d = cur - prev
    arrow = "▲" if d > 0.0005 else ("▼" if d < -0.0005 else "■")
    if abs(d) < 0.0005:
        direction = "flat"
    else:
        direction = "up" if (d > 0) == higher_is_bad else "down"
    return f"{arrow} {fmt_delta_pp(d)} {suffix}", direction


def _delta_rel(cur, prev, suffix: str):
    ch = _pct_change(cur, prev)
    if _isnan(ch):
        return None, "flat"
    arrow = "▲" if ch > 0.0005 else ("▼" if ch < -0.0005 else "■")
    sign = "+" if ch >= 0 else "−"
    return f"{arrow} {sign}{fmt_num(abs(ch) * 100, 1)} % {suffix}", "flat"


def _monthly(d: pd.DataFrame) -> pd.DataFrame:
    t = d.dropna(subset=["periodo"])
    if t.empty:
        return pd.DataFrame(columns=["periodo", "n", "alto", "mora", "monto", "monto_alto"])
    g = t.assign(_alto=(t["y_pred"].astype(str) == "Alto").astype(int),
                 _monto=t["valor_financiacion"].fillna(0))
    g["_monto_alto"] = g["_monto"] * g["_alto"]
    out = g.groupby("periodo").agg(n=("llave2", "size"), alto=("_alto", "mean"), mora=("mora_flag", "mean"),
                                   monto=("_monto", "sum"), monto_alto=("_monto_alto", "sum")).reset_index()
    return out.sort_values("periodo")


def _spark(vals, color: str) -> str:
    v = [float(x) for x in vals if not _isnan(x)]
    if len(v) < 3:
        return ""
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    n = len(v)
    pts = [(i * 100 / (n - 1), 27 - (x - lo) / rng * 23) for i, x in enumerate(v)]
    line = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    return (f"<svg class='rs-spark' viewBox='0 0 100 30' preserveAspectRatio='none' aria-hidden='true'>"
            f"<polygon points='0,30 {line} 100,30' fill='{color}' fill-opacity='0.10'/>"
            f"<polyline points='{line}' fill='none' stroke='{color}' stroke-width='1.8' stroke-linejoin='round' "
            f"stroke-linecap='round' vector-effect='non-scaling-stroke'/></svg>")


def _with_spark(card_html: str, vals, color: str, caption: str) -> str:
    sp = _spark(vals, color)
    if not sp:
        return card_html
    return card_html[:-6] + sp + f"<div class='rs-spark-cap'>{esc(caption)}</div></div>"


def _short(s, n: int = 34) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _seg_label(col: str, v) -> str:
    return f"Scoring {v}" if col == "score_corto" else str(v)


def _bucket_label(v: str, gran: str) -> str:
    v = str(v)
    if gran == "Mes":
        return fmt_period(v)
    if gran == "Trimestre" and "Q" in v:
        y, q = v.split("Q")
        return f"T{q} {y}"
    if gran == "Semestre" and "-" in v:
        y, s = v.split("-")
        return f"{s} {y}"
    return v


def _seg_table(d: pd.DataFrame, col: str, min_n: int, global_alto: float) -> pd.DataFrame:
    s = segment_summary(d, col, min_n=min_n)
    if s.empty:
        return s
    s = s.copy()
    s["segmento"] = s[col].astype(str)
    s["n_alto"] = (s["pct_alto"] * s["creditos"]).round().astype(int)
    s["lift"] = s["pct_alto"] / global_alto if global_alto and not _isnan(global_alto) else NAN
    return s.sort_values(["pct_alto", "creditos"], ascending=[False, False]).reset_index(drop=True)


def _compare(d: pd.DataFrame, mask_a, mask_b) -> dict:
    a = (d["y_pred"].astype(str) == "Alto").to_numpy()
    ma, mb = np.asarray(mask_a, dtype=bool), np.asarray(mask_b, dtype=bool)
    na, nb = int(ma.sum()), int(mb.sum())
    pa = float(a[ma].mean()) if na else NAN
    pb = float(a[mb].mean()) if nb else NAN
    return {"na": na, "nb": nb, "pa": pa, "pb": pb, "alto_a": int(a[ma].sum()), "alto_b": int(a[mb].sum()),
            "lift": pa / pb if nb and pb else NAN}


def _x(v) -> str:
    return "—" if _isnan(v) else f"{fmt_num(v, 1)}×"


# ======================================================================================
# Hallazgos automáticos
# ======================================================================================
def _findings(d: pd.DataFrame, s: dict, min_n: int) -> list[dict]:
    out: list[dict] = []
    g = s["pct_alto"]
    total_alto = max(s["n_Alto"], 1)

    # 1) Segmento más riesgoso (cadena de respaldo si el filtro deja un solo grupo)
    top = None
    for col in ["programa_cluster", "programa", "facultad", "nivel", "sede"]:
        mn = min_n
        tab = _seg_table(d, col, mn, g)
        if len(tab) < 2:
            mn = max(5, min_n // 3)
            tab = _seg_table(d, col, mn, g)
        if len(tab) >= 2:
            top = (col, tab.iloc[0])
            break
    if top is not None and not _isnan(g) and g > 0:
        col, row = top
        out.append({
            "title": "Segmento que más concentra riesgo", "tone": "alto", "icon": "🎯",
            "html": (f"<b>{esc(_seg_label(col, row['segmento']))}</b> ({esc(_DIM_SINGULAR.get(col, col))}) tiene "
                     f"<b>{fmt_pct(row['pct_alto'])}</b> de sus créditos en riesgo Alto: <b>{_x(row['lift'])}</b> el promedio "
                     f"del filtro ({fmt_pct(g)}). Son {fmt_int(row['n_alto'])} alertas sobre {fmt_int(row['creditos'])} créditos "
                     f"y {fmt_cop(row['exposicion_alto'])} financiados en Alto."),
        })
    else:
        out.append({"title": "Segmento que más concentra riesgo", "tone": "info", "icon": "🎯",
                    "html": "El filtro actual no deja al menos dos segmentos comparables o no hay créditos en Alto."})

    # 2) Antigüedad del crédito
    ant = d["antiguedad_meses"]
    c = _compare(d, (ant < 12).fillna(False), (ant >= 12).fillna(False))
    if c["na"] >= 10 and c["nb"] >= 10:
        share_alerts = _div(c["alto_a"], total_alto)
        verbo = "concentran" if c["pa"] > c["pb"] else "muestran"
        out.append({
            "title": "Los créditos recientes pesan más", "tone": "medio" if c["pa"] > c["pb"] else "bajo", "icon": "⏱️",
            "html": (f"Los créditos con <b>menos de 12 meses</b> {verbo} <b>{fmt_pct(c['pa'])}</b> en Alto frente a "
                     f"{fmt_pct(c['pb'])} del resto (<b>{_x(c['lift'])}</b>). Son {fmt_int(c['na'])} créditos "
                     f"({fmt_pct(_div(c['na'], s['n']))} del filtro) y aportan {fmt_pct(share_alerts)} de las alertas."),
        })
    else:
        out.append({"title": "Antigüedad del crédito", "tone": "info", "icon": "⏱️",
                    "html": "No hay suficientes créditos a ambos lados de 12 meses de antigüedad para comparar."})

    # 3) Scoring externo
    sc = d["score_corto"].astype(str)
    c = _compare(d, sc == "≤ 400", sc == "≥ 721")
    if c["na"] >= 10 and c["nb"] >= 10:
        bands = d[~sc.isin(["Sin dato", "Sin score"])].assign(_a=(d["y_pred"].astype(str) == "Alto").astype(int))
        bt = bands.groupby("score_corto")["_a"].agg(["mean", "size"])
        bt = bt[bt["size"] >= 30]
        peak = ""
        if not bt.empty:
            pk = bt["mean"].idxmax()
            if pk not in ("≤ 400", "≥ 721"):
                peak = (f" El pico no está en el extremo: la banda <b>{esc(pk)}</b> llega a {fmt_pct(bt.loc[pk, 'mean'])} "
                        f"({fmt_int(bt.loc[pk, 'size'])} créditos).")
        if c["pa"] > c["pb"]:
            lead = (f"Con scoring externo <b>≤ 400</b> el {fmt_pct(c['pa'])} está en Alto frente a {fmt_pct(c['pb'])} con "
                    f"<b>≥ 721</b> (<b>{_x(c['lift'])}</b>): peor historial crediticio, más riesgo.")
        else:
            lead = (f"Scoring <b>≤ 400</b>: {fmt_pct(c['pa'])} en Alto; <b>≥ 721</b>: {fmt_pct(c['pb'])}. En este filtro el "
                    f"scoring externo no separa el riesgo en la dirección esperada.")
        out.append({"title": "Scoring externo (Datacrédito)", "tone": "medio", "icon": "📉", "html": lead + peak})
    else:
        out.append({"title": "Scoring externo (Datacrédito)", "tone": "info", "icon": "📉",
                    "html": "Muestra insuficiente en las bandas ≤ 400 y ≥ 721 para comparar dentro del filtro."})

    # 4) Tipo de interés
    ti = d["tipo_interes"].astype(str)
    vc = ti[~ti.isin(["Sin dato", "nan"])].value_counts()
    if len(vc) >= 2:
        rates = d.assign(_a=(d["y_pred"].astype(str) == "Alto").astype(int)).groupby("tipo_interes")["_a"].mean()
        rates = rates[rates.index.isin(vc[vc >= 10].index)].sort_values(ascending=False)
        if len(rates) >= 2:
            hi_, lo_ = rates.index[0], rates.index[-1]
            c = _compare(d, ti == hi_, ti == lo_)
            out.append({
                "title": "Tipo de interés", "tone": "accent", "icon": "💳",
                "html": (f"<b>{esc(hi_)}</b>: {fmt_pct(c['pa'])} en Alto frente a {fmt_pct(c['pb'])} en {esc(lo_)} "
                         f"(<b>{_x(c['lift'])}</b>). Esta modalidad reúne {fmt_pct(_div(c['alto_a'], total_alto))} de las "
                         f"alertas con {fmt_pct(_div(c['na'], s['n']))} de los créditos."),
            })
    if len(out) < 4:
        out.append({"title": "Tipo de interés", "tone": "info", "icon": "💳",
                    "html": "El filtro contiene una sola modalidad de interés: no hay contraste disponible."})
    return out


# ======================================================================================
# Gráficas
# ======================================================================================
def _alpha(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _evolution_fig(d: pd.DataFrame, gran: str, measure: str) -> tuple[go.Figure | None, str]:
    p = pal()
    col = _GRAN[gran]
    t = d.dropna(subset=[col]).copy()
    t = t[t[col].astype(str) != "nan"]
    if t.empty:
        return None, ""
    money = measure == "Monto"
    t["_v"] = t["valor_financiacion"].fillna(0) / 1e6 if money else 1.0
    piv = (t.groupby([col, "y_pred"], observed=False)["_v"].sum().unstack("y_pred")
           .reindex(columns=RISK_ORDER).fillna(0).sort_index())
    cnt = t.groupby(col).size().reindex(piv.index).fillna(0)
    tot = piv.sum(axis=1)
    share = (piv["Alto"] / tot.replace(0, np.nan)).astype(float)
    labels = [_bucket_label(x, gran) for x in piv.index]
    avg = float(piv["Alto"].sum() / tot.sum()) if tot.sum() else NAN

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.30, 0.70], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(
        x=labels, y=share, mode="lines+markers" if len(labels) <= 14 else "lines",
        line=dict(color=RISK_COLORS["Alto"], width=2, shape="spline", smoothing=0.5),
        marker=dict(size=7, color=RISK_COLORS["Alto"], line=dict(color=p["bg"], width=2)),
        fill="tozeroy", fillcolor=_alpha(RISK_COLORS["Alto"], 0.10), name="% en Alto", showlegend=False,
        customdata=[fmt_pct(v) for v in share],
        hovertemplate="<b>% en Alto</b>: %{customdata}<extra></extra>"), row=1, col=1)
    if not _isnan(avg):
        fig.add_hline(y=avg, line=dict(color=p["muted"], width=1, dash="dot"), row=1, col=1)
        fig.add_annotation(x=0, xref="x domain", y=avg, yref="y", text=f"Promedio {fmt_pct(avg)}", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=p["muted"]), yshift=2)
    if len(share.dropna()):
        last_i = int(np.where(share.notna())[0][-1])
        fig.add_annotation(x=labels[last_i], y=share.iloc[last_i], xref="x", yref="y", text=f"<b>{fmt_pct(share.iloc[last_i])}</b>",
                           showarrow=False, xanchor="left", xshift=7, font=dict(size=12, color=p["text"]))
    unit = "M" if money else "créditos"
    for r in RISK_ORDER:
        vals = piv[r]
        pct = (vals / tot.replace(0, np.nan)).fillna(0)
        cd = np.c_[[fmt_num(v, 0) if money else fmt_int(v) for v in vals], [fmt_pct(x) for x in pct]]
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=r, marker=dict(color=RISK_COLORS[r] if r != "Bajo" else _alpha(RISK_COLORS["Bajo"], 0.62),
                                                  line=dict(width=0)),
            customdata=cd,
            hovertemplate=f"<b>{r}</b>: " + ("$ %{customdata[0]} M" if money else "%{customdata[0]} créditos")
                          + " (%{customdata[1]})<extra></extra>"), row=2, col=1)
    # Último periodo incompleto
    note = ""
    if len(cnt) >= 4:
        med = float(cnt.iloc[:-1].tail(12).median())
        if med > 0 and cnt.iloc[-1] < 0.35 * med:
            fig.add_annotation(x=labels[-1], y=float(tot.iloc[-1]), xref="x2", yref="y2", text="parcial", showarrow=False,
                               yanchor="bottom", yshift=4, font=dict(size=10, color=p["muted"]))
            note = f"{labels[-1]} tiene solo {fmt_int(cnt.iloc[-1])} créditos (periodo incompleto): lea su % con cautela."
    k = max(1, math.ceil(len(labels) / 12))
    fig.update_xaxes(type="category", tickmode="array", tickvals=labels[::k], tickangle=0, row=2, col=1)
    fig.update_xaxes(type="category", showticklabels=False, row=1, col=1)
    ymax = float(np.nanmax(share.to_numpy())) if share.notna().any() else 0.1
    fig.update_yaxes(tickformat=".0%", range=[0, max(ymax * 1.3, 0.02)], nticks=3, title_text="% Alto", row=1, col=1)
    fig.update_yaxes(title_text="Millones de COP" if money else "Créditos", row=2, col=1)
    fig.update_layout(barmode="stack", bargap=0.22, hovermode="x unified", margin=dict(l=8, r=46, t=34, b=8),
                      legend=dict(y=1.06, x=0, traceorder="normal"))
    return fig, note


def _donut_fig(vals: list[float], center_top: str, center_sub: str) -> go.Figure:
    p = pal()
    tot = sum(vals) or 1
    fig = go.Figure(go.Pie(
        labels=RISK_ORDER, values=vals, hole=0.72, sort=False, direction="clockwise",
        marker=dict(colors=[RISK_COLORS[r] for r in RISK_ORDER], line=dict(color=p["bg"], width=3)),
        textinfo="none", customdata=[fmt_pct(v / tot) for v in vals],
        hovertemplate="<b>%{label}</b><br>%{customdata} del total<extra></extra>"))
    fig.add_annotation(text=f"<b>{center_top}</b>", x=0.5, y=0.54, showarrow=False, font=dict(size=26, color=p["text"]))
    fig.add_annotation(text=center_sub, x=0.5, y=0.41, showarrow=False, font=dict(size=12, color=p["muted"]))
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    return fig


def _expo_fig(values: dict, label: str) -> go.Figure:
    p = pal()
    tot = sum(values.values()) or 1
    x = [values[r] for r in RISK_ORDER]
    fig = go.Figure(go.Bar(
        y=RISK_ORDER, x=x, orientation="h", marker=dict(color=[RISK_COLORS[r] for r in RISK_ORDER], line=dict(width=0)),
        text=[f"<b>{fmt_cop(v)}</b> · {fmt_pct(v / tot)}" for v in x], textposition="outside", cliponaxis=False,
        textfont=dict(color=p["text"], size=12.5), width=0.5,
        customdata=[fmt_cop(v, compact=False) for v in x],
        hovertemplate=f"<b>%{{y}}</b><br>{esc(label)}: %{{customdata}}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  ")
    fig.update_xaxes(range=[0, max(x + [1]) * 1.55], showticklabels=False, showgrid=False)
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    return fig


def _mix_fig(s: dict) -> go.Figure:
    rows = ["Créditos", "Monto financiado", "Exposición ponderada"]
    keys = ["n", "monto", "expo"]
    fig = go.Figure()
    for r in RISK_ORDER:
        shares = [_div(s[f"{k}_{r}"], s[k]) for k in keys]
        shares = [0 if _isnan(v) else v for v in shares]
        fig.add_trace(go.Bar(
            y=rows, x=shares, name=r, orientation="h", marker=dict(color=RISK_COLORS[r], line=dict(width=0)),
            text=[fmt_pct(v) if v >= 0.07 else "" for v in shares], textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#141414" if r == "Medio" else "#FFFFFF", size=12), width=0.52,
            customdata=[fmt_pct(v) for v in shares],
            hovertemplate=f"<b>{r}</b> · %{{y}}: %{{customdata}}<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.35, margin=dict(l=8, r=8, t=34, b=8))
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True)
    fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  ")
    return fig


def _top_fig(tab: pd.DataFrame, col: str, global_alto: float) -> go.Figure:
    p = pal()
    t = tab.iloc[::-1]  # la barra más alta arriba
    labels, seen = [], {}
    for v in t["segmento"]:
        lab = _short(_seg_label(col, v), 34)
        seen[lab] = seen.get(lab, 0) + 1
        labels.append(lab if seen[lab] == 1 else f"{lab} ({seen[lab]})")
    colors = [RISK_COLORS["Alto"] if (not _isnan(l) and l > 1.0) else p["subtle"] for l in t["lift"]]
    cd = np.c_[
        [esc(_seg_label(col, v)) for v in t["segmento"]],
        [fmt_int(v) for v in t["creditos"]], [fmt_int(v) for v in t["n_alto"]], [fmt_pct(v) for v in t["pct_alto"]],
        [_x(v) for v in t["lift"]], [fmt_cop(v) for v in t["exposicion_alto"]], [fmt_pct(v) for v in t["pct_mora"]],
        t["segmento"].astype(str).to_numpy(),
    ]
    fig = go.Figure(go.Bar(
        y=labels, x=t["pct_alto"], orientation="h", marker=dict(color=colors, line=dict(width=0)),
        text=[f"<b>{fmt_pct(a)}</b> · {_x(l)}" for a, l in zip(t["pct_alto"], t["lift"])], textposition="outside",
        cliponaxis=False, textfont=dict(color=p["text"], size=12), customdata=cd,
        hovertemplate=("<b>%{customdata[0]}</b><br>% en Alto: %{customdata[3]} (lift %{customdata[4]})"
                       "<br>Créditos: %{customdata[1]} · en Alto: %{customdata[2]}"
                       "<br>Monto en Alto: %{customdata[5]} · Mora Datacrédito: %{customdata[6]}<extra></extra>")))
    if not _isnan(global_alto):
        fig.add_vline(x=global_alto, line=dict(color=p["text"], width=1.2, dash="dot"))
        fig.add_annotation(x=global_alto, y=1, yref="paper", text=f"Promedio {fmt_pct(global_alto)}", showarrow=False,
                           xanchor="left", yanchor="bottom", xshift=4, font=dict(size=11, color=p["muted"]))
    xmax = float(t["pct_alto"].max()) if len(t) else 0.1
    fig.update_xaxes(range=[0, max(xmax * 1.42, 0.02)], tickformat=".0%", title_text="% de créditos en riesgo Alto")
    fig.update_yaxes(showgrid=False, ticksuffix="  ")
    fig.update_layout(bargap=0.38, margin=dict(l=8, r=8, t=30, b=8), clickmode="event+select")
    return fig


def _confusion_fig(cm: np.ndarray) -> go.Figure:
    p = pal()
    dark = theme_mode() == "dark"
    rt = cm.sum(axis=1, keepdims=True)
    rp = np.divide(cm, rt, out=np.zeros(cm.shape, dtype=float), where=rt > 0)
    scale = ([[0, "#1B202A"], [0.5, "#6B5A12"], [1, YELLOW]] if dark
             else [[0, "#F4F2EC"], [0.5, "#FFE680"], [1, YELLOW]])
    xs = [f"Pred. {r}" for r in RISK_ORDER]
    ys = [f"Real {r}" for r in RISK_ORDER]
    cd = np.vectorize(fmt_int)(cm)
    fig = go.Figure(go.Heatmap(
        z=rp, x=xs, y=ys, colorscale=scale, zmin=0, zmax=1, showscale=False, xgap=4, ygap=4,
        customdata=np.dstack([cd, np.vectorize(fmt_pct)(rp)]),
        hovertemplate="%{y} · %{x}<br>%{customdata[0]} créditos (%{customdata[1]} de la fila)<extra></extra>"))
    for i in range(3):
        for j in range(3):
            fc = "#141414" if (rp[i, j] > 0.45 or not dark) else p["text"]
            fig.add_annotation(x=xs[j], y=ys[i], text=f"<b>{fmt_int(cm[i, j])}</b><br>{fmt_pct(rp[i, j])}", showarrow=False,
                               font=dict(size=12, color=fc))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_layout(margin=dict(l=8, r=8, t=30, b=8))
    return fig


# ======================================================================================
# HTML de tarjetas propias
# ======================================================================================
def _risk_card(r: str, s: dict) -> str:
    n, monto = s[f"n_{r}"], s[f"monto_{r}"]
    pct, pct_m = _div(n, s["n"]), _div(monto, s["monto"])
    w = 0 if _isnan(pct_m) else max(0.0, min(1.0, pct_m)) * 100
    return (f"<div class='rs-risk' style='--rc:{RISK_COLORS[r]}'>"
            f"<div class='rs-risk-top'><span class='rs-dot'></span><span class='rs-risk-name'>Riesgo {r}</span>"
            f"<span class='rs-pill'>{fmt_pct(pct)} de los créditos</span></div>"
            f"<div class='rs-risk-val'>{fmt_int(n)} <small>créditos</small></div>"
            f"<div class='rs-risk-money'><b>{fmt_cop(monto)}</b> financiados · {fmt_pct(pct_m)} del monto</div>"
            f"<div class='rs-meter' title='Participación en el monto financiado'><i style='width:{w:.1f}%'></i></div>"
            f"<div class='rs-risk-foot'><span>Ticket promedio {fmt_cop(s[f'ticket_{r}'])}</span>"
            f"<span>Mora Datacrédito {fmt_pct(s[f'mora_{r}'])}</span></div></div>")


def _legend_html(s: dict, money: bool) -> str:
    rows = []
    for r in RISK_ORDER:
        v = s[f"monto_{r}"] if money else s[f"n_{r}"]
        tot = s["monto"] if money else s["n"]
        rows.append(f"<div class='rs-legend-row'><span class='sw' style='background:{RISK_COLORS[r]}'></span>"
                    f"<span class='nm'>{r}</span><span class='v'>{fmt_cop(v) if money else fmt_int(v)}</span>"
                    f"<span class='s'>{fmt_pct(_div(v, tot))}</span></div>")
    return "<div class='rs-legend'>" + "".join(rows) + "</div>"


def _stack_html(s: dict) -> str:
    parts, caps = [], []
    for r in RISK_ORDER:
        sh = _div(s[f"n_{r}"], s["n"])
        sh = 0 if _isnan(sh) else sh
        if sh > 0:
            parts.append(f"<i style='width:{sh * 100:.2f}%;background:{RISK_COLORS[r]}'></i>")
        caps.append(f"<span style='--c:{RISK_COLORS[r]}'>{r} {fmt_pct(sh)}</span>")
    return f"<div class='rs-stack'>{''.join(parts)}</div><div class='rs-stack-cap'>{''.join(caps)}</div>"


def _segment_detail_html(d: pd.DataFrame, col: str, name: str, global_alto: float, dim_label: str) -> str:
    sub = d[d[col].astype(str) == name]
    s = _stats(sub)
    lift = _div(s["pct_alto"], global_alto)
    kind = "alto" if (not _isnan(lift) and lift > 1.2) else ("medio" if (not _isnan(lift) and lift > 1.0) else "bajo")
    lift_txt = f"{_x(lift)} el promedio" if not _isnan(lift) else "sin referencia"
    badges = badge(lift_txt, kind) + " " + badge(f"{fmt_int(s['est'])} estudiantes", "neutral")
    # Sub-segmentos: dónde se concentra el riesgo dentro del segmento
    sub_col = {"programa_cluster": "programa", "facultad": "programa", "programa": "tipo_interes",
               "tipo_interes": "programa_cluster", "score_corto": "programa_cluster", "cohorte": "programa_cluster",
               "nivel": "programa_cluster"}.get(col, "programa_cluster")
    sub_lbl = {"programa": "programa", "tipo_interes": "tipo de interés", "programa_cluster": "segmento de programa"}[sub_col]
    st_ = segment_summary(sub, sub_col, min_n=max(5, min(20, len(sub) // 20)))
    rows = ""
    if len(st_) >= 1:
        st_ = st_.sort_values(["pct_alto", "creditos"], ascending=[False, False]).head(3)
        mx = max(float(st_["pct_alto"].max()), 1e-9)
        for _, rr in st_.iterrows():
            rows += (f"<div class='rs-sub-row'><span class='nm' title='{esc(rr[sub_col])}'>{esc(_short(rr[sub_col], 30))}"
                     f" <span style='color:var(--sat-subtle)'>· {fmt_int(rr['creditos'])}</span></span>"
                     f"<span class='br'><i style='width:{rr['pct_alto'] / mx * 100:.0f}%'></i></span>"
                     f"<span class='pc'>{fmt_pct(rr['pct_alto'])}</span></div>")
    sub_html = (f"<div class='rs-sub'><div class='rs-sub-t'>Mayor % Alto por {sub_lbl} dentro del segmento</div>{rows}</div>"
                if rows else "")
    return (f"<div class='rs-seg'><div class='rs-seg-k'>{esc(dim_label)} seleccionado</div>"
            f"<div class='rs-seg-name'>{esc(_seg_label(col, name))}</div>{badges}"
            f"<div class='rs-seg-grid'>"
            f"<div><span>Créditos</span><b>{fmt_int(s['n'])}</b></div>"
            f"<div><span>En riesgo Alto</span><b>{fmt_int(s['n_Alto'])} · {fmt_pct(s['pct_alto'])}</b></div>"
            f"<div><span>Monto en Alto</span><b>{fmt_cop(s['monto_Alto'])}</b></div>"
            f"<div><span>Mora Datacrédito</span><b>{fmt_pct(s['pct_mora'])}</b></div></div>"
            f"{_stack_html(s)}{sub_html}</div>")


# ======================================================================================
# Reporte ejecutivo descargable (ME-03)
# ======================================================================================
def _report_html(r: dict) -> str:
    s = r["stats"]
    kpis = "".join(
        f"<div class='k'><div class='kl'>{esc(k['label'])}</div><div class='kv'>{esc(k['value'])}</div>"
        f"<div class='ks'>{esc(k['sub'])}</div>"
        + (f"<div class='kd {k['dir']}'>{esc(k['delta'])}</div>" if k.get("delta") else "") + "</div>"
        for k in r["kpis"])
    bar = "".join(f"<i style='width:{_div(s[f'n_{x}'], s['n']) * 100 if s['n'] else 0:.2f}%;background:{RISK_COLORS[x]}'></i>"
                  for x in RISK_ORDER)
    risk_rows = "".join(
        f"<tr><td><span class='sw' style='background:{RISK_COLORS[x]}'></span>{x}</td><td>{fmt_int(s[f'n_{x}'])}</td>"
        f"<td>{fmt_pct(_div(s[f'n_{x}'], s['n']))}</td><td>{fmt_cop(s[f'monto_{x}'])}</td>"
        f"<td>{fmt_pct(_div(s[f'monto_{x}'], s['monto']))}</td><td>{fmt_pct(s[f'mora_{x}'])}</td></tr>" for x in RISK_ORDER)
    # mini gráfico de % Alto por mes (SVG)
    m = r["monthly"].tail(18)
    svg = ""
    if len(m) >= 2:
        mx = max(float(m["alto"].max()), 0.01)
        bw = 560 / len(m)
        cols = []
        for i, (_, row) in enumerate(m.iterrows()):
            h = row["alto"] / mx * 110
            cols.append(f"<rect x='{i * bw + 3:.1f}' y='{130 - h:.1f}' width='{bw - 6:.1f}' height='{h:.1f}' rx='3' "
                        f"fill='{RISK_COLORS['Alto']}' fill-opacity='{0.45 if i < len(m) - 1 else 0.95}'>"
                        f"<title>{esc(fmt_period(row['periodo']))}: {fmt_pct(row['alto'])}</title></rect>")
        svg = (f"<svg viewBox='0 0 560 150' width='100%' height='150' role='img' aria-label='% en Alto por mes'>"
               f"{''.join(cols)}<line x1='0' x2='560' y1='130.5' y2='130.5' stroke='#D9D6CC'/>"
               f"<text x='0' y='146' font-size='11' fill='#6B6A64'>{esc(fmt_period(m['periodo'].iloc[0]))}</text>"
               f"<text x='560' y='146' font-size='11' fill='#6B6A64' text-anchor='end'>{esc(fmt_period(m['periodo'].iloc[-1]))}"
               f" · {fmt_pct(m['alto'].iloc[-1])}</text></svg>")
    seg_rows = "".join(
        f"<tr><td>{esc(_seg_label(r['dim_col'], row['segmento']))}</td><td>{fmt_int(row['creditos'])}</td>"
        f"<td>{fmt_int(row['n_alto'])}</td><td><b>{fmt_pct(row['pct_alto'])}</b></td><td>{_x(row['lift'])}</td>"
        f"<td>{fmt_cop(row['exposicion_alto'])}</td><td>{fmt_pct(row['pct_mora'])}</td></tr>"
        for _, row in r["top"].iterrows()) or "<tr><td colspan='7'>Sin segmentos con el n mínimo.</td></tr>"
    finds = "".join(f"<div class='f {f['tone']}'><div class='ft'>{esc(f['icon'])} {esc(f['title'])}</div>"
                    f"<div class='fx'>{f['html']}</div></div>" for f in r["findings"])
    truth = r.get("truth")
    truth_html = (
        f"<p>Sobre {fmt_int(truth['n'])} créditos con riesgo observado: concordancia <b>{fmt_pct(truth['acc'])}</b>, "
        f"recall de Alto <b>{fmt_pct(truth['rec'])}</b> y precisión de Alto <b>{fmt_pct(truth['prec'])}</b> "
        f"({fmt_int(truth['tp'])} de {fmt_int(truth['real_alto'])} Alto observados fueron anticipados). "
        f"La regla argmax subdetecta la clase Alto; la meta DE-03 exige recall ≥ {fmt_pct(RECALL_ALTO_TARGET, 0)} y "
        f"se alcanza bajando el umbral de P(Alto) (ver «Desempeño del modelo»).</p>"
        if truth else "<p>El dataset activo no incluye el riesgo observado (y_true): no se calcula concordancia.</p>")
    route_rows = "".join(
        f"<tr><td><span class='sw' style='background:{ACTION_ROUTES[k]['color']}'></span>{k} · {esc(ACTION_ROUTES[k]['nombre'])}</td>"
        f"<td>{esc(ACTION_ROUTES[k]['sla'])}</td><td>{fmt_int(v['n'])}</td><td>{fmt_cop(v['monto'])}</td>"
        f"<td>{esc(ACTION_ROUTES[k]['accion'])}</td></tr>" for k, v in r["routes"].items())
    chips = "".join(f"<span>{esc(k)}: {esc(v)}</span>" for k, v in r["chips"]) or "<span>Sin filtros: toda la cartera</span>"
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reporte ejecutivo · {esc(APP_NAME)}</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#F6F6F3;color:#141414;font-family:Inter,'Segoe UI',Roboto,Arial,sans-serif;font-size:14px;line-height:1.45}}
.w{{max-width:1040px;margin:0 auto;padding:28px 24px 40px}}
.hero{{background:linear-gradient(120deg,#0B0B0C,#26251F);color:#F7F7F2;border-radius:18px;padding:24px 28px;position:relative;overflow:hidden}}
.hero:after{{content:"";position:absolute;right:-70px;top:-70px;width:240px;height:240px;border-radius:50%;background:radial-gradient(circle,rgba(255,209,0,.35),rgba(255,209,0,0) 70%)}}
.ey{{color:#FFD100;font-size:11px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}}
h1{{font-size:26px;margin:6px 0 4px}} .hero p{{margin:0;color:#CFCFC6}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}} .chips span{{font-size:11.5px;padding:3px 9px;border-radius:999px;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.16)}}
.chips span.hl{{background:#FFD100;color:#141414;border-color:#FFD100;font-weight:700}}
h2{{font-size:17px;margin:26px 0 10px;display:flex;align-items:center;gap:8px}} h2:before{{content:"";width:16px;height:3px;border-radius:3px;background:#FFD100}}
.kg{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}
.k{{background:#fff;border:1px solid #E7E5DF;border-radius:14px;padding:12px 14px}} .kl{{font-size:11.5px;color:#6B6A64;font-weight:600}}
.kv{{font-size:22px;font-weight:800;margin-top:4px}} .ks{{font-size:11.5px;color:#6B6A64}}
.kd{{display:inline-block;margin-top:6px;font-size:11px;font-weight:700;padding:2px 7px;border-radius:999px;background:#F1EFE8;color:#6B6A64}}
.kd.up{{background:rgba(229,72,77,.12);color:#C6282D}} .kd.down{{background:rgba(48,164,108,.14);color:#1E7D50}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #E7E5DF;border-radius:12px;overflow:hidden;font-size:13px}}
th{{text-align:left;background:#F1EFE8;color:#44433E;font-weight:700;padding:8px 10px;font-size:12px}} td{{padding:7px 10px;border-top:1px solid #EFEDE7;vertical-align:top}}
td:not(:first-child),th:not(:first-child){{text-align:right}} .rt td:last-child{{text-align:left;color:#6B6A64;font-size:12px}} .rt th:last-child{{text-align:left}}
.sw{{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:7px}}
.bar{{display:flex;gap:2px;height:14px;border-radius:7px;overflow:hidden;margin:4px 0 12px}} .bar i{{display:block;height:100%}}
.two{{display:grid;grid-template-columns:1.1fr 1fr;gap:16px;align-items:start}}
.card{{background:#fff;border:1px solid #E7E5DF;border-radius:14px;padding:14px 16px}}
.fg{{display:grid;grid-template-columns:1fr 1fr;gap:10px}} .f{{background:#fff;border:1px solid #E7E5DF;border-left:4px solid #FFD100;border-radius:12px;padding:11px 13px}}
.f.alto{{border-left-color:#E5484D}} .f.medio{{border-left-color:#F5A524}} .f.bajo{{border-left-color:#30A46C}} .f.info{{border-left-color:#3E63DD}}
.ft{{font-weight:700;margin-bottom:3px}} .fx{{color:#44433E;font-size:13px}}
.muted{{color:#6B6A64;font-size:12px}} .foot{{margin-top:28px;padding-top:12px;border-top:1px solid #E7E5DF;color:#9A988F;font-size:11.5px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px}}
@media (max-width:760px){{.kg{{grid-template-columns:repeat(2,1fr)}} .two,.fg{{grid-template-columns:1fr}}}}
@media print{{body{{background:#fff}} .w{{padding:0}} .hero{{-webkit-print-color-adjust:exact;print-color-adjust:exact}} .k,.f,table,.card{{break-inside:avoid}}}}
</style></head><body><div class="w">
<div class="hero"><div class="ey">◆ {esc(APP_NAME)} · Reporte ejecutivo</div>
<h1>Riesgo de morosidad de la cartera estudiantil</h1>
<p>Resumen para Dirección Financiera y Gestión de Cartera con los criterios de filtro aplicados en el tablero.</p>
<div class="chips"><span class="hl">{fmt_int(s['n_Alto'])} créditos en alerta alta</span><span>Generado {esc(r['generated'])}</span>
<span>Fuente: {esc(r['source'])}</span><span>{esc(r['period'])}</span></div>
<div class="chips">{chips}</div></div>

<h2>Indicadores clave (DB-03)</h2>
<div class="kg">{kpis}</div>
<p class="muted">Tendencias: {esc(r['window'])}.</p>

<h2>Distribución y exposición por riesgo predicho</h2>
<div class="two"><div><div class="bar">{bar}</div>
<table><tr><th>Riesgo</th><th>Créditos</th><th>%</th><th>Monto</th><th>% monto</th><th>Mora DC</th></tr>{risk_rows}</table></div>
<div class="card"><div class="ft">% de créditos en Alto por mes de aprobación</div>{svg or '<p class="muted">Sin serie mensual.</p>'}
<div class="muted">La barra más oscura es el último mes del filtro.</div></div></div>

<h2>Hallazgos automáticos</h2>
<div class="fg">{finds}</div>

<h2>Segmentos con mayor % en riesgo Alto · {esc(r['dim_label'])}</h2>
<table><tr><th>Segmento</th><th>Créditos</th><th>En Alto</th><th>% Alto</th><th>Lift</th><th>Monto en Alto</th><th>Mora DC</th></tr>{seg_rows}</table>
<p class="muted">Segmentos con al menos {fmt_int(r['min_n'])} créditos. Lift = % Alto del segmento ÷ % Alto del filtro ({fmt_pct(s['pct_alto'])}).</p>

<h2>Predicho vs. observado</h2>
<div class="card">{truth_html}</div>

<h2>Plan de gestión preventiva sugerido</h2>
<table class="rt"><tr><th>Ruta</th><th>SLA</th><th>Créditos</th><th>Monto</th><th>Acción</th></tr>{route_rows}</table>

<h2>Notas metodológicas</h2>
<p class="muted">Riesgo predicho por el modelo Random Forest seleccionado (clase más probable). Mora real según el reporte
de Datacrédito. Monto = valor financiado. Exposición ponderada = monto × intensidad de riesgo (proxy para priorizar,
no es una pérdida esperada). Los nombres de estudiantes están anonimizados; este reporte no contiene datos personales.</p>
<div class="foot"><span>{esc(APP_NAME)} · {esc(PROGRAMA_ACADEMICO)} · {esc(UNIVERSIDAD)}</span><span>Uso exclusivo para gestión preventiva de cartera</span></div>
</div></body></html>"""


@st.dialog("Vista previa · Reporte ejecutivo", width="large")
def _preview_dialog(doc: str, fname: str) -> None:
    st.iframe(doc, height=620)
    st.download_button("Descargar reporte (HTML)", doc.encode("utf-8"), file_name=fname, mime="text/html",
                       icon=":material/download:", type="primary", key=f"{P}_report_dl_dialog", width="stretch")


# ======================================================================================
# Página
# ======================================================================================
st.markdown(_CSS, unsafe_allow_html=True)
df_all = get_active_df()
dff = apply_filters(df_all)
p = pal()

n_alto_hero = int((dff["y_pred"].astype(str) == "Alto").sum())
periods = sorted(str(x) for x in dff["periodo"].dropna().unique() if str(x) != "nan")
period_txt = (f"Aprobaciones {fmt_period(periods[0])} – {fmt_period(periods[-1])}" if periods else "Sin fechas de aprobación")
page_header(
    "Resumen ejecutivo",
    "Cuánta cartera está en riesgo, dónde se concentra y qué hacer primero: la vista de decisión para Dirección "
    "Financiera y Gestión de Cartera.",
    eyebrow="Visión general · Dirección y Cartera",
    meta=[period_txt, "Modelo: Random Forest"],
    highlight=f"{fmt_int(n_alto_hero)} créditos en alerta alta",
)
filter_chips(active_chips(df_all))

if dff.empty:
    empty_state("Sin créditos para este filtro", "Ningún crédito cumple los filtros de la barra lateral. "
                "Amplía el periodo o limpia algunos criterios.")
    footer()
    st.stop()

S = _stats(dff)

# ---------- Barra de herramientas: ventana de comparación + reporte ----------
tb1, tb2, tb3, tb4 = st.columns([1.05, 2.2, 0.95, 1.25], vertical_alignment="center")
with tb1:
    win_lbl = st.segmented_control("Comparar tendencia", list(_WINDOWS), default="6 m", key=f"{P}_win",
                                   help="Los deltas comparan los últimos N meses del filtro contra los N meses "
                                        "inmediatamente anteriores (mismos filtros, sin el filtro de periodo).")
N = _WINDOWS.get(win_lbl or "6 m", 6)
cmp_ok = False
rec_s = prev_s = None
window_txt = "Sin periodo comparable"
if periods:
    end = periods[-1]
    r_from, p_to, p_from = _shift(end, -(N - 1)), _shift(end, -N), _shift(end, -(2 * N - 1))
    base_np = apply_filters(df_all, skip={"periodo"})
    pp = base_np["periodo"].astype(str)
    rec_df = base_np[(pp >= r_from) & (pp <= end)]
    prev_df = base_np[(pp >= p_from) & (pp <= p_to)]
    if len(rec_df) >= 20 and len(prev_df) >= 20:
        cmp_ok = True
        rec_s, prev_s = _stats(rec_df), _stats(prev_df)
        window_txt = (f"últimos {N} meses ({_window_label(r_from, end)}, {fmt_int(len(rec_df))} créditos) vs. "
                      f"{N} meses previos ({_window_label(p_from, p_to)}, {fmt_int(len(prev_df))} créditos)")
    else:
        window_txt = f"sin {N} meses previos con volumen suficiente para comparar"
with tb2:
    st.markdown(f"<div class='rs-window'>Deltas de tendencia: <b>{esc(window_txt)}</b>.</div>", unsafe_allow_html=True)
preview_slot = tb3.empty()
download_slot = tb4.empty()

# ---------- (a) KPI principales ----------
suffix = f"vs {N} m previos"
if cmp_ok:
    d_n = _delta_rel(rec_s["n"], prev_s["n"], suffix)
    d_alto = _delta_pp(rec_s["pct_alto"], prev_s["pct_alto"], suffix)
    d_mora = _delta_pp(rec_s["pct_mora"], prev_s["pct_mora"], suffix)
    d_monto = _delta_rel(rec_s["monto"], prev_s["monto"], suffix)
    d_malto = _delta_pp(rec_s["pct_monto_alto"], prev_s["pct_monto_alto"], suffix)
else:
    d_n = d_alto = d_mora = d_monto = d_malto = (None, "flat")

mon = _monthly(dff).tail(12)
spark_cap = f"Últimos {len(mon)} meses del filtro"
ink = p["text"]
kpis = [
    {"label": "Créditos filtrados", "value": fmt_int(S["n"]),
     "sub": f"{fmt_int(S['est'])} estudiantes · {fmt_pct(_div(S['n'], len(df_all)))} de la cartera",
     "tone": "ink", "delta": d_n[0], "dir": d_n[1], "spark": mon["n"], "color": ink, "icon": "📄",
     "help": "Número total de créditos que cumplen los filtros activos."},
    {"label": "En riesgo Alto (predicho)", "value": fmt_int(S["n_Alto"]),
     "sub": f"{fmt_pct(S['pct_alto'])} de los créditos filtrados", "tone": "alto", "delta": d_alto[0], "dir": d_alto[1],
     "spark": mon["alto"], "color": RISK_COLORS["Alto"], "icon": "🔴",
     "help": "Cantidad y porcentaje de créditos que el modelo clasifica en riesgo Alto."},
    {"label": "Mora real · Datacrédito", "value": fmt_pct(S["pct_mora"]),
     "sub": f"{fmt_int(S['n_mora'])} créditos con reporte de mora", "tone": "medio", "delta": d_mora[0], "dir": d_mora[1],
     "spark": mon["mora"], "color": RISK_COLORS["Medio"], "icon": "🏦",
     "help": "Porcentaje de créditos con mora reportada por Datacrédito (dato observado, no predicción)."},
    {"label": "Monto financiado total", "value": fmt_cop(S["monto"]),
     "sub": f"Ticket promedio {fmt_cop(S['ticket'])}", "tone": "accent", "delta": d_monto[0], "dir": d_monto[1],
     "spark": mon["monto"], "color": ink, "icon": "💰", "help": "Suma del valor financiado de los créditos filtrados."},
    {"label": "Monto en riesgo Alto", "value": fmt_cop(S["monto_Alto"]),
     "sub": f"{fmt_pct(S['pct_monto_alto'])} del monto financiado", "tone": "alto", "delta": d_malto[0], "dir": d_malto[1],
     "spark": mon["monto_alto"], "color": RISK_COLORS["Alto"], "icon": "⚠️",
     "help": "Valor financiado de los créditos clasificados en riesgo Alto (exposición)."},
]
cols = st.columns(len(kpis), gap="small")
for c_, k in zip(cols, kpis):
    html_ = kpi_card(k["label"], k["value"], k["sub"], tone=k["tone"], delta=k["delta"], delta_dir=k["dir"],
                     icon=k["icon"], help=k["help"])
    c_.markdown(_with_spark(html_, k["spark"], k["color"], spark_cap), unsafe_allow_html=True)

# ---------- (b) Tarjetas por categoría de riesgo ----------
section("Monto financiado por categoría de riesgo", "Número de créditos, participación y dinero comprometido en "
        "cada nivel de riesgo predicho. La barra mide la participación en el monto total.", kicker="Exposición")
rc = st.columns(3, gap="small")
for c_, r in zip(rc, RISK_ORDER):
    c_.markdown(_risk_card(r, S), unsafe_allow_html=True)

# ---------- Hallazgos automáticos ----------
min_n = int(st.session_state.get(f"{P}_minn", 30) or 30)
F = _findings(dff, S, min_n)
section("Hallazgos automáticos", "Se recalculan con cada cambio de filtro; las cifras corresponden a la cartera "
        "filtrada.", kicker="Lectura ejecutiva")
for i in range(0, len(F), 2):
    cc = st.columns(2, gap="small")
    for c_, f in zip(cc, F[i:i + 2]):
        c_.markdown(insight(f["title"], f["html"], tone=f["tone"], icon=f["icon"]), unsafe_allow_html=True)
    if i + 2 < len(F):
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ---------- Dinámica: evolución + dona ----------
section("Evolución de la cartera por riesgo predicho", "Volumen aprobado por periodo y proporción en riesgo Alto. "
        "Pase el cursor para ver el detalle de cada periodo.", kicker="Tendencia")
cc1, cc2 = st.columns([2.15, 1], gap="medium")
with cc1:
    with st.container(border=True):
        h1, h2 = st.columns([1, 1], vertical_alignment="center")
        gran = h1.segmented_control("Granularidad", list(_GRAN), default="Mes", key=f"{P}_gran",
                                    label_visibility="collapsed") or "Mes"
        measure = h2.segmented_control("Medida", ["Créditos", "Monto"], default="Créditos", key=f"{P}_measure",
                                       label_visibility="collapsed") or "Créditos"
        fig_ev, ev_note = _evolution_fig(dff, gran, measure)
        if fig_ev is None:
            empty_state("Sin fechas de aprobación", "El dataset activo no trae fechas válidas para construir la serie.", "🗓️")
        else:
            show_fig(fig_ev, key=f"{P}_evolucion", height=430)
            st.markdown(f"<div class='rs-note'>Franja superior: % de {'monto' if measure == 'Monto' else 'créditos'} en "
                        f"Alto; la línea punteada es el promedio del filtro. {esc(ev_note)}</div>", unsafe_allow_html=True)
with cc2:
    with st.container(border=True):
        money = measure == "Monto"
        vals = [S[f"monto_{r}"] if money else S[f"n_{r}"] for r in RISK_ORDER]
        st.markdown(f"<div class='rs-link-t'>Distribución por riesgo</div><div class='rs-note' style='margin:2px 0 0 0'>"
                    f"{'Monto financiado' if money else 'Créditos'} según riesgo predicho.</div>", unsafe_allow_html=True)
        show_fig(_donut_fig(vals, fmt_cop(S["monto"]) if money else fmt_int(S["n"]),
                            "financiados" if money else "créditos"), key=f"{P}_dona", height=250, legend=False)
        st.markdown(_legend_html(S, money), unsafe_allow_html=True)

# ---------- Exposición ----------
section("Exposición financiera por riesgo", "¿Pesa más el riesgo Alto en dinero que en número de créditos? "
        "Compare la participación en cada medida.", kicker="Exposición")
ec1, ec2 = st.columns([1, 1.15], gap="medium")
with ec1:
    with st.container(border=True):
        expo_m = st.segmented_control("Medida de exposición", ["Monto financiado", "Exposición ponderada"],
                                      default="Monto financiado", key=f"{P}_expo", label_visibility="collapsed") \
            or "Monto financiado"
        key_ = "monto" if expo_m == "Monto financiado" else "expo"
        show_fig(_expo_fig({r: S[f"{key_}_{r}"] for r in RISK_ORDER}, expo_m), key=f"{P}_expo_fig", height=240,
                 legend=False)
        st.markdown("<div class='rs-note'>" + (
            "Valor financiado total de cada categoría de riesgo predicho." if key_ == "monto" else
            "Monto × intensidad de riesgo (0–1, según clase y confianza del modelo): proxy para priorizar, "
            "no es una pérdida esperada.") + "</div>", unsafe_allow_html=True)
with ec2:
    with st.container(border=True):
        st.markdown("<div class='rs-link-t'>Participación por medida</div>", unsafe_allow_html=True)
        show_fig(_mix_fig(S), key=f"{P}_mix", height=250)
        st.markdown(f"<div class='rs-note'>El riesgo Alto es {fmt_pct(S['pct_alto'])} de los créditos y "
                    f"{fmt_pct(S['pct_monto_alto'])} del monto. Si el rojo crece hacia abajo, las alertas se concentran en "
                    f"créditos más grandes o de mayor intensidad.</div>", unsafe_allow_html=True)

# ---------- Top segmentos ----------
section("Dónde se concentra el riesgo", "Segmentos ordenados por % de créditos en riesgo Alto. Lift = % Alto del "
        "segmento ÷ promedio del filtro. Haga clic en una barra para ver su detalle.", kicker="Concentración")
with st.container(border=True):
    k1, k2, k3 = st.columns([3.2, 0.9, 0.9], vertical_alignment="bottom")
    dim_lbl = k1.segmented_control("Dimensión", list(_DIMS), default="Segmento de programa", key=f"{P}_dim") \
        or "Segmento de programa"
    min_n = int(k2.number_input("n mínimo", min_value=1, max_value=2000, value=30, step=5, key=f"{P}_minn",
                                help="Solo se muestran segmentos con al menos este número de créditos."))
    top_n = int(k3.number_input("Top", min_value=3, max_value=25, value=10, step=1, key=f"{P}_topn",
                                help="Número de segmentos a mostrar."))
    dim_col = _DIMS[dim_lbl]
    seg_all = _seg_table(dff, dim_col, min_n, S["pct_alto"])
    seg_top = seg_all.head(top_n)
    if seg_top.empty:
        empty_state("Ningún segmento cumple el n mínimo",
                    f"No hay valores de {dim_lbl.lower()} con al menos {fmt_int(min_n)} créditos en el filtro. "
                    "Reduce el n mínimo.", "🧩")
    else:
        g1, g2 = st.columns([1.75, 1], gap="medium")
        with g1:
            ev = show_fig(_top_fig(seg_top, dim_col, S["pct_alto"]), key=f"{P}_top_{dim_col}",
                          height=max(260, 44 + 34 * len(seg_top)), legend=False, on_select="rerun",
                          selection_mode=("points",))
        options = seg_top["segmento"].tolist()
        pick_key, click_key = f"{P}_seg_{dim_col}", f"_{P}_click_{dim_col}"
        clicked = None
        try:
            pts = (ev or {}).get("selection", {}).get("points", []) if isinstance(ev, dict) else ev.selection.points
            if pts:
                cdp = pts[0].get("customdata")
                clicked = str(cdp[-1]) if cdp is not None else None
        except Exception:
            clicked = None
        if clicked and clicked in options and st.session_state.get(click_key) != clicked:
            st.session_state[pick_key] = clicked
            st.session_state[click_key] = clicked
        if st.session_state.get(pick_key) not in options:
            st.session_state.pop(pick_key, None)
        with g2:
            pick = st.selectbox("Segmento", options, key=pick_key, format_func=lambda v: _seg_label(dim_col, v),
                                label_visibility="collapsed")
            st.markdown(_segment_detail_html(dff, dim_col, pick, S["pct_alto"], dim_lbl), unsafe_allow_html=True)
            if can_access("segmentos"):
                st.page_link(PAGES["segmentos"][0], label="Profundizar en Segmentos y perfiles",
                             icon=":material/arrow_forward:")
        with st.expander(f"Tabla completa · {fmt_int(len(seg_all))} segmentos de {dim_lbl.lower()} con n ≥ {fmt_int(min_n)}"):
            tbl = pd.DataFrame({
                dim_lbl: [_seg_label(dim_col, v) for v in seg_all["segmento"]],
                "Créditos": seg_all["creditos"].astype(int),
                "Estudiantes": seg_all["estudiantes"].astype(int),
                "En Alto": seg_all["n_alto"].astype(int),
                "% Alto": (seg_all["pct_alto"] * 100).round(2),
                "Lift": seg_all["lift"].round(2),
                "% Alto observado": (seg_all["pct_alto_real"] * 100).round(2),
                "% mora Datacrédito": (seg_all["pct_mora"] * 100).round(2),
                "Monto (M COP)": (seg_all["exposicion"] / 1e6).round(1),
                "Monto en Alto (M COP)": (seg_all["exposicion_alto"] / 1e6).round(1),
            })
            vmax = float(tbl["% Alto"].max() or 1)
            st.dataframe(tbl, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(tbl)), column_config={
                "% Alto": st.column_config.ProgressColumn("% Alto", format="%.1f %%", min_value=0,
                                                          max_value=max(vmax, 1)),
                "Lift": st.column_config.NumberColumn("Lift", format="%.2f×"),
                "% Alto observado": st.column_config.NumberColumn(format="%.1f %%"),
                "% mora Datacrédito": st.column_config.NumberColumn(format="%.1f %%"),
                "Monto (M COP)": st.column_config.NumberColumn(format="$ %.1f M"),
                "Monto en Alto (M COP)": st.column_config.NumberColumn(format="$ %.1f M"),
            })
            download_bar(tbl, f"segmentos_{dim_col}", key=f"{P}_seg_dl", label="Exportar")

# ---------- Predicho vs observado ----------
section("Predicho vs. observado", "Qué tanto coincide la clasificación del modelo con el riesgo observado en la "
        "cartera filtrada.", kicker="Confiabilidad")
truth = None
tdf = dff[dff["has_truth"] & dff["y_pred"].notna()]
if tdf.empty:
    st.markdown(insight("Sin riesgo observado en el dataset activo",
                        "El archivo cargado no trae la columna <b>y_true</b>: la concordancia no se puede calcular. "
                        "Los KPI y gráficas anteriores usan solo el riesgo predicho.", tone="info", icon="ℹ️"),
                unsafe_allow_html=True)
else:
    cm = confusion(tdf["y_true"], tdf["y_pred"])
    tp = int(cm[0, 0])
    real_alto, pred_alto = int(cm[0].sum()), int(cm[:, 0].sum())
    acc = _div(np.trace(cm), cm.sum())
    rec = _div(tp, real_alto)
    prec = _div(tp, pred_alto)
    truth = {"n": int(cm.sum()), "acc": acc, "rec": rec, "prec": prec, "tp": tp, "real_alto": real_alto}
    art = load_artifacts() or {}
    thr = ((art.get("alto") or {}).get("rf") or {}).get("umbral_objetivo") or {}
    with st.container(border=True):
        v1, v2, v3 = st.columns([1, 1.25, 1.2], gap="medium")
        with v1:
            w_rec = 0 if _isnan(rec) else min(rec, 1) * 100
            st.markdown(
                f"<div class='rs-mini'>"
                f"<div class='rs-mini-t'><div class='l'>Concordancia (accuracy)</div><div class='v'>{fmt_pct(acc)}</div>"
                f"<div class='s'>{fmt_int(np.trace(cm))} de {fmt_int(cm.sum())} créditos coinciden</div></div>"
                f"<div class='rs-mini-t'><div class='l'>Recall de Alto · meta DE-03 {fmt_pct(RECALL_ALTO_TARGET, 0)}</div>"
                f"<div class='v'>{fmt_pct(rec)}</div>"
                f"<div class='rs-target'><i style='width:{w_rec:.1f}%'></i><b style='left:{RECALL_ALTO_TARGET * 100:.0f}%'></b></div>"
                f"<div class='s'>{fmt_int(tp)} de {fmt_int(real_alto)} Alto observados fueron anticipados</div></div>"
                f"<div class='rs-mini-t'><div class='l'>Precisión de Alto</div><div class='v'>{fmt_pct(prec)}</div>"
                f"<div class='s'>de {fmt_int(pred_alto)} alertas Alto, {fmt_int(tp)} son Alto real</div></div></div>",
                unsafe_allow_html=True)
        with v2:
            st.markdown("<div class='rs-link-t'>Matriz de confusión</div><div class='rs-note' style='margin:2px 0 0 0'>"
                        "Filas: riesgo observado · columnas: predicho · % por fila (recall).</div>", unsafe_allow_html=True)
            show_fig(_confusion_fig(cm), key=f"{P}_confusion", height=300, legend=False)
        with v3:
            thr_txt = ""
            if thr:
                thr_txt = (f" En la muestra de prueba, alertar cuando <b>P(Alto) ≥ {fmt_num(thr.get('umbral', 0.3), 2)}</b> "
                           f"eleva el recall a <b>{fmt_pct(thr.get('recall'))}</b> (precisión {fmt_pct(thr.get('precision'))}), "
                           f"a costa de alertar al {fmt_pct(thr.get('pct_alertas'), 0)} de la cartera.")
            tone = "alto" if (not _isnan(rec) and rec < RECALL_ALTO_TARGET) else "bajo"
            ttl = "La regla argmax subdetecta el riesgo Alto" if tone == "alto" else "Recall de Alto dentro de la meta"
            st.markdown(insight(ttl, f"Clasificar por la clase más probable recupera solo <b>{fmt_pct(rec)}</b> de los Alto "
                                     f"observados en este filtro; la meta DE-03 es {fmt_pct(RECALL_ALTO_TARGET, 0)}.{thr_txt} "
                                     f"Hay {fmt_int(real_alto - tp)} créditos Alto que hoy no reciben alerta.",
                                tone=tone, icon="⚠️" if tone == "alto" else "✅"), unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if can_access("modelo"):
                st.page_link(PAGES["modelo"][0], label="Abrir el simulador de umbral", icon=":material/tune:",
                             width="stretch")
            else:
                st.caption("El simulador de umbral está en «Desempeño del modelo» (roles Dirección y Analítica).")

# ---------- Plan de acción + accesos ----------
section("Siguientes pasos", "Rutas de gestión preventiva sugeridas para la cartera filtrada y accesos directos a "
        "las vistas operativas.", kicker="Acción")
routes = {}
rt = dff["ruta"].astype(str)
for k_ in ACTION_ROUTES:
    m_ = (rt == k_).to_numpy()
    routes[k_] = {"n": int(m_.sum()), "monto": float(dff["valor_financiacion"].fillna(0)[m_].sum())}
rcols = st.columns(4, gap="small")
for c_, (k_, v_) in zip(rcols, routes.items()):
    info = ACTION_ROUTES[k_]
    c_.markdown(
        f"<div class='rs-route' style='--rc:{info['color']}'><div class='rs-route-h'><span class='rs-route-code'>{k_}</span>"
        f"<span class='rs-route-sla'>SLA {esc(info['sla'])}</span></div>"
        f"<div class='rs-route-name'>{esc(info['nombre'])}</div>"
        f"<div class='rs-route-n'>{fmt_int(v_['n'])} <small>créditos · {fmt_pct(_div(v_['n'], S['n']))}</small></div>"
        f"<div class='rs-route-a'>{fmt_cop(v_['monto'])} financiados. {esc(info['accion'])}</div></div>",
        unsafe_allow_html=True)
links = [q for q in _QUICK_LINKS if can_access(q[0])]
if links:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    lcols = st.columns(len(links), gap="small")
    for c_, (k_, ttl, desc) in zip(lcols, links):
        with c_:
            with st.container(border=True):
                st.markdown(f"<div class='rs-link-t'>{esc(ttl)}</div><div class='rs-link-d'>{esc(desc)}</div>",
                            unsafe_allow_html=True)
                st.page_link(PAGES[k_][0], label=PAGES[k_][1], icon=PAGES[k_][2], width="stretch")

# ---------- Reporte ejecutivo (se llena al final, se muestra arriba) ----------
report = {
    "stats": S, "kpis": kpis, "monthly": _monthly(dff), "findings": F, "top": seg_top, "dim_col": dim_col,
    "dim_label": dim_lbl, "min_n": min_n, "truth": truth, "routes": routes, "chips": active_chips(df_all),
    "generated": datetime.now().strftime("%d/%m/%Y %H:%M"), "source": get_active_meta().get("name", ""),
    "period": period_txt, "window": window_txt,
}
doc = _report_html(report)
fname = f"reporte_ejecutivo_SAT_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
if preview_slot.button("Vista previa", icon=":material/visibility:", key=f"{P}_report_preview", width="stretch"):
    _preview_dialog(doc, fname)
download_slot.download_button("Reporte ejecutivo", doc.encode("utf-8"), file_name=fname, mime="text/html",
                              icon=":material/download:", type="primary", key=f"{P}_report_dl", width="stretch",
                              help="HTML autocontenido con KPI, hallazgos, top segmentos, filtros y fecha (ME-03). "
                                   "Se puede imprimir a PDF desde el navegador.")

footer()
