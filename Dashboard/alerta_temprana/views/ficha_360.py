"""Ficha 360° del estudiante — ¿quién es, qué tan riesgoso es su crédito, cómo se compara con sus pares y qué
hacemos hoy con él?

Vista individual para el gestor de Cartera. Reúne en una sola pantalla:

1. Selector con búsqueda y navegación anterior/siguiente en el orden de prioridad (o el que elija el gestor).
2. Perfil anonimizado (sin dirección ni fecha de nacimiento).
3. Riesgo del crédito analizado: confianza del modelo (gauge), ruta, SLA, índice de prioridad descompuesto,
   fiabilidad histórica de la etiqueta y riesgo observado cuando existe.
4. Historial completo de créditos (tabla + línea de tiempo).
5. Comparación con pares (percentil del estudiante frente al programa, segmento, facultad o nivel).
6. Señales del perfil: lift de la tasa histórica de riesgo Alto observado de cada segmento al que pertenece
   (asociaciones de segmento con IC 95 % de Wilson, no causalidad ni explicación individual del modelo).
7. Plan de acción: checklist y guion según la ruta + bitácora de gestiones de la sesión (DB-04 es la integración
   pendiente con el sistema de cartera).
8. Exportación de la ficha como HTML autocontenido (y de sus tablas en CSV/Excel, DB-02).
"""
from __future__ import annotations

import calendar
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import (badge, download_bar, empty_state, esc, filter_chips, footer, insight, insight_row,
                             kpi_card, kpi_row, page_header, risk_badge)
from core.config import (ACTION_ROUTES, APP_NAME, MAX_FINANCIACION_PCT, PROGRAMA_ACADEMICO, RISK_BANDS_TEXT,
                         RISK_COLORS, RISK_ICONS, RISK_ORDER, SCORE_ORDER, SCORE_SHORT, UNIVERSIDAD)
from core.data import DEFAULT_WEIGHTS, get_active_df, get_active_meta, load_base
from core.filters import active_chips, apply_filters
from core.metrics import load_artifacts
from core.nav import FICHA_KEY, PAGES, can_access
from core.theme import MESES_CORTOS, YELLOW, fmt_cop, fmt_date, fmt_int, fmt_num, fmt_pct, fmt_period, pal, \
    show_fig, theme_mode

ss = st.session_state

# ============================================================================================
# Constantes de la página
# ============================================================================================
SEL_KEY = "ficha_sel"                 # widget: estudiante seleccionado
CONSUMED_KEY = "_ficha_consumed"      # último id de FICHA_KEY ya aplicado al selector
FORCE_KEY = "_ficha_force"            # estudiante enviado desde la Cola que no cumple los filtros
LOG_KEY = "_ficha_bitacora"           # bitácora de gestiones de la sesión (lista de dicts)

ORDER_MODES = ["Prioridad · crédito más reciente", "Prioridad máxima del estudiante", "Exposición en riesgo",
               "Aprobación más reciente", "Nombre (A–Z)"]
ORDER_HELP = ("Define el orden de los botones anterior/siguiente y el crédito de referencia de cada estudiante: "
              "el más reciente dentro del filtro o, con «Prioridad máxima», el de mayor índice de prioridad.")
PEER_GROUPS = {"Programa": "programa", "Segmento": "programa_cluster", "Facultad": "facultad", "Nivel": "nivel",
               "Toda la base": None}
MIN_PEERS = 30          # mínimo de créditos para que un grupo de pares sea un referente estable
MIN_SIGNAL_N = 30       # mínimo de créditos para leer una señal de segmento

ROUTE_SHORT = {"R1": "Inmediato", "R2": "Preventivo", "R3": "Automático", "R4": "Monitoreo"}
ROUTE_TONE = {"R1": "alto", "R2": "medio", "R3": "accent", "R4": "bajo"}
ROUTE_CANAL = {"R1": "Llamada", "R2": "WhatsApp", "R3": "Mensaje automático", "R4": "Correo electrónico"}
CANALES = ["Llamada", "WhatsApp", "SMS", "Correo electrónico", "Presencial", "Mensaje automático"]
RESULTADOS = ["Contactado · con compromiso de pago", "Contactado · sin compromiso", "Contactado · ya pagó",
              "No contesta / buzón de voz", "Dato de contacto errado", "Remitido a consejería financiera",
              "Solicita ajuste de fecha o cuotas", "Otro"]

CHECKLISTS = {
    "R1": ["Revisar esta ficha y el historial de créditos antes de contactar.",
           "Llamar al estudiante (hasta 3 intentos en 48 h, en franjas horarias distintas).",
           "Explicar el estado del crédito y la cuota de referencia ({cuota}, día {dia} de cada mes).",
           "Acordar un compromiso de pago preventivo (monto y fecha).",
           "Ofrecer consejería financiera o alternativas de fecha de pago.",
           "Registrar la gestión y el compromiso en la bitácora."],
    "R2": ["Enviar WhatsApp/SMS 5 días antes de la fecha de pago (día {dia}).",
           "Enviar correo con el valor de la cuota ({cuota}) y los canales de pago.",
           "Ofrecer una cita de consejería financiera gratuita.",
           "Si no responde en 48 h, escalar a llamada del gestor (ruta R1).",
           "Registrar la gestión en la bitácora."],
    "R3": ["Verificar que el celular y el correo del estudiante estén vigentes.",
           "Confirmar que el recordatorio automático quedó programado 3 días antes del día {dia}.",
           "Revisar el pago tras el vencimiento; si no se registra, escalar a la ruta R2."],
    "R4": ["Mantener el seguimiento mensual en el corte de cartera.",
           "Revisar cambios de riesgo en la próxima puntuación del modelo.",
           "Sin contacto adicional salvo novedad reportada por el estudiante."],
}
GUIONES = {
    "R1": ("Hola, {nombre}. Te habla {gestor} del equipo de Crédito Estudiantil. Te llamamos para acompañarte antes "
           "de tu próxima cuota (día {dia} de cada mes) y acordar contigo un plan de pago preventivo que se ajuste a "
           "tu situación. ¿Tienes unos minutos?"),
    "R2": ("Hola, {nombre} 👋 Te recordamos que tu cuota del crédito educativo vence el día {dia}. Si necesitas "
           "orientación, agenda una consejería financiera gratuita respondiendo este mensaje. — {gestor}, Crédito "
           "Estudiantil"),
    "R3": ("Recordatorio: {nombre}, tu cuota del crédito educativo vence en 3 días (día {dia}). Paga a tiempo por "
           "los canales habituales y evita recargos."),
    "R4": "Seguimiento estándar mensual: sin contacto adicional para {nombre}.",
}
CANAL_GUION = {"R1": "Llamada del gestor", "R2": "WhatsApp / SMS + correo", "R3": "Mensaje automático",
               "R4": "Sin contacto adicional"}

# Métricas de la comparación con pares: (columna, etiqueta, tipo de formato)
PEER_METRICS = [
    ("valor_financiacion", "Valor financiado", "cop"),
    ("vr_neto_matricula", "Matrícula neta", "cop"),
    ("ratio_financiacion", "% financiado de la matrícula", "pct"),
    ("valor_primera_cuota", "Primera cuota", "cop"),
    ("cuotas", "Número de cuotas", "int"),
    ("score_rank", "Scoring externo (rango)", "score"),
    ("antiguedad_meses", "Antigüedad del crédito (meses)", "int"),
]
SIGNAL_ORDER = ["Tipo de interés", "Número de cuotas", "Scoring externo", "Antigüedad del crédito",
                "Mora en Datacrédito", "Segmento de programa", "Tipo de estudiante", "Nivel académico",
                "Rango de edad", "Financiación / matrícula"]
_LOWER_WORDS = {"de", "del", "la", "las", "los", "y", "e", "en", "el"}


# ============================================================================================
# Utilidades de formato y color
# ============================================================================================
def _dark() -> bool:
    return theme_mode() == "dark"


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _nb(text) -> str:
    return str(text).replace(" ", "\u00a0")


def _is_na(x) -> bool:
    try:
        return x is None or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _txt(x, default: str = "Sin dato") -> str:
    if _is_na(x):
        return default
    t = str(x).strip()
    return default if not t or t.lower() in {"nan", "none", "<na>"} else t


def _cap(x) -> str:
    t = _txt(x)
    if t == "Sin dato":
        return t
    return (t[:1].upper() + t[1:].lower()).replace(" (a)", "(a)")


def _title(x) -> str:
    t = _txt(x)
    if t == "Sin dato":
        return t
    if t.lower().startswith("bogota") or t.lower().startswith("bogotá"):
        return "Bogotá D.C."
    words = t.lower().split()
    return " ".join(w if (i and w in _LOWER_WORDS) else w[:1].upper() + w[1:] for i, w in enumerate(words))


def _initials(name: str) -> str:
    parts = [p for p in str(name).split() if p]
    return ("".join(p[0] for p in parts[:2]) or "·").upper()


def _risk(x) -> str:
    t = _txt(x, "")
    return t if t in RISK_ORDER else "Sin dato"


def _risk_text_color(r: str) -> str:
    """Color de texto legible para cada nivel de riesgo en el tema actual."""
    dark = _dark()
    return {"Alto": "#FF6B70" if dark else "#D13438", "Medio": "#FFB547" if dark else "#A8650A",
            "Bajo": "#46C487" if dark else "#218358"}.get(r, "inherit")


def _rc(r: str) -> str:
    """Color de marca de la ruta."""
    if r == "R3" and _dark():
        return "#7C9CFF"
    return ACTION_ROUTES.get(r, {}).get("color", "#888888")


def _ri(r: str) -> str:
    """Color de texto legible de la ruta."""
    dark = _dark()
    return {"R1": "#FF6B70" if dark else "#D13438", "R2": "#FFB547" if dark else "#A8650A",
            "R3": "#8FA8FF" if dark else "#3E63DD", "R4": "#46C487" if dark else "#218358"}.get(r, "inherit")


def _route_chip(r: str, with_name: bool = True) -> str:
    a = ACTION_ROUTES.get(r, {})
    label = f"{r} · {a.get('nombre', '')}" if with_name else f"{r} · {ROUTE_SHORT.get(r, r)}"
    return (f"<span class='sat-badge' style='color:{_ri(r)};background:{_rgba(_rc(r), .14)};"
            f"border-color:{_rgba(_rc(r), .4)}'>{esc(label)}</span>")


def _es_x(x: float, d: int = 2) -> str:
    return f"×{fmt_num(x, d)}"


def _fmt_metric(v, kind: str) -> str:
    if _is_na(v):
        return "—"
    if kind == "cop":
        return fmt_cop(v)
    if kind == "pct":
        return fmt_pct(v, 0)
    if kind == "score":
        i = int(round(float(v)))
        return SCORE_SHORT.get(SCORE_ORDER[i], "—") if 0 <= i < len(SCORE_ORDER) else "—"
    return fmt_int(v)


def _period_label(ts) -> str:
    if _is_na(ts):
        return "—"
    return f"{MESES_CORTOS[ts.month - 1]} {ts.year}"


def _anchor_section(anchor: str, title: str, subtitle: str, kicker: str) -> None:
    st.markdown(
        f"""<div class="sat-section fx-anchor" id="{anchor}"><div class="kicker">{esc(kicker)}</div>
        <h3>{esc(title)}</h3><p>{esc(subtitle)}</p></div>""",
        unsafe_allow_html=True,
    )


def _note(html_text: str) -> None:
    st.markdown(f"<div class='fx-note'>{html_text}</div>", unsafe_allow_html=True)


def _date_ticks(start: pd.Timestamp, end: pd.Timestamp, max_ticks: int = 8) -> tuple[list, list]:
    """Marcas de fecha en español (evita los meses en inglés de Plotly)."""
    months = max(1, (end.year - start.year) * 12 + end.month - start.month)
    step = next(s for s in (1, 2, 3, 6, 12, 24) if months / s <= max_ticks)
    first = pd.Timestamp(year=start.year, month=start.month, day=1)
    if step >= 3:
        m0 = ((first.month - 1) // step) * step + 1 if step < 12 else 1
        first = pd.Timestamp(year=first.year, month=m0, day=1)
    vals, cur = [], first
    while cur <= end + pd.DateOffset(months=1):
        if cur >= start - pd.DateOffset(days=20):
            vals.append(cur)
        cur = cur + pd.DateOffset(months=step)
    return vals, [f"{MESES_CORTOS[v.month - 1]} {v.year}" for v in vals]


def _add_business_days(d: date, n: int) -> date:
    return pd.Timestamp(np.busday_offset(np.datetime64(d), n, roll="forward")).date()


def _next_payment(day: int, today: date) -> date:
    for k in range(3):
        mm = (today.month - 1 + k) % 12 + 1
        yy = today.year + (today.month - 1 + k) // 12
        d = date(yy, mm, min(int(day), calendar.monthrange(yy, mm)[1]))
        if d >= today:
            return d
    return today


def _wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def _pctile(values: np.ndarray, v) -> float:
    """Percentil (rango medio) de ``v`` dentro de ``values`` — robusto a empates (p. ej. cuotas)."""
    if _is_na(v):
        return np.nan
    x = values[~np.isnan(values)]
    if len(x) == 0:
        return np.nan
    return float(100 * (np.sum(x < v) + 0.5 * np.sum(x == v)) / len(x))


def _data_token(df: pd.DataFrame) -> str:
    meta = get_active_meta()
    s = float(pd.to_numeric(df["prioridad"], errors="coerce").fillna(0).sum()) if len(df) else 0.0
    return f"{meta.get('source')}|{meta.get('name')}|{len(df)}|{s:.3f}"


# ============================================================================================
# Datos: estudiantes, señales y pares
# ============================================================================================
def _students(d: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Una fila por estudiante con su crédito de referencia dentro del filtro, en el orden de navegación."""
    d = d.drop_duplicates("llave2")
    if mode == "Prioridad máxima del estudiante":
        s = d.sort_values(["prioridad", "fecha_aprobacion"], ascending=[False, False], na_position="last")
    else:
        s = d.sort_values(["fecha_aprobacion", "prioridad"], ascending=[False, False], na_position="last")
    ref = s.drop_duplicates("id_estudiante", keep="first").copy()
    agg = d.groupby("id_estudiante", observed=True).agg(n_filtro=("llave2", "size"),
                                                         expo_total=("exposicion_riesgo", "sum"))
    ref = ref.join(agg, on="id_estudiante")
    if mode.startswith("Prioridad"):
        ref = ref.sort_values(["prioridad", "exposicion_riesgo", "id_estudiante"], ascending=[False, False, True],
                              na_position="last")
    elif mode == "Exposición en riesgo":
        ref = ref.sort_values(["expo_total", "prioridad"], ascending=[False, False], na_position="last")
    elif mode == "Aprobación más reciente":
        ref = ref.sort_values(["fecha_aprobacion", "prioridad"], ascending=[False, False], na_position="last")
    else:
        ref = ref.sort_values(["nombre", "id_estudiante"])
    return ref.reset_index(drop=True)


def _sig_values(d: pd.DataFrame) -> pd.DataFrame:
    """Valor de cada atributo de señal por fila (mismo cálculo para la base y para el estudiante)."""
    cuotas = pd.to_numeric(d["cuotas"], errors="coerce")
    ant = pd.to_numeric(d["antiguedad_meses"], errors="coerce")
    ratio = pd.to_numeric(d["ratio_financiacion"], errors="coerce")
    out = pd.DataFrame(index=d.index)
    out["Tipo de interés"] = d["tipo_interes"].astype(str)
    out["Número de cuotas"] = np.where(cuotas.notna(), cuotas.fillna(0).round().astype(int).astype(str) + " cuotas",
                                       "Sin dato")
    out["Scoring externo"] = d["score_corto"].astype(str)
    out["Antigüedad del crédito"] = np.where(ant.isna(), "Sin dato", np.where(ant < 12, "< 12 meses", "≥ 12 meses"))
    out["Mora en Datacrédito"] = d["mora_txt"].astype(str)
    out["Segmento de programa"] = d["programa_cluster"].astype(str)
    out["Tipo de estudiante"] = d["tipo_estudiante"].astype(str)
    out["Nivel académico"] = d["nivel"].astype(str).str.capitalize()
    out["Rango de edad"] = d["rango_edad"].astype(str)
    out["Financiación / matrícula"] = np.where(ratio.isna(), "Sin dato",
                                               np.where(ratio > MAX_FINANCIACION_PCT, "> 85 % (DA-02)", "≤ 85 %"))
    return out


@st.cache_data(show_spinner=False, max_entries=6)
def _reference_tables(token: str, _ref: pd.DataFrame) -> dict:
    """Tasas históricas de Alto observado por atributo + fiabilidad de cada etiqueta predicha y confianza media."""
    y = (_ref["y_true"].astype(str) == "Alto").to_numpy()
    sv = _sig_values(_ref)
    tables = {}
    for col in sv.columns:
        g = pd.DataFrame({"v": sv[col].to_numpy(), "y": y}).groupby("v")["y"].agg(["sum", "size"])
        tables[col] = g
    yp = _ref["y_pred"].astype(str)
    yt = _ref["y_true"].astype(str)
    precision = {r: float((yt[yp == r] == r).mean()) if (yp == r).any() else np.nan for r in RISK_ORDER}
    support = {r: int((yp == r).sum()) for r in RISK_ORDER}
    return {"tables": tables, "base": float(y.mean()) if len(y) else np.nan, "n": int(len(y)),
            "precision": precision, "support": support}


def _signals(rec_df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    base = ref["base"]
    sv = _sig_values(rec_df).iloc[0]
    rows = []
    for attr in SIGNAL_ORDER:
        val = str(sv[attr])
        tb = ref["tables"].get(attr)
        if tb is None or val == "Sin dato" or val not in tb.index:
            continue
        k, n = float(tb.loc[val, "sum"]), float(tb.loc[val, "size"])
        if n <= 0 or not base or _is_na(base):
            continue
        rate = k / n
        lo, hi = _wilson(k, n)
        small = n < MIN_SIGNAL_N
        sig = (not small) and (lo > base or hi < base)
        rows.append({"atributo": attr, "valor": val, "n": int(n), "altos": int(k), "tasa": rate, "ic_lo": lo,
                     "ic_hi": hi, "lift": rate / base, "lift_lo": lo / base, "lift_hi": hi / base,
                     "significativa": sig, "muestra_pequena": small})
    return pd.DataFrame(rows)


def _peer_table(rec: pd.Series, peers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label, kind in PEER_METRICS:
        v = rec.get(col)
        x = pd.to_numeric(peers[col], errors="coerce").to_numpy(dtype=float) if col in peers else np.array([])
        x = x[~np.isnan(x)]
        if _is_na(v) or len(x) == 0:
            continue
        v = float(v)
        med, q1, q3 = np.median(x), np.percentile(x, 25), np.percentile(x, 75)
        rows.append({"col": col, "metrica": label, "kind": kind, "valor": v, "mediana": med, "p25": q1, "p75": q3,
                     "percentil": _pctile(x, v), "n": int(len(x)),
                     "dif_pct": (v / med - 1) if med not in (0, 0.0) else np.nan})
    return pd.DataFrame(rows)


# ============================================================================================
# Figuras
# ============================================================================================
def _fig_gauge(conf: float, color: str, class_mean: float | None) -> go.Figure:
    p = pal()
    gauge = dict(
        shape="angular",
        axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], tickwidth=1, tickcolor=p["border"],
                  tickfont=dict(color=p["muted"], size=10), ticksuffix=""),
        bar=dict(color=color, thickness=0.32),
        bgcolor=p["surface_2"], borderwidth=1, bordercolor=p["border"],
        steps=[dict(range=[0, 50], color=p["surface_2"]), dict(range=[50, 75], color=_rgba(color, .07)),
               dict(range=[75, 100], color=_rgba(color, .13))],
    )
    if class_mean is not None and not _is_na(class_mean):
        gauge["threshold"] = dict(line=dict(color=p["text"], width=3), thickness=0.9, value=class_mean * 100)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=conf * 100, gauge=gauge, domain=dict(x=[0, 1], y=[0, 1]),
        number=dict(suffix=" %", valueformat=".0f",
                    font=dict(size=38, color=p["text"], family="Space Grotesk, Inter, sans-serif")),
    ))
    fig.update_layout(margin=dict(l=28, r=28, t=18, b=4))
    return fig


def _fig_timeline(cr: pd.DataFrame, sel_llave: str) -> go.Figure:
    p = pal()
    fig = go.Figure()
    c = cr.dropna(subset=["fecha_aprobacion"]).copy()
    c["_fin"] = [f + pd.DateOffset(months=int(n)) if not _is_na(n) and n > 0 else f
                 for f, n in zip(c["fecha_aprobacion"], c["cuotas"])]
    # Plazo estimado de cada plan (supone cuotas mensuales).
    first_seg = True
    for _, r in c.iterrows():
        if r["_fin"] <= r["fecha_aprobacion"]:
            continue
        col = RISK_COLORS.get(_risk(r["y_pred"]), p["subtle"])
        fig.add_trace(go.Scatter(
            x=[r["fecha_aprobacion"], r["_fin"]], y=[r["valor_financiacion"]] * 2, mode="lines",
            line=dict(color=_rgba(col, .38), width=5), hoverinfo="skip", legendgroup="plazo",
            name="Plazo estimado (cuotas mensuales)", showlegend=first_seg))
        first_seg = False
    for rk in RISK_ORDER + ["Sin dato"]:
        s = c[c["y_pred"].astype(str).map(_risk) == rk]
        if s.empty:
            continue
        cd = np.stack([
            s["llave2"].astype(str),
            [fmt_date(f) for f in s["fecha_aprobacion"]],
            [fmt_cop(v, compact=False) for v in s["valor_financiacion"]],
            [fmt_int(v) if not _is_na(v) else "—" for v in s["cuotas"]],
            s["tipo_interes"].astype(str),
            [fmt_pct(v, 0) for v in s["proba_pred"]],
            [_risk(v) if not _is_na(v) else "—" for v in s["y_true"]],
            [_period_label(f) for f in s["_fin"]],
            s["ruta"].astype(str),
        ], axis=-1)
        fig.add_trace(go.Scatter(
            x=s["fecha_aprobacion"], y=s["valor_financiacion"], mode="markers", name=f"Riesgo {rk}",
            marker=dict(color=RISK_COLORS.get(rk, p["subtle"]), size=15, line=dict(color=p["surface"], width=2)),
            customdata=cd,
            hovertemplate=("<b>Crédito %{customdata[0]}</b><br>Aprobado: %{customdata[1]}<br>"
                           "Valor financiado: %{customdata[2]}<br>Cuotas: %{customdata[3]} · %{customdata[4]}<br>"
                           "Riesgo predicho: " + rk + " (confianza %{customdata[5]})<br>"
                           "Riesgo observado: %{customdata[6]}<br>Ruta: %{customdata[8]}<br>"
                           "Fin estimado del plan: %{customdata[7]}<extra></extra>")))
    sel = c[c["llave2"] == sel_llave]
    if not sel.empty:
        fig.add_trace(go.Scatter(
            x=sel["fecha_aprobacion"], y=sel["valor_financiacion"], mode="markers", name="Crédito analizado",
            marker=dict(size=30, color="rgba(0,0,0,0)", line=dict(color=YELLOW, width=3)), hoverinfo="skip"))
    for _, r in c.iterrows():
        is_sel = r["llave2"] == sel_llave
        fig.add_annotation(x=r["fecha_aprobacion"], y=r["valor_financiacion"], text=f"<b>{fmt_cop(r['valor_financiacion'])}</b>"
                           if is_sel else fmt_cop(r["valor_financiacion"]), showarrow=False, yshift=24 if is_sel else 18,
                           font=dict(size=11.5, color=p["text"]))
    if not c.empty:
        x0 = c["fecha_aprobacion"].min() - pd.DateOffset(months=2)
        x1 = max(c["_fin"].max(), c["fecha_aprobacion"].max()) + pd.DateOffset(months=2)
        tv, tt = _date_ticks(x0, x1)
        ymax = float(c["valor_financiacion"].max() or 1)
        fig.update_xaxes(range=[x0, x1], tickvals=tv, ticktext=tt, title_text="Fecha de aprobación")
        fig.update_yaxes(range=[0, ymax * 1.32], tickprefix="$ ", tickformat="~s", title_text="Valor financiado",
                         rangemode="tozero")
    fig.update_layout(margin=dict(l=8, r=12, t=48, b=8), hovermode="closest")
    return fig


def _fig_peers(pt: pd.DataFrame) -> go.Figure:
    p = pal()
    d = pt.iloc[::-1].reset_index(drop=True)
    labels = d["metrica"].tolist()
    fig = go.Figure()
    fig.add_vrect(x0=25, x1=75, fillcolor=_rgba(YELLOW, .10 if _dark() else .16), line_width=0, layer="below")
    fig.add_vline(x=50, line=dict(color=p["muted"], width=1.4, dash="dot"))
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(x=[50, r["percentil"]], y=[r["metrica"]] * 2, mode="lines",
                                 line=dict(color=p["muted"], width=2), hoverinfo="skip", showlegend=False))
    outlier = (d["percentil"] >= 90) | (d["percentil"] <= 10)
    colors = [YELLOW if o else p["text"] for o in outlier]
    texts = [f"<b>p{fmt_num(v, 0)}</b> · {_fmt_metric(val, k)}" for v, val, k in zip(d["percentil"], d["valor"],
                                                                                     d["kind"])]
    pos = ["top center"] * len(d)
    cd = np.stack([
        [_fmt_metric(v, k) for v, k in zip(d["valor"], d["kind"])],
        [_fmt_metric(v, k) for v, k in zip(d["mediana"], d["kind"])],
        [f"{_fmt_metric(a, k)} – {_fmt_metric(b, k)}" for a, b, k in zip(d["p25"], d["p75"], d["kind"])],
        [fmt_num(v, 0) for v in d["percentil"]],
        [fmt_int(v) for v in d["n"]],
    ], axis=-1)
    fig.add_trace(go.Scatter(
        x=d["percentil"], y=labels, mode="markers+text", text=texts, textposition=pos,
        textfont=dict(color=p["text"], size=12), showlegend=False, cliponaxis=False,
        marker=dict(size=15, color=colors, line=dict(color=p["text"], width=1.5)), customdata=cd,
        hovertemplate=("<b>%{y}</b><br>Estudiante: %{customdata[0]} · percentil %{customdata[3]}<br>"
                       "Mediana de pares: %{customdata[1]}<br>Rango típico (p25–p75): %{customdata[2]}<br>"
                       "Pares con dato: %{customdata[4]}<extra></extra>")))
    fig.add_annotation(x=50, y=1.0, yref="paper", yanchor="bottom", text="Mediana de pares", showarrow=False,
                       font=dict(size=11, color=p["muted"]))
    fig.add_annotation(x=25, y=1.0, yref="paper", yanchor="bottom", xanchor="left", text="p25", showarrow=False,
                       font=dict(size=10, color=p["subtle"]))
    fig.add_annotation(x=75, y=1.0, yref="paper", yanchor="bottom", xanchor="right", text="p75", showarrow=False,
                       font=dict(size=10, color=p["subtle"]))
    fig.update_xaxes(range=[-3, 103], tickvals=[0, 10, 25, 50, 75, 90, 100],
                     ticktext=["0", "10", "25", "50", "75", "90", "100"], showgrid=True, gridcolor=p["grid"],
                     title_text="Percentil del estudiante dentro de sus pares")
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=labels, automargin=True)
    fig.update_layout(margin=dict(l=8, r=16, t=30, b=8))
    return fig


def _fig_signals(sig: pd.DataFrame, base: float) -> go.Figure:
    p = pal()
    d = sig.sort_values("lift").reset_index(drop=True)
    labels = [f"{a} · <b>{esc(v)}</b>" for a, v in zip(d["atributo"], d["valor"])]
    x = (d["lift"] - 1) * 100
    colors = []
    for _, r in d.iterrows():
        if r["muestra_pequena"]:
            colors.append(_rgba("#8A8A80", .45))
            continue
        base_c = RISK_COLORS["Alto"] if r["lift"] >= 1 else RISK_COLORS["Bajo"]
        colors.append(base_c if r["significativa"] else _rgba(base_c, .42))
    cd = np.stack([
        [fmt_pct(v) for v in d["tasa"]], [fmt_int(v) for v in d["n"]], [_es_x(v) for v in d["lift"]],
        [f"{_es_x(a)} – {_es_x(b)}" for a, b in zip(d["lift_lo"], d["lift_hi"])],
        ["Sí (el IC 95 % excluye el promedio)" if s else ("Muestra pequeña (n < 30)" if m else "No (el IC 95 % "
                                                                                                "incluye el promedio)")
         for s, m in zip(d["significativa"], d["muestra_pequena"])],
        [fmt_int(v) for v in d["altos"]],
    ], axis=-1)
    fig = go.Figure(go.Bar(
        x=x, y=labels, orientation="h", marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        error_x=dict(type="data", array=(d["lift_hi"] - d["lift"]) * 100, arrayminus=(d["lift"] - d["lift_lo"]) * 100,
                     color=p["muted"], thickness=1.2, width=4),
        customdata=cd, width=0.62,
        hovertemplate=("<b>%{y}</b><br>Tasa de Alto observado: %{customdata[0]} (%{customdata[5]} de "
                       "%{customdata[1]} créditos)<br>Promedio de la base: " + fmt_pct(base) +
                       "<br>Lift: %{customdata[2]} · IC 95 %: %{customdata[3]}<br>Diferencia significativa: "
                       "%{customdata[4]}<extra></extra>")))
    fig.add_vline(x=0, line=dict(color=p["text"], width=1.5))
    for i, r in d.iterrows():
        fig.add_annotation(x=1.0, xref="paper", xanchor="left", y=labels[i], showarrow=False, xshift=10,
                           text=f"<b>{_es_x(r['lift'])}</b>  <span style='color:{p['muted']}'>{fmt_pct(r['tasa'])}</span>",
                           font=dict(size=12, color=p["text"]), align="left")
    fig.add_annotation(x=1.0, xref="paper", xanchor="left", y=1.0, yref="paper", yanchor="bottom", xshift=10,
                       showarrow=False, text="Lift · tasa", font=dict(size=11, color=p["muted"]))
    fig.add_annotation(x=0, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"Promedio de la base · {fmt_pct(base)}", font=dict(size=11, color=p["muted"]))
    lim = float(np.nanmax(np.abs(np.r_[(d["lift_hi"] - 1) * 100, (d["lift_lo"] - 1) * 100, 10]))) * 1.12
    fig.update_xaxes(range=[-lim, lim], ticksuffix=" %", zeroline=False, showgrid=True, gridcolor=p["grid"],
                     title_text="Diferencia de la tasa de Alto observado frente al promedio de la base")
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_layout(margin=dict(l=8, r=118, t=30, b=8), bargap=0.35)
    return fig


# ============================================================================================
# Exportación HTML autocontenida
# ============================================================================================
def _ficha_html(ctx: dict) -> str:
    rec = ctx["rec"]
    rk = ctx["rk"]
    rcol = RISK_COLORS.get(rk, "#8A8A80")
    route = ctx["route"]
    a = ACTION_ROUTES.get(route, {})
    conf = ctx["conf"]
    css = """
    *{box-sizing:border-box}body{margin:0;background:#F6F6F3;color:#141414;font-family:Inter,'Segoe UI',Roboto,Arial,
    sans-serif;font-size:13.5px;line-height:1.45}main{max-width:1040px;margin:0 auto;padding:28px 22px 40px}
    .hero{background:linear-gradient(120deg,#0B0B0C,#26251F);color:#F7F7F2;border-radius:18px;padding:22px 26px}
    .eyebrow{color:#FFD100;font-size:11px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}
    .hero h1{margin:6px 0 4px;font-size:26px;letter-spacing:-.02em}.hero p{margin:0;color:#CFCFC6}
    .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}.chip{font-size:11.5px;padding:3px 10px;
    border-radius:999px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);color:#EDEDE6}
    .chip.hl{background:#FFD100;color:#141414;border-color:#FFD100;font-weight:700}
    h2{font-size:16px;margin:26px 0 10px;display:flex;align-items:center;gap:8px}h2:before{content:"";width:16px;
    height:3px;border-radius:3px;background:#FFD100}.card{background:#fff;border:1px solid #E7E5DF;border-radius:14px;
    padding:14px 16px}.grid2{display:grid;grid-template-columns:1.3fr 1fr;gap:14px}.kv{display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));gap:10px 14px}.k{font-size:10px;text-transform:uppercase;
    letter-spacing:.08em;color:#9A988F;font-weight:700}.v{font-weight:600}.risk{display:flex;gap:16px;align-items:center}
    .ring{width:118px;height:118px;border-radius:50%;display:grid;place-items:center;flex:0 0 auto}
    .ring span{width:86px;height:86px;border-radius:50%;background:#fff;display:grid;place-items:center;font-size:22px;
    font-weight:800}.badge{display:inline-block;font-size:11.5px;font-weight:700;padding:2px 9px;border-radius:999px;
    border:1px solid}.muted{color:#6B6A64}table{width:100%;border-collapse:collapse;font-size:12.5px}th{text-align:left;
    font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:#6B6A64;border-bottom:1px solid #E7E5DF;
    padding:6px 8px}td{padding:6px 8px;border-bottom:1px solid #F0EEE8;vertical-align:middle}.track{position:relative;
    height:10px;background:#F3F1EB;border-radius:6px;min-width:140px}.track .band{position:absolute;left:25%;width:50%;
    top:0;bottom:0;background:rgba(255,209,0,.28)}.track .mid{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;
    background:#6B6A64}.track .dot{position:absolute;top:50%;width:12px;height:12px;border-radius:50%;
    transform:translate(-50%,-50%);border:1.5px solid #141414}.div{position:relative;height:10px;min-width:150px}
    .div .axis{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:#141414}.div i{position:absolute;
    top:0;bottom:0;border-radius:3px}ul.chk{list-style:none;padding:0;margin:6px 0}ul.chk li{padding:3px 0}
    .quote{border-left:3px solid #FFD100;padding:4px 0 4px 10px;font-style:italic}.foot{margin-top:28px;font-size:11px;
    color:#9A988F;border-top:1px solid #E7E5DF;padding-top:10px}@media print{body{background:#fff}main{padding:0}
    .card{break-inside:avoid}}@media (max-width:760px){.grid2{grid-template-columns:1fr}.kv{grid-template-columns:
    repeat(2,minmax(0,1fr))}}
    """

    def b(level: str) -> str:
        c = RISK_COLORS.get(level, "#8A8A80")
        return (f"<span class='badge' style='color:{c};background:{_rgba(c, .1)};border-color:{_rgba(c, .4)}'>"
                f"{esc(RISK_ICONS.get(level, '⚪'))} {esc(level)}</span>")

    prof = "".join(f"<div><div class='k'>{esc(k)}</div><div class='v'>{esc(v)}</div></div>" for k, v in ctx["profile"])
    conf_txt = fmt_pct(conf, 0) if not _is_na(conf) else "—"
    ring_p = 0 if _is_na(conf) else conf * 100
    yt = ctx["yt"]
    obs = b(yt) if yt in RISK_ORDER else "<span class='muted'>Sin riesgo observado</span>"
    risk_html = (
        f"<div class='risk'><div class='ring' style='background:conic-gradient({rcol} {ring_p:.1f}%, #EEECE6 0)'>"
        f"<span>{conf_txt}</span></div><div><div class='k'>Riesgo predicho (confianza del modelo)</div>"
        f"<div style='margin:3px 0 8px'>{b(rk)}</div><div class='k'>Ruta y SLA</div><div class='v'>{esc(route)} · "
        f"{esc(a.get('nombre', ''))} · SLA {esc(a.get('sla', ''))}</div><div class='k' style='margin-top:6px'>"
        f"Índice de prioridad</div><div class='v'>{fmt_num(rec.get('prioridad'), 1)} / 100 · {esc(ctx['rank_txt'])}"
        f"</div><div class='k' style='margin-top:6px'>Riesgo observado (validación histórica)</div>"
        f"<div>{obs}</div></div></div>")

    cr_rows = "".join(
        f"<tr><td>{'◆ ' if r['llave2'] == ctx['sel_llave'] else ''}{esc(fmt_date(r['fecha_aprobacion']))}</td>"
        f"<td>{esc(r['llave2'])}</td><td>{esc(r['programa'])}</td><td>{esc(fmt_cop(r['valor_financiacion'], False))}</td>"
        f"<td>{esc(fmt_int(r['cuotas']) if not _is_na(r['cuotas']) else '—')}</td><td>{b(_risk(r['y_pred']))} "
        f"<span class='muted'>{esc(fmt_pct(r['proba_pred'], 0))}</span></td>"
        f"<td>{b(_risk(r['y_true'])) if _risk(r['y_true']) in RISK_ORDER else '—'}</td>"
        f"<td>{esc(fmt_num(r['prioridad'], 1))}</td><td>{esc(r['ruta'])}</td></tr>"
        for _, r in ctx["credits"].iterrows())

    pt = ctx["peers"]
    peer_rows = "".join(
        f"<tr><td>{esc(r['metrica'])}</td><td><b>{esc(_fmt_metric(r['valor'], r['kind']))}</b></td>"
        f"<td>{esc(_fmt_metric(r['mediana'], r['kind']))}</td><td>p{esc(fmt_num(r['percentil'], 0))}</td>"
        f"<td><div class='track'><div class='band'></div><div class='mid'></div><div class='dot' style='left:"
        f"{r['percentil']:.1f}%;background:{'#FFD100' if r['percentil'] >= 90 or r['percentil'] <= 10 else '#141414'}'>"
        f"</div></div></td></tr>" for _, r in pt.iterrows()) if not pt.empty else ""

    sg = ctx["signals"]
    sig_rows = ""
    if not sg.empty:
        lim = max(10.0, float(np.nanmax(np.abs((sg["lift"] - 1) * 100))))
        for _, r in sg.sort_values("lift", ascending=False).iterrows():
            xv = (r["lift"] - 1) * 100
            w = min(abs(xv) / lim, 1) * 50
            c = "#8A8A80" if r["muestra_pequena"] else (RISK_COLORS["Alto"] if xv >= 0 else RISK_COLORS["Bajo"])
            op = 1 if r["significativa"] else .45
            left = 50 if xv >= 0 else 50 - w
            sig_rows += (f"<tr><td>{esc(r['atributo'])} · <b>{esc(r['valor'])}</b></td><td>{esc(fmt_pct(r['tasa']))}</td>"
                         f"<td>{esc(fmt_int(r['n']))}</td><td><b>{esc(_es_x(r['lift']))}</b></td><td><div class='div'>"
                         f"<div class='axis'></div><i style='left:{left:.1f}%;width:{w:.1f}%;background:{c};"
                         f"opacity:{op}'></i></div></td><td>{'Sí' if r['significativa'] else 'No'}</td></tr>")

    chk = "".join(f"<li>{'☑' if done else '☐'} {esc(t)}</li>" for t, done in ctx["checklist"])
    talk = "".join(f"<li>{esc(t)}</li>" for t in ctx["talking"])
    logs = ctx["log"]
    log_html = ("<p class='muted'>Sin gestiones registradas en la sesión.</p>" if logs.empty else
                "<table><tr><th>Fecha</th><th>Canal</th><th>Resultado</th><th>Compromiso</th><th>Notas</th>"
                "<th>Gestor</th></tr>" + "".join(
                    f"<tr><td>{esc(r['Fecha de gestión'])}</td><td>{esc(r['Canal'])}</td><td>{esc(r['Resultado'])}</td>"
                    f"<td>{esc(r['Compromiso'])}</td><td>{esc(r['Notas'])}</td><td>{esc(r['Gestor'])}</td></tr>"
                    for _, r in logs.iterrows()) + "</table>")

    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Ficha 360° · {esc(rec['nombre'])}</title>
<style>{css}</style></head><body><main>
<div class="hero"><div class="eyebrow">◆ {esc(APP_NAME)} · Ficha 360° del estudiante</div>
<h1>{esc(rec['nombre'])}</h1><p>ID {esc(rec['id_estudiante'])} · {esc(rec['programa'])} · {esc(rec['facultad'])}</p>
<div class="chips"><span class="chip hl">{esc(RISK_ICONS.get(rk, '⚪'))} Riesgo {esc(rk)} · {esc(route)}</span>
<span class="chip">Generada {esc(ctx['generated'])}</span><span class="chip">Por {esc(ctx['gestor'])}</span>
<span class="chip">Fuente: {esc(ctx['source'])}</span><span class="chip">Crédito analizado {esc(ctx['sel_llave'])}</span></div></div>
<div class="grid2" style="margin-top:14px"><div class="card"><h2 style="margin-top:0">Perfil</h2><div class="kv">{prof}</div>
<p class="muted" style="margin:10px 0 0;font-size:11.5px">Nombre anonimizado · sin dirección ni fecha de nacimiento.</p></div>
<div class="card"><h2 style="margin-top:0">Riesgo del crédito analizado</h2>{risk_html}
<p class="muted" style="margin:10px 0 0;font-size:11.5px">{esc(ctx['risk_note'])}</p></div></div>
<h2>Créditos del estudiante ({len(ctx['credits'])})</h2><div class="card"><table><tr><th>Aprobación</th><th>Crédito</th>
<th>Programa</th><th>Valor</th><th>Cuotas</th><th>Riesgo predicho</th><th>Observado</th><th>Prioridad</th><th>Ruta</th></tr>
{cr_rows}</table></div>
<h2>Comparación con pares · {esc(ctx['peer_label'])}</h2><div class="card">{'<table><tr><th>Métrica</th><th>Estudiante</th><th>Mediana pares</th><th>Percentil</th><th>Posición (banda p25–p75)</th></tr>' + peer_rows + '</table>' if peer_rows else '<p class="muted">Sin métricas comparables.</p>'}</div>
<h2>Señales del perfil (asociaciones de segmento)</h2><div class="card">{'<table><tr><th>Segmento</th><th>Tasa Alto obs.</th><th>n</th><th>Lift</th><th>Frente al promedio (' + esc(fmt_pct(ctx['base'])) + ')</th><th>Signif.</th></tr>' + sig_rows + '</table>' if sig_rows else '<p class="muted">Sin señales calculables.</p>'}
<p class="muted" style="font-size:11.5px;margin:8px 0 0">{esc(ctx['signal_note'])}</p></div>
<h2>Plan de acción · {esc(route)} {esc(a.get('nombre', ''))}</h2><div class="grid2"><div class="card">
<div class="k">Acción recomendada</div><div class="v">{esc(a.get('accion', ''))}</div>
<div class="k" style="margin-top:8px">Fecha límite sugerida</div><div class="v">{esc(ctx['deadline'])}</div>
<div class="k" style="margin-top:8px">Checklist</div><ul class="chk">{chk}</ul></div>
<div class="card"><div class="k">Guion sugerido · {esc(CANAL_GUION.get(route, ''))}</div><p class="quote">{esc(ctx['guion'])}</p>
<div class="k">Puntos de conversación basados en datos</div><ul>{talk}</ul></div></div>
<h2>Bitácora de gestiones (sesión)</h2><div class="card">{log_html}</div>
<div class="foot">{esc(APP_NAME)} · {esc(PROGRAMA_ACADEMICO)} · {esc(UNIVERSIDAD)}. Documento de uso interno para gestión
preventiva de cartera. Las probabilidades son la confianza del modelo en la clase predicha, no P(Alto). Las señales
son asociaciones históricas de segmento, no causalidad ni explicación individual del modelo. La bitácora es de la sesión
del tablero: la integración con el sistema de gestión de cartera corresponde al requerimiento DB-04.</div>
</main></body></html>"""


# ============================================================================================
# CSS propio de la página
# ============================================================================================
def _css() -> None:
    st.markdown(
        """
        <style>
        @media (max-width: 1180px){
          [data-testid="stMain"] [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important;}
          [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
            flex:1 1 100% !important;width:100% !important;min-width:100% !important;}
        }
        .fx-anchor{scroll-margin-top:70px;}
        .fx-steps{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:4px 0 6px 0;}
        .fx-steps a{display:inline-flex;align-items:center;gap:8px;text-decoration:none !important;font-size:12.5px;
          font-weight:600;color:var(--sat-text) !important;background:var(--sat-surface);border:1px solid var(--sat-border);
          border-radius:999px;padding:4px 12px 4px 4px;transition:border-color .15s, transform .15s;}
        .fx-steps a:hover{border-color:var(--sat-accent);transform:translateY(-1px);}
        .fx-steps a b{width:22px;height:22px;border-radius:50%;background:var(--sat-accent);color:#141414;
          display:inline-flex;align-items:center;justify-content:center;font-size:11.5px;font-weight:800;}
        .fx-note{font-size:12.5px;color:var(--sat-muted);line-height:1.5;margin:2px 0 6px 0;}
        .fx-note b{color:var(--sat-text);}
        .fx-pos{font-size:12.5px;color:var(--sat-muted);display:flex;gap:8px;align-items:center;flex-wrap:wrap;
          min-height:38px;}
        .fx-pos b{color:var(--sat-text);}
        .fx-pos .pill{font-family:'Space Grotesk','Inter',sans-serif;font-weight:700;font-size:13px;color:#141414;
          background:var(--sat-accent);border-radius:999px;padding:2px 10px;}
        .fx-banner{display:flex;gap:12px;align-items:flex-start;border-radius:14px;padding:10px 14px;
          background:color-mix(in srgb, #3E63DD 10%, var(--sat-surface));border:1px solid color-mix(in srgb, #3E63DD 40%, transparent);
          color:var(--sat-text);font-size:13px;line-height:1.5;margin:6px 0 4px 0;}
        .fx-banner .tag{font-size:10.5px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;background:#3E63DD;
          color:#fff;padding:3px 9px;border-radius:999px;white-space:nowrap;}

        /* ---- Perfil ---- */
        .fx-profile{position:relative;background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:18px;
          padding:22px 22px 16px 22px;box-shadow:var(--sat-shadow);overflow:hidden;}
        .fx-profile:before{content:"";position:absolute;left:0;right:0;top:0;height:5px;
          background:linear-gradient(90deg, var(--rc) 0 28%, var(--sat-accent) 28% 100%);}
        .fx-top{display:flex;gap:16px;align-items:center;}
        .fx-avatar{width:66px;height:66px;border-radius:50%;background:var(--sat-accent);color:#141414;flex:0 0 auto;
          font-family:'Space Grotesk','Inter',sans-serif;font-weight:700;font-size:24px;display:flex;align-items:center;
          justify-content:center;box-shadow:0 0 0 3px var(--sat-surface), 0 0 0 6px var(--rc);letter-spacing:.02em;}
        .fx-name{font-family:'Space Grotesk','Inter',sans-serif;font-size:25px;font-weight:700;color:var(--sat-text);
          line-height:1.12;letter-spacing:-.02em;}
        .fx-sub{font-size:13px;color:var(--sat-muted);margin-top:3px;}
        .fx-sub b{color:var(--sat-text);}
        .fx-tags{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;}
        .fx-prog{margin-top:16px;padding:12px 14px;border-radius:12px;background:var(--sat-surface-2);
          border:1px solid var(--sat-border);display:grid;grid-template-columns:1.5fr 1fr;gap:10px 16px;}
        .fx-k{font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--sat-subtle);font-weight:700;}
        .fx-v{font-size:14px;font-weight:600;color:var(--sat-text);margin-top:2px;overflow-wrap:anywhere;line-height:1.3;}
        .fx-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px 16px;margin-top:14px;}
        .fx-rel{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:16px;}
        .fx-rel > div{border:1px solid var(--sat-border);border-radius:12px;padding:8px 10px;background:var(--sat-surface);}
        .fx-rel .fx-v{font-family:'Space Grotesk','Inter',sans-serif;font-size:16px;font-weight:700;}
        @media (max-width: 1320px){ .fx-rel{grid-template-columns:repeat(2,minmax(0,1fr));} }
        .fx-foot{margin-top:14px;font-size:11.5px;color:var(--sat-subtle);}

        /* ---- Riesgo ---- */
        .fx-rhead{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;}
        .fx-rhead .fx-v{font-size:13px;}
        .fx-cap{text-align:center;font-size:12px;color:var(--sat-muted);margin:-6px 0 10px 0;line-height:1.4;}
        .fx-cap b{color:var(--sat-text);}
        .fx-r2{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
        .fx-r2 > div{background:var(--sat-surface-2);border:1px solid var(--sat-border);border-radius:12px;padding:8px 10px;}
        .fx-conc{display:block;font-size:11.5px;font-weight:700;margin-top:4px;}
        .fx-route{position:relative;margin-top:8px;border:1px solid var(--sat-border);border-radius:12px;
          padding:10px 12px 10px 16px;background:var(--sat-surface);overflow:hidden;}
        .fx-route:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--rc);}
        .fx-route .hd{display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap;}
        .fx-route .nm{font-weight:700;font-size:14px;color:var(--sat-text);}
        .fx-route .sla{font-size:11.5px;font-weight:700;color:var(--sat-muted);border:1px solid var(--sat-border);
          padding:2px 9px;border-radius:999px;white-space:nowrap;}
        .fx-route .act{font-size:12.5px;color:var(--sat-muted);margin-top:4px;line-height:1.45;}
        .fx-prio{margin-top:10px;}
        .fx-prio .row{display:flex;justify-content:space-between;align-items:baseline;gap:8px;font-size:12.5px;
          color:var(--sat-muted);}
        .fx-prio .row span:last-child{white-space:nowrap;}
        .fx-prio .row b{font-family:'Space Grotesk','Inter',sans-serif;font-size:20px;color:var(--sat-text);}
        .fx-stack{display:flex;height:10px;border-radius:6px;overflow:hidden;gap:2px;background:var(--sat-surface-2);
          margin-top:6px;border:1px solid var(--sat-border);}
        .fx-stack i{display:block;height:100%;}
        .fx-lg{display:flex;gap:12px;flex-wrap:wrap;font-size:11.5px;color:var(--sat-muted);margin-top:6px;}
        .fx-lg span:before{content:"";display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:5px;
          background:var(--c);vertical-align:-1px;}
        .fx-lg b{color:var(--sat-text);}
        .fx-p3{display:flex;height:12px;border-radius:6px;overflow:hidden;gap:2px;margin-top:6px;}
        .fx-p3 i{display:block;height:100%;}

        /* ---- Tarjetas de lectura / plan ---- */
        .fx-side{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:16px;padding:14px 16px;
          box-shadow:var(--sat-shadow);}
        .fx-side .t{font-size:13px;font-weight:700;color:var(--sat-text);margin-bottom:6px;}
        .fx-side ul{margin:4px 0 0 0;padding-left:18px;}
        .fx-side li{font-size:12.5px;color:var(--sat-muted);line-height:1.5;margin:3px 0;}
        .fx-side li b{color:var(--sat-text);}
        .fx-side .row{display:flex;justify-content:space-between;gap:10px;font-size:12.5px;color:var(--sat-muted);
          padding:6px 0;border-bottom:1px dashed var(--sat-border);}
        .fx-side .row:last-child{border-bottom:none;}
        .fx-side .row b{color:var(--sat-text);text-align:right;}
        .fx-leg{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--sat-muted);margin-top:8px;}
        .fx-leg span{display:inline-flex;align-items:center;gap:6px;}
        .fx-leg i{display:inline-block;width:12px;height:12px;border-radius:3px;}
        .fx-plan{position:relative;background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:16px;
          padding:16px 18px 14px 22px;box-shadow:var(--sat-shadow);overflow:hidden;}
        .fx-plan:before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;background:var(--rc);}
        .fx-plan .hd{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;}
        .fx-plan .code{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13px;letter-spacing:.04em;
          padding:2px 10px;border-radius:7px;color:var(--ri);background:color-mix(in srgb, var(--rc) 15%, transparent);}
        .fx-plan .nm{font-family:'Space Grotesk','Inter',sans-serif;font-size:20px;font-weight:700;color:var(--sat-text);
          margin-top:8px;}
        .fx-plan .act{font-size:13px;color:var(--sat-muted);margin-top:4px;line-height:1.5;}
        .fx-plan .g3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px;}
        .fx-plan .g3 > div{background:var(--sat-surface-2);border:1px solid var(--sat-border);border-radius:10px;padding:7px 10px;}
        .fx-plan .g3 .fx-v{font-size:13.5px;}
        .fx-script{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:14px;padding:12px 14px;
          box-shadow:var(--sat-shadow);margin-top:10px;}
        .fx-script .t{font-size:12.5px;font-weight:700;color:var(--sat-text);margin-bottom:6px;}
        .fx-script .q{font-size:13.5px;line-height:1.5;color:var(--sat-text);border-left:3px solid var(--sat-accent);
          padding:2px 0 2px 10px;font-style:italic;}
        .fx-script .f{font-size:11.5px;color:var(--sat-subtle);margin-top:8px;}
        .fx-status{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:12.5px;color:var(--sat-muted);
          margin:0 0 8px 0;}
        .fx-status b{color:var(--sat-text);}
        .fx-export{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:16px;padding:14px 16px;
          box-shadow:var(--sat-shadow);font-size:12.5px;color:var(--sat-muted);line-height:1.55;}
        .fx-export b{color:var(--sat-text);}
        .fx-export .ttl{font-size:14px;font-weight:700;color:var(--sat-text);margin-bottom:4px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================================
# Estado: callbacks de navegación y bitácora
# ============================================================================================
def _step(ids: list[str], delta: int) -> None:
    cur = ss.get(SEL_KEY)
    i = ids.index(cur) if cur in ids else 0
    ss[SEL_KEY] = ids[max(0, min(len(ids) - 1, i + delta))]


def _goto_first(ids: list[str]) -> None:
    if ids:
        ss[SEL_KEY] = ids[0]


def _undo_last(sid: str) -> None:
    logs = ss.get(LOG_KEY, [])
    for i in range(len(logs) - 1, -1, -1):
        if logs[i].get("id_estudiante") == sid:
            logs.pop(i)
            break


def _log_frame(entries: list[dict]) -> pd.DataFrame:
    cols = {"registrado": "Registrado", "id_estudiante": "ID estudiante", "nombre": "Estudiante",
            "llave2": "Crédito", "ruta": "Ruta", "fecha_gestion": "Fecha de gestión", "canal": "Canal",
            "resultado": "Resultado", "compromiso": "Compromiso", "monto_compromiso": "Monto comprometido",
            "fecha_compromiso": "Fecha del compromiso", "notas": "Notas", "gestor": "Gestor"}
    if not entries:
        return pd.DataFrame(columns=list(cols.values()))
    return pd.DataFrame(entries)[list(cols)].rename(columns=cols)


# ============================================================================================
# Página
# ============================================================================================
df_all = get_active_df()
dff = apply_filters(df_all)
n_est_filtro = int(dff["id_estudiante"].nunique()) if len(dff) else 0

page_header(
    "Ficha 360° del estudiante",
    "Todo lo que el gestor necesita para entender un caso y actuar hoy: perfil, riesgo de cada crédito, "
    "comparación con pares, señales de segmento y plan de acción con bitácora.",
    eyebrow="Gestión de cartera",
    highlight=f"{fmt_int(n_est_filtro)} estudiantes en el filtro",
)
filter_chips(active_chips(df_all))

if dff.empty:
    empty_state("Sin estudiantes para mostrar", "Los filtros actuales no dejan ningún crédito. Ajústalos o límpialos "
                "en la barra lateral para abrir una ficha.", icon="🪪")
    footer()
    st.stop()

_css()
p = pal()
truth_active = bool(df_all["has_truth"].any())
user = ss.get("sat_user") or {}
gestor_name = str(user.get("name") or "Gestor de Cartera")

# --------------------------------------------------------------------------------------------
# 1 · Selector y navegación
# --------------------------------------------------------------------------------------------
all_ids = set(df_all["id_estudiante"].astype(str))
nav_id = ss.get(FICHA_KEY)
nav_id = str(nav_id) if nav_id is not None else None
filter_ids = set(dff["id_estudiante"].astype(str))
if nav_id and nav_id != ss.get(CONSUMED_KEY):          # llegada desde la Cola (u otra página)
    ss[CONSUMED_KEY] = nav_id
    if nav_id in filter_ids:
        ss[SEL_KEY] = nav_id
        ss.pop(FORCE_KEY, None)
    elif nav_id in all_ids:
        ss[FORCE_KEY] = nav_id
        ss[SEL_KEY] = nav_id
force_id = ss.get(FORCE_KEY)
if force_id and ss.get(SEL_KEY, force_id) != force_id:
    ss.pop(FORCE_KEY, None)
    force_id = None

with st.container(border=True):
    c_sel, c_ord, c_prev, c_next = st.columns([3.7, 1.9, 1.15, 1.25], vertical_alignment="bottom", gap="small")
    with c_ord:
        mode = st.selectbox("Ordenar la navegación por", ORDER_MODES, key="ficha_orden", help=ORDER_HELP,
                            persist_state="session")
    base_nav = dff if not force_id else pd.concat([dff, df_all[df_all["id_estudiante"].astype(str) == force_id]])
    stu = _students(base_nav, mode or ORDER_MODES[0])
    ids = stu["id_estudiante"].astype(str).tolist()
    rk_lab = stu["y_pred"].astype(str).map(lambda r: f"{RISK_ICONS.get(r, '⚪')} {r if r in RISK_ORDER else 'Sin dato'}")
    labels = dict(zip(ids, (stu["nombre"].astype(str) + " · " + stu["id_estudiante"].astype(str) + " · "
                            + stu["programa"].astype(str) + " · " + rk_lab).tolist()))
    if force_id in labels:
        labels[force_id] += " · fuera del filtro"
    cur = ss.get(SEL_KEY)
    if cur not in labels:
        cur = nav_id if nav_id in labels else ids[0]
    ss[SEL_KEY] = cur
    with c_sel:
        st.selectbox("Estudiante · escribe nombre, ID o programa para buscar", ids, key=SEL_KEY,
                     format_func=lambda i: labels.get(i, i), filter_mode="contains",
                     help="Lista de los estudiantes que cumplen los filtros globales, en el orden elegido.")
    sid = str(ss[SEL_KEY])
    pos = ids.index(sid)
    with c_prev:
        st.button("Anterior", icon=":material/chevron_left:", key="ficha_prev", width="stretch", on_click=_step,
                  args=(ids, -1), disabled=pos == 0, help="Estudiante anterior en el orden de navegación")
    with c_next:
        st.button("Siguiente", icon=":material/chevron_right:", key="ficha_next", width="stretch", on_click=_step,
                  args=(ids, 1), disabled=pos >= len(ids) - 1, help="Siguiente estudiante en el orden de navegación")
ss[FICHA_KEY] = sid
ss[CONSUMED_KEY] = sid

# --------------------------------------------------------------------------------------------
# Datos del estudiante y del crédito analizado
# --------------------------------------------------------------------------------------------
D = df_all.drop_duplicates("llave2")
n_dups_all = int(len(df_all) - len(D))
credits = D[D["id_estudiante"].astype(str) == sid].sort_values("fecha_aprobacion", ascending=False,
                                                                 na_position="last")
n_rows_student = int((df_all["id_estudiante"].astype(str) == sid).sum())
filter_llaves = set(dff.loc[dff["id_estudiante"].astype(str) == sid, "llave2"].astype(str))
ref_row = stu.loc[stu["id_estudiante"].astype(str) == sid].iloc[0]
ref_llave = str(ref_row["llave2"])
llaves = credits["llave2"].astype(str).tolist()

with st.container():
    t1, t2, t3 = st.columns([2.2, 2.6, 1.5], vertical_alignment="center", gap="small")
    with t1:
        rank_mode = "prioridad" if mode and mode.startswith("Prioridad") else (mode or "").lower()
        st.markdown(f"<div class='fx-pos'><span class='pill'>{fmt_int(pos + 1)}</span> de <b>{fmt_int(len(ids))}</b> "
                    f"estudiantes · orden: {esc(rank_mode)}</div>", unsafe_allow_html=True)
    with t2:
        if len(llaves) > 1:
            def _cred_label(l: str) -> str:
                r = credits.loc[credits["llave2"].astype(str) == l].iloc[0]
                rr = _risk(r["y_pred"])
                tag = "" if l in filter_llaves else " · fuera del filtro"
                return (f"{fmt_date(r['fecha_aprobacion'])} · {fmt_cop(r['valor_financiacion'])} · "
                        f"{RISK_ICONS.get(rr, '⚪')} {rr} · prioridad {fmt_num(r['prioridad'], 1)}{tag}")
            sel_llave = st.selectbox("Crédito analizado", llaves, index=llaves.index(ref_llave) if ref_llave in llaves
                                     else 0, key=f"ficha_cred_{sid}", format_func=_cred_label,
                                     help="Por defecto, el crédito de referencia según el orden elegido. Cambia "
                                          "el crédito para recalcular riesgo, pares, señales y plan.")
        else:
            sel_llave = llaves[0] if llaves else ref_llave
            st.markdown(f"<div class='fx-pos'>Único crédito · <b>{esc(sel_llave)}</b></div>", unsafe_allow_html=True)
    with t3:
        if can_access("cola"):
            st.page_link(PAGES["cola"][0], label="Volver a la Cola de gestión", icon=":material/arrow_back:",
                         width="stretch")

if force_id == sid:
    fb1, fb2 = st.columns([4.2, 1.3], vertical_alignment="center", gap="small")
    with fb1:
        st.markdown("<div class='fx-banner'><span class='tag'>Desde la cola</span><div>Este estudiante llegó desde la "
                    "<b>Cola de gestión</b> pero no cumple los filtros actuales de la barra lateral; se muestra "
                    "igualmente para no perder el caso.</div></div>", unsafe_allow_html=True)
    with fb2:
        in_filter = [i for i in ids if i != force_id]
        st.button("Ir al #1 del filtro", icon=":material/first_page:", key="ficha_first", width="stretch",
                  on_click=_goto_first, args=(in_filter,), disabled=not in_filter,
                  help="Abre el estudiante de mayor prioridad que sí cumple los filtros actuales.")

rec = credits.loc[credits["llave2"].astype(str) == str(sel_llave)].iloc[0]
rec_df = credits.loc[credits["llave2"].astype(str) == str(sel_llave)]
rk = _risk(rec["y_pred"])
yt = _risk(rec["y_true"]) if bool(rec.get("has_truth", False)) else "Sin dato"
conf = rec["proba_pred"]
route = str(rec["ruta"]) if str(rec["ruta"]) in ACTION_ROUTES else "R4"
act = ACTION_ROUTES[route]
rcolor = RISK_COLORS.get(rk, "#8A8A80")
dia = int(rec["fecha_de_pago"]) if not _is_na(rec.get("fecha_de_pago")) else None
cuota_ref = rec.get("valor_primera_cuota")
cuota_txt = fmt_cop(cuota_ref, compact=False) if not _is_na(cuota_ref) else "—"
first_name = str(rec["nombre"]).split(" ")[0]

# Rango de prioridad dentro del filtro (a nivel de estudiante, crédito de referencia).
prio_vals = stu["prioridad"].to_numpy(dtype=float)
prio = float(rec["prioridad"]) if not _is_na(rec["prioridad"]) else np.nan
rank_short = "fuera del filtro"
if force_id == sid:
    rank_txt = "fuera del filtro"
else:
    rank_pos = int(np.sum(prio_vals > prio)) + 1 if not _is_na(prio) else None
    top_pct = rank_pos / max(len(prio_vals), 1) if rank_pos else None
    top_s = (fmt_pct(top_pct, 0) if top_pct is not None and top_pct >= 0.01 else
             f"{fmt_pct(max(top_pct or 0, 0.001), 1)}")
    rank_txt = (f"#{fmt_int(rank_pos)} de {fmt_int(len(prio_vals))} en el filtro · top {top_s}" if rank_pos else "—")
    rank_short = f"#{fmt_int(rank_pos)} de {fmt_int(len(prio_vals))} · top {top_s}" if rank_pos else "—"

# Descomposición del índice de prioridad (mismos pesos por defecto de core.data).
w = DEFAULT_WEIGHTS
tot_w = max(sum(w.values()), 1e-9)
vf = pd.to_numeric(D["valor_financiacion"], errors="coerce").to_numpy(dtype=float)
e_pct = (_pctile(vf, rec["valor_financiacion"]) / 100) if not _is_na(rec["valor_financiacion"]) else 0.5
comp = {"Riesgo": 100 * w["riesgo"] * float(rec["intensidad_riesgo"]) / tot_w,
        "Exposición": 100 * w["exposicion"] * float(e_pct) / tot_w,
        "Mora": 100 * w["mora"] * float(rec["mora_flag"]) / tot_w}

# Referencia histórica (señales y fiabilidad): la base activa si trae riesgo observado; si no, la base oficial.
ref_df = D[D["has_truth"]]
ref_source = "base activa"
if len(ref_df) < 200:
    try:
        bb = load_base().drop_duplicates("llave2")
        ref_df = bb[bb["has_truth"]]
        ref_source = "base oficial histórica"
    except Exception:
        ref_df = ref_df
ref_tables = _reference_tables(_data_token(ref_df) + "|" + ref_source, ref_df) if len(ref_df) else None
base_rate = ref_tables["base"] if ref_tables else np.nan
sig = _signals(rec_df, ref_tables) if ref_tables else pd.DataFrame()
class_mean = D.loc[D["y_pred"].astype(str) == rk, "proba_pred"].mean() if rk in RISK_ORDER else np.nan
label_prec = ref_tables["precision"].get(rk, np.nan) if ref_tables else np.nan

# --------------------------------------------------------------------------------------------
# KPI del estudiante
# --------------------------------------------------------------------------------------------
tot_fin = float(pd.to_numeric(credits["valor_financiacion"], errors="coerce").sum())
first_date = credits["fecha_aprobacion"].min()
conf_txt = fmt_pct(conf, 0) if not _is_na(conf) else "—"
kpi_row([
    kpi_card("Riesgo predicho", f"{rk}", f"Confianza {_nb(conf_txt)} · {_nb(_period_label(rec['fecha_aprobacion']))}",
             tone={"Alto": "alto", "Medio": "medio", "Bajo": "bajo"}.get(rk, "ink"), icon=RISK_ICONS.get(rk, "⚪"),
             help="Clase predicha por el modelo para el crédito analizado; la confianza es la probabilidad de esa clase."),
    kpi_card("Índice de prioridad", fmt_num(prio, 1) if not _is_na(prio) else "—", rank_txt, tone="accent",
             bar=(prio / 100) if not _is_na(prio) else None, icon="🎯",
             help="Combinación ponderada de intensidad de riesgo (60 %), exposición (25 %) y mora histórica (15 %)."),
    kpi_card("Exposición en riesgo", fmt_cop(rec["exposicion_riesgo"]), "Valor financiado × intensidad de riesgo",
             tone="ink", icon="💰"),
    kpi_card("Historial", f"{len(credits)} crédito{'s' if len(credits) != 1 else ''}",
             f"{_nb(fmt_cop(tot_fin))} financiados desde {_nb(_period_label(first_date))}", tone="ink", icon="🗂️"),
    kpi_card("Ruta y SLA", route, f"{act['nombre']} · SLA {_nb(act['sla'])}", tone=ROUTE_TONE.get(route, "ink"), icon="🧭",
             help=act["accion"]),
])

# --------------------------------------------------------------------------------------------
# Comparación con pares (se calcula antes para la lectura rápida)
# --------------------------------------------------------------------------------------------
own = set(llaves)
peer_counts = {}
for gname, gcol in PEER_GROUPS.items():
    if gcol is None:
        peer_counts[gname] = int((~D["llave2"].isin(own)).sum())
    else:
        peer_counts[gname] = int(((D[gcol].astype(str) == str(rec[gcol])) & ~D["llave2"].isin(own)).sum())
default_group = next((g for g in PEER_GROUPS if peer_counts[g] >= MIN_PEERS), "Toda la base")
peer_opts = list(PEER_GROUPS)

# --------------------------------------------------------------------------------------------
# Índice de secciones
# --------------------------------------------------------------------------------------------
steps = [("ficha-perfil", "Perfil y riesgo"), ("ficha-creditos", "Créditos"), ("ficha-pares", "Pares"),
         ("ficha-senales", "Señales"), ("ficha-plan", "Plan de acción"), ("ficha-exportar", "Exportar")]
st.markdown("<div class='fx-steps'>" + "".join(f"<a href='#{a}'><b>{i}</b>{esc(t)}</a>" for i, (a, t) in
                                               enumerate(steps, 1)) + "</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------
# 2 · Perfil + 3 · Riesgo
# --------------------------------------------------------------------------------------------
_anchor_section("ficha-perfil", "Perfil y riesgo del crédito analizado",
                "Quién es el estudiante (datos anonimizados) y qué dice el modelo sobre su crédito: confianza, ruta, "
                "SLA, prioridad y, si existe, el riesgo observado.", "Paso 1 · Perfil y riesgo")

score_txt = _txt(rec.get("score_corto"))
profile_pairs = [
    ("Nivel", _cap(rec["nivel"])), ("Sede", _txt(rec["sede"])),
    ("Rango de edad", f"{_txt(rec['rango_edad'])} años" if _txt(rec["rango_edad"]) != "Sin dato" else "Sin dato"),
    ("Género", _txt(rec["genero_txt"])), ("Estado civil", _cap(rec["estado_civil"])),
    ("Tipo de estudiante", _txt(rec["tipo_estudiante"])), ("Condición académica", _cap(rec["tipoestudiante"])),
    ("Ciudad", _title(rec["ciudad_norm"])), ("Departamento", _txt(rec["departamento_limpio"])),
    ("Cohorte", _txt(rec["cohorte"])), ("Tipo de cliente", _txt(rec["cliente_limpio"])),
    ("Nacionalidad", _cap(rec.get("nacionalidad"))),
]
grid_html = "".join(f"<div><div class='fx-k'>{esc(k)}</div><div class='fx-v'>{esc(v)}</div></div>"
                    for k, v in profile_pairs)
mora_si = int(rec["mora_flag"]) == 1
tags = (badge(_txt(rec["programa_cluster"]), "neutral") + badge(_cap(rec["nivel"]), "neutral")
        + badge(f"Sede {_txt(rec['sede'])}", "neutral") + badge(_txt(rec["tipo_estudiante"]), "neutral"))
profile_html = (
    f"<div class='fx-profile' style='--rc:{rcolor}'>"
    f"<div class='fx-top'><div class='fx-avatar'>{esc(_initials(rec['nombre']))}</div><div>"
    f"<div class='fx-name'>{esc(rec['nombre'])}</div>"
    f"<div class='fx-sub'>ID <b>{esc(sid)}</b> · {len(credits)} crédito{'s' if len(credits) != 1 else ''} · "
    f"cliente desde {esc(_period_label(first_date))}</div><div class='fx-tags'>{tags}</div></div></div>"
    f"<div class='fx-prog'><div><div class='fx-k'>Programa</div><div class='fx-v'>{esc(_txt(rec['programa']))}</div></div>"
    f"<div><div class='fx-k'>Facultad</div><div class='fx-v'>{esc(_txt(rec['facultad']))}</div></div></div>"
    f"<div class='fx-grid'>{grid_html}</div>"
    f"<div class='fx-rel'>"
    f"<div><div class='fx-k'>Total financiado</div><div class='fx-v'>{_nb(fmt_cop(tot_fin))}</div></div>"
    f"<div><div class='fx-k'>Mora Datacrédito</div><div class='fx-v' style='color:{_risk_text_color('Alto') if mora_si else 'inherit'}'>"
    f"{'Sí' if mora_si else 'No'}</div></div>"
    f"<div><div class='fx-k'>Scoring externo</div><div class='fx-v'>{esc(score_txt)}</div></div>"
    f"<div><div class='fx-k'>Día de pago</div><div class='fx-v'>{f'Día {dia}' if dia else '—'}</div></div></div>"
    f"<div class='fx-foot'>🔒 Nombre anonimizado · no se muestran dirección ni fecha de nacimiento.</div></div>"
)

cp, cr_ = st.columns([1.18, 1], gap="medium")
with cp:
    st.markdown(profile_html, unsafe_allow_html=True)
with cr_:
    with st.container(border=True):
        st.markdown(
            f"<div class='fx-rhead'><div><div class='fx-k'>Crédito analizado</div><div class='fx-v'>{esc(sel_llave)} · "
            f"{esc(fmt_date(rec['fecha_aprobacion']))}</div></div><div>{risk_badge(rk)}</div></div>",
            unsafe_allow_html=True)
        if _is_na(conf):
            st.markdown("<div class='sat-empty' style='padding:24px'>El motor no entregó la confianza de este crédito."
                        "</div>", unsafe_allow_html=True)
        else:
            show_fig(_fig_gauge(float(conf), rcolor, class_mean), key="ficha_gauge", height=196, legend=False)
            cm_txt = (f" · la marca indica la confianza media de su clase (<b>{fmt_pct(class_mean, 0)}</b>)"
                      if not _is_na(class_mean) else "")
            st.markdown(f"<div class='fx-cap'>Confianza del modelo en la clase predicha «{esc(rk)}»{cm_txt}. "
                        f"No es P(Alto).</div>", unsafe_allow_html=True)
        # Riesgo observado + fiabilidad de la etiqueta
        if yt in RISK_ORDER:
            order = {r: i for i, r in enumerate(RISK_ORDER)}
            if rk not in order:
                conc = ""
            elif yt == rk:
                conc = f"<span class='fx-conc' style='color:{_risk_text_color('Bajo')}'>✓ Coincide con el modelo</span>"
            elif order[yt] < order[rk]:
                conc = (f"<span class='fx-conc' style='color:{_risk_text_color('Alto')}'>▲ El modelo subestimó el "
                        f"riesgo</span>")
            else:
                conc = (f"<span class='fx-conc' style='color:{_risk_text_color('Medio')}'>▼ El modelo sobrestimó el "
                        f"riesgo</span>")
            obs_html = f"{risk_badge(yt)}{conc}"
        else:
            obs_html = "<span class='fx-conc' style='color:var(--sat-muted)'>Sin riesgo observado en este dataset</span>"
        prec_html = (f"<div class='fx-v'>{fmt_pct(label_prec, 0)}</div><span class='fx-conc' style='color:var(--sat-muted);"
                     f"font-weight:500'>de los créditos predichos «{esc(rk)}» resultaron {esc(rk)} ({ref_source})</span>"
                     if not _is_na(label_prec) else "<div class='fx-v'>—</div>")
        stack = "".join(f"<i style='width:{v:.2f}%;background:{c}'></i>" for (k, v), c in
                        zip(comp.items(), [p["text"], YELLOW, "#3E63DD"]) if v > 0)
        legend = "".join(f"<span style='--c:{c}'>{k} <b>{fmt_num(v, 1)}</b></span>" for (k, v), c in
                         zip(comp.items(), [p["text"], YELLOW, "#3E63DD"]))
        p3 = ""
        if not _is_na(rec.get("proba_alto")):
            pa, pm, pb = (float(rec.get(c) or 0) for c in ("proba_alto", "proba_medio", "proba_bajo"))
            art = load_artifacts() or {}
            thr = (((art.get("alto") or {}).get("rf") or {}).get("umbral_objetivo") or {}).get("umbral", 0.30)
            alert = pa >= thr
            p3 = (f"<div class='fx-prio'><div class='row'><span>Probabilidades por clase (motor local)</span>"
                  f"<span>P(Alto) <b style='font-size:15px'>{fmt_pct(pa, 0)}</b></span></div><div class='fx-p3'>"
                  f"<i style='width:{pa * 100:.1f}%;background:{RISK_COLORS['Alto']}'></i>"
                  f"<i style='width:{pm * 100:.1f}%;background:{RISK_COLORS['Medio']}'></i>"
                  f"<i style='width:{pb * 100:.1f}%;background:{RISK_COLORS['Bajo']}'></i></div>"
                  f"<div class='fx-lg'><span style='--c:{RISK_COLORS['Alto']}'>Alto {fmt_pct(pa, 0)}</span>"
                  f"<span style='--c:{RISK_COLORS['Medio']}'>Medio {fmt_pct(pm, 0)}</span>"
                  f"<span style='--c:{RISK_COLORS['Bajo']}'>Bajo {fmt_pct(pb, 0)}</span></div>"
                  f"<div class='fx-note' style='margin-top:4px'>Con el umbral DE-03 (P(Alto) ≥ {fmt_num(thr, 2)}) este "
                  f"crédito <b>{'sí' if alert else 'no'}</b> se marcaría como alerta Alta.</div></div>")
        st.markdown(
            f"<div class='fx-r2'><div><div class='fx-k'>Riesgo observado · validación</div>{obs_html}</div>"
            f"<div><div class='fx-k'>Fiabilidad histórica de la etiqueta</div>{prec_html}</div></div>"
            f"<div class='fx-route' style='--rc:{_rc(route)}'><div class='hd'>{_route_chip(route)}"
            f"<span class='sla'>SLA · {esc(act['sla'])}</span></div><div class='act'>{esc(act['accion'])}</div></div>"
            f"<div class='fx-prio'><div class='row'><span>Índice de prioridad · {esc(rank_short)}</span>"
            f"<span><b>{fmt_num(prio, 1) if not _is_na(prio) else '—'}</b> / 100</span></div>"
            f"<div class='fx-stack'>{stack}</div><div class='fx-lg'>{legend}"
            f"<span style='--c:transparent;margin-left:auto'>pesos 60 · 25 · 15</span></div></div>{p3}",
            unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------
# Lectura rápida (hallazgos automáticos)
# --------------------------------------------------------------------------------------------
peer_group = ss.get("ficha_pares") or default_group
if peer_group not in PEER_GROUPS:
    peer_group = default_group
gcol = PEER_GROUPS[peer_group]
peers = D[~D["llave2"].isin(own)] if gcol is None else D[(D[gcol].astype(str) == str(rec[gcol]))
                                                         & ~D["llave2"].isin(own)]
pt = _peer_table(rec, peers)
peer_label = ("toda la base" if gcol is None else f"{peer_group.lower()} {_txt(rec[gcol])}")

ins = []
t_obs = ""
if yt in RISK_ORDER:
    t_obs = f" Riesgo observado: <b>{esc(yt)}</b>."
ins.append(insight(
    f"{rk} · {route} {act['nombre']}",
    f"El modelo clasifica el crédito de {esc(_period_label(rec['fecha_aprobacion']))} como <b>{esc(rk)}</b> con "
    f"<b>{conf_txt}</b> de confianza. Prioridad <b>{fmt_num(prio, 1)}</b>/100 ({esc(rank_txt)}). Atender en "
    f"<b>{esc(act['sla'])}</b>.{t_obs}",
    tone={"Alto": "alto", "Medio": "medio", "Bajo": "bajo"}.get(rk, "accent"), icon="🎯"))
if not pt.empty:
    ext = pt.assign(dev=(pt["percentil"] - 50).abs()).sort_values("dev", ascending=False)
    ext_top = ext[ext["dev"] >= 25].head(2)
    if ext_top.empty:
        txt_p = (f"Frente a {fmt_int(len(peers))} créditos de su {esc(peer_label)}, todas sus métricas están dentro "
                 f"del rango típico (p25–p75): un perfil financiero <b>habitual</b> para sus pares.")
    else:
        parts = [f"su <b>{esc(r['metrica'].lower())}</b> está en el <b>percentil {fmt_num(r['percentil'], 0)}</b> "
                 f"({esc(_fmt_metric(r['valor'], r['kind']))} vs mediana {esc(_fmt_metric(r['mediana'], r['kind']))})"
                 for _, r in ext_top.iterrows()]
        txt_p = f"Frente a {fmt_int(len(peers))} créditos de su {esc(peer_label)}, " + " y ".join(parts) + "."
    ins.append(insight("Cómo se compara con sus pares", txt_p, tone="info", icon="👥"))
if not sig.empty:
    ok = sig[~sig["muestra_pequena"]]
    up = ok[ok["significativa"] & (ok["lift"] > 1)].sort_values("lift", ascending=False)
    dn = ok[ok["significativa"] & (ok["lift"] < 1)].sort_values("lift")
    bits = []
    if not up.empty:
        r = up.iloc[0]
        bits.append(f"la señal que más eleva el riesgo es <b>{esc(r['atributo'].lower())} {esc(r['valor'])}</b> "
                    f"({_es_x(r['lift'])} la tasa base)")
    if not dn.empty:
        r = dn.iloc[0]
        bits.append(f"la más protectora, <b>{esc(r['atributo'].lower())} {esc(r['valor'])}</b> ({_es_x(r['lift'])})")
    head = f"{len(up)} de {len(sig)} señales elevan y {len(dn)} reducen el riesgo histórico de su segmento"
    txt_s = head + (": " + "; ".join(bits) if bits else "") + ". Son asociaciones de segmento, no causas."
    ins.append(insight("Señales del perfil", txt_s, tone="alto" if len(up) > len(dn) else "bajo", icon="🧭"))
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
insight_row(ins)

# --------------------------------------------------------------------------------------------
# 4 · Créditos
# --------------------------------------------------------------------------------------------
_anchor_section("ficha-creditos", "Todos los créditos del estudiante",
                "Historial completo en la base activa (incluye créditos fuera del filtro). Cada punto es un crédito: "
                "color = riesgo predicho, franja = plazo estimado del plan; el anillo amarillo marca el analizado.",
                "Paso 2 · Créditos")
cc1, cc2 = st.columns([1.65, 1], gap="medium")
with cc1:
    with st.container(border=True):
        show_fig(_fig_timeline(credits, str(sel_llave)), key="ficha_timeline", height=340)
with cc2:
    n_alto = int((credits["y_pred"].astype(str) == "Alto").sum())
    n_alto_obs = int((credits["y_true"].astype(str) == "Alto").sum())
    max_prio_row = credits.sort_values("prioridad", ascending=False).iloc[0]
    rows_html = (
        f"<div class='row'><span>Créditos en la base</span><b>{len(credits)}</b></div>"
        f"<div class='row'><span>Dentro del filtro actual</span><b>{len(filter_llaves)}</b></div>"
        f"<div class='row'><span>Predichos «Alto»</span><b>{n_alto}</b></div>"
        + (f"<div class='row'><span>Observados «Alto»</span><b>{n_alto_obs}</b></div>" if truth_active else "")
        + f"<div class='row'><span>Total financiado</span><b>{_nb(fmt_cop(tot_fin, compact=False))}</b></div>"
        f"<div class='row'><span>Ticket promedio</span><b>{_nb(fmt_cop(tot_fin / max(len(credits), 1), compact=False))}"
        f"</b></div><div class='row'><span>Crédito de mayor prioridad</span><b>{esc(fmt_date(max_prio_row['fecha_aprobacion']))}"
        f" · {fmt_num(max_prio_row['prioridad'], 1)}</b></div>")
    st.markdown(f"<div class='fx-side'><div class='t'>Resumen del historial</div>{rows_html}</div>",
                unsafe_allow_html=True)
    if str(max_prio_row["llave2"]) != str(sel_llave) and float(max_prio_row["prioridad"]) > float(rec["prioridad"]):
        _note(f"⚠️ Otro crédito de este estudiante tiene <b>mayor prioridad</b> ({fmt_num(max_prio_row['prioridad'], 1)}"
              f" frente a {fmt_num(rec['prioridad'], 1)}): selecciónalo en «Crédito analizado».")
    if n_rows_student > len(credits):
        _note(f"🧹 Se consolidaron <b>{n_rows_student - len(credits)}</b> filas duplicadas de la misma llave de crédito "
              f"(en la base activa hay {fmt_int(n_dups_all)} duplicados de llave; hallazgo de calidad de datos).")

tb = pd.DataFrame({
    "Analizado": ["◆" if str(l) == str(sel_llave) else "" for l in credits["llave2"]],
    "Aprobación": credits["fecha_aprobacion"].dt.strftime("%d/%m/%Y").fillna("—"),
    "Crédito": credits["llave2"].astype(str),
    "Programa": credits["programa"].astype(str),
    "Valor financiado": [fmt_cop(v, compact=False) for v in credits["valor_financiacion"]],
    "% de la matrícula": pd.to_numeric(credits["ratio_financiacion"], errors="coerce").clip(0, 1.5),
    "Cuotas": pd.to_numeric(credits["cuotas"], errors="coerce").astype("Int64"),
    "Tipo de interés": credits["tipo_interes"].astype(str),
    "Riesgo predicho": [f"{RISK_ICONS.get(_risk(v), '⚪')} {_risk(v)}" for v in credits["y_pred"]],
    "Confianza": pd.to_numeric(credits["proba_pred"], errors="coerce"),
    "Riesgo observado": [f"{RISK_ICONS.get(_risk(v), '⚪')} {_risk(v)}" if _risk(v) in RISK_ORDER else "—"
                         for v in credits["y_true"]],
    "Prioridad": pd.to_numeric(credits["prioridad"], errors="coerce"),
    "Ruta": [f"{r} · {ROUTE_SHORT.get(str(r), str(r))}" for r in credits["ruta"]],
    "En el filtro": [str(l) in filter_llaves for l in credits["llave2"]],
})
st.dataframe(
    tb, hide_index=True, width="stretch", key="ficha_tabla_creditos",
    column_order=["Analizado", "Aprobación", "Riesgo predicho", "Confianza", "Riesgo observado", "Prioridad", "Ruta",
                  "Valor financiado", "% de la matrícula", "Cuotas", "Tipo de interés", "Programa", "Crédito",
                  "En el filtro"],
    column_config={
        "Aprobación": st.column_config.TextColumn("Aprobación", width=92),
        "Riesgo predicho": st.column_config.TextColumn("Riesgo predicho", width=112),
        "Riesgo observado": st.column_config.TextColumn("Riesgo observado", width=118),
        "Ruta": st.column_config.TextColumn("Ruta", width=118),
        "Cuotas": st.column_config.NumberColumn("Cuotas", width=64),
        "Analizado": st.column_config.TextColumn("", width=34, help="Crédito analizado en esta ficha"),
        "% de la matrícula": st.column_config.ProgressColumn("% matrícula", format="percent", min_value=0,
                                                             max_value=1, help="Valor financiado / matrícula neta "
                                                                               "(DA-02: máximo 85 %)"),
        "Confianza": st.column_config.ProgressColumn("Confianza", format="percent", min_value=0, max_value=1,
                                                     help="Probabilidad de la clase predicha"),
        "Prioridad": st.column_config.ProgressColumn("Prioridad", format="%.1f", min_value=0, max_value=100),
        "En el filtro": st.column_config.CheckboxColumn("En filtro", width="small"),
        "Programa": st.column_config.TextColumn("Programa", width="medium"),
    },
)

# --------------------------------------------------------------------------------------------
# 5 · Comparación con pares
# --------------------------------------------------------------------------------------------
_anchor_section("ficha-pares", "Comparación con sus pares",
                "Percentil del estudiante en cada métrica frente a los créditos de su grupo de referencia (sin contar "
                "sus propios créditos). La franja amarilla es el rango típico p25–p75; los puntos amarillos son "
                "atípicos (≤ p10 o ≥ p90).", "Paso 3 · Pares")
g1, g2 = st.columns([3.0, 1], vertical_alignment="bottom")
with g1:
    choice = st.segmented_control(
        "Grupo de pares", peer_opts, key="ficha_pares", default=peer_group, required=True,
        format_func=lambda g: f"{g} · {fmt_int(peer_counts[g])}",
        help="El número indica cuántos créditos de pares hay en cada grupo (base activa completa).")
with g2:
    if peer_counts.get(peer_group, 0) < MIN_PEERS:
        _note(f"⚠️ Solo <b>{fmt_int(peer_counts.get(peer_group, 0))}</b> pares en este grupo: la comparación es "
              f"poco estable. Prueba un grupo más amplio.")
    else:
        _note(f"Referente: <b>{fmt_int(len(peers))}</b> créditos de {esc(peer_label)}.")

if pt.empty:
    empty_state("Sin métricas comparables", "El crédito analizado o sus pares no tienen datos numéricos suficientes.",
                icon="👥")
else:
    pc1, pc2 = st.columns([1.65, 1], gap="medium")
    with pc1:
        with st.container(border=True):
            show_fig(_fig_peers(pt), key="ficha_pares_fig", height=max(300, 58 * len(pt) + 70), legend=False)
    with pc2:
        rows = ""
        for _, r in pt.iterrows():
            dv = r["dif_pct"]
            dv_txt = "" if _is_na(dv) or r["kind"] in ("score",) else (
                f" <span style='color:var(--sat-muted);font-weight:500'>({'+' if dv >= 0 else '−'}"
                f"{fmt_num(abs(dv) * 100, 0)} %)</span>")
            rows += (f"<div class='row'><span>{esc(r['metrica'])}</span><b>{_nb(_fmt_metric(r['valor'], r['kind']))} "
                     f"<span style='color:var(--sat-muted);font-weight:500'>vs</span> "
                     f"{_nb(_fmt_metric(r['mediana'], r['kind']))}{dv_txt}</b></div>")
        if peers["has_truth"].any():
            pa_obs = float((peers.loc[peers["has_truth"], "y_true"].astype(str) == "Alto").mean())
            pa_pred = float((peers["y_pred"].astype(str) == "Alto").mean())
            pr_html = (f"<div class='row'><span>% Alto observado en pares</span><b>{fmt_pct(pa_obs)} "
                       f"<span style='color:var(--sat-muted);font-weight:500'>vs {fmt_pct(base_rate)} base</span></b></div>"
                       f"<div class='row'><span>% Alto predicho en pares</span><b>{fmt_pct(pa_pred)}</b></div>")
        else:
            pa_pred = float((peers["y_pred"].astype(str) == "Alto").mean()) if len(peers) else np.nan
            pr_html = f"<div class='row'><span>% Alto predicho en pares</span><b>{fmt_pct(pa_pred)}</b></div>"
        st.markdown(f"<div class='fx-side'><div class='t'>Estudiante vs mediana de pares</div>{rows}{pr_html}</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='fx-leg'><span><i style='background:{_rgba(YELLOW, .35)}'></i>Rango típico p25–p75</span>"
            f"<span><i style='background:{p['text']};border-radius:50%'></i>Estudiante</span>"
            f"<span><i style='background:{YELLOW};border-radius:50%;border:1.5px solid {p['text']}'></i>Atípico</span></div>",
            unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------
# 6 · Señales del perfil
# --------------------------------------------------------------------------------------------
_anchor_section("ficha-senales", "Señales del perfil basadas en datos",
                "Para cada atributo del crédito analizado, la tasa histórica de riesgo Alto observado del grupo al "
                "que pertenece frente al promedio de la base (lift). Rojo: el segmento tiene más riesgo; verde: menos.",
                "Paso 4 · Señales")
if sig.empty:
    empty_state("Sin señales calculables", "No hay riesgo observado suficiente para estimar tasas de segmento.",
                icon="🧭")
else:
    s1, s2 = st.columns([1.75, 1], gap="medium")
    with s1:
        with st.container(border=True):
            show_fig(_fig_signals(sig, base_rate), key="ficha_senales_fig", height=max(320, 40 * len(sig) + 90),
                     legend=False)
    with s2:
        n_up = int((sig["significativa"] & (sig["lift"] > 1)).sum())
        n_dn = int((sig["significativa"] & (sig["lift"] < 1)).sum())
        n_ns = int(len(sig) - n_up - n_dn)
        top_up = sig.sort_values("lift", ascending=False).iloc[0]
        top_dn = sig.sort_values("lift").iloc[0]
        st.markdown(
            f"<div class='fx-side'><div class='t'>Cómo leer las señales</div>"
            f"<div class='row'><span>Elevan el riesgo (significativas)</span><b style='color:{_risk_text_color('Alto')}'>"
            f"{n_up}</b></div><div class='row'><span>Reducen el riesgo (significativas)</span>"
            f"<b style='color:{_risk_text_color('Bajo')}'>{n_dn}</b></div>"
            f"<div class='row'><span>Sin diferencia clara</span><b>{n_ns}</b></div>"
            f"<div class='row'><span>Mayor lift</span><b>{esc(top_up['valor'])} · {_es_x(top_up['lift'])}</b></div>"
            f"<div class='row'><span>Menor lift</span><b>{esc(top_dn['valor'])} · {_es_x(top_dn['lift'])}</b></div>"
            f"<ul><li><b>Lift</b> = tasa de Alto observado del segmento ÷ promedio de la base "
            f"({fmt_pct(base_rate)}, {fmt_int(ref_tables['n'])} créditos · {esc(ref_source)}).</li>"
            f"<li>Bigotes: intervalo de confianza del 95 % (Wilson). Color tenue = el intervalo incluye el promedio; "
            f"gris = menos de {MIN_SIGNAL_N} créditos.</li>"
            f"<li><b>Asociaciones de segmento, no causalidad</b> ni explicación individual del modelo (no son "
            f"valores SHAP): describen el historial del grupo, no la razón de esta predicción.</li></ul></div>",
            unsafe_allow_html=True)
        st.markdown(
            f"<div class='fx-leg'><span><i style='background:{RISK_COLORS['Alto']}'></i>Más riesgo</span>"
            f"<span><i style='background:{RISK_COLORS['Bajo']}'></i>Menos riesgo</span>"
            f"<span><i style='background:{_rgba(RISK_COLORS['Alto'], .42)}'></i>No significativa</span>"
            f"<span><i style='background:{_rgba('#8A8A80', .45)}'></i>Muestra pequeña</span></div>",
            unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------
# 7 · Plan de acción + bitácora
# --------------------------------------------------------------------------------------------
_anchor_section("ficha-plan", "Plan de acción",
                "La ruta recomendada convertida en pasos concretos, un guion listo para usar y el registro de cada "
                "gestión realizada con el estudiante.", "Paso 5 · Actuar")
today = date.today()
if route == "R1":
    deadline = _add_business_days(today, 2)
    deadline_txt = f"{fmt_date(deadline)} · 2 días hábiles"
elif route == "R2":
    deadline = _add_business_days(today, 5)
    deadline_txt = f"{fmt_date(deadline)} · 5 días hábiles"
elif route == "R3":
    deadline = (_next_payment(dia, today) - timedelta(days=3)) if dia else today
    deadline = max(deadline, today)
    deadline_txt = f"{fmt_date(deadline)} · recordatorio automático"
else:
    deadline = today + timedelta(days=30)
    deadline_txt = f"{fmt_date(deadline)} · próximo corte mensual"
next_pay_txt = fmt_date(_next_payment(dia, today)) if dia else "—"

talking = []
if dia:
    talking.append(f"Cuota de referencia de <b>{cuota_txt}</b>; paga el <b>día {dia}</b> de cada mes "
                   f"(próximo: {next_pay_txt}).")
rt = rec.get("ratio_financiacion")
if not _is_na(rt):
    over = float(rt) > MAX_FINANCIACION_PCT
    talking.append(f"Financia el <b>{fmt_pct(rt, 0)}</b> de la matrícula neta"
                   + (" — <b>por encima del límite DA-02 (85 %)</b>." if over else "."))
if mora_si:
    talking.append("Tiene <b>reporte de mora en Datacrédito</b>: indagar por otras obligaciones y su capacidad de pago.")
if not _is_na(rec.get("score_rank")) and float(rec["score_rank"]) <= 2:
    talking.append(f"Scoring externo bajo (<b>{esc(score_txt)}</b>): priorizar alternativas de pago realistas.")
if not _is_na(rec.get("antiguedad_meses")) and float(rec["antiguedad_meses"]) < 12:
    talking.append(f"Crédito reciente (<b>{fmt_int(rec['antiguedad_meses'])} meses</b>): momento clave para "
                   "consolidar el hábito de pago.")
if len(credits) > 1:
    talking.append(f"Relación de <b>{len(credits)} créditos</b> con la institución desde {esc(_period_label(first_date))}.")
if not talking:
    talking.append("Sin alertas adicionales en los datos del crédito.")

chk_items = [t.format(cuota=cuota_txt, dia=dia if dia else "—") for t in CHECKLISTS.get(route, [])]
chk_keys = [f"ficha_chk_{sid}_{sel_llave}_{i}" for i in range(len(chk_items))]
guion = GUIONES.get(route, "").format(nombre=first_name, gestor=gestor_name, dia=dia if dia else "—")

logs_all = ss.setdefault(LOG_KEY, [])
logs_student = [e for e in logs_all if e.get("id_estudiante") == sid]

pl1, pl2 = st.columns([1.12, 1], gap="medium")
with pl1:
    st.markdown(
        f"<div class='fx-plan' style='--rc:{_rc(route)};--ri:{_ri(route)}'><div class='hd'><span class='code'>{route}</span>"
        f"<span class='sat-badge neutral'>SLA · {esc(act['sla'])}</span></div><div class='nm'>{esc(act['nombre'])}</div>"
        f"<div class='act'>{esc(act['accion'])}</div><div class='g3'>"
        f"<div><div class='fx-k'>Fecha límite sugerida</div><div class='fx-v'>{esc(deadline_txt)}</div></div>"
        f"<div><div class='fx-k'>Canal sugerido</div><div class='fx-v'>{esc(CANAL_GUION.get(route, ''))}</div></div>"
        f"<div><div class='fx-k'>Responsable</div><div class='fx-v'>{esc(gestor_name)}</div></div></div></div>",
        unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        done_prev = sum(bool(ss.get(k, False)) for k in chk_keys)
        st.markdown(f"<div class='fx-status'><b>Checklist de la ruta {route}</b> · {done_prev} de {len(chk_items)} "
                    f"pasos completados</div>", unsafe_allow_html=True)
        st.progress(done_prev / max(len(chk_items), 1))
        for txt_i, k in zip(chk_items, chk_keys):
            st.checkbox(txt_i, key=k, persist_state="session")
    st.markdown(
        f"<div class='fx-script'><div class='t'>💬 Guion sugerido · {esc(CANAL_GUION.get(route, ''))}</div>"
        f"<div class='q'>{esc(guion)}</div><div class='f'>Personalizado con el nombre, el gestor y el día de pago. "
        f"Ajústalo al protocolo de Cartera.</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='fx-side' style='margin-top:10px'><div class='t'>🗣️ Puntos de conversación basados en datos"
                "</div><ul>" + "".join(f"<li>{t}</li>" for t in talking) + "</ul></div>", unsafe_allow_html=True)

with pl2:
    last = logs_student[-1] if logs_student else None
    status_html = (f"Última gestión: <b>{esc(last['fecha_gestion'])}</b> · {esc(last['canal'])} · {esc(last['resultado'])}"
                   if last else "Sin gestiones registradas para este estudiante en la sesión.")
    with st.form("ficha_form", clear_on_submit=True, border=True):
        st.markdown(f"<div class='fx-status'><b>📝 Registrar gestión</b></div><div class='fx-note'>{status_html}</div>",
                    unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        f_fecha = f1.date_input("Fecha de la gestión", value=today, max_value=today + timedelta(days=1),
                                format="DD/MM/YYYY", key="ficha_f_fecha")
        f_canal = f2.selectbox("Canal", CANALES, index=CANALES.index(ROUTE_CANAL.get(route, "Llamada")),
                               key="ficha_f_canal")
        f_res = st.selectbox("Resultado", RESULTADOS, key="ficha_f_res")
        f3, f4 = st.columns(2)
        f_monto = f3.number_input("Monto comprometido (COP)", min_value=0, step=50_000,
                                  value=int(round(float(cuota_ref))) if not _is_na(cuota_ref) else 0,
                                  key="ficha_f_monto", help="Solo aplica si hay compromiso de pago.")
        f_fcomp = f4.date_input("Fecha del compromiso", value=today + timedelta(days=7), format="DD/MM/YYYY",
                                key="ficha_f_fcomp")
        f_comp = st.checkbox("El estudiante asumió un compromiso de pago", key="ficha_f_comp",
                             help="Se marca automáticamente si el resultado es «con compromiso de pago».")
        f_notas = st.text_area("Notas", placeholder="Resumen de la conversación, acuerdos y próximos pasos…",
                               max_chars=600, key="ficha_f_notas", height=92)
        submitted = st.form_submit_button("Registrar gestión", type="primary", icon=":material/add_task:",
                                          width="stretch", key="ficha_f_submit")
    if submitted:
        f_comp = bool(f_comp or f_res == RESULTADOS[0])
        if f_comp and (not f_monto or f_monto <= 0):
            st.error("Indica el monto del compromiso de pago (mayor que cero) o desmarca el compromiso.", icon="⚠️")
        else:
            entry = {
                "registrado": datetime.now().strftime("%Y-%m-%d %H:%M"), "id_estudiante": sid,
                "nombre": str(rec["nombre"]), "llave2": str(sel_llave), "ruta": route,
                "fecha_gestion": f_fecha.isoformat() if f_fecha else today.isoformat(), "canal": f_canal,
                "resultado": f_res, "compromiso": "Sí" if f_comp else "No",
                "monto_compromiso": int(f_monto) if f_comp else None,
                "fecha_compromiso": f_fcomp.isoformat() if (f_comp and f_fcomp) else "",
                "notas": (f_notas or "").strip(), "gestor": gestor_name,
            }
            logs_all.append(entry)
            logs_student.append(entry)
            st.toast("Gestión registrada en la bitácora de la sesión", icon="✅")
    st.markdown("<div class='fx-note'>🗒️ <b>Bitácora de sesión</b>: vive mientras dure tu sesión en el tablero. "
                "La integración con el sistema de gestión de cartera es el requerimiento <b>DB-04</b> (parcial).</div>",
                unsafe_allow_html=True)

log_df_student = _log_frame(logs_student)
log_df_all = _log_frame(logs_all)
st.markdown(f"<div class='fx-status' style='margin-top:10px'><b>Bitácora del estudiante</b> · {len(logs_student)} "
            f"gestión{'es' if len(logs_student) != 1 else ''} · {len(logs_all)} en la sesión para "
            f"{len({e['id_estudiante'] for e in logs_all})} estudiante(s)</div>", unsafe_allow_html=True)
if log_df_student.empty:
    st.markdown("<div class='sat-empty' style='padding:18px'>Aún no hay gestiones para este estudiante. Usa el "
                "formulario «Registrar gestión».</div>", unsafe_allow_html=True)
else:
    st.dataframe(log_df_student.drop(columns=["ID estudiante", "Estudiante"]).iloc[::-1], hide_index=True,
                 width="stretch", key="ficha_bitacora_tabla",
                 column_order=["Fecha de gestión", "Canal", "Resultado", "Compromiso", "Monto comprometido",
                               "Fecha del compromiso", "Notas", "Ruta", "Crédito", "Gestor", "Registrado"],
                 column_config={"Monto comprometido": st.column_config.NumberColumn(format="localized"),
                                "Notas": st.column_config.TextColumn("Notas", width="large"),
                                "Compromiso": st.column_config.TextColumn("Compromiso", width=96)})
if logs_all:
    b1, b2 = st.columns([4, 1.2], vertical_alignment="center")
    with b1:
        download_bar(log_df_all, f"bitacora_sesion_{today.isoformat()}", key="ficha_bit", label="Bitácora")
    with b2:
        st.button("Deshacer último registro", icon=":material/undo:", key="ficha_undo", width="stretch",
                  on_click=_undo_last, args=(sid,), disabled=not logs_student)

# --------------------------------------------------------------------------------------------
# 8 · Exportar
# --------------------------------------------------------------------------------------------
_anchor_section("ficha-exportar", "Exportar la ficha",
                "Un archivo HTML autocontenido (sin conexión, listo para imprimir o guardar como PDF) con el perfil, "
                "el riesgo, los créditos, los pares, las señales, el plan y la bitácora; y las tablas en CSV/Excel.",
                "Paso 6 · Exportar")
rows_peer_export = pt.drop(columns=["col", "kind"]).rename(columns={
    "metrica": "Métrica", "valor": "Estudiante", "mediana": "Mediana pares", "p25": "p25", "p75": "p75",
    "percentil": "Percentil", "n": "Pares con dato", "dif_pct": "Diferencia vs mediana"}) if not pt.empty else pd.DataFrame()
sig_export = sig.rename(columns={
    "atributo": "Atributo", "valor": "Valor", "n": "Créditos", "altos": "Alto observado", "tasa": "Tasa Alto",
    "ic_lo": "IC95 inf", "ic_hi": "IC95 sup", "lift": "Lift", "lift_lo": "Lift inf", "lift_hi": "Lift sup",
    "significativa": "Significativa", "muestra_pequena": "Muestra pequeña"}) if not sig.empty else pd.DataFrame()
ctx = {
    "rec": rec, "rk": rk, "yt": yt, "conf": conf, "route": route, "sel_llave": str(sel_llave), "rank_txt": rank_txt,
    "profile": profile_pairs + [("Scoring externo", score_txt), ("Mora Datacrédito", "Sí" if mora_si else "No"),
                                ("Día de pago", f"Día {dia}" if dia else "—")],
    "credits": credits, "peers": pt, "peer_label": peer_label, "signals": sig, "base": base_rate,
    "checklist": [(t, bool(ss.get(k, False))) for t, k in zip(chk_items, chk_keys)],
    "talking": [t.replace("<b>", "").replace("</b>", "") for t in talking], "guion": guion,
    "deadline": deadline_txt, "log": log_df_student, "gestor": gestor_name,
    "generated": datetime.now().strftime("%d/%m/%Y %H:%M"), "source": get_active_meta().get("name", ""),
    "risk_note": (f"Confianza = probabilidad de la clase predicha. Fiabilidad histórica de la etiqueta «{rk}»: "
                  f"{fmt_pct(label_prec, 0)}." if not _is_na(label_prec) else "Confianza = probabilidad de la clase "
                                                                                 "predicha."),
    "signal_note": (f"Lift = tasa de Alto observado del segmento ÷ promedio ({fmt_pct(base_rate)}; {ref_source}). "
                    "Significativa = el IC 95 % de Wilson excluye el promedio. Asociaciones, no causalidad."),
}
ficha_html = _ficha_html(ctx)

e1, e2 = st.columns([1.5, 1], gap="medium")
with e1:
    st.markdown(
        f"<div class='fx-export'><div class='ttl'>📄 Ficha 360° de {esc(rec['nombre'])}</div>"
        f"Incluye {len(credits)} crédito(s), {len(pt)} métricas de pares, {len(sig)} señales, el checklist "
        f"({sum(d for _, d in ctx['checklist'])}/{len(chk_items)} pasos) y {len(logs_student)} gestión(es) de la "
        f"bitácora. Se abre en cualquier navegador <b>sin conexión</b> y se imprime en PDF desde el mismo "
        f"navegador.</div>", unsafe_allow_html=True)
with e2:
    st.download_button("Descargar ficha (HTML)", ficha_html.encode("utf-8"), file_name=f"ficha_360_{sid}.html",
                       mime="text/html", key="ficha_dl_html", type="primary", icon=":material/download:",
                       width="stretch")

    @st.dialog("Vista previa de la ficha", width="large")
    def _preview(doc: str) -> None:
        st.iframe(doc, height=640) if hasattr(st, "iframe") else st.components.v1.html(doc, height=640, scrolling=True)

    if st.button("Vista previa", icon=":material/visibility:", key="ficha_preview", width="stretch"):
        _preview(ficha_html)

cred_export = credits[["llave2", "fecha_aprobacion", "programa", "facultad", "valor_financiacion", "vr_neto_matricula",
                       "ratio_financiacion", "cuotas", "tipo_interes", "valor_primera_cuota", "fecha_de_pago",
                       "y_pred", "proba_pred", "y_true", "prioridad", "ruta", "exposicion_riesgo"]].rename(
    columns={"llave2": "Crédito", "fecha_aprobacion": "Aprobación", "programa": "Programa", "facultad": "Facultad",
             "valor_financiacion": "Valor financiado", "vr_neto_matricula": "Matrícula neta",
             "ratio_financiacion": "% financiado", "cuotas": "Cuotas", "tipo_interes": "Tipo de interés",
             "valor_primera_cuota": "Primera cuota", "fecha_de_pago": "Día de pago", "y_pred": "Riesgo predicho",
             "proba_pred": "Confianza", "y_true": "Riesgo observado", "prioridad": "Prioridad", "ruta": "Ruta",
             "exposicion_riesgo": "Exposición en riesgo"})
extra = {}
if not rows_peer_export.empty:
    extra["pares"] = rows_peer_export
if not sig_export.empty:
    extra["senales"] = sig_export
if not log_df_student.empty:
    extra["bitacora"] = log_df_student
download_bar(cred_export, f"ficha_360_{sid}_creditos", key="ficha_dl", extra_sheets=extra, label="Créditos")

footer()
