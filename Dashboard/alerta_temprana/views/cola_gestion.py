"""Cola de gestión — ¿a quién contactamos esta semana, por qué ruta, con qué gestor y con qué impacto?

Corazón operativo del SAT para el equipo de Cartera: convierte las predicciones del modelo en una
cola priorizada de gestión preventiva con rutas de cobro (R1–R4), la dimensiona contra la capacidad
del equipo, la reparte entre gestores, proyecta un escenario ilustrativo de recaudo y la exporta
(DB-02) en el archivo de intercambio con el sistema de gestión de cartera (DB-04).
"""
from __future__ import annotations

import io
import json
import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import (badge, empty_state, esc, filter_chips, footer, insight, kpi_card, page_header,
                             risk_badge)
from core.config import ACTION_ROUTES, RECAUDO_BASELINE, RISK_COLORS, RISK_ICONS
from core.data import DEFAULT_WEIGHTS, compute_priority, get_active_df, to_excel_bytes
from core.filters import active_chips, apply_filters
from core.nav import can_access, open_ficha
from core.theme import fmt_cop, fmt_delta_pp, fmt_int, fmt_num, fmt_pct, pal, show_fig, theme_mode

# ============================================================================================
# Constantes de la página
# ============================================================================================
ROUTES = list(ACTION_ROUTES)                       # ["R1", "R2", "R3", "R4"]
ROUTE_SHORT = {"R1": "Inmediato", "R2": "Preventivo", "R3": "Automático", "R4": "Monitoreo"}
ROUTE_EMOJI = {"R1": "🟥", "R2": "🟧", "R3": "🟦", "R4": "🟩"}
HUMAN_ROUTES = {"R1", "R2"}                        # rutas que consumen tiempo de gestor por diseño
DIAS_HABILES = 5
SLA_DIAS = {"R1": 2, "R2": 5}                      # 48 h ≈ 2 días hábiles · 5 días

_CFG_KEY = "_cola_cfg"
_DEFAULTS = {
    "w_riesgo": int(round(DEFAULT_WEIGHTS["riesgo"] * 100)),
    "w_exposicion": int(round(DEFAULT_WEIGHTS["exposicion"] * 100)),
    "w_mora": int(round(DEFAULT_WEIGHTS["mora"] * 100)),
    "rutas": ["R1", "R2"],
    "capacidad": 200,
    "gestores": 4,
    "reparto": "Round-robin",
    "vista": "Plan de la semana",
    "gestor_ver": "Todos los gestores",
    "horizonte": "12 semanas",
    # Escenario ilustrativo de recaudo (supuestos editables, en %)
    "sc_semanas": 4,
    "sc_contacto": 70,
    "sc_ef_R1": 40,
    "sc_ef_R2": 30,
    "sc_ef_R3": 15,
    "sc_ef_R4": 5,
    "sc_auto_r3": True,
}
_SCENARIO_KEYS = ["sc_semanas", "sc_contacto", "sc_ef_R1", "sc_ef_R2", "sc_ef_R3", "sc_ef_R4", "sc_auto_r3"]

# Esquema fijo del archivo de intercambio con el sistema de gestión de cartera (DB-04).
DB04_SCHEMA = [
    ("id_estudiante", "texto", True, "Identificador institucional del estudiante.", "100046229"),
    ("llave2", "texto", True, "Llave única del crédito (estudiante_periodo_consecutivo).", "100046229_202508_93130"),
    ("nombre", "texto", True, "Nombre del estudiante (anonimizado en este prototipo).", "Laura Gómez Rojas"),
    ("programa", "texto", True, "Programa académico asociado al crédito.", "Psicología"),
    ("riesgo_predicho", "texto {Alto, Medio, Bajo}", True, "Clase de riesgo predicha por el modelo (Random Forest).",
     "Alto"),
    ("confianza", "decimal [0, 1] · punto decimal", False,
     "Probabilidad de la clase predicha (confianza del modelo). Vacío si el motor no la entrega.", "0.6123"),
    ("prioridad", "decimal [0, 100] · punto decimal", True,
     "Índice de prioridad ponderado (riesgo, exposición, mora) con los pesos vigentes.", "71.4"),
    ("ruta", "texto {R1, R2, R3, R4}", True, "Ruta de gestión preventiva asignada.", "R1"),
    ("accion", "texto", True, "Acción recomendada para la ruta.", ACTION_ROUTES["R1"]["accion"]),
    ("sla", "texto", True, "Tiempo máximo de atención de la ruta.", ACTION_ROUTES["R1"]["sla"]),
    ("valor_financiado", "entero (COP)", True, "Valor financiado del crédito en pesos colombianos.", "2345000"),
    ("gestor_asignado", "texto 'Gestor k'", True, "Gestor responsable según el reparto de la semana.", "Gestor 1"),
    ("fecha_asignacion", "fecha ISO 8601 (AAAA-MM-DD)", True, "Fecha en que se genera la asignación.",
     date.today().isoformat()),
    ("estado", "texto {PENDIENTE}", True, "Estado inicial del caso en el sistema de cartera.", "PENDIENTE"),
]
DB04_COLUMNS = [c[0] for c in DB04_SCHEMA]

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


# ============================================================================================
# Estado persistente de la configuración (sobrevive a la navegación entre páginas)
# ============================================================================================
def _cfg() -> dict:
    if _CFG_KEY not in st.session_state:
        st.session_state[_CFG_KEY] = {k: (list(v) if isinstance(v, list) else v) for k, v in _DEFAULTS.items()}
    cfg = st.session_state[_CFG_KEY]
    for k, v in _DEFAULTS.items():
        cfg.setdefault(k, list(v) if isinstance(v, list) else v)
    return cfg


def _sync(name: str) -> None:
    st.session_state[_CFG_KEY][name] = st.session_state.get(f"cola_{name}")


def _bind(name: str) -> dict:
    """Prepara el estado del widget desde la configuración persistente y devuelve sus kwargs."""
    k = f"cola_{name}"
    st.session_state[k] = _cfg()[name]
    return {"key": k, "on_change": _sync, "args": (name,)}


def _reset(names: list[str]) -> None:
    cfg = _cfg()
    for n in names:
        v = _DEFAULTS[n]
        cfg[n] = list(v) if isinstance(v, list) else v


# ============================================================================================
# Utilidades visuales
# ============================================================================================
def _dark() -> bool:
    return theme_mode() == "dark"


def _rc(r: str) -> str:
    """Color de marca de la ruta (marcas gráficas)."""
    if r == "R3" and _dark():
        return "#7C9CFF"
    return ACTION_ROUTES.get(r, {}).get("color", "#888888")


def _ri(r: str) -> str:
    """Color de texto legible de la ruta (chips y cifras)."""
    dark = _dark()
    return {"R1": "#FF6B70" if dark else "#D13438", "R2": "#FFB547" if dark else "#A8650A",
            "R3": "#8FA8FF" if dark else "#3E63DD", "R4": "#46C487" if dark else "#218358"}.get(r, "inherit")


def _route_label(r: str) -> str:
    return f"{r} · {ROUTE_SHORT.get(r, r)}"


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _anchor_section(step: int, title: str, subtitle: str, kicker: str) -> None:
    """Sección con la misma estética de ``core.components.section`` + ancla para el recorrido."""
    st.markdown(
        f"""<div class="sat-section cola-anchor" id="paso-{step}"><div class="kicker">{esc(kicker)}</div>
        <h3>{esc(title)}</h3><p>{esc(subtitle)}</p></div>""",
        unsafe_allow_html=True,
    )


def _es(x: float, d: int = 0) -> str:
    return fmt_num(x, d)


def _nb(text: str) -> str:
    """Texto con espacios no separables (evita cortes como '$' / '632,1 M' en dos líneas)."""
    return str(text).replace(" ", "&nbsp;")


def _grid(items: list[str], min_px: int = 200) -> None:
    """Rejilla CSS responsiva: reacomoda tarjetas HTML según el ancho disponible."""
    st.markdown(f"<div class='cola-grid' style='--min:{min_px}px'>" + "".join(f"<div>{h}</div>" for h in items)
                + "</div>", unsafe_allow_html=True)


def _css() -> None:
    st.markdown(
        """
        <style>
        /* ---- Rejillas responsivas (se reacomodan según el ancho disponible) ---- */
        .cola-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(var(--min,200px),100%),1fr));
          gap:12px;align-items:stretch;}
        .cola-grid > div{min-width:0;}
        /* En pantallas medianas/pequeñas las columnas de Streamlit se apilan en lugar de comprimirse. */
        @media (max-width: 1180px){
          [data-testid="stMain"] [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important;}
          [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
            flex:1 1 100% !important;width:100% !important;min-width:100% !important;}
        }
        .st-key-cola_rutas [role="toolbar"]{flex-wrap:wrap !important;overflow:visible !important;row-gap:6px;}
        .cola-steps{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:2px 0 12px 0;}
        .cola-steps a{display:inline-flex;align-items:center;gap:8px;text-decoration:none !important;font-size:12.5px;
          font-weight:600;color:var(--sat-text) !important;background:var(--sat-surface);border:1px solid var(--sat-border);
          border-radius:999px;padding:4px 12px 4px 4px;transition:border-color .15s, transform .15s;}
        .cola-steps a:hover{border-color:var(--sat-accent);transform:translateY(-1px);}
        .cola-steps a b{width:22px;height:22px;border-radius:50%;background:var(--sat-accent);color:#141414;
          display:inline-flex;align-items:center;justify-content:center;font-size:11.5px;font-weight:800;}
        .cola-steps .sep{color:var(--sat-subtle);font-size:12px;}
        .cola-anchor{scroll-margin-top:70px;}

        .cola-cfg{font-size:12.5px;color:var(--sat-muted);line-height:1.6;}
        .cola-cfg b{color:var(--sat-text);}
        .cola-cfg .ln{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
        .cola-wbar{display:inline-flex;width:120px;height:8px;border-radius:6px;overflow:hidden;gap:2px;
          background:var(--sat-surface-2);vertical-align:middle;}
        .cola-wbar i{display:block;height:100%;}

        .cola-route{position:relative;background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:16px;
          padding:14px 16px 14px 20px;box-shadow:var(--sat-shadow);height:100%;overflow:hidden;}
        .cola-route:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--rc);}
        .cola-route.off{opacity:.58;}
        .cola-route .top{display:flex;justify-content:space-between;align-items:center;gap:8px;}
        .cola-route .code{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:12.5px;letter-spacing:.04em;
          padding:2px 9px;border-radius:7px;color:var(--ri);background:color-mix(in srgb, var(--rc) 15%, transparent);}
        .cola-route .sla{font-size:11.5px;font-weight:700;color:var(--sat-muted);border:1px solid var(--sat-border);
          padding:2px 9px;border-radius:999px;white-space:nowrap;}
        .cola-route .name{font-weight:700;font-size:14px;margin-top:8px;color:var(--sat-text);white-space:nowrap;
          overflow:hidden;text-overflow:ellipsis;}
        .cola-route .big{font-family:'Space Grotesk','Inter',sans-serif;font-size:30px;font-weight:700;color:var(--sat-text);
          line-height:1.1;margin-top:4px;letter-spacing:-.02em;}
        .cola-route .big small{font-family:'Inter',sans-serif;font-size:12.5px;font-weight:500;color:var(--sat-muted);
          margin-left:6px;letter-spacing:0;}
        .cola-route .grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;margin-top:10px;}
        .cola-route .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--sat-subtle);font-weight:700;}
        .cola-route .v{font-size:14px;font-weight:700;color:var(--sat-text);}
        .cola-route .obs{margin-top:12px;padding-top:10px;border-top:1px solid var(--sat-border);}
        .cola-route .row{display:flex;justify-content:space-between;align-items:baseline;gap:8px;font-size:12.5px;
          color:var(--sat-muted);}
        .cola-route .row b{color:var(--sat-text);font-size:15px;}
        .cola-route .lift{font-weight:800;color:var(--ri);white-space:nowrap;}
        .cola-route .meter{position:relative;height:8px;border-radius:6px;background:var(--sat-surface-2);
          margin:7px 0 3px 0;border:1px solid var(--sat-border);}
        .cola-route .meter i{position:absolute;left:0;top:0;bottom:0;border-radius:6px;background:var(--rc);}
        .cola-route .meter u{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--sat-text);border-radius:2px;}
        .cola-route .legend{font-size:11px;color:var(--sat-subtle);}
        .cola-route .act{font-size:12.5px;color:var(--sat-muted);margin-top:10px;line-height:1.45;min-height:36px;}
        .cola-route .foot{margin-top:10px;display:flex;gap:6px;flex-wrap:wrap;}

        .cola-stat{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:14px;padding:12px 14px;
          box-shadow:var(--sat-shadow);height:100%;}
        .cola-stat .k{font-size:12px;font-weight:600;color:var(--sat-muted);}
        .cola-stat .v{font-family:'Space Grotesk','Inter',sans-serif;font-size:26px;font-weight:700;color:var(--sat-text);
          line-height:1.15;letter-spacing:-.02em;}
        .cola-stat .s{font-size:12px;color:var(--sat-muted);margin-top:2px;}
        .cola-stat .s b{color:var(--sat-text);}
        .cola-stat.hl{border-left:4px solid var(--sat-accent);}

        .cola-case{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:16px;padding:16px 18px;
          box-shadow:var(--sat-shadow);}
        .cola-case .hd{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;}
        .cola-case .nm{font-family:'Space Grotesk','Inter',sans-serif;font-size:20px;font-weight:700;color:var(--sat-text);}
        .cola-case .meta{font-size:12.5px;color:var(--sat-muted);margin-top:2px;}
        .cola-case .chips{display:flex;gap:6px;flex-wrap:wrap;align-items:center;}
        .cola-case .g4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:14px;}
        .cola-case .g4 > div{background:var(--sat-surface-2);border:1px solid var(--sat-border);border-radius:12px;padding:8px 10px;}
        .cola-case .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--sat-subtle);font-weight:700;}
        .cola-case .v{font-size:16px;font-weight:700;color:var(--sat-text);margin-top:2px;}
        .cola-case .why{margin-top:14px;}
        .cola-case .why .t{font-size:12.5px;font-weight:700;color:var(--sat-text);margin-bottom:6px;}
        .cola-case .stack{display:flex;height:12px;border-radius:7px;overflow:hidden;gap:2px;background:var(--sat-surface-2);}
        .cola-case .stack i{display:block;height:100%;}
        .cola-case .lg{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--sat-muted);margin-top:6px;}
        .cola-case .lg span:before{content:"";display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:6px;
          background:var(--c);vertical-align:-1px;}
        .cola-case .act{margin-top:12px;font-size:13px;color:var(--sat-muted);line-height:1.5;}
        .cola-case .act b{color:var(--sat-text);}
        @media (max-width: 1100px){ .cola-case .g4{grid-template-columns:repeat(2,minmax(0,1fr));} }

        .cola-script{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:14px;padding:12px 14px;
          margin-top:10px;box-shadow:var(--sat-shadow);}
        .cola-script .t{font-size:12.5px;font-weight:700;color:var(--sat-text);margin-bottom:6px;}
        .cola-script .q{font-size:13.5px;line-height:1.5;color:var(--sat-text);border-left:3px solid var(--sat-accent);
          padding:2px 0 2px 10px;font-style:italic;}
        .cola-script .f{font-size:11.5px;color:var(--sat-subtle);margin-top:8px;}
        .cola-banner{display:flex;gap:12px;align-items:flex-start;border-radius:14px;padding:12px 16px;
          background:color-mix(in srgb, var(--sat-medio) 11%, var(--sat-surface));border:1px solid color-mix(in srgb, var(--sat-medio) 45%, transparent);
          color:var(--sat-text);font-size:13.5px;line-height:1.5;margin-bottom:10px;}
        .cola-banner .tag{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;background:var(--sat-medio);
          color:#141414;padding:3px 9px;border-radius:999px;white-space:nowrap;}
        .cola-tests{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:13px;color:var(--sat-muted);margin:4px 0 8px 0;}
        .cola-tests b{color:var(--sat-text);}
        .cola-mini{font-size:12.5px;color:var(--sat-muted);margin:-2px 0 6px 0;}
        .cola-files{display:flex;flex-direction:column;gap:6px;margin:10px 0 8px 0;}
        .cola-files div{display:grid;grid-template-columns:1fr auto;gap:2px 8px;align-items:baseline;font-size:12.5px;
          color:var(--sat-muted);padding:7px 10px;border:1px solid var(--sat-border);border-radius:10px;background:var(--sat-surface-2);}
        .cola-files code{font-size:12px;color:var(--sat-text);background:transparent;padding:0;font-weight:600;}
        .cola-files b{color:var(--sat-text);font-weight:700;text-align:right;}
        .cola-files span{grid-column:1 / -1;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================================
# Cálculos
# ============================================================================================
def _weights(cfg: dict) -> tuple[dict, bool, bool]:
    """Pesos normalizados, ¿son personalizados?, ¿se usó el respaldo por pesos en cero?"""
    raw = {"riesgo": cfg["w_riesgo"], "exposicion": cfg["w_exposicion"], "mora": cfg["w_mora"]}
    total = sum(float(v or 0) for v in raw.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS), False, True
    w = {k: float(v or 0) / total for k, v in raw.items()}
    custom = any(abs(w[k] - DEFAULT_WEIGHTS[k]) > 1e-6 for k in w)
    return w, custom, False


def _working_frame(df_all: pd.DataFrame, dff: pd.DataFrame, w: dict, custom: bool) -> pd.DataFrame:
    """Subconjunto filtrado con la prioridad vigente y los componentes del índice, ordenado por prioridad."""
    cols = ["llave2", "id_estudiante", "nombre", "programa", "facultad", "sede", "nivel", "y_pred", "y_true",
            "has_truth", "proba_pred", "valor_financiacion", "mora_flag", "intensidad_riesgo", "exposicion_riesgo",
            "ruta", "fecha_de_pago", "cuotas", "tipo_interes", "fecha_aprobacion", "prioridad"]
    W = dff[[c for c in cols if c in dff.columns]].copy()
    if custom:
        W["prioridad"] = compute_priority(df_all, w).reindex(W.index)
    # Percentil del monto sobre el dataset activo completo (igual que compute_priority).
    W["_e"] = df_all["valor_financiacion"].rank(pct=True).fillna(0.5).reindex(W.index)
    W["exposicion_riesgo"] = W["exposicion_riesgo"].fillna(0.0)
    W["valor_financiacion"] = W["valor_financiacion"].fillna(0.0)
    W["_alto_obs"] = (W["has_truth"] & (W["y_true"].astype(str) == "Alto")).astype(int)
    W["_idx"] = W.index
    W = W.sort_values(["prioridad", "exposicion_riesgo", "llave2"], ascending=[False, False, True],
                      kind="mergesort").reset_index(drop=True)
    return W


def _assign(queue: pd.DataFrame, capacity: int, n_gestores: int, strategy: str) -> pd.DataFrame:
    """Programa la cola priorizada por semanas (capacidad) y la reparte entre gestores."""
    q = queue.copy()
    pos = np.arange(len(q))
    q["_pos"] = pos
    q["semana"] = pos // capacity + 1
    in_week = pos % capacity
    if strategy == "Serpentina":
        rnd, off = in_week // n_gestores, in_week % n_gestores
        g = np.where(rnd % 2 == 0, off, n_gestores - 1 - off)
    else:  # Round-robin
        g = in_week % n_gestores
    q["_g"] = g + 1
    q["gestor"] = "Gestor " + q["_g"].astype(str)
    return q


def _compare_default(W: pd.DataFrame, df_all: pd.DataFrame, sel: list[str], capacity: int) -> dict:
    """Compara el plan de la semana con pesos personalizados frente al plan con los pesos por defecto."""
    m = W["ruta"].isin(sel).to_numpy()
    Q = W[m].assign(_p0=df_all["prioridad"].reindex(W.loc[m, "_idx"]).to_numpy())
    now = Q.head(capacity)
    dft = Q.sort_values(["_p0", "exposicion_riesgo", "llave2"], ascending=[False, False, True], kind="mergesort").head(capacity)
    inter = len(set(now["llave2"]) & set(dft["llave2"]))
    return {"overlap": inter / max(len(now), 1), "alto_now": int(now["_alto_obs"].sum()),
            "alto_def": int(dft["_alto_obs"].sum()), "exp_now": float(now["exposicion_riesgo"].sum()),
            "exp_def": float(dft["exposicion_riesgo"].sum())}


def _route_stats(W: pd.DataFrame, truth: bool) -> pd.DataFrame:
    rows = []
    n_all = max(len(W), 1)
    base = W["_alto_obs"].sum() / max(W["has_truth"].sum(), 1) if truth else np.nan
    tot_alto = W["_alto_obs"].sum()
    for r in ROUTES:
        s = W[W["ruta"] == r]
        nt = int(s["has_truth"].sum())
        rate = s["_alto_obs"].sum() / nt if (truth and nt) else np.nan
        rows.append({
            "ruta": r, "n": len(s), "share": len(s) / n_all, "exposicion": s["valor_financiacion"].sum(),
            "exp_riesgo": s["exposicion_riesgo"].sum(), "alto_rate": rate, "base": base,
            "lift": rate / base if (truth and base and not np.isnan(rate)) else np.nan,
            "alto_share": s["_alto_obs"].sum() / tot_alto if (truth and tot_alto) else np.nan,
            "alto_pred": (s["y_pred"].astype(str) == "Alto").mean() if len(s) else np.nan,
            "mora": s["mora_flag"].mean() if len(s) else np.nan,
            "prioridad": s["prioridad"].mean() if len(s) else np.nan,
        })
    return pd.DataFrame(rows)


def _scenario_gain(queue: pd.DataFrame, W: pd.DataFrame, sel: list[str], contacto: float, ef: dict,
                   auto_r3: bool) -> tuple[np.ndarray, pd.DataFrame, float]:
    """Ganancia acumulada (COP) de la gestión humana en el orden de la cola + aporte del canal automático R3."""
    g_row = (queue["exposicion_riesgo"].to_numpy() * contacto * queue["ruta"].map(ef).fillna(0).to_numpy())
    cum = np.concatenate([[0.0], np.cumsum(g_row)])
    auto = 0.0
    if auto_r3 and "R3" not in sel:
        auto = float(W.loc[W["ruta"] == "R3", "exposicion_riesgo"].sum() * ef.get("R3", 0))
    return cum, pd.DataFrame({"ruta": queue["ruta"].to_numpy(), "g": g_row}), auto


def _db04_frame(plan: pd.DataFrame, today: date) -> pd.DataFrame:
    return pd.DataFrame({
        "id_estudiante": plan["id_estudiante"].astype("string").fillna("").astype(str).to_numpy(),
        "llave2": plan["llave2"].astype("string").fillna("").astype(str).to_numpy(),
        "nombre": plan["nombre"].astype("string").fillna("").astype(str).to_numpy(),
        "programa": plan["programa"].astype("string").fillna("").astype(str).to_numpy(),
        "riesgo_predicho": plan["y_pred"].astype("string").fillna("").astype(str).to_numpy(),
        "confianza": plan["proba_pred"].astype(float).round(4).to_numpy(),
        "prioridad": plan["prioridad"].astype(float).round(1).to_numpy(),
        "ruta": plan["ruta"].astype(str).to_numpy(),
        "accion": plan["ruta"].map(lambda r: ACTION_ROUTES[r]["accion"]).to_numpy(),
        "sla": plan["ruta"].map(lambda r: ACTION_ROUTES[r]["sla"]).to_numpy(),
        "valor_financiado": pd.array(plan["valor_financiacion"].round(0).to_numpy(), dtype="Int64"),
        "gestor_asignado": plan["gestor"].astype(str).to_numpy(),
        "fecha_asignacion": today.isoformat(),
        "estado": "PENDIENTE",
    }, columns=DB04_COLUMNS)


def _db04_csv(ex: pd.DataFrame) -> bytes:
    return ex.to_csv(sep=";", index=False, decimal=".").encode("utf-8-sig")


def _db04_json(ex: pd.DataFrame, meta: dict) -> bytes:
    recs = json.loads(ex.to_json(orient="records", force_ascii=False))
    payload = {"esquema": "SAT-cola-gestion", "version": "1.0", "generado": datetime.now().isoformat(timespec="seconds"),
               **meta, "total_registros": len(recs), "campos": DB04_COLUMNS, "registros": recs}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _db04_tests(ex: pd.DataFrame, csv_b: bytes, json_b: bytes) -> list[tuple[str, bool, str]]:
    """Casos de prueba de integración automáticos sobre el archivo generado (criterio DB-04 ≥ 90 %)."""
    req = [c[0] for c in DB04_SCHEMA if c[2]]
    t = []
    t.append(("Columnas exactas y en el orden del esquema (14)", list(ex.columns) == DB04_COLUMNS,
              f"{len(ex.columns)} columnas"))
    nulls = int(ex[req].isna().sum().sum()) + int((ex[req].astype(str).apply(lambda s: s.str.strip()) == "").sum().sum())
    t.append(("Campos obligatorios sin vacíos", nulls == 0, f"{nulls} vacíos"))
    t.append(("llave2 única por registro", ex["llave2"].is_unique, f"{ex['llave2'].nunique()} únicas"))
    t.append(("riesgo_predicho ∈ {Alto, Medio, Bajo}", ex["riesgo_predicho"].isin(["Alto", "Medio", "Bajo"]).all(), ""))
    conf = ex["confianza"].dropna()
    t.append(("confianza ∈ [0, 1] (o vacía)", bool(((conf >= 0) & (conf <= 1)).all()), f"{len(conf)} con valor"))
    t.append(("prioridad ∈ [0, 100]", bool(ex["prioridad"].between(0, 100).all()), ""))
    t.append(("ruta ∈ {R1, R2, R3, R4}", ex["ruta"].isin(ROUTES).all(), ""))
    t.append(("accion y sla coherentes con la ruta",
              bool((ex["accion"] == ex["ruta"].map(lambda r: ACTION_ROUTES[r]["accion"])).all()
                   and (ex["sla"] == ex["ruta"].map(lambda r: ACTION_ROUTES[r]["sla"])).all()), ""))
    t.append(("valor_financiado entero ≥ 0", bool((ex["valor_financiado"].dropna() >= 0).all()), ""))
    t.append(("gestor_asignado con formato 'Gestor k'", bool(ex["gestor_asignado"].str.fullmatch(r"Gestor \d+").all()), ""))
    t.append(("fecha_asignacion en formato ISO (AAAA-MM-DD)",
              bool(pd.to_datetime(ex["fecha_asignacion"], format="%Y-%m-%d", errors="coerce").notna().all()), ""))
    t.append(("estado inicial = PENDIENTE", bool((ex["estado"] == "PENDIENTE").all()), ""))
    try:
        back = pd.read_csv(io.BytesIO(csv_b), sep=";", dtype=str, encoding="utf-8-sig")
        ok = list(back.columns) == DB04_COLUMNS and len(back) == len(ex) and back["llave2"].tolist() == ex["llave2"].tolist()
    except Exception:
        ok = False
    t.append(("CSV ';' se relee sin pérdida (ida y vuelta)", ok, "UTF-8 con BOM"))
    try:
        js = json.loads(json_b.decode("utf-8"))
        ok = js["total_registros"] == len(ex) and len(js["registros"]) == len(ex) and js["campos"] == DB04_COLUMNS
    except Exception:
        ok = False
    t.append(("JSON válido y completo (ida y vuelta)", ok, "UTF-8"))
    return [(a, bool(b), c) for a, b, c in t]


def _lazy_download(label: str, fn, file_name: str, mime: str, key: str, icon: str | None = None,
                   typ: str = "secondary") -> None:
    """Botón de descarga con generación diferida (el archivo se construye solo al hacer clic)."""
    st.download_button(label, data=fn, file_name=file_name, mime=mime, key=key, width="stretch", icon=icon,
                       type=typ, on_click="ignore")


# ============================================================================================
# Página
# ============================================================================================
df_all = get_active_df()
dff = apply_filters(df_all)
cfg = _cfg()
_css()

n_r1_hdr = int((dff["ruta"] == "R1").sum()) if len(dff) else 0
page_header(
    "Cola de gestión",
    "Convierte las predicciones en una cola priorizada con rutas de cobro, la dimensiona contra la capacidad del "
    "equipo, la reparte entre gestores y la envía al sistema de cartera.",
    eyebrow="Gestión de cartera",
    highlight=f"{fmt_int(n_r1_hdr)} casos para contacto inmediato",
)
filter_chips(active_chips(df_all))

if dff.empty:
    empty_state("La cartera filtrada está vacía", "Ajusta o limpia los filtros de la barra lateral para armar la cola.")
    footer()
    st.stop()

# --------------------------------------------------------------------------------------------
# Recorrido (anclas) + barra de control
# --------------------------------------------------------------------------------------------
steps = ["Priorizar", "Rutas", "Capacidad", "Cola", "Gestores", "Recaudo", "Exportar"]
st.markdown(
    "<div class='cola-steps cola-anchor' id='paso-1'>" + "<span class='sep'>›</span>".join(
        f"<a href='#paso-{i}' target='_self'><b>{i}</b>{esc(s)}</a>" for i, s in enumerate(steps, start=1)) + "</div>",
    unsafe_allow_html=True,
)

with st.container(border=True):
    c_r, c_w = st.columns([2.7, 1], vertical_alignment="bottom", gap="medium")
    with c_r:
        st.pills("Rutas que entran a la cola de gestores", ROUTES, selection_mode="multi",
                 format_func=lambda r: f"{ROUTE_EMOJI[r]} {_route_label(r)}",
                 help="Por diseño R1 y R2 requieren gestor. R3 se envía por canal automático y R4 es monitoreo; "
                      "inclúyelas si quieres gestionarlas manualmente.", **_bind("rutas"))
    with c_w:
        with st.popover("⚙️ Pesos de priorización", width="stretch",
                        help="Ajusta cuánto pesa cada componente en el índice de prioridad (0–100)"):
            st.markdown("**Índice de prioridad** = combinación ponderada de riesgo, exposición y mora "
                        "(los pesos se normalizan para sumar 100 %).")
            st.slider("Riesgo del modelo (clase + confianza)", 0, 100, step=5, format="%d %%", **_bind("w_riesgo"))
            st.slider("Exposición (percentil del valor financiado)", 0, 100, step=5, format="%d %%",
                      **_bind("w_exposicion"))
            st.slider("Mora histórica en Datacrédito", 0, 100, step=5, format="%d %%", **_bind("w_mora"))
            st.button("↺ Restablecer pesos", key="cola_reset_w", on_click=_reset,
                      args=(["w_riesgo", "w_exposicion", "w_mora"],), width="stretch")
            pop_cmp = st.empty()
    c_c, c_g, c_s = st.columns([1, 1, 2.7], vertical_alignment="center", gap="medium")
    with c_c:
        st.number_input("Gestiones por semana", min_value=10, max_value=20000, step=10,
                        help="Capacidad total de gestiones que el equipo puede realizar por semana.", **_bind("capacidad"))
    with c_g:
        st.number_input("Gestores", min_value=1, max_value=50, step=1,
                        help="Personas disponibles para la gestión preventiva.", **_bind("gestores"))

    w, custom_w, w_fallback = _weights(cfg)
    wcols = {"riesgo": "var(--sat-text)", "exposicion": "var(--sat-accent)", "mora": "#3E63DD"}
    wbar = "".join(f"<i style='width:{w[k] * 100:.1f}%;background:{wcols[k]}'></i>" for k in w if w[k] > 0)
    sel_routes = [r for r in (cfg["rutas"] or []) if r in ROUTES] or list(ROUTES)
    capacity = int(max(10, cfg["capacidad"] or 10))
    n_gest = int(max(1, cfg["gestores"] or 1))
    per_g = capacity / n_gest
    tag = badge("Pesos personalizados", "accent") if custom_w else badge("Pesos por defecto", "neutral")
    with c_s:
        st.markdown(
            f"<div class='cola-cfg'><div class='ln'>{tag}<span class='cola-wbar'>{wbar}</span>"
            f"<span>Riesgo <b>{fmt_pct(w['riesgo'], 0)}</b> · Exposición <b>{fmt_pct(w['exposicion'], 0)}</b> · "
            f"Mora <b>{fmt_pct(w['mora'], 0)}</b></span></div>"
            f"<div>Cola de gestores: <b>{esc(', '.join(sel_routes))}</b>"
            f"{' (todas: no seleccionaste rutas)' if not cfg['rutas'] else ''} · <b>{fmt_int(capacity)}</b> gestiones/semana "
            f"≈ <b>{_es(per_g, 1)}</b> por gestor (≈ {_es(per_g / DIAS_HABILES, 1)} al día)</div></div>",
            unsafe_allow_html=True,
        )
        if w_fallback:
            st.caption("⚠️ Todos los pesos están en 0: se usan los pesos por defecto.")
        sum_cmp = st.empty()

# --------------------------------------------------------------------------------------------
# Cálculo central
# --------------------------------------------------------------------------------------------
W = _working_frame(df_all, dff, w, custom_w)
truth = bool(W["has_truth"].sum() >= 30 and W.loc[W["has_truth"], "_alto_obs"].sum() > 0)
n_all = len(W)
tot_exp_r = float(W["exposicion_riesgo"].sum())
tot_alto = int(W["_alto_obs"].sum())
queue = _assign(W[W["ruta"].isin(sel_routes)], capacity, n_gest, cfg["reparto"])
n_q = len(queue)
plan = queue[queue["semana"] == 1]
n_plan = len(plan)
weeks_q = math.ceil(n_q / capacity) if n_q else 0
stats = _route_stats(W, truth)
rs = stats.set_index("ruta")

exp_q = float(queue["exposicion_riesgo"].sum())
exp_plan = float(plan["exposicion_riesgo"].sum())
alto_plan = int(plan["_alto_obs"].sum())
share_plan_rows = n_plan / max(n_all, 1)

if custom_w and n_q:
    cmpd = _compare_default(W, df_all, sel_routes, capacity)
    alto_txt = (f" · Alto observados capturados <b>{fmt_int(cmpd['alto_def'])} → {fmt_int(cmpd['alto_now'])}</b>"
                if truth else "")
    cmp_html = (f"<div class='cola-cfg'>Frente a los pesos por defecto, el plan de la semana comparte "
                f"<b>{fmt_pct(cmpd['overlap'], 0)}</b> de los casos{alto_txt} · exposición en riesgo "
                f"<b>{_nb(fmt_cop(cmpd['exp_def']))} → {_nb(fmt_cop(cmpd['exp_now']))}</b>.</div>")
    pop_cmp.markdown(cmp_html, unsafe_allow_html=True)
    sum_cmp.markdown(cmp_html, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------
# KPI
# --------------------------------------------------------------------------------------------
cap_share_exp = exp_plan / tot_exp_r if tot_exp_r else np.nan
if truth:
    cap_alto = alto_plan / tot_alto if tot_alto else np.nan
    lift_plan = cap_alto / share_plan_rows if share_plan_rows else np.nan
    k4 = kpi_card("Riesgo real capturado esta semana", fmt_pct(cap_alto), f"de los Alto observados · "
                  f"{_es(lift_plan, 1)}× un orden aleatorio", tone="bajo", bar=cap_alto, icon="🎯",
                  help="Porcentaje de los créditos con riesgo Alto observado (y_true) que quedan en el plan de la semana.")
else:
    k4 = kpi_card("Exposición en riesgo capturada", fmt_pct(cap_share_exp), "con el plan de la semana",
                  tone="bajo", bar=cap_share_exp, icon="🎯")
r1_q = int((queue["ruta"] == "R1").sum())
w_r1 = math.ceil(r1_q / capacity) if r1_q else 0
_grid([
    kpi_card("Cola de gestores", fmt_int(n_q), f"{', '.join(sel_routes)} · {fmt_pct(n_q / n_all)} de "
             f"{fmt_int(n_all)} créditos", tone="alto", bar=n_q / n_all, icon="📋"),
    kpi_card("Plan de esta semana", fmt_int(n_plan), f"{n_gest} gestores × {_es(per_g, 0)} gestiones",
             tone="accent", bar=(n_plan / n_q) if n_q else 0, icon="🗓️"),
    kpi_card("Exposición en riesgo en la cola", fmt_cop(exp_q), f"{fmt_pct(exp_q / tot_exp_r if tot_exp_r else np.nan)} "
             f"del total en riesgo ({fmt_cop(tot_exp_r)})", tone="medio",
             bar=(exp_q / tot_exp_r) if tot_exp_r else 0, icon="💰",
             help="Exposición en riesgo = valor financiado × intensidad de riesgo del modelo."),
    k4,
    kpi_card("Semanas para cubrir la cola", fmt_int(weeks_q) if n_q else "—",
             (f"R1 en {w_r1} semana{'s' if w_r1 != 1 else ''} a esta capacidad" if r1_q else "a la capacidad actual"),
             tone="ink", icon="⏱️"),
], min_px=180)

# ============================================================================================
# 2 · Rutas de cobro
# ============================================================================================
_anchor_section(2, "Rutas de cobro preventivo", "Cada crédito recibe una ruta según riesgo predicho, confianza y mora. "
                "Si la priorización funciona, el riesgo observado debe concentrarse en R1.", "Paso 2 · Rutas")

cards = []
max_rate = float(np.nanmax(stats["alto_rate"])) if truth and stats["alto_rate"].notna().any() else 0.0
scale = max(0.4, max_rate * 1.15) if truth else 1.0
for r in ROUTES:
    s = rs.loc[r]
    a = ACTION_ROUTES[r]
    in_q = r in sel_routes
    if truth and s["n"] > 0 and not np.isnan(s["alto_rate"]):
        lift_txt = f"{_es(s['lift'], 1)}× la base" if not np.isnan(s["lift"]) else ""
        obs = (f"<div class='obs'><div class='row'><span>Alto observado</span><span class='lift'>{esc(lift_txt)}</span></div>"
               f"<div class='row'><b>{fmt_pct(s['alto_rate'])}</b><span>{fmt_pct(s['alto_share'], 0)} de los Alto reales"
               f"</span></div><div class='meter'><i style='width:{min(s['alto_rate'] / scale, 1) * 100:.1f}%'></i>"
               f"<u style='left:{min(s['base'] / scale, 1) * 100:.1f}%'></u></div>"
               f"<div class='legend'>▮ tasa de la ruta · │ tasa base {fmt_pct(s['base'])}</div></div>")
    else:
        obs = (f"<div class='obs'><div class='row'><span>Alto predicho</span><b>{fmt_pct(s['alto_pred'])}</b></div>"
               f"<div class='legend'>Sin riesgo observado (y_true) en este dataset.</div></div>")
    chan = ("Cola de gestores" if in_q else ("Canal automático" if r == "R3" else "Fuera de la cola"))
    chan_kind = "accent" if in_q else "neutral"
    cards.append(
        f"<div class='cola-route{'' if in_q else ' off'}' style='--rc:{_rc(r)};--ri:{_ri(r)}'>"
        f"<div class='top'><span class='code'>{r}</span><span class='sla'>SLA · {esc(a['sla'])}</span></div>"
        f"<div class='name' title='{esc(a['nombre'])}'>{esc(a['nombre'])}</div>"
        f"<div class='big'>{fmt_int(s['n'])}<small>créditos · {fmt_pct(s['share'])}</small></div>"
        f"<div class='grid'><div><div class='k'>Financiado</div><div class='v'>{_nb(fmt_cop(s['exposicion']))}</div></div>"
        f"<div><div class='k'>En riesgo</div><div class='v'>{_nb(fmt_cop(s['exp_riesgo']))}</div></div>"
        f"<div><div class='k'>Prioridad</div><div class='v'>{_es(s['prioridad'], 1) if s['n'] else '—'}</div></div>"
        f"<div><div class='k'>Mora DC</div><div class='v'>{fmt_pct(s['mora']) if s['n'] else '—'}</div></div></div>"
        f"{obs}<div class='act'>{esc(a['accion'])}</div>"
        f"<div class='foot'>{badge(chan, chan_kind)}</div></div>"
    )
_grid(cards, min_px=225)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# Hallazgos automáticos
ins = []
r1 = rs.loc["R1"]
if truth and r1["n"] > 0:
    ins.append(insight("R1 concentra el riesgo real",
                       f"Con solo <b>{fmt_pct(r1['share'])}</b> de los créditos, R1 reúne <b>{fmt_pct(r1['alto_share'])}</b> "
                       f"de los Alto observados: su tasa es <b>{fmt_pct(r1['alto_rate'])}</b> frente a "
                       f"<b>{fmt_pct(r1['base'])}</b> de base (<b>{_es(r1['lift'], 1)}×</b>).", tone="alto", icon="🎯"))
elif r1["n"] > 0:
    ins.append(insight("R1 concentra la exposición en riesgo",
                       f"R1 reúne <b>{fmt_pct(r1['exp_riesgo'] / tot_exp_r if tot_exp_r else np.nan)}</b> de la "
                       f"exposición en riesgo con <b>{fmt_pct(r1['share'])}</b> de los créditos.", tone="alto", icon="🎯"))
if n_q:
    need_txt = ""
    if r1_q:
        need = math.ceil(r1_q / (per_g * SLA_DIAS["R1"] / DIAS_HABILES))
        need_txt = (f" Atender todo R1 ({fmt_int(r1_q)} casos) dentro de su SLA de 48 h requeriría ≈ <b>{fmt_int(need)} "
                    f"gestores</b> con la productividad actual.")
    ins.append(insight("Capacidad frente a la cola",
                       f"A <b>{fmt_int(capacity)}</b> gestiones/semana, la cola de gestores ({fmt_int(n_q)} casos) se cubre "
                       f"en <b>{fmt_int(weeks_q)} semana{'s' if weeks_q != 1 else ''}</b>.{need_txt}",
                       tone="accent", icon="⏱️"))
r2s, r3s = rs.loc["R2"], rs.loc["R3"]
if truth and r3s["n"] > 0 and r2s["n"] > 0 and r3s["alto_rate"] > r2s["alto_rate"]:
    ins.append(insight("R3 esconde riesgo real",
                       f"R3 (recordatorio automático) tiene <b>{fmt_pct(r3s['alto_rate'])}</b> de Alto observado, más que R2 "
                       f"(<b>{fmt_pct(r2s['alto_rate'])}</b>), y <b>{fmt_pct(r3s['mora'])}</b> con mora en Datacrédito. "
                       f"Considera llevar a gestión humana los R3 con mora.", tone="medio", icon="🔎"))
elif r3s["n"] > 0:
    ins.append(insight("Canal automático",
                       f"R3 cubre <b>{fmt_int(r3s['n'])}</b> créditos con recordatorio automático sin consumir capacidad; "
                       f"<b>{fmt_pct(r3s['mora'])}</b> tiene mora en Datacrédito.", tone="info", icon="🤖"))
if ins:
    _grid(ins, min_px=260)

# ============================================================================================
# 3 · Capacidad vs riesgo capturado
# ============================================================================================
_anchor_section(3, "¿Cuánto riesgo captura tu capacidad?",
                "Curva de ganancia operativa: si se gestionan los N créditos de mayor prioridad, qué parte de la exposición "
                "en riesgo y de los Alto observados queda cubierta, frente a un orden aleatorio.", "Paso 3 · Capacidad")

p = pal()
exp_arr = W["exposicion_riesgo"].to_numpy(float)
alto_arr = W["_alto_obs"].to_numpy(int)
cum_exp = np.cumsum(exp_arr) / tot_exp_r if tot_exp_r else np.zeros(n_all)
cum_alto = np.cumsum(alto_arr) / tot_alto if tot_alto else np.zeros(n_all)
k_cap = min(capacity, n_all)
cap_exp = float(cum_exp[k_cap - 1]) if k_cap else 0.0
cap_alto_c = float(cum_alto[k_cap - 1]) if (k_cap and truth) else np.nan
rand = k_cap / n_all

cc1, cc2 = st.columns([2.35, 1], gap="medium")
with cc1:
    with st.container(border=True):
        hz_opts = ["4 semanas", "12 semanas", "Toda la cartera"]
        h1, h2 = st.columns([1.05, 1.5], vertical_alignment="center")
        h1.markdown("<div class='cola-mini' style='margin:0'>Cada línea vertical es una semana de gestión a la capacidad "
                    "actual; la amarilla es la primera.</div>", unsafe_allow_html=True)
        with h2:
            st.segmented_control("Horizonte de la curva", hz_opts, selection_mode="single",
                                 label_visibility="collapsed", width="stretch", **_bind("horizonte"))
        hz = cfg["horizonte"] if cfg["horizonte"] in hz_opts else "12 semanas"
        xmax = {"4 semanas": capacity * 4, "12 semanas": capacity * 12}.get(hz, n_all)
        xmax = int(min(max(xmax, 10), n_all))
        idx = np.unique(np.concatenate([np.linspace(0, xmax - 1, min(xmax, 400)).astype(int), [k_cap - 1]]))
        idx = idx[(idx >= 0) & (idx < n_all)]
        xs = np.concatenate([[0], idx + 1])
        pct_cart = [fmt_pct(v / n_all) for v in xs]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=np.concatenate([[0], cum_exp[idx]]) * 100, mode="lines", legendrank=1,
                                 name="Exposición en riesgo capturada", line=dict(color=p["text"], width=2.5),
                                 customdata=pct_cart,
                                 hovertemplate="Exposición en riesgo: %{y:.1f} % <span style='opacity:.7'>"
                                               "(%{customdata} de la cartera)</span><extra></extra>"))
        if truth:
            fig.add_trace(go.Scatter(x=xs, y=np.concatenate([[0], cum_alto[idx]]) * 100, mode="lines", legendrank=2,
                                     name="Alto observados capturados", line=dict(color=RISK_COLORS["Alto"], width=2.5),
                                     hovertemplate="Alto observados: %{y:.1f} %<extra></extra>"))
            fig.add_trace(go.Scatter(x=xs, y=np.minimum(xs, tot_alto) / tot_alto * 100, mode="lines", legendrank=3,
                                     name="Techo teórico (Alto)",
                                     line=dict(color=_rgba(RISK_COLORS["Alto"], .4), width=1.5, dash="dash"),
                                     hovertemplate="Techo teórico: %{y:.1f} %<extra></extra>"))
        fig.add_trace(go.Scatter(x=xs, y=xs / n_all * 100, mode="lines", name="Orden aleatorio", legendrank=4,
                                 line=dict(color=p["subtle"], width=1.5, dash="dot"),
                                 hovertemplate="Orden aleatorio: %{y:.1f} %<extra></extra>"))
        n_weeks_lines = int(xmax // capacity)
        if n_weeks_lines <= 12:
            for kk in range(2, n_weeks_lines + 1):
                fig.add_vline(x=kk * capacity, line=dict(color=p["grid"], width=1))
                fig.add_annotation(x=kk * capacity, y=1.0, yref="paper", text=f"S{kk}", showarrow=False,
                                   font=dict(size=10, color=p["subtle"]), yanchor="bottom")
        fig.add_vline(x=k_cap, line=dict(color="#FFD100", width=2.5))
        fig.add_annotation(x=k_cap, y=1.0, yref="paper", text="<b>S1</b>", showarrow=False,
                           font=dict(size=11, color=p["text"]), yanchor="bottom")
        lab = dict(showarrow=False, xanchor="left", xshift=9, borderpad=2, bgcolor=p["surface"])
        fig.add_trace(go.Scatter(x=[k_cap], y=[cap_exp * 100], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=10, color=p["text"], line=dict(color=p["surface"], width=2))))
        fig.add_annotation(x=k_cap, y=cap_exp * 100, text=f"<b>{fmt_pct(cap_exp)}</b>", yshift=10,
                           font=dict(color=p["text"], size=12), **lab)
        if truth:
            fig.add_trace(go.Scatter(x=[k_cap], y=[cap_alto_c * 100], mode="markers", showlegend=False,
                                     hoverinfo="skip", marker=dict(size=10, color=RISK_COLORS["Alto"],
                                                                   line=dict(color=p["surface"], width=2))))
            fig.add_annotation(x=k_cap, y=cap_alto_c * 100, text=f"<b>{fmt_pct(cap_alto_c)}</b>", yshift=-10,
                               font=dict(color=_ri("R1"), size=12), **lab)
        fig.update_layout(hovermode="x unified", margin=dict(l=8, r=16, t=56, b=8),
                          xaxis=dict(title="Gestiones realizadas (créditos de mayor prioridad)", range=[0, xmax * 1.02],
                                     tickformat=",d", hoverformat=",d"),
                          yaxis=dict(title="% capturado del total", ticksuffix=" %", range=[0, 104]),
                          legend=dict(y=1.1))
        show_fig(fig, key="cola_curva", height=410)

with cc2:
    lift_e = cap_exp / rand if rand else np.nan
    blocks = [
        f"<div class='cola-stat hl'><div class='k'>Semana 1 · {fmt_int(k_cap)} gestiones</div>"
        f"<div class='v'>{fmt_pct(rand)}</div><div class='s'>de la cartera filtrada ({fmt_int(n_all)} créditos)</div></div>",
        f"<div class='cola-stat'><div class='k'>Exposición en riesgo capturada</div><div class='v'>{fmt_pct(cap_exp)}</div>"
        f"<div class='s'><b>{_es(lift_e, 1)}×</b> lo que lograría un orden aleatorio · {_nb(fmt_cop(cap_exp * tot_exp_r))}</div></div>",
    ]
    if truth:
        lift_a = cap_alto_c / rand if rand else np.nan
        blocks.append(
            f"<div class='cola-stat'><div class='k'>Alto observados capturados</div><div class='v'>{fmt_pct(cap_alto_c)}</div>"
            f"<div class='s'><b>{_es(lift_a, 1)}×</b> un orden aleatorio · {fmt_int(round(cap_alto_c * tot_alto))} de "
            f"{fmt_int(tot_alto)} casos</div></div>")
        k50 = int(np.searchsorted(cum_alto, 0.5) + 1) if tot_alto else 0
        blocks.append(
            f"<div class='cola-stat'><div class='k'>Para capturar la mitad de los Alto</div>"
            f"<div class='v'>{fmt_int(k50)}</div><div class='s'>gestiones ≈ <b>{_es(k50 / capacity, 1)} semanas</b> "
            f"a la capacidad actual</div></div>")
    else:
        k50 = int(np.searchsorted(cum_exp, 0.5) + 1) if tot_exp_r else 0
        blocks.append(
            f"<div class='cola-stat'><div class='k'>Para cubrir la mitad de la exposición en riesgo</div>"
            f"<div class='v'>{fmt_int(k50)}</div><div class='s'>gestiones ≈ <b>{_es(k50 / capacity, 1)} semanas</b></div></div>")
    _grid(blocks, min_px=200)

# Deciles de prioridad
with st.container(border=True):
    nb = int(min(10, n_all))
    dec = (np.arange(n_all) * nb // max(n_all, 1)) + 1
    Wd = W.assign(_dec=dec)
    if truth:
        g = Wd[Wd["has_truth"]].groupby("_dec").agg(rate=("_alto_obs", "mean"), n=("llave2", "size"),
                                                    exp=("exposicion_riesgo", "sum")).reset_index()
        base_rate = W.loc[W["has_truth"], "_alto_obs"].mean()
        yv, ytitle = g["rate"], "Tasa de Alto observado"
        sub = (f"Tasa observada de riesgo Alto en cada decil de prioridad (D1 = 10 % más prioritario) frente a la tasa base "
               f"de {fmt_pct(base_rate)}. Una buena priorización dibuja una escalera descendente.")
    else:
        g = Wd.groupby("_dec").agg(n=("llave2", "size"), exp=("exposicion_riesgo", "sum")).reset_index()
        g["rate"] = g["exp"] / tot_exp_r if tot_exp_r else 0
        base_rate = 1 / nb
        yv, ytitle = g["rate"], "% de la exposición en riesgo"
        sub = "Participación de cada decil de prioridad en la exposición en riesgo (D1 = 10 % más prioritario)."
    st.markdown(f"<div class='cola-mini'><b>Escalera de riesgo por decil de prioridad.</b> {esc(sub)}</div>",
                unsafe_allow_html=True)
    lbl = [f"D{int(d)}" for d in g["_dec"]]
    colors = [RISK_COLORS["Alto"] if v >= base_rate else _rgba(RISK_COLORS["Alto"], .35) for v in yv]
    figd = go.Figure(go.Bar(
        x=lbl, y=yv, marker=dict(color=colors, cornerradius=4, line=dict(width=0)),
        customdata=np.stack([g["n"].map(fmt_int), g["exp"].map(fmt_cop), (yv / base_rate if base_rate else yv * 0)
                             .map(lambda v: _es(v, 1))], axis=-1),
        hovertemplate="<b>%{x}</b><br>" + ytitle + ": %{y:.1%}<br>Lift: %{customdata[2]}×<br>Créditos: %{customdata[0]}"
                      "<br>Exposición en riesgo: %{customdata[1]}<extra></extra>",
        text=[fmt_pct(v) if i in (0, len(yv) - 1) else "" for i, v in enumerate(yv)], textposition="outside",
        textfont=dict(color=p["text"], size=12), cliponaxis=False, constraintext="none",
    ))
    figd.add_hline(y=base_rate, line=dict(color=p["muted"], width=1.5, dash="dash"),
                   annotation_text=f"Base {fmt_pct(base_rate)}", annotation_position="top right",
                   annotation_font=dict(color=p["muted"], size=11))
    figd.update_layout(bargap=0.35, margin=dict(l=8, r=8, t=8, b=8),
                       xaxis=dict(title="Decil de prioridad"), yaxis=dict(title=ytitle, tickformat=".0%",
                                                                          range=[0, max(float(np.nanmax(yv)) if len(yv) else 0, 0.01) * 1.15]))
    show_fig(figd, key="cola_deciles", height=280, legend=False)

# ============================================================================================
# 4 · Cola priorizada
# ============================================================================================
_anchor_section(4, "Cola priorizada de gestión", "Ordenada por índice de prioridad. Selecciona una fila para ver por qué "
                "está en la cola, el guion sugerido y abrir su ficha 360°.", "Paso 4 · Cola")


@st.fragment
def _queue_block(queue: pd.DataFrame, w: dict, truth: bool, n_gest: int) -> None:
    if queue.empty:
        empty_state("La cola de gestores está vacía", "Ninguna de las rutas seleccionadas tiene créditos con los filtros "
                    "actuales. Incluye más rutas o amplía los filtros.", icon="📭")
        return
    v1, v2, v3 = st.columns([1.45, 1.05, 2.1], vertical_alignment="center")
    with v1:
        st.segmented_control("Vista de la cola", ["Plan de la semana", "Cola completa"], selection_mode="single",
                             width="stretch", label_visibility="collapsed", **_bind("vista"))
    gopts = ["Todos los gestores"] + [f"Gestor {i}" for i in range(1, n_gest + 1)]
    if _cfg()["gestor_ver"] not in gopts:
        _cfg()["gestor_ver"] = gopts[0]
    with v2:
        st.selectbox("Ver la cola de", gopts, label_visibility="collapsed", **_bind("gestor_ver"))
    vista = _cfg()["vista"] or "Plan de la semana"
    view = queue[queue["semana"] == 1] if vista == "Plan de la semana" else queue
    gv = _cfg()["gestor_ver"]
    if gv != "Todos los gestores":
        view = view[view["gestor"] == gv]
    with v3:
        st.markdown(f"<div class='cola-mini' style='text-align:right;margin:0'>Mostrando <b>{fmt_int(len(view))}</b> casos · "
                    f"<b>{_nb(fmt_cop(view['exposicion_riesgo'].sum()))}</b> en riesgo · "
                    f"{_nb(fmt_cop(view['valor_financiacion'].sum()))} financiados</div>", unsafe_allow_html=True)
    if view.empty:
        empty_state("Sin casos para esta vista", "Este gestor no tiene casos asignados en la vista elegida.", icon="📭")
        return
    view = view.reset_index(drop=True)
    tbl = pd.DataFrame({
        "#": view["_pos"] + 1,
        "Prioridad": view["prioridad"].astype(float),
        "Riesgo": view["y_pred"].astype(str).map(lambda x: f"{RISK_ICONS.get(x, '⚪')} {x}"),
        "Confianza": (view["proba_pred"] * 100).round(0),
        "Estudiante": view["nombre"],
        "Ruta": view["ruta"].map(lambda r: f"{ROUTE_EMOJI[r]} {_route_label(r)}"),
        "SLA": view["ruta"].map(lambda r: ACTION_ROUTES[r]["sla"]),
        "Gestor": view["gestor"],
        "Semana": "S" + view["semana"].astype(str),
        "Valor financiado": view["valor_financiacion"].round(0).astype("int64"),
        "Exp. en riesgo": view["exposicion_riesgo"].round(0).astype("int64"),
        "Mora DC": view["mora_flag"].astype(bool),
        "ID": view["id_estudiante"].astype(str),
        "Programa": view["programa"],
    })
    key = f"cola_tabla_{vista}_{gv}_{len(view)}"
    ev = st.dataframe(
        tbl, key=key, hide_index=True, on_select="rerun", selection_mode="single-row", height=430,
        column_config={
            "#": st.column_config.NumberColumn("#", format="%d", width=48, help="Posición en la cola priorizada"),
            "Prioridad": st.column_config.ProgressColumn("Prioridad", min_value=0, max_value=100, format="%.0f",
                                                         width=105, help="Índice de prioridad 0–100"),
            "Riesgo": st.column_config.TextColumn("Riesgo", width=82, help="Clase predicha por el modelo"),
            "Confianza": st.column_config.NumberColumn("Confianza", format="%d %%", width=88,
                                                       help="Probabilidad de la clase predicha"),
            "Estudiante": st.column_config.TextColumn("Estudiante", width=190, help="Nombre anonimizado"),
            "Ruta": st.column_config.TextColumn("Ruta", width=140),
            "SLA": st.column_config.TextColumn("SLA", width=86),
            "Gestor": st.column_config.TextColumn("Gestor", width=82),
            "Semana": st.column_config.TextColumn("Semana", width=72, help="Semana programada según capacidad"),
            "Valor financiado": st.column_config.NumberColumn("Financiado (COP)", format="localized", width=122),
            "Exp. en riesgo": st.column_config.NumberColumn("En riesgo (COP)", format="localized", width=118,
                                                            help="Valor financiado × intensidad de riesgo"),
            "Mora DC": st.column_config.CheckboxColumn("Mora DC", width=74, help="Mora reportada en Datacrédito"),
            "ID": st.column_config.TextColumn("ID", width=96),
            "Programa": st.column_config.TextColumn("Programa", width=260),
        },
    )
    rows = list(getattr(getattr(ev, "selection", None), "rows", []) or [])
    if not rows:
        st.markdown("<div class='cola-mini'>☝️ Selecciona una fila (casilla de la izquierda) para ver el detalle del caso "
                    "y abrir su ficha 360°.</div>", unsafe_allow_html=True)
        return
    rec = view.iloc[int(rows[0])]
    _case_detail(rec, w, truth)


def _case_detail(rec: pd.Series, w: dict, truth: bool) -> None:
    r = str(rec["ruta"])
    a = ACTION_ROUTES[r]
    comp = {"Riesgo": 100 * w["riesgo"] * float(rec["intensidad_riesgo"]),
            "Exposición": 100 * w["exposicion"] * float(rec["_e"]),
            "Mora": 100 * w["mora"] * float(rec["mora_flag"])}
    ccol = {"Riesgo": "var(--sat-text)", "Exposición": "var(--sat-accent)", "Mora": "#3E63DD"}
    stack = "".join(f"<i style='width:{v:.2f}%;background:{ccol[k]}'></i>" for k, v in comp.items() if v > 0)
    legend = "".join(f"<span style='--c:{ccol[k]}'>{k}: <b>{_es(v, 1)}</b> pts</span>" for k, v in comp.items())
    conf = rec["proba_pred"]
    conf_txt = fmt_pct(conf, 0) if pd.notna(conf) else "—"
    dia = int(rec["fecha_de_pago"]) if pd.notna(rec.get("fecha_de_pago")) else None
    truth_chip = ""
    if truth and bool(rec["has_truth"]):
        truth_chip = (f"<span class='cola-mini' style='margin:0 0 0 4px'>Observado (validación):</span> "
                      f"{risk_badge(str(rec['y_true']))}")
    html = (
        f"<div class='cola-case'><div class='hd'><div><div class='nm'>{esc(rec['nombre'])}</div>"
        f"<div class='meta'>ID {esc(rec['id_estudiante'])} · Crédito {esc(rec['llave2'])} · {esc(rec['programa'])}"
        f" · {esc(rec.get('sede', ''))}</div></div>"
        f"<div class='chips'>{risk_badge(str(rec['y_pred']))}"
        f"<span class='sat-badge' style='color:{_ri(r)};background:{_rgba(_rc(r), .14)};border-color:{_rgba(_rc(r), .4)}'>"
        f"{r} · {esc(a['nombre'])}</span>{truth_chip}</div></div>"
        f"<div class='g4'><div><div class='k'>Prioridad</div><div class='v'>{_es(rec['prioridad'], 1)} / 100</div></div>"
        f"<div><div class='k'>Confianza</div><div class='v'>{conf_txt}</div></div>"
        f"<div><div class='k'>Valor financiado</div><div class='v'>{fmt_cop(rec['valor_financiacion'], compact=False)}</div></div>"
        f"<div><div class='k'>Exp. en riesgo</div><div class='v'>{fmt_cop(rec['exposicion_riesgo'], compact=False)}</div></div></div>"
        f"<div class='why'><div class='t'>¿Por qué está en la cola? Descomposición del índice ({_es(rec['prioridad'], 1)} pts)</div>"
        f"<div class='stack'>{stack}</div><div class='lg'>{legend}</div></div>"
        f"<div class='act'><b>Acción ({esc(a['sla'])}):</b> {esc(a['accion'])}<br>"
        f"<b>Responsable:</b> {esc(rec['gestor'])} · semana S{int(rec['semana'])} · posición #{int(rec['_pos']) + 1} en la cola"
        f"{f' · pago el día {dia} de cada mes' if dia else ''} · {int(rec['cuotas']) if pd.notna(rec.get('cuotas')) else '—'} cuotas · "
        f"mora Datacrédito: {'sí' if int(rec['mora_flag']) == 1 else 'no'}</div></div>"
    )
    d1, d2 = st.columns([2.6, 1], gap="medium")
    with d1:
        st.markdown(html, unsafe_allow_html=True)
    with d2:
        allowed = can_access("ficha")
        if st.button("Abrir ficha 360°", icon=":material/person_search:", type="primary", width="stretch",
                     key="cola_open_ficha", disabled=not allowed,
                     help=None if allowed else "Tu rol no tiene acceso a la Ficha 360°"):
            open_ficha(str(rec["id_estudiante"]))
        guion = GUIONES.get(r, "").format(nombre=str(rec["nombre"]).split(" ")[0], gestor=str(rec["gestor"]),
                                          dia=dia if dia else "—")
        canal = {"R1": "Llamada del gestor", "R2": "WhatsApp / SMS + correo", "R3": "Mensaje automático",
                 "R4": "Sin contacto adicional"}.get(r, "")
        st.markdown(f"<div class='cola-script'><div class='t'>💬 Guion sugerido · {esc(canal)}</div>"
                    f"<div class='q'>{esc(guion)}</div><div class='f'>Personalizado con el nombre, el gestor y el día de "
                    f"pago. Ajústalo al protocolo de Cartera.</div></div>", unsafe_allow_html=True)


_queue_block(queue, w, truth, n_gest)

# ============================================================================================
# 5 · Asignación a gestores
# ============================================================================================
_anchor_section(5, "Asignación a gestores", "Reparto de la cola priorizada entre el equipo respetando la capacidad semanal: "
                "cada gestor recibe una mezcla equilibrada de casos críticos y exposición.", "Paso 5 · Gestores")

if queue.empty:
    empty_state("Sin casos para asignar", "La cola de gestores está vacía con las rutas y filtros actuales.", icon="📭")
else:
    a1, a2 = st.columns([1.3, 3], vertical_alignment="center")
    with a1:
        st.segmented_control("Estrategia de reparto", ["Round-robin", "Serpentina"], selection_mode="single",
                             width="stretch", label_visibility="collapsed",
                             help="Round-robin: 1, 2, …, N, 1, 2, … · Serpentina: 1…N, N…1 "
                             "(compensa que el primer gestor siempre toma el caso más prioritario de cada vuelta).",
                             **_bind("reparto"))
    strategy = cfg["reparto"] or "Round-robin"
    with a2:
        st.markdown(
            f"<div class='cola-mini' style='margin:0'><b>Estrategia de reparto.</b> Round-robin asigna 1, 2, …, N, 1, 2, …; "
            f"la serpentina alterna el sentido (1…N, N…1) para compensar que el primer gestor siempre toma el caso más "
            f"prioritario de cada vuelta. Capacidad individual: <b>{_es(per_g, 1)}</b> gestiones/semana.</div>",
            unsafe_allow_html=True)

    def _balance(pl: pd.DataFrame) -> float:
        e = pl.groupby("gestor")["exposicion_riesgo"].sum().reindex([f"Gestor {i}" for i in range(1, n_gest + 1)],
                                                                    fill_value=0.0)
        return float((e.max() - e.min()) / e.mean()) if len(e) > 1 and e.mean() > 0 else 0.0

    other = "Serpentina" if strategy != "Serpentina" else "Round-robin"
    plan_other = _assign(W[W["ruta"].isin(sel_routes)].head(min(capacity, n_q)), capacity, n_gest, other)
    bal, bal_o = _balance(plan), _balance(plan_other)

    gcols = [f"Gestor {i}" for i in range(1, n_gest + 1)]
    load = plan.groupby(["gestor", "ruta"]).size().unstack(fill_value=0).reindex(index=gcols, fill_value=0)
    ge = plan.groupby("gestor").agg(exp=("exposicion_riesgo", "sum"), val=("valor_financiacion", "sum"),
                                    prio=("prioridad", "mean"), n=("llave2", "size")).reindex(gcols)
    ge = ge.fillna({"exp": 0, "val": 0, "n": 0})
    tick_angle = 0 if n_gest <= 8 else -45
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        with st.container(border=True):
            st.markdown(f"<div class='cola-mini'><b>Carga de la semana por gestor</b> · casos por ruta; la línea punteada "
                        f"marca la capacidad individual ({_es(per_g, 0)}).</div>", unsafe_allow_html=True)
            figl = go.Figure()
            for r in ROUTES:
                if r in load.columns and load[r].sum() > 0:
                    figl.add_trace(go.Bar(x=load.index, y=load[r], name=_route_label(r),
                                          marker=dict(color=_rc(r), line=dict(color=p["surface"], width=1.5)),
                                          hovertemplate="%{x}<br>" + _route_label(r) + ": %{y:,d} casos<extra></extra>"))
            tot = load.sum(axis=1)
            figl.add_trace(go.Scatter(x=load.index, y=tot, mode="text", text=[fmt_int(v) for v in tot],
                                      textposition="top center", textfont=dict(color=p["text"], size=12),
                                      showlegend=False, hoverinfo="skip", cliponaxis=False))
            figl.add_hline(y=per_g, line=dict(color=p["muted"], width=1.5, dash="dash"))
            figl.update_layout(barmode="stack", bargap=0.4, margin=dict(l=8, r=8, t=40, b=8),
                               legend=dict(traceorder="normal"),
                               yaxis=dict(title="Casos asignados", range=[0, max(float(tot.max()), per_g) * 1.22]),
                               xaxis=dict(title=None, tickangle=tick_angle))
            show_fig(figl, key="cola_carga", height=320)
    with g2:
        with st.container(border=True):
            st.markdown("<div class='cola-mini'><b>Exposición en riesgo asignada por gestor</b> · la línea punteada marca "
                        "el promedio del equipo.</div>", unsafe_allow_html=True)
            figx = go.Figure(go.Bar(
                x=ge.index, y=ge["exp"], marker=dict(color="#FFD100" if _dark() else p["text"], cornerradius=4),
                customdata=np.stack([ge["exp"].map(fmt_cop), ge["val"].map(fmt_cop), ge["prio"].map(lambda v: _es(v, 1)),
                                     ge["n"].map(fmt_int)], axis=-1),
                hovertemplate="<b>%{x}</b><br>Exposición en riesgo: %{customdata[0]}<br>Valor financiado: "
                              "%{customdata[1]}<br>Prioridad media: %{customdata[2]}<br>Casos: %{customdata[3]}<extra></extra>",
                text=ge["exp"].map(fmt_cop) if n_gest <= 8 else None, textposition="inside", insidetextanchor="end",
                textangle=0, textfont=dict(color="#141414" if _dark() else "#FFFFFF", size=11), constraintext="none",
            ))
            figx.add_hline(y=float(ge["exp"].mean()), line=dict(color=p["muted"], width=1.5, dash="dash"))
            figx.update_layout(bargap=0.4, margin=dict(l=8, r=8, t=40, b=8), showlegend=False,
                               yaxis=dict(title="Exposición en riesgo (COP)", tickformat="~s",
                                          range=[0, float(ge["exp"].max()) * 1.22 if ge["exp"].max() > 0 else 1]),
                               xaxis=dict(title=None, tickangle=tick_angle))
            show_fig(figx, key="cola_expo_gestor", height=320, legend=False)

    s1, s2 = st.columns([1.2, 1], gap="medium")
    with s1:
        with st.container(border=True):
            st.markdown("<div class='cola-mini'><b>Programación de la cola por semana</b> · cómo se evacúa la cola a la "
                        "capacidad actual (hasta 12 semanas; el resto se agrupa en S13+).</div>", unsafe_allow_html=True)
            qq = queue.assign(_s=np.where(queue["semana"] <= 12, "S" + queue["semana"].astype(str), "S13+"))
            order = [f"S{i}" for i in range(1, min(int(queue["semana"].max()), 12) + 1)] + \
                    (["S13+"] if queue["semana"].max() > 12 else [])
            wk = qq.groupby(["_s", "ruta"]).size().unstack(fill_value=0).reindex(order, fill_value=0)
            figw = go.Figure()
            for r in ROUTES:
                if r in wk.columns and wk[r].sum() > 0:
                    figw.add_trace(go.Bar(x=wk.index, y=wk[r], name=_route_label(r),
                                          marker=dict(color=_rc(r), line=dict(color=p["surface"], width=1.5)),
                                          hovertemplate="%{x}<br>" + _route_label(r) + ": %{y:,d} casos<extra></extra>"))
            figw.add_hline(y=capacity, line=dict(color=p["muted"], width=1.5, dash="dash"))
            if queue["semana"].max() > 12:
                figw.add_annotation(x="S13+", y=float(wk.loc["S13+"].sum()), text=f"{fmt_int(wk.loc['S13+'].sum())}",
                                    showarrow=False, yshift=10, font=dict(color=p["text"], size=11))
            figw.update_layout(barmode="stack", bargap=0.3, margin=dict(l=8, r=8, t=40, b=8),
                               legend=dict(traceorder="normal"),
                               yaxis=dict(title="Casos programados", range=[0, max(float(wk.sum(axis=1).max()),
                                                                                  capacity) * 1.15]),
                               xaxis=dict(title="Semana de gestión"))
            show_fig(figw, key="cola_semanas", height=320)
    with s2:
        stat_bal = (f"<div class='cola-stat hl'><div class='k'>Brecha de exposición entre gestores</div>"
                    f"<div class='v'>{fmt_pct(bal)}</div><div class='s'>del promedio con <b>{esc(strategy)}</b> · "
                    f"{fmt_pct(bal_o)} con {esc(other)}</div></div>")
        stat_cap = (f"<div class='cola-stat'><div class='k'>Uso de la capacidad semanal</div>"
                    f"<div class='v'>{fmt_pct(n_plan / capacity, 0)}</div><div class='s'>{fmt_int(n_plan)} de "
                    f"{fmt_int(capacity)} gestiones · {fmt_int(max(n_q - n_plan, 0))} casos quedan para las semanas "
                    f"siguientes</div></div>")
        _grid([stat_bal, stat_cap], min_px=200)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        summ = pd.DataFrame({
            "Gestor": gcols,
            "Casos": load.sum(axis=1).to_numpy().astype(int),
            "Prioridad": ge["prio"].round(1).to_numpy(),
            "En riesgo (COP)": ge["exp"].round(0).astype("int64").to_numpy(),
            "Participación": (ge["exp"] / ge["exp"].sum() * 100).to_numpy() if ge["exp"].sum() else 0.0,
        })
        st.dataframe(summ, hide_index=True, height=min(38 + 35 * len(summ), 250), key="cola_tbl_gestores",
                     column_config={
                         "Gestor": st.column_config.TextColumn(width=78),
                         "Casos": st.column_config.NumberColumn(format="%d", width=56),
                         "Prioridad": st.column_config.NumberColumn(format="localized", width=74,
                                                                    help="Prioridad media de los casos asignados"),
                         "En riesgo (COP)": st.column_config.NumberColumn(format="localized", width=112),
                         "Participación": st.column_config.ProgressColumn("Particip.", min_value=0, max_value=100,
                                                                          format="%.0f %%", width=96),
                     })

# ============================================================================================
# 6 · Escenario ilustrativo de recaudo
# ============================================================================================
_anchor_section(6, "Impacto en recaudo · escenario ilustrativo",
                "Proyección del indicador institucional de Recaudo de Cartera frente a la línea base del 85 % "
                "(según el reporte) bajo supuestos editables de efectividad de la gestión preventiva.", "Paso 6 · Recaudo")


@st.fragment
def _scenario_block(queue: pd.DataFrame, W: pd.DataFrame, sel: list[str], capacity: int, n_gest: int,
                    truth: bool) -> None:
    st.markdown(
        "<div class='cola-banner'><span class='tag'>Escenario ilustrativo</span><div>La base no contiene datos de recaudo: "
        "este módulo <b>no es una predicción</b>, sino una simulación transparente para dimensionar el efecto potencial de la "
        "gestión preventiva. Todos los supuestos están a la vista y son editables; la fórmula está al final.</div></div>",
        unsafe_allow_html=True)
    cfg = _cfg()
    left, right = st.columns([1, 2.45], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("**Supuestos del escenario**")
            st.slider("Horizonte de gestión (semanas)", 1, 26, step=1, **_bind("sc_semanas"),
                      help="Semanas de gestión humana que se simulan a la capacidad actual.")
            st.slider("Contactabilidad efectiva de los gestores", 0, 100, step=5, format="%d %%",
                      help="% de los casos asignados que efectivamente se logra contactar.", **_bind("sc_contacto"))
            st.markdown("<div class='cola-mini' style='margin-top:4px'><b>Efectividad por ruta</b> · % de la exposición "
                        "en riesgo que la gestión evita que se convierta en no recaudo.</div>", unsafe_allow_html=True)
            for r in ROUTES:
                st.slider(f"{ROUTE_EMOJI[r]} {_route_label(r)}", 0, 100, step=5, format="%d %%", **_bind(f"sc_ef_{r}"))
            st.toggle("R3 por canal automático (no consume capacidad)", **_bind("sc_auto_r3"),
                      help="Si R3 no está en la cola de gestores, se asume que recibe el recordatorio automático.")
            st.button("↺ Restablecer supuestos", key="cola_reset_sc", on_click=_reset, args=(_SCENARIO_KEYS,),
                      width="stretch")

    H = int(cfg["sc_semanas"])
    contacto = cfg["sc_contacto"] / 100
    ef = {r: cfg[f"sc_ef_{r}"] / 100 for r in ROUTES}
    auto_r3 = bool(cfg["sc_auto_r3"])
    V = float(W["valor_financiacion"].sum())
    base = RECAUDO_BASELINE
    cum, grow, auto = _scenario_gain(queue, W, sel, contacto, ef, auto_r3)
    n_q = len(queue)
    k_h = int(min(n_q, capacity * H))
    gain_h = float(cum[k_h]) + auto
    gap = 1 - base

    def proj(gain: float) -> float:
        return base + min(gain / V, gap) if V > 0 else base

    rec_proj = proj(gain_h)
    per_g = capacity / n_gest
    k_plus = int(min(n_q, (capacity + per_g) * H))
    marg = float(cum[k_plus] - cum[k_h])
    by_route = grow.iloc[:k_h].groupby("ruta")["g"].sum().reindex(ROUTES, fill_value=0.0)
    if auto_r3 and "R3" not in sel:
        by_route["R3"] += auto
    managed = queue.iloc[:k_h]
    alto_m = int(managed["_alto_obs"].sum())

    with right:
        _grid([
            kpi_card("Recaudo adicional estimado", fmt_cop(gain_h), f"en {H} semana{'s' if H != 1 else ''} · "
                     f"{fmt_delta_pp(rec_proj - base)} sobre la línea base", tone="bajo", icon="💵"),
            kpi_card("Casos con gestión humana", fmt_int(k_h),
                     (f"incluye {fmt_int(alto_m)} Alto observados" if truth else f"de {fmt_int(n_q)} en la cola")
                     + (f" · + {fmt_int((W['ruta'] == 'R3').sum())} R3 automáticos" if auto_r3 and "R3" not in sel
                        else ""), tone="accent", icon="📞"),
            kpi_card("Valor de +1 gestor", fmt_cop(marg), f"{fmt_delta_pp(marg / V if V else 0, 2)} de recaudo en el "
                     f"horizonte (+{_es(per_g, 0)} gestiones/sem.)", tone="ink", icon="➕",
                     help="Recaudo adicional que aportaría un gestor más con la misma productividad."),
        ], min_px=190)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        gcol1, gcol2 = st.columns([1.05, 1.15], gap="medium")
        with gcol1:
            with st.container(border=True):
                st.markdown(f"<div class='cola-mini'><b>Recaudo proyectado</b> · la marca vertical es la línea base "
                            f"institucional ({fmt_pct(base, 0)}).</div>", unsafe_allow_html=True)
                figg = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=rec_proj * 100,
                    number=dict(suffix=" %", valueformat=".1f", font=dict(size=32, color=p["text"])),
                    delta=dict(reference=base * 100, valueformat=".2f", suffix=" pp", position="bottom",
                               increasing=dict(color=RISK_COLORS["Bajo"]), font=dict(size=15)),
                    gauge=dict(
                        axis=dict(range=[70, 100], ticksuffix=" %", tickcolor=p["border"], nticks=4,
                                  tickfont=dict(color=p["muted"], size=11)),
                        bar=dict(color="#FFD100", thickness=0.3), bgcolor=p["surface_2"], borderwidth=0,
                        steps=[dict(range=[70, base * 100], color=_rgba(RISK_COLORS["Alto"], .10)),
                               dict(range=[base * 100, 100], color=_rgba(RISK_COLORS["Bajo"], .12))],
                        threshold=dict(line=dict(color=p["text"], width=3), thickness=0.9, value=base * 100),
                    ),
                    domain=dict(x=[0.12, 0.88], y=[0, 1]),
                ))
                figg.update_layout(margin=dict(l=10, r=10, t=22, b=4))
                show_fig(figg, key="cola_gauge", height=196, legend=False)
                closed = (rec_proj - base) / gap if gap > 0 else 0.0
                st.markdown(f"<div class='cola-mini' style='text-align:center;margin:0'>La gestión cerraría "
                            f"<b>{fmt_pct(closed)}</b> de la brecha de no recaudo ({fmt_pct(gap, 0)} de la cartera)."
                            f"</div>", unsafe_allow_html=True)
        with gcol2:
            with st.container(border=True):
                st.markdown("<div class='cola-mini'><b>Puente de recaudo por ruta</b> · puntos porcentuales que aporta "
                            "cada ruta sobre la línea base.</div>", unsafe_allow_html=True)
                xs = ["Base"] + ROUTES + ["Total"]
                pp = [by_route[r] / V * 100 if V else 0 for r in ROUTES]
                tf = dict(color=p["text"], size=11)
                figb = go.Figure()
                figb.add_trace(go.Bar(x=["Base"], y=[base * 100], marker=dict(color=p["subtle"], cornerradius=4),
                                      text=[f"{_es(base * 100, 1)} %"], textposition="outside", textangle=0,
                                      textfont=tf, cliponaxis=False, constraintext="none",
                                      hovertemplate="Línea base: %{y:.1f} %<extra></extra>", showlegend=False))
                acc = base * 100
                for r, v in zip(ROUTES, pp):
                    figb.add_trace(go.Bar(x=[r], y=[max(v, 0.0)], base=[acc], marker=dict(color=_rc(r), cornerradius=2),
                                          text=[f"+{_es(v, 2)}"] if v > 0.004 else [""], textposition="outside",
                                          textangle=0, textfont=tf, cliponaxis=False, constraintext="none",
                                          customdata=[[fmt_cop(by_route[r])]],
                                          hovertemplate=f"{_route_label(r)}<br>+%{{y:.2f}} pp · %{{customdata[0]}}"
                                                        f"<extra></extra>", showlegend=False))
                    acc += v
                figb.add_trace(go.Bar(x=["Total"], y=[rec_proj * 100], marker=dict(color="#FFD100", cornerradius=4),
                                      text=[f"<b>{_es(rec_proj * 100, 1)} %</b>"], textposition="outside", textangle=0,
                                      textfont=tf, cliponaxis=False, constraintext="none",
                                      hovertemplate="Proyectado: %{y:.2f} %<extra></extra>", showlegend=False))
                span = max((rec_proj - base) * 100, 0.3)
                lo = base * 100 - max(1.0, span * 0.9)
                hi = rec_proj * 100 + max(0.5, span * 0.45)
                figb.update_layout(barmode="overlay", bargap=0.3, margin=dict(l=8, r=8, t=18, b=8),
                                   yaxis=dict(title="Recaudo", range=[lo, hi], ticksuffix=" %"),
                                   xaxis=dict(categoryorder="array", categoryarray=xs, tickangle=0))
                show_fig(figb, key="cola_puente", height=236, legend=False)

        with st.container(border=True):
            cmax = int(max(capacity * 3, 50))
            grid = np.unique(np.linspace(0, cmax, 61).astype(int))
            ys = [proj(float(cum[int(min(n_q, c * H))]) + auto) * 100 for c in grid]
            st.markdown(f"<div class='cola-mini'><b>Sensibilidad a la capacidad.</b> Recaudo proyectado según gestiones "
                        f"semanales en un horizonte de {H} semanas; los rendimientos decrecen porque la cola atiende "
                        f"primero lo más prioritario.</div>", unsafe_allow_html=True)
            figs = go.Figure()
            figs.add_trace(go.Scatter(x=grid, y=ys, mode="lines", line=dict(color=p["text"], width=2.5), name="Proyectado",
                                      fill="tozeroy", fillcolor=_rgba("#FFD100", .08 if _dark() else .14),
                                      hovertemplate="%{x:,d} gestiones/sem.<br>Recaudo: %{y:.2f} %<extra></extra>"))
            figs.add_hline(y=base * 100, line=dict(color=p["muted"], width=1.5, dash="dash"),
                           annotation_text=f"Línea base {fmt_pct(base, 0)}", annotation_position="bottom right",
                           annotation_font=dict(color=p["muted"], size=11))
            figs.add_trace(go.Scatter(x=[capacity], y=[rec_proj * 100], mode="markers+text",
                                      marker=dict(size=11, color="#FFD100", line=dict(color=p["text"], width=2)),
                                      text=[f"Actual · {_es(rec_proj * 100, 1)} %"], textposition="top left",
                                      textfont=dict(color=p["text"], size=12), showlegend=False,
                                      hovertemplate="Capacidad actual: %{x:,d}<br>%{y:.2f} %<extra></extra>"))
            yplus = proj(float(cum[k_plus]) + auto) * 100
            figs.add_trace(go.Scatter(x=[capacity + per_g], y=[yplus], mode="markers",
                                      marker=dict(size=9, color=p["surface"], line=dict(color=p["text"], width=2)),
                                      showlegend=False, hovertemplate="+1 gestor: %{x:,.0f}<br>%{y:.2f} %<extra></extra>"))
            ylo = base * 100 - 0.3
            yhi = max(ys) + max(0.4, (max(ys) - base * 100) * 0.25)
            figs.update_layout(margin=dict(l=8, r=16, t=16, b=8), showlegend=False,
                               xaxis=dict(title="Capacidad semanal (gestiones)", tickformat=",d"),
                               yaxis=dict(title="Recaudo proyectado (%)", range=[ylo, yhi], ticksuffix=" %"))
            show_fig(figs, key="cola_sensibilidad", height=260, legend=False)

    with st.expander("🧮 ¿Cómo se calcula el escenario? Fórmula y supuestos"):
        st.markdown("El recaudo proyectado parte de la **línea base institucional del 85 %** (según el reporte) y suma el "
                    "monto en riesgo que la gestión preventiva evitaría que se convierta en no recaudo:")
        st.latex(r"\text{Recaudo}_{proy} = \min\left(100\,\%,\; 85\,\% + \frac{\sum_{i \in G} E_i\,c\,e_{r(i)} \;+\; "
                 r"\sum_{i \in A} E_i\,e_{R3}}{V}\right)")
        st.markdown(
            f"- **Eᵢ = valor financiadoᵢ × intensidad de riesgoᵢ** (exposición en riesgo del modelo).\n"
            f"- **G** = los primeros min(n, C × H) casos de la cola priorizada = **{fmt_int(k_h)}** casos "
            f"(C = {fmt_int(capacity)} gestiones/semana, H = {H} semanas).\n"
            f"- **c** = contactabilidad efectiva = **{fmt_pct(contacto, 0)}**; **e_r** = efectividad de la ruta "
            f"(R1 {fmt_pct(ef['R1'], 0)}, R2 {fmt_pct(ef['R2'], 0)}, R3 {fmt_pct(ef['R3'], 0)}, R4 {fmt_pct(ef['R4'], 0)}).\n"
            f"- **A** = créditos R3 atendidos por canal automático (entrega del 100 %, sin consumir capacidad): "
            f"{'activo' if auto_r3 and 'R3' not in sel else 'no aplica'}.\n"
            f"- **V** = valor financiado de la cartera filtrada = **{fmt_cop(V)}** (proxy de la cartera exigible: la base no "
            f"trae saldos ni pagos).\n"
            f"- El aporte se limita a la brecha de {fmt_pct(gap, 0)} (el recaudo no puede superar 100 %). "
            f"El valor de **+1 gestor** repite el cálculo con C + C/N gestiones por semana.\n\n"
            "Para convertir este escenario en una medición real, registre el resultado de cada gestión en el sistema de "
            "cartera (estado del archivo DB-04) y compare el recaudo de los casos gestionados contra un grupo de control."
        )


_scenario_block(queue, W, sel_routes, capacity, n_gest, truth)

# ============================================================================================
# 7 · Exportar (DB-02) e integración (DB-04)
# ============================================================================================
_anchor_section(7, "Exportar y enviar al sistema de cartera", "DB-02: descarga de la cola programada · DB-04: archivo de "
                "intercambio con esquema fijo para el sistema de gestión de cartera.", "Paso 7 · Exportar")

today = date.today()
e1, e2 = st.columns([1, 1.35], gap="medium")
with e1:
    with st.container(border=True):
        st.markdown(f"{badge('DB-02', 'accent')} &nbsp;**Cola priorizada completa**", unsafe_allow_html=True)
        st.markdown(f"<div class='cola-mini'>{fmt_int(n_q)} casos con posición, ruta, SLA, gestor y semana programada. "
                    f"El Excel incluye hojas de rutas, gestores y parámetros de la corrida.</div>", unsafe_allow_html=True)
        exp_df = pd.DataFrame({
            "posicion": queue["_pos"] + 1, "semana": queue["semana"], "gestor": queue["gestor"],
            "prioridad": queue["prioridad"].round(1), "ruta": queue["ruta"],
            "ruta_nombre": queue["ruta"].map(lambda r: ACTION_ROUTES[r]["nombre"]),
            "sla": queue["ruta"].map(lambda r: ACTION_ROUTES[r]["sla"]),
            "accion": queue["ruta"].map(lambda r: ACTION_ROUTES[r]["accion"]),
            "riesgo_predicho": queue["y_pred"].astype(str), "confianza": queue["proba_pred"].round(4),
            "id_estudiante": queue["id_estudiante"].astype(str), "llave2": queue["llave2"], "nombre": queue["nombre"],
            "programa": queue["programa"], "facultad": queue["facultad"], "sede": queue["sede"],
            "valor_financiado": queue["valor_financiacion"].round(0), "exposicion_riesgo": queue["exposicion_riesgo"].round(0),
            "mora_datacredito": np.where(queue["mora_flag"] == 1, "Sí", "No"),
            "dia_pago": queue["fecha_de_pago"], "cuotas": queue["cuotas"],
        })
        rutas_df = stats.assign(nombre=stats["ruta"].map(lambda r: ACTION_ROUTES[r]["nombre"]),
                                sla=stats["ruta"].map(lambda r: ACTION_ROUTES[r]["sla"])).rename(columns={
            "n": "creditos", "share": "participacion", "exposicion": "valor_financiado", "exp_riesgo": "exposicion_riesgo",
            "alto_rate": "tasa_alto_observado", "base": "tasa_base", "alto_share": "participacion_alto_observado",
            "alto_pred": "pct_alto_predicho", "mora": "pct_mora_datacredito", "prioridad": "prioridad_media"})
        params_df = pd.DataFrame({
            "parametro": ["fecha", "origen", "creditos_filtrados", "rutas_en_cola", "capacidad_semanal", "gestores",
                          "reparto", "peso_riesgo", "peso_exposicion", "peso_mora"],
            "valor": [today.isoformat(), str(dff["origen"].iloc[0]) if "origen" in dff.columns else "",
                      n_all, ", ".join(sel_routes), capacity, n_gest, cfg["reparto"], round(w["riesgo"], 4),
                      round(w["exposicion"], 4), round(w["mora"], 4)],
        })
        params_df["valor"] = params_df["valor"].astype(str)
        gest_df = queue[queue["semana"] == 1].groupby("gestor").agg(
            casos=("llave2", "size"), exposicion_riesgo=("exposicion_riesgo", "sum"),
            valor_financiado=("valor_financiacion", "sum"), prioridad_media=("prioridad", "mean")).reset_index()
        stamp = today.strftime("%Y%m%d")

        def _csv_q() -> bytes:
            return exp_df.to_csv(index=False).encode("utf-8-sig")

        def _xlsx_q() -> bytes:
            return to_excel_bytes({"cola": exp_df, "rutas": rutas_df, "gestores_semana1": gest_df,
                                   "parametros": params_df})

        b1, b2 = st.columns(2)
        with b1:
            _lazy_download("CSV", _csv_q, f"cola_gestion_{stamp}.csv", "text/csv", "cola_dl_csv",
                           icon=":material/download:", typ="primary")
        with b2:
            _lazy_download("Excel", _xlsx_q, f"cola_gestion_{stamp}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "cola_dl_xlsx",
                           icon=":material/table_view:")
        files = [("cola", len(exp_df), "posición, semana, gestor, ruta, SLA, acción y datos de cada crédito"),
                 ("rutas", len(rutas_df), "resumen R1–R4: créditos, exposición, tasa observada y lift"),
                 ("gestores_semana1", len(gest_df), "carga, exposición y prioridad media por gestor"),
                 ("parametros", len(params_df), "pesos, capacidad, gestores, rutas y fecha de la corrida")]
        st.markdown("<div class='cola-files'>" + "".join(
            f"<div><code>{esc(n)}</code><b>{fmt_int(k)} fila{'s' if k != 1 else ''}</b><span>{esc(d)}</span></div>"
            for n, k, d in files) + "</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='cola-mini'>Parámetros de la corrida: pesos {fmt_pct(w['riesgo'], 0)} / "
            f"{fmt_pct(w['exposicion'], 0)} / {fmt_pct(w['mora'], 0)} · rutas {esc(', '.join(sel_routes))} · "
            f"{fmt_int(capacity)} gestiones/semana · {n_gest} gestores · reparto {esc(cfg['reparto'] or 'Round-robin')}. "
            f"Los archivos se generan solo al hacer clic.</div>", unsafe_allow_html=True)

with e2:
    with st.container(border=True):
        st.markdown(f"{badge('DB-04', 'accent')} &nbsp;**Archivo de intercambio para el sistema de gestión de cartera**",
                    unsafe_allow_html=True)
        if plan.empty:
            st.markdown("<div class='cola-mini'>No hay casos asignados esta semana: el archivo de intercambio se genera "
                        "cuando la cola tiene casos.</div>", unsafe_allow_html=True)
        else:
            ex = _db04_frame(plan, today)
            csv_b = _db04_csv(ex)
            json_b = _db04_json(ex, {"origen": "SAT · Cola de gestión", "fecha_asignacion": today.isoformat(),
                                     "capacidad_semanal": capacity, "gestores": n_gest, "rutas": sel_routes})
            tests = _db04_tests(ex, csv_b, json_b)
            ok = sum(1 for t in tests if t[1])
            share_ok = ok / len(tests)
            st.markdown(
                f"<div class='cola-tests'>{badge(f'{ok}/{len(tests)} casos de prueba superados', 'bajo' if share_ok >= .9 else 'alto')}"
                f"<span>Criterio DB-04: ≥ 90 % · <b>{fmt_pct(share_ok, 0)}</b> · {fmt_int(len(ex))} registros del plan de "
                f"la semana · {len(DB04_COLUMNS)} campos · CSV con «;» y JSON UTF-8</span></div>",
                unsafe_allow_html=True)
            st.dataframe(ex.head(5), hide_index=True, height=213, key="cola_db04_preview",
                         column_config={"confianza": st.column_config.NumberColumn(format="%.4f"),
                                        "prioridad": st.column_config.NumberColumn(format="%.1f"),
                                        "valor_financiado": st.column_config.NumberColumn(format="%d")})
            d1, d2 = st.columns(2)
            with d1:
                st.download_button("CSV ; (intercambio)", csv_b, file_name=f"sat_intercambio_cartera_{stamp}.csv",
                                   mime="text/csv", key="cola_db04_csv", width="stretch", type="primary",
                                   icon=":material/sync_alt:", on_click="ignore")
            with d2:
                st.download_button("JSON (intercambio)", json_b, file_name=f"sat_intercambio_cartera_{stamp}.json",
                                   mime="application/json", key="cola_db04_json", width="stretch",
                                   icon=":material/data_object:", on_click="ignore")

# Expansores a todo el ancho (el diccionario del esquema necesita espacio).
with st.expander("📘 DB-04 · Diccionario del esquema (14 campos, orden fijo)"):
    dic = pd.DataFrame(DB04_SCHEMA, columns=["Campo", "Tipo", "Obligatorio", "Descripción", "Ejemplo"])
    dic.insert(0, "#", range(1, len(dic) + 1))
    dic["Obligatorio"] = dic["Obligatorio"].map({True: "Sí", False: "No"})
    st.dataframe(dic, hide_index=True, key="cola_db04_dic", height=38 + 35 * len(dic),
                 column_config={"#": st.column_config.NumberColumn(width=36),
                                "Campo": st.column_config.TextColumn(width=130),
                                "Tipo": st.column_config.TextColumn(width=190),
                                "Obligatorio": st.column_config.TextColumn(width=86),
                                "Descripción": st.column_config.TextColumn(width=330),
                                "Ejemplo": st.column_config.TextColumn(width=180)})
    st.markdown("- **CSV:** separador `;`, punto decimal, codificación UTF-8 con BOM (abre directo en Excel es-CO), "
                "una fila por crédito.\n"
                "- **JSON:** objeto con metadatos (`esquema`, `version`, `generado`, `total_registros`, `campos`) y la "
                "lista `registros` con los mismos 14 campos.\n"
                "- **Llave de integración:** `llave2` (única por crédito). `estado` inicia en `PENDIENTE` y lo actualiza "
                "el sistema de cartera (p. ej. CONTACTADO, ACUERDO, NO CONTACTADO).")
with st.expander("🧪 DB-04 · Casos de prueba de integración ejecutados"):
    if plan.empty:
        st.caption("Sin archivo que validar: la cola de la semana está vacía.")
    else:
        st.dataframe(pd.DataFrame([{"OK": "✅" if t[1] else "❌", "Caso de prueba": t[0], "Detalle": t[2]}
                                   for t in tests]), hide_index=True, key="cola_db04_tests",
                     height=38 + 35 * len(tests),
                     column_config={"OK": st.column_config.TextColumn(width=44),
                                    "Caso de prueba": st.column_config.TextColumn(width=300)})

footer()
