"""Segmentos y perfiles — ¿qué segmentos concentran el riesgo, qué tan sólida es cada diferencia y el
modelo trata por igual a todos los grupos?

Materializa el *enfoque descriptivo* del reporte (comportamiento demográfico, académico y financiero):

* **Explorador de segmentos**: ranking por cualquier dimensión con intervalos de confianza de Wilson al
  95 % (opcional: corrección de Bonferroni), lift contra el promedio, radiografía del segmento elegido
  (qué lo distingue del resto) y cruces de dos dimensiones en un mapa de calor.
* **Académico**: treemap facultad → programa, evolución del % Alto por segmento de programa y tabla de
  programas con tendencia.
* **Financiero**: distribuciones por clase de riesgo (violín + δ de Cliff), antigüedad por clase,
  riesgo por deciles y la regla DA-02 (financiación > 85 % de la matrícula).
* **Scoring externo**: relación monotónica por rango de score (ρ de Spearman) y efecto de los valores
  de Datacrédito (0 = sin dato antes de 2024).
* **Perfil y equidad**: auditoría de sesgo con variables sensibles (razón de disparidad en [0,8; 1,25]).
* **Predicho vs observado**: dónde el modelo sobre o subestima el riesgo por segmento.

Las variables sensibles (género, edad, estado civil, grupo étnico) se excluyen de los hallazgos de
concentración: en esta página solo se usan para auditar el sesgo del modelo.
"""
from __future__ import annotations

import math
import re
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import download_bar, empty_state, esc, filter_chips, footer, insight, kpi_card, page_header, section
from core.config import MAX_FINANCIACION_PCT, MODEL_FEATURES, RISK_COLORS, RISK_ORDER, SCORE_ORDER, SCORE_SHORT
from core.data import get_active_df, nrm
from core.filters import MULTI, PKEY, active_chips, apply_filters
from core.nav import PAGES, can_access, goto
from core.theme import SEQ_RISK, YELLOW, fmt_cop, fmt_int, fmt_num, fmt_pct, pal, show_fig, theme_mode

P = "segmentos"
NAN = float("nan")
Z95 = 1.959964
BAND = (0.80, 1.25)  # rango aceptable de la razón de disparidad (regla de los cuatro quintos, simétrica)

# ======================================================================================
# Registro de dimensiones
# ======================================================================================
_DIM_GROUPS = [
    ("Académico", [("programa_cluster", "Segmento de programa"), ("facultad", "Facultad"), ("programa", "Programa"),
                   ("nivel", "Nivel académico"), ("sede", "Sede"), ("cohorte", "Cohorte"),
                   ("tipo_estudiante", "Tipo de estudiante"), ("tipoestudiante", "Condición académica")]),
    ("Crédito", [("tipo_interes", "Tipo de interés"), ("cuotas", "Número de cuotas"), ("nombre_linea", "Línea de crédito"),
                 ("fecha_de_pago", "Día de pago"), ("operacion_limpia", "Operación comercial"),
                 ("cliente_limpio", "Tipo de cliente"), ("anio", "Año de aprobación"),
                 ("semestre", "Semestre de aprobación")]),
    ("Scoring", [("score_corto", "Scoring externo"), ("plataforma", "Plataforma de scoring"),
                 ("mora_txt", "Mora Datacrédito")]),
    ("Perfil", [("rango_edad", "Rango de edad"), ("genero_txt", "Género"), ("estado_civil", "Estado civil")]),
]
DIMS = {c: lbl for _, items in _DIM_GROUPS for c, lbl in items}
DIM_GROUP = {c: g for g, items in _DIM_GROUPS for c, _ in items}
DIMS["grupo_etnico"] = "Grupo étnico"
DIM_GROUP["grupo_etnico"] = "Perfil"
EXPLORER_DIMS = [c for _, items in _DIM_GROUPS for c, _ in items]
ORDINAL = {"cuotas", "fecha_de_pago", "anio", "semestre", "cohorte", "score_corto", "rango_edad"}
SENSITIVE = {"genero_txt": "Género", "rango_edad": "Rango de edad", "estado_civil": "Estado civil",
             "grupo_etnico": "Grupo étnico"}
# Dimensiones no sensibles que alimentan los hallazgos y el conteo de segmentos significativos.
PANEL = ["programa_cluster", "facultad", "nivel", "sede", "cohorte", "tipo_estudiante", "tipo_interes", "cuotas",
         "nombre_linea", "fecha_de_pago", "score_corto", "plataforma", "mora_txt", "cliente_limpio"]
GLOBAL_FILTER = {col: name for name, (_, col) in MULTI.items()}

CLUSTER_COLORS = {
    "light": {"Educación": "#2a78d6", "Psicología y Humanidades": "#eb6834", "Negocios y Administración": "#1baf7a",
              "Ingeniería y TI": "#eda100", "Salud": "#e87ba4", "Derecho": "#4a3aa7", "Otros": "#9A988F"},
    "dark": {"Educación": "#3987e5", "Psicología y Humanidades": "#d95926", "Negocios y Administración": "#199e70",
             "Ingeniería y TI": "#c98500", "Salud": "#d55181", "Derecho": "#9085e9", "Otros": "#7B7F88"},
}
_AGE_ORDER = ["≤ 24", "25–29", "30–34", "35–44", "45 +"]
_SCORE_LABELS = [SCORE_SHORT[s] for s in SCORE_ORDER]  # peor → mejor
_MISSING = {"nan", "none", "<na>", "", "sin dato", "nat"}
_PRETTY = {
    "SOLTERO (A)": "Soltero(a)", "UNION LIBRE": "Unión libre", "CASADO(A)": "Casado(a)", "SEPARADO (A)": "Separado(a)",
    "DIVORCIADO (A)": "Divorciado(a)", "RELIGIOSO (A)": "Religioso(a)", "VIUDO (A)": "Viudo(a)",
    "NO DECLARADO": "No declarado", "NO PERTENECE": "No pertenece", "NO INFORMA": "No informa",
    "AFRCOLOMBIANO": "Afrocolombiano", "MESTIZO": "Mestizo", "PUEBLO INDIGENA": "Pueblo indígena",
    "COMUNIDAD NEGRA": "Comunidad negra", "PUEBLO RROM": "Pueblo Rrom", "MULATO": "Mulato",
}

METRICS = {
    "% Alto predicho": {"val": "pct_pred", "lo": "lo_pred", "hi": "hi_pred", "kind": "prop", "truth": False,
                        "short": "Alto predicho"},
    "% Alto observado": {"val": "pct_obs", "lo": "lo_obs", "hi": "hi_obs", "kind": "prop", "truth": True,
                         "short": "Alto observado"},
    "% mora": {"val": "pct_mora", "lo": "lo_mora", "hi": "hi_mora", "kind": "prop", "truth": False,
               "short": "Mora Datacrédito"},
    "Exposición en Alto": {"val": "monto_alto", "kind": "money", "truth": False, "short": "Monto en Alto predicho"},
    "Ticket promedio": {"val": "ticket", "lo": "lo_ticket", "hi": "hi_ticket", "kind": "mean", "truth": False,
                        "short": "Ticket promedio"},
}

# ======================================================================================
# Estilos de la página (tokens --sat-* de core.theme)
# ======================================================================================
_CSS = """
<style>
.sg-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:4px 0 2px 0}
.sg-kpis .sat-kpi .sub{min-height:36px;line-height:1.45}
.sg-kpis .sat-kpi .val{white-space:nowrap}
.sg-ins{display:grid;grid-template-columns:repeat(auto-fit,minmax(265px,1fr));gap:12px;margin-top:2px}
.sg-ins .sat-insight{height:auto}
.sg-note{font-size:12.5px;color:var(--sat-muted);line-height:1.5;margin:2px 0 4px 0}
.sg-note b{color:var(--sat-text)}
.sg-cap{font-size:13.5px;font-weight:700;color:var(--sat-text);margin:0 0 2px 0}
.sg-legend{display:flex;gap:6px 16px;flex-wrap:wrap;font-size:12px;color:var(--sat-muted);margin:0 0 2px 0}
.sg-legend span{display:inline-flex;align-items:center;gap:6px}
.sg-legend i{display:inline-block;width:11px;height:11px;border-radius:3px;background:var(--c)}
.sg-legend i.ln{height:3px;width:16px;border-radius:2px}
.sg-legend i.dt{border-radius:50%}
.sg-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.sg-tile{background:var(--sat-surface-2);border:1px solid var(--sat-border);border-radius:14px;padding:11px 14px;position:relative;overflow:hidden}
.sg-tile.tone-alto{border-left:4px solid var(--sat-alto)}
.sg-tile.tone-bajo{border-left:4px solid var(--sat-bajo)}
.sg-tile.tone-medio{border-left:4px solid var(--sat-medio)}
.sg-tile.tone-accent{border-left:4px solid var(--sat-accent)}
.sg-tile .l{font-size:11.5px;color:var(--sat-muted);font-weight:700;letter-spacing:.02em}
.sg-tile .v{font-family:'Space Grotesk','Inter',sans-serif;font-size:23px;font-weight:700;color:var(--sat-text);line-height:1.2;margin-top:3px}
.sg-tile .v small{font-family:'Inter',sans-serif;font-size:12.5px;font-weight:500;color:var(--sat-muted)}
.sg-tile .s{font-size:12px;color:var(--sat-muted);line-height:1.4;margin-top:2px}
.sg-tile .s b{color:var(--sat-text)}
.sg-stack{display:flex;flex-direction:column;gap:10px}
.sg-prof{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:var(--sat-radius);padding:16px 18px;
  box-shadow:var(--sat-shadow);position:relative;overflow:hidden}
.sg-prof:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--sat-accent)}
.sg-prof .k{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--sat-muted)}
.sg-prof .n{font-family:'Space Grotesk','Inter',sans-serif;font-size:21px;font-weight:700;color:var(--sat-text);margin:3px 0 2px 0;
  line-height:1.2;word-break:break-word}
.sg-prof .m{font-size:12.5px;color:var(--sat-muted)}
.sg-grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0 10px 0}
.sg-grid2 div{background:var(--sat-surface-2);border:1px solid var(--sat-border);border-radius:12px;padding:8px 10px}
.sg-grid2 span{display:block;font-size:11.5px;color:var(--sat-muted);font-weight:600}
.sg-grid2 b{display:block;font-size:17px;color:var(--sat-text);margin-top:1px}
.sg-grid2 em{font-style:normal;font-size:11.5px;font-weight:700;margin-left:4px}
.sg-grid2 em.up{color:var(--sat-alto)} .sg-grid2 em.down{color:var(--sat-bajo)} .sg-grid2 em.flat{color:var(--sat-muted)}
.sg-mix{display:flex;height:12px;border-radius:7px;overflow:hidden;gap:2px;background:var(--sat-surface-2)}
.sg-mix i{display:block;height:100%}
.sg-mixcap{display:flex;gap:12px;flex-wrap:wrap;font-size:11.5px;color:var(--sat-muted);margin-top:5px}
.sg-mixcap span:before{content:"";display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;background:var(--c)}
.sg-dist{display:flex;flex-direction:column;gap:7px}
.sg-dist-row{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(90px,1fr) 118px;gap:12px;align-items:center;
  font-size:12.5px;color:var(--sat-muted);padding:7px 10px;border-radius:10px;background:var(--sat-surface-2);border:1px solid var(--sat-border)}
.sg-dist-row .nm{color:var(--sat-text);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sg-dist-row .nm small{display:block;color:var(--sat-muted);font-weight:500;font-size:11px;letter-spacing:.02em}
.sg-bars{display:flex;flex-direction:column;gap:3px}
.sg-bars i{display:block;height:6px;border-radius:6px}
.sg-bars i.a{background:var(--sat-text)} .sg-bars i.b{background:color-mix(in srgb,var(--sat-muted) 45%,transparent)}
.sg-dist-row .pc{text-align:right;font-variant-numeric:tabular-nums;color:var(--sat-text);font-weight:700;white-space:nowrap}
.sg-dist-row .pc small{display:block;font-weight:500;color:var(--sat-muted);font-size:11px}
.sg-med{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:4px}
.sg-med th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--sat-muted);text-align:right;font-weight:700;
  padding:4px 6px;border-bottom:1px solid var(--sat-border)}
.sg-med th:first-child,.sg-med td:first-child{text-align:left}
.sg-med td{padding:6px;text-align:right;color:var(--sat-text);border-bottom:1px solid var(--sat-border);font-variant-numeric:tabular-nums}
.sg-med td.d{font-weight:700}
.sg-ethic{display:flex;gap:16px;align-items:flex-start;border-radius:var(--sat-radius);padding:16px 20px;margin:4px 0 6px 0;
  background:linear-gradient(120deg,color-mix(in srgb,#3E63DD 9%,var(--sat-surface)) 0%,var(--sat-surface) 70%);
  border:1px solid color-mix(in srgb,#3E63DD 30%,var(--sat-border));box-shadow:var(--sat-shadow)}
.sg-ethic .ic{font-size:26px;line-height:1}
.sg-ethic .t{font-weight:800;color:var(--sat-text);font-size:15px;margin-bottom:4px}
.sg-ethic ul{margin:4px 0 0 0;padding-left:18px;color:var(--sat-muted);font-size:13px;line-height:1.55}
.sg-ethic li b{color:var(--sat-text)}
.sg-warn{display:flex;gap:12px;align-items:flex-start;border-radius:14px;padding:12px 16px;margin:2px 0 8px 0;
  background:color-mix(in srgb,var(--sat-medio) 11%,var(--sat-surface));border:1px solid color-mix(in srgb,var(--sat-medio) 40%,var(--sat-border))}
.sg-warn .ic{font-size:18px;line-height:1.2}
.sg-warn .tx{font-size:13px;color:var(--sat-text);line-height:1.5}
.sg-fair{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:8px 0 4px 0}
.sg-chipline{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.sg-mono{font-variant-numeric:tabular-nums}
[data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(> [class^="sg-"], > .sat-insight){margin-bottom:0}
div[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--sat-border);flex-wrap:wrap}
div[data-testid="stTabs"] button[role="tab"]{padding:8px 12px;border-radius:10px 10px 0 0}
div[data-testid="stTabs"] button[role="tab"] p{font-size:14.5px}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{background:var(--sat-accent-soft)}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background-color:var(--sat-accent) !important;height:3px}
@media (max-width:1180px){
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has([data-testid="stPlotlyChart"]) > [data-testid="stColumn"],
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.sg-prof) > [data-testid="stColumn"]{min-width:100%}
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]{row-gap:12px;flex-wrap:wrap}
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has([data-testid="stSelectbox"]) > [data-testid="stColumn"],
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has([data-testid="stButtonGroup"]) > [data-testid="stColumn"]{min-width:calc(50% - 12px)}
  [data-testid="stMain"] [data-testid="stButtonGroup"] div:has(> button){flex-wrap:wrap;row-gap:6px}
}
@media (max-width:1000px){
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has([data-testid="stButtonGroup"]) > [data-testid="stColumn"],
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has([data-testid="stSelectbox"]) > [data-testid="stColumn"],
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has([data-testid="stButton"]) > [data-testid="stColumn"]{min-width:100%}
  .sg-dist-row{grid-template-columns:minmax(0,1fr) 80px 104px}
}
</style>
"""


# ======================================================================================
# Utilidades
# ======================================================================================
def _isnan(x) -> bool:
    return x is None or (isinstance(x, (float, np.floating)) and math.isnan(x))


def _div(a, b) -> float:
    try:
        a, b = float(a), float(b)
        return a / b if b else NAN
    except (TypeError, ValueError):
        return NAN


def _x(v, d: int = 1) -> str:
    return "—" if _isnan(v) else f"{fmt_num(v, d)}×"


def _pp(v, d: int = 1) -> str:
    if _isnan(v):
        return "—"
    return f"{'+' if v >= 0 else '−'}{fmt_num(abs(v) * 100, d)} pp"


def _short(s: str, n: int = 30) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _dedupe(labels: list[str]) -> list[str]:
    seen, out = {}, []
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
        out.append(lab if seen[lab] == 1 else f"{lab} ({seen[lab]})")
    return out


def _wilson(k, n, z: float = Z95):
    """Proporción e intervalo de Wilson (vectorizado). Devuelve (p, lo, hi)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        den = 1 + z ** 2 / n
        c = (p + z ** 2 / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def _z_bonferroni(m: int, alpha: float = 0.05) -> float:
    m = max(int(m), 1)
    return NormalDist().inv_cdf(1 - alpha / (2 * m))


def _spearman(a, b) -> float:
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return NAN
    ra, rb = a[ok].rank(), b[ok].rank()
    if ra.std() == 0 or rb.std() == 0:
        return NAN
    return float(np.corrcoef(ra, rb)[0, 1])


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """δ de Cliff = 2·P(A > B) + P(A = B) − 1, vía la U de Mann-Whitney."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return NAN
    ranks = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    u = ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2
    return float(2 * u / (len(a) * len(b)) - 1)


def _cliff_txt(d: float) -> str:
    if _isnan(d):
        return "sin datos suficientes"
    a = abs(d)
    return "insignificante" if a < 0.147 else "pequeño" if a < 0.33 else "mediano" if a < 0.474 else "grande"


def _labels(d: pd.DataFrame, col: str) -> pd.Series:
    """Etiquetas legibles (str) de una dimensión, con 'Sin dato' para faltantes."""
    s = d[col]
    if col in ("cuotas", "fecha_de_pago", "anio"):
        v = pd.to_numeric(s, errors="coerce")
        tpl = {"cuotas": "{} cuotas", "fecha_de_pago": "Día {}", "anio": "{}"}[col]
        uniq = {x: (tpl.format(int(x)) if pd.notna(x) else "Sin dato") for x in pd.unique(v)}
        return v.map(uniq).fillna("Sin dato").astype(str)
    out = s.astype(str).str.strip()
    out = out.where(~out.str.lower().isin(_MISSING), "Sin dato")
    if col in ("estado_civil", "grupo_etnico"):
        uniq = {x: _PRETTY.get(x, x[:1].upper() + x[1:].lower() if x.isupper() else x) for x in out.unique()}
        out = out.map(uniq)
    return out


def _nat_key(col: str):
    if col in ("cuotas", "fecha_de_pago", "anio", "cohorte"):
        def k(x):
            num = re.sub(r"\D", "", str(x))
            return (x == "Sin dato", float(num) if num else 1e9, str(x))
        return k
    if col == "score_corto":
        return lambda x: (_SCORE_LABELS.index(x) if x in _SCORE_LABELS else 99, str(x))
    if col == "rango_edad":
        return lambda x: (_AGE_ORDER.index(x) if x in _AGE_ORDER else 99, str(x))
    return lambda x: (x == "Sin dato", nrm(x))


def _dim_title(col: str) -> str:
    return f"{DIM_GROUP.get(col, '')} · {DIMS.get(col, col)}"


def _fmt_val(v, kind: str) -> str:
    if kind == "prop":
        return fmt_pct(v)
    return fmt_cop(v)


def _set_global(name: str, value: str) -> None:
    s = st.session_state.get(PKEY)
    if isinstance(s, dict):
        s[name] = [value]


# ======================================================================================
# Agregaciones
# ======================================================================================
def _prep(d: pd.DataFrame) -> pd.DataFrame:
    """Marco numérico mínimo para agregar rápido cualquier segmentación."""
    yp = d["y_pred"].astype(str).to_numpy()
    yt = d["y_true"].astype(str).to_numpy()
    t = d["has_truth"].to_numpy(dtype=bool)
    pa, oa = yp == "Alto", yt == "Alto"
    v = pd.to_numeric(d["valor_financiacion"], errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame({
        "_pn": d["y_pred"].notna().to_numpy(dtype=float),
        "_ap": pa.astype(float),
        "_t": t.astype(float),
        "_ao": np.where(t, oa, np.nan),
        "_tp": np.where(t, pa & oa, np.nan),
        "_pt": np.where(t, pa, np.nan),
        "_fp": np.where(t, pa & ~oa, np.nan),
        "_neg": np.where(t, ~oa, np.nan),
        "_m": pd.to_numeric(d["mora_flag"], errors="coerce").fillna(0).to_numpy(dtype=float),
        "_v": v,
        "_va": np.where(pa, np.nan_to_num(v), 0.0),
        "_id": d["id_estudiante"].astype(str).to_numpy(),
    }, index=d.index)


def _intervals(t: pd.DataFrame, z: float = Z95) -> pd.DataFrame:
    t = t.copy()
    t["pct_pred"], t["lo_pred"], t["hi_pred"] = _wilson(t["k_pred"], t["n_pred"], z)
    t["pct_obs"], t["lo_obs"], t["hi_obs"] = _wilson(t["k_obs"], t["n_obs"], z)
    t["pct_mora"], t["lo_mora"], t["hi_mora"] = _wilson(t["k_mora"], t["n"], z)
    se = t["ticket_sd"] / np.sqrt(t["n_v"].clip(lower=1))
    t["lo_ticket"] = t["ticket"] - z * se
    t["hi_ticket"] = t["ticket"] + z * se
    return t


def _agg(x: pd.DataFrame, keys, students: bool = False, z: float = Z95) -> pd.DataFrame:
    """Conteos, tasas con IC de Wilson y métricas de clasificación de Alto por grupo."""
    spec = dict(n=("_pn", "size"), n_pred=("_pn", "sum"), k_pred=("_ap", "sum"), n_obs=("_t", "sum"),
                k_obs=("_ao", "sum"), tp=("_tp", "sum"), pt=("_pt", "sum"), fp=("_fp", "sum"), neg=("_neg", "sum"),
                k_mora=("_m", "sum"), monto=("_v", "sum"), monto_alto=("_va", "sum"), ticket=("_v", "mean"),
                ticket_sd=("_v", "std"), n_v=("_v", "count"))
    if students:
        spec["est"] = ("_id", "nunique")
    g = x.groupby(keys, observed=True, sort=False).agg(**spec).reset_index()
    g = _intervals(g, z)
    with np.errstate(invalid="ignore", divide="ignore"):
        g["recall"] = np.where(g["k_obs"] > 0, g["tp"] / g["k_obs"], np.nan)
        g["precision"] = np.where(g["pt"] > 0, g["tp"] / g["pt"], np.nan)
        g["fpr"] = np.where(g["neg"] > 0, g["fp"] / g["neg"], np.nan)
    g["gap"] = g["pct_pred"] - g["pct_obs"]
    return g


def _total(x: pd.DataFrame) -> dict:
    return _agg(x.assign(_all="Cartera"), "_all", students=True).iloc[0].to_dict()


def _segments(dff: pd.DataFrame, x: pd.DataFrame, col: str, students: bool = False) -> pd.DataFrame:
    return _agg(x.assign(seg=_labels(dff, col).to_numpy()), "seg", students=students)


def _panel(dff: pd.DataFrame, x: pd.DataFrame, min_n: int = 50) -> pd.DataFrame:
    rows = []
    for c in PANEL:
        if c not in dff.columns:
            continue
        t = _segments(dff, x, c)
        t = t[(t["n"] >= min_n) & (t["seg"] != "Sin dato")]
        if len(t) > 1:
            rows.append(t.assign(dim=c))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fold(labels: pd.Series, min_n: int, other: str = "Otros") -> pd.Series:
    vc = labels.value_counts()
    small = set(vc[vc < min_n].index)
    if len(small) <= 1 and not (small and vc.get(next(iter(small)), 0) >= 1 and len(vc) > 1):
        pass
    return labels.where(~labels.isin(small), other) if small else labels


def _fold_top(labels: pd.Series, top: int, other: str = "Otros") -> pd.Series:
    vc = labels.value_counts()
    if len(vc) <= top:
        return labels
    keep = set(vc.index[:top])
    return labels.where(labels.isin(keep), other)


# ======================================================================================
# Figuras
# ======================================================================================
def _seq_scale(m: float) -> list:
    """SEQ_RISK con el color central anclado en ``m`` (posición del promedio en [0, 1])."""
    m = min(max(m, 0.12), 0.88)
    c = SEQ_RISK
    return [[0, c[0]], [m * 0.5, c[1]], [m, c[2]], [m + (1 - m) * 0.4, c[3]], [m + (1 - m) * 0.75, c[4]], [1, c[5]]]


def _obs_color() -> str:
    return "#3987e5" if theme_mode() == "dark" else "#2a78d6"


def _sig_colors() -> dict:
    p = pal()
    return {"sobre": RISK_COLORS["Alto"], "bajo": RISK_COLORS["Bajo"], "nc": p["subtle"]}


def _rank_fig(t: pd.DataFrame, mdef: dict, mname: str, ref: float, ci: bool) -> go.Figure:
    p = pal()
    t = t.iloc[::-1].reset_index(drop=True)
    kind = mdef["kind"]
    sig_c = _sig_colors()
    if kind == "money":
        colors = [RISK_COLORS["Alto"]] * len(t)
    elif kind == "mean":
        cmap = {"sobre": p["text"], "bajo": p["subtle"], "nc": p["subtle"]}
        colors = [cmap[s] for s in t["sig"]]
    else:
        colors = [sig_c[s] for s in t["sig"]]
    labels = _dedupe([_short(s, 34) for s in t["seg"]])
    val = t["val"].to_numpy(dtype=float)
    has_ci = ci and kind in ("prop", "mean")
    hi = t["hi"].to_numpy(dtype=float) if has_ci else val
    lo = t["lo"].to_numpy(dtype=float) if has_ci else val
    cd = np.c_[
        [esc(s) for s in t["seg"]], [fmt_int(v) for v in t["n"]], [_fmt_val(v, kind) for v in val],
        [f"{_fmt_val(a, kind)} – {_fmt_val(b, kind)}" if kind in ("prop", "mean") else "—"
         for a, b in zip(t["lo"] if "lo" in t else val, t["hi"] if "hi" in t else val)],
        [_x(v, 2) for v in t["lift"]], [fmt_pct(v) for v in t["pct_pred"]], [fmt_pct(v) for v in t["pct_obs"]],
        [fmt_pct(v) for v in t["pct_mora"]], [fmt_cop(v) for v in t["monto"]], [fmt_cop(v) for v in t["ticket"]],
        [{"sobre": "sobre el promedio (significativo)", "bajo": "bajo el promedio (significativo)",
          "nc": "diferencia no concluyente"}[s] for s in t["sig"]],
    ]
    ci_line = "<br>IC 95 %: %{customdata[3]}" if kind in ("prop", "mean") else ""
    lift_line = "<br>Lift: %{customdata[4]} · %{customdata[10]}" if kind != "money" else ""
    fig = go.Figure(go.Bar(
        y=labels, x=val, orientation="h", marker=dict(color=colors, line=dict(width=0)), customdata=cd,
        error_x=dict(type="data", symmetric=False, array=np.nan_to_num(hi - val), arrayminus=np.nan_to_num(val - lo),
                     color=p["muted"], thickness=1.4, width=4, visible=has_ci),
        hovertemplate=(f"<b>%{{customdata[0]}}</b><br>{esc(mname)}: <b>%{{customdata[2]}}</b>{ci_line}{lift_line}"
                       "<br>Créditos: %{customdata[1]}"
                       "<br>Alto predicho %{customdata[5]} · observado %{customdata[6]} · mora %{customdata[7]}"
                       "<br>Monto %{customdata[8]} · ticket %{customdata[9]}<extra></extra>"),
    ))
    if kind == "money":
        tot = float(np.nansum(t["val"])) or 1.0
        txt = [f"<b>{fmt_cop(v)}</b> · {fmt_pct(v / tot, 0)}" for v in val]
    else:
        txt = [f"<b>{_fmt_val(v, kind)}</b> · {_x(l, 1)}" for v, l in zip(val, t["lift"])]
    xpos = np.where(np.isnan(hi), val, hi)
    fig.add_trace(go.Scatter(x=xpos, y=labels, mode="text", text=txt, textposition="middle right",
                             textfont=dict(size=11.5, color=p["text"]), hoverinfo="skip", cliponaxis=False,
                             showlegend=False))
    if not _isnan(ref) and kind != "money":
        fig.add_vline(x=ref, line=dict(color=p["text"], width=1.2, dash="dot"), layer="below")
        fig.add_annotation(x=ref, y=1, yref="paper", text=f"Promedio {_fmt_val(ref, kind)}", showarrow=False,
                           xanchor="left", yanchor="bottom", xshift=4, font=dict(size=11, color=p["muted"]))
    xmax = float(np.nanmax(xpos)) if len(t) else 1.0
    fig.update_xaxes(range=[0, max(xmax * 1.32, 1e-3)], tickformat=".0%" if kind == "prop" else "~s",
                     tickprefix="" if kind == "prop" else "$ ", title_text=mname, showgrid=True, gridcolor=p["grid"])
    fig.update_yaxes(showgrid=False, ticksuffix="  ", tickfont=dict(size=12, color=p["text"]))
    fig.update_layout(bargap=0.35, barcornerradius=4, margin=dict(l=8, r=16, t=28, b=8), showlegend=False,
                      clickmode="event+select")
    return fig


def _heatmap_fig(z, text, cd, rows, cols, avg: float, zmax: float, rlab: str, clab: str, mname: str) -> go.Figure:
    p = pal()
    zmax = max(zmax, avg * 1.6, 1e-3)
    fig = go.Figure(go.Heatmap(
        z=z, x=[_short(c, 22) for c in cols], y=_dedupe([_short(r, 30) for r in rows]), text=text,
        texttemplate="%{text}", textfont=dict(size=11.5), customdata=cd, zmin=0, zmax=zmax,
        colorscale=_seq_scale(avg / zmax if zmax else 0.5), xgap=3, ygap=3, hoverongaps=False,
        colorbar=dict(title=dict(text=mname, side="right", font=dict(size=11, color=p["muted"])), tickformat=".0%",
                      thickness=10, len=0.9, outlinewidth=0, tickfont=dict(color=p["muted"], size=10)),
        hovertemplate=(f"<b>{esc(rlab)}:</b> %{{customdata[0]}}<br><b>{esc(clab)}:</b> %{{customdata[1]}}"
                       f"<br>{esc(mname)}: <b>%{{customdata[2]}}</b> (IC 95 %: %{{customdata[3]}})"
                       "<br>Créditos: %{customdata[4]} · lift %{customdata[5]}<extra></extra>"),
    ))
    fig.update_xaxes(side="top", showgrid=False, tickangle=0, tickfont=dict(size=11.5, color=p["text"]))
    fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  ", tickfont=dict(size=11.5, color=p["text"]))
    fig.update_layout(margin=dict(l=8, r=8, t=36, b=8), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _pred_obs_bars(xlabels, t: pd.DataFrame, ref_pred: float, ref_obs: float, has_truth: bool,
                   ytitle: str = "% de créditos en riesgo Alto", hatch_first: bool = False) -> go.Figure:
    """Barras = % Alto predicho, puntos con IC = % Alto observado (un solo eje en %)."""
    p = pal()
    oc = _obs_color()
    cd = np.c_[[esc(s) for s in t["hover_lbl"]], [fmt_int(v) for v in t["n"]], [fmt_pct(v) for v in t["pct_pred"]],
               [f"{fmt_pct(a)} – {fmt_pct(b)}" for a, b in zip(t["lo_pred"], t["hi_pred"])],
               [fmt_pct(v) for v in t["pct_obs"]], [f"{fmt_pct(a)} – {fmt_pct(b)}" for a, b in zip(t["lo_obs"], t["hi_obs"])],
               [fmt_pct(v) for v in t["pct_mora"]]]
    patt = ["/" if (hatch_first and i == 0) else "" for i in range(len(t))]
    fig = go.Figure()
    fig.add_bar(x=xlabels, y=t["pct_pred"], name="Alto predicho (modelo)",
                marker=dict(color=p["text"], opacity=0.88, line=dict(width=0),
                            pattern=dict(shape=patt, fgcolor=p["surface"], size=6)),
                customdata=cd, width=0.52,
                hovertemplate=("<b>%{customdata[0]}</b><br>Créditos: %{customdata[1]}"
                               "<br>Alto predicho: <b>%{customdata[2]}</b> (IC %{customdata[3]})"
                               "<br>Alto observado: %{customdata[4]} (IC %{customdata[5]})"
                               "<br>Mora Datacrédito: %{customdata[6]}<extra></extra>"))
    if has_truth:
        po = t["pct_obs"].to_numpy(dtype=float)
        fig.add_trace(go.Scatter(
            x=xlabels, y=po, name="Alto observado (IC 95 %)", mode="markers+lines",
            line=dict(color=oc, width=2), marker=dict(size=10, color=oc, line=dict(color=p["bg"], width=2)),
            error_y=dict(type="data", symmetric=False, array=np.nan_to_num(t["hi_obs"] - po),
                         arrayminus=np.nan_to_num(po - t["lo_obs"]), color=oc, thickness=1.5, width=5),
            customdata=cd, hovertemplate=("<b>%{customdata[0]}</b><br>Alto observado: <b>%{customdata[4]}</b>"
                                          " (IC %{customdata[5]})<br>Créditos: %{customdata[1]}<extra></extra>")))
    ref = ref_obs if has_truth and not _isnan(ref_obs) else ref_pred
    if not _isnan(ref):
        fig.add_hline(y=ref, line=dict(color=p["muted"], width=1.1, dash="dot"), layer="below")
        fig.add_annotation(x=1, xref="paper", y=ref, text=f"Promedio {'observado' if has_truth else 'predicho'} "
                           f"{fmt_pct(ref)}", showarrow=False, xanchor="right", yanchor="bottom",
                           font=dict(size=11, color=p["muted"]))
    top = np.nanmax(np.r_[t["pct_pred"].to_numpy(dtype=float), t["hi_obs"].to_numpy(dtype=float) if has_truth else [0],
                          [0.01]])
    fig.update_yaxes(tickformat=".0%", title_text=ytitle, range=[0, top * 1.18])
    fig.update_xaxes(type="category", tickfont=dict(size=11.5, color=p["text"]))
    fig.update_layout(barcornerradius=4, margin=dict(l=8, r=8, t=36, b=8), hovermode="closest")
    return fig


# ======================================================================================
# HTML
# ======================================================================================
def _legend(items: list[tuple[str, str, str]]) -> str:
    """items: (color, texto, forma: sq|ln|dt)."""
    return "<div class='sg-legend'>" + "".join(
        f"<span><i class='{shape}' style='--c:{c}'></i>{esc(txt)}</span>" for c, txt, shape in items) + "</div>"


def _note(html_text: str) -> None:
    st.markdown(f"<div class='sg-note'>{html_text}</div>", unsafe_allow_html=True)


def _tile(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (f"<div class='sg-tile tone-{tone}'><div class='l'>{esc(label)}</div><div class='v'>{value}</div>"
            f"<div class='s'>{sub}</div></div>")


def _mix_html(d: pd.DataFrame) -> str:
    vc = d["y_pred"].astype(str).value_counts(normalize=True)
    bars = "".join(f"<i style='width:{vc.get(r, 0) * 100:.2f}%;background:{RISK_COLORS[r]}'></i>" for r in RISK_ORDER
                   if vc.get(r, 0) > 0)
    cap = "".join(f"<span style='--c:{RISK_COLORS[r]}'>{r} {fmt_pct(vc.get(r, 0))}</span>" for r in RISK_ORDER)
    return f"<div class='sg-mix'>{bars}</div><div class='sg-mixcap'>{cap}</div>"


def _delta_em(v: float, ref: float) -> str:
    if _isnan(v) or _isnan(ref):
        return ""
    d = v - ref
    cls = "up" if d > 0.0005 else "down" if d < -0.0005 else "flat"
    return f"<em class='{cls}'>{_pp(d)}</em>"


# ======================================================================================
# Página
# ======================================================================================
st.markdown(_CSS, unsafe_allow_html=True)

df_all = get_active_df()
dff = apply_filters(df_all)
has_truth_all = bool(dff["has_truth"].any()) if len(dff) else False

X = _prep(dff) if len(dff) else None
TOT = _total(X) if len(dff) else {}
PANEL_T = _panel(dff, X) if len(dff) else pd.DataFrame()
BASIS = "obs" if has_truth_all else "pred"
BASIS_TXT = "observado" if has_truth_all else "predicho"

n_sig = 0
if len(PANEL_T):
    ref_b = TOT.get(f"pct_{BASIS}", NAN)
    n_sig = int((PANEL_T[f"lo_{BASIS}"] > ref_b).sum())

page_header(
    "Segmentos y perfiles",
    "Qué grupos de la cartera concentran el riesgo, qué tan sólida es cada diferencia (intervalos de confianza) "
    "y si el modelo trata por igual a todos los perfiles: el enfoque descriptivo del proyecto convertido en "
    "criterios para focalizar la gestión preventiva.",
    eyebrow="Análisis de riesgo · enfoque descriptivo",
    highlight=f"{n_sig} segmentos sobre el promedio" if len(dff) else None,
    meta=["IC de Wilson 95 %", "Auditoría de equidad"],
)
filter_chips(active_chips(df_all))

if dff.empty:
    empty_state("Sin créditos para analizar", "Los filtros actuales no dejan créditos. Ajusta o limpia los filtros "
                "de la barra lateral.")
    footer()
    st.stop()


# ---------------------------------------------------------------- KPI
def _kpis() -> None:
    n = int(TOT["n"])
    cards = [kpi_card("Créditos analizados", fmt_int(n),
                      f"{fmt_int(TOT['est'])} estudiantes · {dff['programa_cluster'].nunique()} segmentos de programa",
                      tone="ink", icon="📄")]
    cards.append(kpi_card("En riesgo Alto (predicho)", fmt_pct(TOT["pct_pred"]),
                          f"IC 95 %: {fmt_pct(TOT['lo_pred'])} – {fmt_pct(TOT['hi_pred'])} · "
                          f"{fmt_int(TOT['k_pred'])} créditos", tone="alto", icon="🔴",
                          bar=min(TOT["pct_pred"] / 0.25, 1) if not _isnan(TOT["pct_pred"]) else None))
    if has_truth_all:
        gap = TOT["pct_obs"] - TOT["pct_pred"]
        cards.append(kpi_card("Riesgo Alto observado", fmt_pct(TOT["pct_obs"]),
                              f"IC 95 %: {fmt_pct(TOT['lo_obs'])} – {fmt_pct(TOT['hi_obs'])} · "
                              f"{fmt_int(TOT['k_obs'])} créditos", tone="medio", icon="🎯",
                              delta=f"{_pp(gap)} vs. predicho", delta_dir="up" if gap > 0.001 else "down" if gap < -0.001 else "flat",
                              help="Diferencia positiva: el modelo (regla argmax) marca menos Alto de los que se observan."))
    else:
        cards.append(kpi_card("Riesgo Alto observado", "—", "El dataset activo no trae riesgo observado (y_true)",
                              tone="medio", icon="🎯"))
    # dispersión entre segmentos de programa
    seg = _segments(dff, X, "programa_cluster")
    seg = seg[seg["n"] >= 30]
    vcol = f"pct_{BASIS}"
    if len(seg) >= 2 and seg[vcol].min() > 0:
        hi_r, lo_r = seg.loc[seg[vcol].idxmax()], seg.loc[seg[vcol].idxmin()]
        cards.append(kpi_card("Brecha entre segmentos", _x(hi_r[vcol] / lo_r[vcol], 1),
                              f"{esc(hi_r['seg'])} {fmt_pct(hi_r[vcol])} vs. {esc(lo_r['seg'])} {fmt_pct(lo_r[vcol])} "
                              f"(Alto {BASIS_TXT})", tone="accent", icon="↕️"))
    else:
        cards.append(kpi_card("Brecha entre segmentos", "—", "Se necesitan ≥ 2 segmentos con 30 créditos", tone="accent",
                              icon="↕️"))
    n_eval = len(PANEL_T)
    cards.append(kpi_card("Segmentos sobre el promedio", f"{n_sig} de {n_eval}",
                          f"IC 95 % por encima del promedio (Alto {BASIS_TXT}) · {len(PANEL)} dimensiones, n ≥ 50",
                          tone="alto", icon="📌", bar=n_sig / n_eval if n_eval else None))
    st.markdown("<div class='sg-kpis'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


_kpis()


# ---------------------------------------------------------------- Hallazgos
def _fair_summary(d: pd.DataFrame, x: pd.DataFrame, attr: str, gmin: int = 100) -> dict | None:
    if attr not in d.columns:
        return None
    lab = _fold(_labels(d, attr), gmin)
    t = _agg(x.assign(g=lab.to_numpy()), "g")
    t = t[t["n"] >= 1]
    if len(t) < 2:
        return None
    ref = t.loc[t["n"].idxmax()]
    t = t.assign(r_pred=t["pct_pred"] / ref["pct_pred"] if ref["pct_pred"] > 0 else np.nan,
                 r_obs=t["pct_obs"] / ref["pct_obs"] if ref["pct_obs"] > 0 else np.nan)
    others = t[t["g"] != ref["g"]]
    out = others[(others["r_pred"] < BAND[0]) | (others["r_pred"] > BAND[1])]
    far = others.iloc[(others["r_pred"] - 1).abs().argsort()[::-1]].iloc[0] if len(others) else None
    return {"attr": attr, "ref": ref["g"], "n_groups": len(t), "n_out": len(out), "table": t, "far": far,
            "rmin": float(others["r_pred"].min()), "rmax": float(others["r_pred"].max())}


def _findings() -> list[str]:
    cards = []
    vcol, locol = f"pct_{BASIS}", f"lo_{BASIS}"
    ref = TOT.get(vcol, NAN)
    # 1) concentración
    if len(PANEL_T) and not _isnan(ref) and ref > 0:
        sig = PANEL_T[PANEL_T[locol] > ref].copy()
        if len(sig):
            sig["lift"] = sig[vcol] / ref
            r = sig.sort_values(["lift", "n"], ascending=False).iloc[0]
            extra = (f"; el modelo le asigna {fmt_pct(r['pct_pred'])}" if has_truth_all else "")
            cards.append(insight("Dónde se concentra el riesgo",
                                 f"<b>{esc(r['seg'])}</b> ({esc(DIMS[r['dim']].lower())}) registra "
                                 f"<b>{fmt_pct(r[vcol])}</b> de riesgo Alto {BASIS_TXT}, <b>{_x(r['lift'], 1)}</b> el "
                                 f"promedio (IC 95 %: {fmt_pct(r[locol])} – {fmt_pct(r[f'hi_{BASIS}'])}; "
                                 f"n = {fmt_int(r['n'])}){extra}.", tone="alto", icon="🎯"))
        else:
            cards.append(insight("Sin concentraciones significativas",
                                 "Ningún segmento supera el promedio con un IC 95 % completo por encima; las "
                                 "diferencias observadas pueden ser azar muestral.", tone="bajo", icon="🎯"))
    # 2) scoring externo
    sc = _segments(dff, X, "score_corto")
    sc = sc[(sc["n"] >= 30) & sc["seg"].isin(_SCORE_LABELS[1:])].copy()
    if len(sc) >= 3:
        sc["rk"] = sc["seg"].map(_SCORE_LABELS.index)
        sc = sc.sort_values("rk")
        rho = _spearman(sc["rk"], sc[vcol])
        worst = sc.loc[sc[vcol].idxmax()]
        best = sc.iloc[-1]
        rr = _div(worst[vcol], best[vcol])
        cards.append(insight("Scoring externo y riesgo",
                             f"El riesgo Alto {BASIS_TXT} pasa de <b>{fmt_pct(worst[vcol])}</b> en el rango "
                             f"<b>{esc(worst['seg'])}</b> a <b>{fmt_pct(best[vcol])}</b> en <b>{esc(best['seg'])}</b> "
                             f"({_x(rr, 1)}). ρ de Spearman = <b>{fmt_num(rho, 2)}</b>: "
                             f"{'a mejor score, menos riesgo' if rho < -0.3 else 'relación débil en este filtro'}.",
                             tone="info", icon="📉"))
    # 3) antigüedad
    cls = dff["y_true"] if has_truth_all else dff["y_pred"]
    ant = pd.to_numeric(dff["antiguedad_meses"], errors="coerce")
    ma, mb = ant[cls.astype(str) == "Alto"].median(), ant[cls.astype(str) == "Bajo"].median()
    if not (_isnan(ma) or _isnan(mb)):
        rec = ma < mb
        cards.append(insight("Antigüedad del crédito",
                             f"Los créditos en Alto {BASIS_TXT} tienen una mediana de <b>{fmt_num(ma, 0)} meses</b> "
                             f"frente a <b>{fmt_num(mb, 0)}</b> en Bajo: "
                             + ("el riesgo se concentra en créditos recientes, como anticipa el EDA del reporte."
                                if rec else "en este filtro el riesgo no se concentra en créditos recientes."),
                             tone="medio", icon="⏱️"))
    # 4) subestimación del modelo
    if has_truth_all and len(PANEL_T):
        pv = PANEL_T[PANEL_T["n"] >= 100].copy()
        if len(pv):
            pv["under"] = pv["pct_obs"] - pv["pct_pred"]
            r = pv.sort_values("under", ascending=False).iloc[0]
            if r["under"] > 0:
                cards.append(insight("Donde el modelo se queda corto",
                                     f"En <b>{esc(r['seg'])}</b> ({esc(DIMS[r['dim']].lower())}) se observa "
                                     f"<b>{fmt_pct(r['pct_obs'])}</b> de Alto y el modelo marca "
                                     f"<b>{fmt_pct(r['pct_pred'])}</b> ({_pp(-r['under'])}). Revísalo en "
                                     f"<i>Predicho vs observado</i>.", tone="accent", icon="🧭"))
    # 5) equidad
    fs = [f for f in (_fair_summary(dff, X, a) for a in SENSITIVE) if f]
    if fs:
        n_out = sum(1 for f in fs if f["n_out"])
        worst = max(fs, key=lambda f: max(abs(f["rmax"] - 1), abs(1 - f["rmin"])) if not _isnan(f["rmax"]) else 0)
        far = worst["far"]
        det = (f"Mayor brecha: {esc(SENSITIVE[worst['attr']].lower())}, <b>{esc(far['g'])}</b> vs. "
               f"{esc(worst['ref'])} = <b>{fmt_num(far['r_pred'], 2)}</b>.") if far is not None and not _isnan(far["r_pred"]) else ""
        cards.append(insight("Auditoría de equidad",
                             f"<b>{n_out} de {len(fs)}</b> atributos sensibles con alguna razón de disparidad de "
                             f"alertas fuera de [0,8; 1,25]. {det}", tone="alto" if n_out else "bajo", icon="⚖️"))
    return cards


section("Hallazgos automáticos", "Se recalculan con cada filtro. Las concentraciones solo usan variables no sensibles; "
        "las sensibles se reservan para la auditoría de equidad.", kicker="Lectura rápida")
_cards = _findings()
if _cards:
    st.markdown("<div class='sg-ins'>" + "".join(_cards) + "</div>", unsafe_allow_html=True)


# ======================================================================================
# Pestaña 1 · Explorador de segmentos
# ======================================================================================
_HM_PRESETS = {
    "Interés × cuotas": ("tipo_interes", "cuotas"),
    "Cohorte × nivel": ("cohorte", "nivel"),
    "Segmento × año": ("programa_cluster", "anio"),
    "Scoring × interés": ("score_corto", "tipo_interes"),
    "Nivel × tipo de estudiante": ("nivel", "tipo_estudiante"),
}


def _apply_preset() -> None:
    v = st.session_state.get(f"{P}_hm_preset")
    if v in _HM_PRESETS:
        st.session_state[f"{P}_hm_r"], st.session_state[f"{P}_hm_c"] = _HM_PRESETS[v]


def _clear_preset() -> None:
    st.session_state[f"{P}_hm_preset"] = None


def _profile_block(dim: str, seg_value: str, labels: pd.Series, tab: pd.DataFrame, mdef: dict) -> None:
    """Radiografía del segmento: KPI, mezcla de riesgo y rasgos que lo distinguen del resto."""
    mask = (labels == seg_value).to_numpy()
    d_seg, d_rest = dff[mask], dff[~mask]
    row = tab[tab["seg"] == seg_value]
    if row.empty or d_seg.empty:
        return
    r = row.iloc[0]
    left, right = st.columns([1, 1.45], gap="medium")
    with left:
        obs_html = (f"<div><span>Alto observado</span><b>{fmt_pct(r['pct_obs'])}{_delta_em(r['pct_obs'], TOT['pct_obs'])}</b></div>"
                    if has_truth_all else "<div><span>Alto observado</span><b>—</b></div>")
        st.markdown(
            f"<div class='sg-prof'><div class='k'>{esc(_dim_title(dim))}</div><div class='n'>{esc(seg_value)}</div>"
            f"<div class='m'>{fmt_int(r['n'])} créditos · {fmt_int(d_seg['id_estudiante'].nunique())} estudiantes · "
            f"{fmt_pct(r['n'] / TOT['n'])} de la cartera filtrada</div>"
            f"<div class='sg-grid2'>"
            f"<div><span>Alto predicho</span><b>{fmt_pct(r['pct_pred'])}{_delta_em(r['pct_pred'], TOT['pct_pred'])}</b></div>"
            f"{obs_html}"
            f"<div><span>Mora Datacrédito</span><b>{fmt_pct(r['pct_mora'])}{_delta_em(r['pct_mora'], TOT['pct_mora'])}</b></div>"
            f"<div><span>Ticket promedio</span><b>{fmt_cop(r['ticket'])}</b></div>"
            f"</div><div class='sg-cap' style='font-size:12px;margin-bottom:5px'>Mezcla de riesgo predicho</div>"
            f"{_mix_html(d_seg)}</div>", unsafe_allow_html=True)
        if dim in GLOBAL_FILTER and seg_value != "Sin dato":
            name = GLOBAL_FILTER[dim]
            b1, b2 = st.columns(2)
            b1.button("Filtrar el tablero", icon=":material/filter_alt:", key=f"{P}_prof_filter", width="stretch",
                      on_click=_set_global, args=(name, seg_value),
                      help="Aplica este segmento como filtro global en la barra lateral (todas las páginas).")
            if can_access("cola"):
                if b2.button("Ver en la cola", icon=":material/format_list_numbered:", key=f"{P}_prof_cola",
                             width="stretch", help="Abre la cola de gestión filtrada por este segmento."):
                    _set_global(name, seg_value)
                    goto("cola")
    with right:
        st.markdown("<div class='sg-cap'>¿Qué distingue a este segmento del resto de la cartera?</div>",
                    unsafe_allow_html=True)
        if d_rest.empty:
            _note("El segmento abarca toda la cartera filtrada: no hay un resto con el cual compararlo.")
            return
        feats = [c for c in ["tipo_interes", "cuotas", "cohorte", "nivel", "programa_cluster", "score_corto",
                             "tipo_estudiante", "sede", "plataforma", "nombre_linea", "cliente_limpio", "mora_txt"]
                 if c != dim and c in dff.columns]
        rows = []
        for c in feats:
            lab = _labels(dff, c)
            s_seg = lab[mask].value_counts(normalize=True)
            s_rest = lab[~mask].value_counts(normalize=True)
            diff = s_seg.sub(s_rest, fill_value=0)
            if diff.empty:
                continue
            top = diff.idxmax()
            rows.append((c, top, float(s_seg.get(top, 0)), float(s_rest.get(top, 0)), float(diff.max())))
        rows = sorted(rows, key=lambda t: -t[4])[:6]
        html_rows = "".join(
            f"<div class='sg-dist-row'><div class='nm' title='{esc(DIMS.get(c, c))}: {esc(v)}'><small>"
            f"{esc(DIMS.get(c, c))}</small>{esc(_short(v, 40))}</div>"
            f"<div class='sg-bars'><i class='a' style='width:{a * 100:.1f}%'></i><i class='b' style='width:{b * 100:.1f}%'></i></div>"
            f"<div class='pc'>{fmt_pct(a, 0)} vs {fmt_pct(b, 0)}<small>{_pp(dlt, 0)}</small></div></div>"
            for c, v, a, b, dlt in rows if dlt > 0.005)
        if html_rows:
            st.markdown(f"<div class='sg-dist'>{html_rows}</div>", unsafe_allow_html=True)
            st.markdown(_legend([(pal()["text"], "Segmento", "ln"), (pal()["subtle"], "Resto de la cartera", "ln")]),
                        unsafe_allow_html=True)
        num = [("antiguedad_meses", "Antigüedad (meses)", lambda v: fmt_num(v, 0)),
               ("valor_financiacion", "Valor financiado", fmt_cop),
               ("ratio_financiacion", "Financiación / matrícula", lambda v: fmt_pct(v, 0)),
               ("valor_primera_cuota", "Primera cuota", fmt_cop), ("edad", "Edad (años)", lambda v: fmt_num(v, 0))]
        trs = ""
        for c, lbl, f in num:
            if c not in dff.columns:
                continue
            a = pd.to_numeric(d_seg[c], errors="coerce").median()
            b = pd.to_numeric(d_rest[c], errors="coerce").median()
            rel = _div(a - b, abs(b)) if not (_isnan(a) or _isnan(b)) else NAN
            trs += (f"<tr><td>{esc(lbl)}</td><td>{f(a)}</td><td>{f(b)}</td>"
                    f"<td class='d'>{'—' if _isnan(rel) else ('+' if rel >= 0 else '−') + fmt_num(abs(rel) * 100, 0) + ' %'}</td></tr>")
        st.markdown(f"<table class='sg-med'><tr><th>Mediana</th><th>Segmento</th><th>Resto</th><th>Dif.</th></tr>"
                    f"{trs}</table>", unsafe_allow_html=True)


def _tab_explorer() -> None:
    section("Ranking de segmentos", "Elige una dimensión y una métrica. Las barras rojas superan el promedio con "
            "significancia (todo su IC 95 % por encima); las verdes están por debajo; las grises no son concluyentes.",
            kicker="Explorador")
    metric_opts = [m for m, d in METRICS.items() if has_truth_all or not d["truth"]]
    if st.session_state.get(f"{P}_metric") not in metric_opts:
        st.session_state.pop(f"{P}_metric", None)
    with st.container(border=True):
        c1, c2 = st.columns([1.25, 2.4], vertical_alignment="bottom")
        dim = c1.selectbox("Dimensión", EXPLORER_DIMS, index=0, format_func=_dim_title, key=f"{P}_dim")
        mname = c2.segmented_control("Métrica", metric_opts, default="% Alto predicho", key=f"{P}_metric",
                                     required=True) or "% Alto predicho"
        c3, c4, c5, c6, c7 = st.columns([1.05, 1.05, 1.5, 1, 1.1], vertical_alignment="bottom")
        nmin = c3.select_slider("n mínimo por segmento", [1, 10, 20, 30, 50, 100, 200, 500], value=30, key=f"{P}_nmin")
        topn = c4.select_slider("Mostrar", [10, 15, 20, 30, 50, "Todos"], value=15, key=f"{P}_topn",
                                format_func=lambda v: f"Top {v}" if v != "Todos" else "Todos")
        order = c5.segmented_control("Orden", ["Métrica", "Tamaño", "Natural"], key=f"{P}_order_{dim}",
                                     default="Natural" if dim in ORDINAL else "Métrica", required=True) or "Métrica"
        ci = c6.toggle("IC 95 %", value=True, key=f"{P}_ci", help="Intervalo de Wilson para proporciones; normal "
                       "(media ± 1,96·EE) para el ticket promedio.")
        bonf = c7.toggle("Bonferroni", value=False, key=f"{P}_bonf",
                         help="Ajusta el nivel de confianza por el número de segmentos comparados (α = 5 % familiar). "
                              "Útil con dimensiones de muchas categorías, como Programa.")
        mdef = METRICS[mname]
        labels = _labels(dff, dim)
        tab = _agg(X.assign(seg=labels.to_numpy()), "seg", students=True)
        n_all = len(tab)
        tab = tab[tab["n"] >= nmin].copy()
        if tab.empty:
            empty_state("Ningún segmento cumple el n mínimo", "Reduce el n mínimo o amplía los filtros.", icon="📏")
            return
        z = _z_bonferroni(len(tab)) if bonf else Z95
        if bonf:
            tab = _intervals(tab, z)
        kind = mdef["kind"]
        ref = TOT["ticket"] if kind == "mean" else TOT.get(mdef["val"], NAN)
        tab["val"] = tab[mdef["val"]]
        if kind in ("prop", "mean"):
            tab["lo"], tab["hi"] = tab[mdef["lo"]], tab[mdef["hi"]]
            tab["sig"] = np.where(tab["lo"] > ref, "sobre", np.where(tab["hi"] < ref, "bajo", "nc"))
            tab["lift"] = tab["val"] / ref if ref else np.nan
        else:
            tab["lo"], tab["hi"] = tab["val"], tab["val"]
            tab["sig"] = "nc"
            tab["lift"] = (tab["monto_alto"] / tab["monto"]) / _div(TOT["monto_alto"], TOT["monto"])
        by_metric = tab.sort_values(["val", "n"], ascending=False)
        chosen = by_metric if order != "Tamaño" else tab.sort_values("n", ascending=False)
        if topn != "Todos":
            chosen = chosen.head(int(topn))
        if order == "Natural":
            chosen = chosen.assign(_k=chosen["seg"].map(_nat_key(dim))).sort_values("_k").drop(columns="_k")
        elif order == "Tamaño":
            chosen = chosen.sort_values("n", ascending=False)
        h = int(np.clip(34 * len(chosen) + 86, 300, 980))
        sc = _sig_colors()
        leg = ([(sc["sobre"], "Sobre el promedio (significativo)", "sq"), (sc["nc"], "No concluyente", "sq"),
                (sc["bajo"], "Bajo el promedio (significativo)", "sq")] if kind == "prop" else
               [(pal()["text"], "Sobre el promedio (significativo)", "sq"), (sc["nc"], "No concluyente o por debajo", "sq")]
               if kind == "mean" else [(RISK_COLORS["Alto"], "Monto financiado de créditos en Alto predicho", "sq")])
        conf = f"{(1 - 0.05 / len(tab)) * 100:.2f}".replace(".", ",") if bonf else "95"
        st.markdown(_legend(leg + ([(pal()["muted"], f"IC {conf} %", "ln")] if ci and kind != "money" else [])),
                    unsafe_allow_html=True)
        ev = show_fig(_rank_fig(chosen, mdef, mname, ref, ci), key=f"{P}_rank", height=h, on_select="rerun",
                      selection_mode=("points",))
        hidden = n_all - len(tab)
        n_sig_up = int((tab["sig"] == "sobre").sum())
        extra = (f" Se muestran {len(chosen)} de {len(tab)} segmentos con n ≥ {nmin}"
                 + (f" ({hidden} con menos créditos quedan fuera)." if hidden else "."))
        if kind == "prop":
            _note(f"<b>{n_sig_up}</b> de {len(tab)} segmentos superan el promedio de {fmt_pct(ref)} con significancia"
                  f"{' tras Bonferroni' if bonf else ''}. Lift = tasa del segmento ÷ promedio de la cartera filtrada."
                  f"{extra} Haz clic en una barra para ver su radiografía.")
        elif kind == "mean":
            _note(f"Ticket promedio de la cartera: <b>{fmt_cop(ref)}</b>. Intervalo normal para la media.{extra} "
                  "Haz clic en una barra para ver su radiografía.")
        else:
            _note(f"Monto en Alto predicho de la cartera filtrada: <b>{fmt_cop(TOT['monto_alto'])}</b>. El porcentaje "
                  f"es la participación del segmento en ese monto; el lift compara su proporción de monto en Alto con la "
                  f"de la cartera.{extra}")

    # --- tabla + descarga
    out = tab.sort_values(["val", "n"], ascending=False)
    table = pd.DataFrame({
        DIMS[dim]: out["seg"], "Créditos": out["n"].astype(int), "Estudiantes": out["est"].astype(int),
        "% Alto predicho": out["pct_pred"], "IC inf. (pred.)": out["lo_pred"], "IC sup. (pred.)": out["hi_pred"],
        "% Alto observado": out["pct_obs"], "Lift (métrica)": out["lift"], "% mora": out["pct_mora"],
        "Monto financiado": out["monto"], "Monto en Alto": out["monto_alto"], "Ticket promedio": out["ticket"],
        "Significancia": out["sig"].map({"sobre": "Sobre el promedio", "bajo": "Bajo el promedio",
                                         "nc": "No concluyente"}),
    })
    if not has_truth_all:
        table = table.drop(columns=["% Alto observado"])
    with st.expander(f"Tabla completa · {len(table)} segmentos ({DIMS[dim].lower()})", expanded=False):
        vmax = float(np.nanmax([table["% Alto predicho"].max(), 0.01]))
        st.dataframe(table, hide_index=True, width="stretch", height=min(38 * len(table) + 40, 420), column_config={
            "% Alto predicho": st.column_config.ProgressColumn("% Alto predicho", format="percent", min_value=0,
                                                               max_value=vmax),
            "IC inf. (pred.)": st.column_config.NumberColumn(format="percent"),
            "IC sup. (pred.)": st.column_config.NumberColumn(format="percent"),
            "% Alto observado": st.column_config.NumberColumn(format="percent"),
            "Lift (métrica)": st.column_config.NumberColumn(format="%.2f×"),
            "% mora": st.column_config.NumberColumn(format="percent"),
            "Monto financiado": st.column_config.NumberColumn(format="accounting"),
            "Monto en Alto": st.column_config.NumberColumn(format="accounting"),
            "Ticket promedio": st.column_config.NumberColumn(format="accounting"),
        })
        defs = pd.DataFrame({"campo": ["IC inf./sup.", "Lift", "Significancia", "Métrica del ranking", "Filtros"],
                             "definicion": [f"Intervalo de Wilson al {conf} % del % Alto predicho",
                                            f"{mname} del segmento ÷ promedio de la cartera filtrada",
                                            "Sobre/Bajo: el IC completo queda por encima/por debajo del promedio",
                                            mname, "; ".join(f"{k}: {v}" for k, v in active_chips(df_all)) or "Sin filtros"]})
        download_bar(table, f"segmentos_{dim}", key=f"{P}_dl_rank", extra_sheets={"definiciones": defs})

    # --- radiografía
    section("Radiografía del segmento", "Perfil del segmento elegido frente al resto de la cartera filtrada: qué "
            "categorías están sobrerrepresentadas y cómo cambian las medianas financieras.", kicker="Perfil")
    options = by_metric["seg"].tolist()
    skey = f"{P}_prof_{dim}"
    try:
        pts = (ev.selection.points if ev is not None else []) or []
    except AttributeError:
        pts = (ev or {}).get("selection", {}).get("points", []) if isinstance(ev, dict) else []
    if pts:
        pt = pts[0]
        idx = pt.get("point_index", pt.get("pointIndex")) if isinstance(pt, dict) else None
        rev = chosen.iloc[::-1].reset_index(drop=True)
        if idx is not None and 0 <= int(idx) < len(rev):
            clicked = rev.loc[int(idx), "seg"]
            if st.session_state.get(f"{skey}_last") != clicked and clicked in options:
                st.session_state[skey] = clicked
                st.session_state[f"{skey}_last"] = clicked
    if st.session_state.get(skey) not in options:
        st.session_state.pop(skey, None)
    with st.container(border=True):
        seg_value = st.selectbox("Segmento", options, key=skey, help="También puedes hacer clic en una barra del ranking.")
        _profile_block(dim, seg_value, labels, tab, mdef)

    # --- heatmap
    section("Cruce de dos dimensiones", "Mapa de calor del riesgo en la intersección de dos dimensiones. El color "
            "central de la escala corresponde al promedio de la cartera; las celdas con pocos créditos se ocultan.",
            kicker="Mapa de calor")
    hm_dims = [c for c in EXPLORER_DIMS if c in dff.columns]
    st.session_state.setdefault(f"{P}_hm_r", "tipo_interes")
    st.session_state.setdefault(f"{P}_hm_c", "cuotas")
    hm_metric_opts = [m for m in ("% Alto predicho", "% Alto observado", "% mora") if has_truth_all or m != "% Alto observado"]
    if st.session_state.get(f"{P}_hm_metric") not in hm_metric_opts:
        st.session_state.pop(f"{P}_hm_metric", None)
    with st.container(border=True):
        st.pills("Cruces sugeridos", list(_HM_PRESETS), key=f"{P}_hm_preset", on_change=_apply_preset,
                 selection_mode="single")
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.9, 1.1], vertical_alignment="bottom")
        rdim = c1.selectbox("Filas", hm_dims, format_func=_dim_title, key=f"{P}_hm_r", on_change=_clear_preset)
        cdim = c2.selectbox("Columnas", hm_dims, format_func=_dim_title, key=f"{P}_hm_c", on_change=_clear_preset)
        hm_m = c3.segmented_control("Métrica del mapa", hm_metric_opts, default="% Alto predicho",
                                    key=f"{P}_hm_metric", required=True) or "% Alto predicho"
        hm_min = c4.select_slider("n mínimo por celda", [5, 10, 20, 30, 50, 100], value=20, key=f"{P}_hm_min")
        if rdim == cdim:
            st.info("Elige dos dimensiones distintas para el cruce.", icon=":material/info:")
            return
        md = METRICS[hm_m]
        rl = _fold_top(_labels(dff, rdim), 18, "Otros (resto)")
        cl = _fold_top(_labels(dff, cdim), 12, "Otros (resto)")
        g = _agg(X.assign(r=rl.to_numpy(), c=cl.to_numpy()), ["r", "c"])
        rows_t = _agg(X.assign(r=rl.to_numpy()), "r")
        cols_t = _agg(X.assign(c=cl.to_numpy()), "c")
        avg = TOT.get(md["val"], NAN)
        rkey, ckey = _nat_key(rdim), _nat_key(cdim)
        rows = (sorted(rows_t["r"], key=rkey) if rdim in ORDINAL
                else rows_t.sort_values(md["val"], ascending=False)["r"].tolist())
        cols = (sorted(cols_t["c"], key=ckey) if cdim in ORDINAL
                else cols_t.sort_values("n", ascending=False)["c"].tolist())
        rows = [r for r in rows if r != "Otros (resto)"] + (["Otros (resto)"] if "Otros (resto)" in rows else [])
        cols = [c for c in cols if c != "Otros (resto)"] + (["Otros (resto)"] if "Otros (resto)" in cols else [])
        cols_all = cols + ["Total"]
        gi = g.set_index(["r", "c"])
        ri, ci_ = rows_t.set_index("r"), cols_t.set_index("c")
        z = np.full((len(rows) + 1, len(cols_all)), np.nan)
        text = np.full(z.shape, "", dtype=object)
        cd = np.empty(z.shape + (6,), dtype=object)
        cd[:] = "—"
        n_hidden = 0
        row_names = rows + ["Total"]
        for i, rname in enumerate(row_names):
            for j, cname in enumerate(cols_all):
                if rname == "Total" and cname == "Total":
                    rec = TOT
                elif rname == "Total":
                    rec = ci_.loc[cname] if cname in ci_.index else None
                elif cname == "Total":
                    rec = ri.loc[rname] if rname in ri.index else None
                else:
                    rec = gi.loc[(rname, cname)] if (rname, cname) in gi.index else None
                if rec is None:
                    continue
                n_c = int(rec["n"])
                v = float(rec[md["val"]])
                cd[i, j] = [esc(rname), esc(cname), fmt_pct(v), f"{fmt_pct(rec[md['lo']])} – {fmt_pct(rec[md['hi']])}",
                            fmt_int(n_c), _x(_div(v, avg), 2)]
                if n_c < hm_min:
                    n_hidden += 1
                    text[i, j] = "·"
                    continue
                z[i, j] = v
                text[i, j] = f"<b>{fmt_pct(v)}</b>" if (i == len(rows) or j == len(cols)) else fmt_pct(v)
        finite = z[np.isfinite(z)]
        zmax = float(np.nanpercentile(finite, 95)) if finite.size else avg
        fig = _heatmap_fig(z, text, cd, row_names, cols_all, avg, zmax, DIMS[rdim], DIMS[cdim], hm_m)
        hh = int(np.clip(34 * (len(row_names)) + 90, 260, 760))
        show_fig(fig, key=f"{P}_heatmap", height=hh, legend=False)
        _note(f"Fila y columna <b>Total</b> = tasas marginales. Promedio de la cartera: <b>{fmt_pct(avg)}</b> "
              f"(color ámbar central). {n_hidden} celdas con n < {hm_min} ocultas (·). "
              + ("Las dimensiones con muchas categorías agrupan la cola en «Otros (resto)». "
                 if (rl == "Otros (resto)").any() or (cl == "Otros (resto)").any() else "")
              + "Pasa el cursor para ver el n, el IC 95 % y el lift de cada celda.")
        mat = g.rename(columns={"r": DIMS[rdim], "c": DIMS[cdim]})[[DIMS[rdim], DIMS[cdim], "n", "pct_pred", "pct_obs",
                                                                      "pct_mora", "monto"]]
        mat = mat.rename(columns={"n": "Créditos", "pct_pred": "% Alto predicho", "pct_obs": "% Alto observado",
                                  "pct_mora": "% mora", "monto": "Monto financiado"})
        download_bar(mat, f"cruce_{rdim}_{cdim}", key=f"{P}_dl_hm", label="Cruce")


# ======================================================================================
# Pestaña 2 · Académico
# ======================================================================================
def _treemap_fig(basis: str, gmin: int) -> go.Figure:
    p = pal()
    fac = dff["facultad"].astype(str).str.strip().replace({"nan": "Sin dato"})
    prog = dff["programa"].astype(str).str.strip().replace({"nan": "Sin dato"})
    sz = prog.groupby([fac, prog]).transform("size")
    prog = prog.where(sz >= gmin, f"Otros programas (< {gmin})")
    xx = X.assign(f=fac.to_numpy(), pr=prog.to_numpy())
    tp = _agg(xx, ["f", "pr"])
    tf = _agg(xx, "f")
    vcol = f"pct_{basis}"
    avg = TOT.get(vcol, NAN)
    cap = max(avg * 2.4 if not _isnan(avg) else 0.2, 0.02)
    ids, labels, parents, values, colors, cd = [], [], [], [], [], []

    def add(i, lab, par, rec):
        ids.append(i)
        labels.append(lab)
        parents.append(par)
        values.append(int(rec["n"]))
        v = float(rec[vcol]) if not _isnan(rec[vcol]) else 0.0
        colors.append(min(v, cap))
        cd.append([fmt_int(rec["n"]), fmt_pct(rec["n"] / TOT["n"]), fmt_pct(rec["pct_pred"]), fmt_pct(rec["pct_obs"]),
                   _x(_div(rec[vcol], avg), 2), fmt_cop(rec["monto"]), fmt_pct(rec[vcol]),
                   f"{fmt_pct(rec[f'lo_{basis}'])} – {fmt_pct(rec[f'hi_{basis}'])}"])

    add("root", "Cartera filtrada", "", TOT)
    for _, r in tf.iterrows():
        add(f"F|{r['f']}", str(r["f"]), "root", r)
    for _, r in tp.iterrows():
        add(f"P|{r['f']}|{r['pr']}", str(r["pr"]), f"F|{r['f']}", r)
    m = avg / cap if cap and not _isnan(avg) else 0.5
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total", customdata=cd,
        marker=dict(colors=colors, colorscale=_seq_scale(m), cmin=0, cmax=cap, line=dict(color=p["bg"], width=2),
                    colorbar=dict(title=dict(text=f"% Alto {'observado' if basis == 'obs' else 'predicho'}",
                                             side="right", font=dict(size=11, color=p["muted"])),
                                  tickvals=[0, avg, cap] if not _isnan(avg) else None,
                                  ticktext=["0 %", f"Prom. {fmt_pct(avg)}", f"≥ {fmt_pct(cap, 0)}"] if not _isnan(avg) else None,
                                  thickness=10, len=0.85, outlinewidth=0, tickfont=dict(color=p["muted"], size=10))),
        texttemplate="<b>%{label}</b><br>%{customdata[6]}", textfont=dict(size=12.5), tiling=dict(pad=2),
        pathbar=dict(visible=True, thickness=22, textfont=dict(size=12)), maxdepth=3,
        hovertemplate=("<b>%{label}</b><br>Créditos: %{customdata[0]} (%{customdata[1]} de la cartera)"
                       "<br>% Alto predicho: %{customdata[2]} · observado: %{customdata[3]}"
                       "<br>IC 95 %: %{customdata[7]} · lift %{customdata[4]}<br>Monto financiado: %{customdata[5]}"
                       "<extra></extra>"),
    ))
    fig.update_layout(margin=dict(l=4, r=4, t=30, b=4), uniformtext=dict(minsize=10, mode="hide"))
    return fig


def _trend_fig(basis: str, gran: str) -> tuple[go.Figure, pd.DataFrame]:
    p = pal()
    colors = CLUSTER_COLORS[theme_mode()]
    per = dff[gran].astype(str)
    cl = dff["programa_cluster"].astype(str)
    g = _agg(X.assign(c=cl.to_numpy(), t=per.to_numpy()), ["c", "t"])
    gt = _agg(X.assign(t=per.to_numpy()), "t").sort_values("t")
    gt = gt[gt["t"].str.lower() != "nan"]
    periods = gt["t"].tolist()
    vcol = f"pct_{basis}"
    fig = go.Figure()
    order = [c for c in CLUSTER_COLORS["light"] if c in set(cl)]
    for c in order:
        s = g[g["c"] == c].set_index("t").reindex(periods)
        yv = np.where(s["n"].fillna(0) >= 30, s[vcol], np.nan)
        cd = np.c_[[fmt_pct(v) for v in s[vcol]], [fmt_int(v) if not _isnan(v) else "0" for v in s["n"]]]
        fig.add_trace(go.Scatter(x=periods, y=yv, name=c, mode="lines+markers", line=dict(color=colors[c], width=2),
                                 marker=dict(size=7, color=colors[c], line=dict(color=p["bg"], width=1.5)),
                                 customdata=cd, connectgaps=False,
                                 hovertemplate=f"{esc(c)}: <b>%{{customdata[0]}}</b> (n = %{{customdata[1]}})<extra></extra>"))
    cdt = np.c_[[fmt_pct(v) for v in gt[vcol]], [fmt_int(v) for v in gt["n"]]]
    fig.add_trace(go.Scatter(x=periods, y=gt[vcol], name="Cartera", mode="lines", line=dict(color=p["text"], width=3.2),
                             customdata=cdt, opacity=0.85,
                             hovertemplate="Cartera: <b>%{customdata[0]}</b> (n = %{customdata[1]})<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", title_text=f"% Alto {'observado' if basis == 'obs' else 'predicho'}", rangemode="tozero")
    fig.update_xaxes(type="category", title_text="Semestre de aprobación" if gran == "semestre" else "Año de aprobación")
    fig.update_layout(hovermode="x unified", margin=dict(l=8, r=8, t=40, b=8))
    return fig, g


def _tab_academico() -> None:
    section("Mapa de facultades y programas", "Tamaño = número de créditos; color = % de riesgo Alto con el tono "
            "central anclado en el promedio de la cartera (más rojo = más riesgo que el promedio). Haz clic en una "
            "facultad para acercarte.", kicker="Treemap")
    bopts = ["Alto predicho", "Alto observado"] if has_truth_all else ["Alto predicho"]
    if st.session_state.get(f"{P}_tm_basis") not in bopts:
        st.session_state.pop(f"{P}_tm_basis", None)
    with st.container(border=True):
        c1, c2, _ = st.columns([1.4, 1.4, 2], vertical_alignment="bottom")
        b = c1.segmented_control("Color según", bopts, default="Alto predicho", key=f"{P}_tm_basis", required=True)
        basis = "obs" if b == "Alto observado" else "pred"
        gmin = c2.select_slider("Agrupar programas con menos de", [1, 10, 20, 30, 50, 100], value=20,
                                key=f"{P}_tm_min", format_func=lambda v: f"{v} créditos")
        show_fig(_treemap_fig(basis, gmin), key=f"{P}_treemap", height=540, legend=False)
        _note("Los programas pequeños se agrupan en «Otros programas» dentro de su facultad para no sobreinterpretar "
              "tasas con pocos créditos. El color se satura en 2,4× el promedio; el hover muestra la tasa exacta y su IC 95 %.")

    section("Evolución por segmento de programa", "% de riesgo Alto por periodo de aprobación para cada segmento. "
            "Los puntos con menos de 30 créditos se omiten.", kicker="Tendencia")
    with st.container(border=True):
        c1, c2, _ = st.columns([1.4, 1.4, 2], vertical_alignment="bottom")
        b2 = c1.segmented_control("Riesgo", bopts, default="Alto predicho", key=f"{P}_tr_basis", required=True)
        gran = c2.segmented_control("Periodo", ["Semestre", "Año"], default="Semestre", key=f"{P}_tr_gran", required=True)
        fig, _g = _trend_fig("obs" if b2 == "Alto observado" else "pred", "anio" if gran == "Año" else "semestre")
        show_fig(fig, key=f"{P}_trend", height=400)
        _note("La línea gruesa es la cartera filtrada. Una línea que se separa hacia arriba es un segmento cuyo riesgo "
              "crece más rápido que el de la cartera: candidato a una campaña preventiva focalizada.")

    section("Programas con mayor riesgo", "Ranking de programas con al menos el n mínimo del treemap; la tendencia muestra "
            "el % Alto predicho por semestre (semestres con menos de 5 créditos quedan vacíos).", kicker="Tabla")
    progs = _agg(X.assign(pr=dff["programa"].astype(str).to_numpy()), "pr", students=True)
    progs = progs[progs["n"] >= max(gmin, 1)]
    if progs.empty:
        _note("No hay programas con suficientes créditos en el filtro actual.")
        return
    avg = TOT["pct_pred"]
    progs = progs.sort_values(["pct_pred", "n"], ascending=False)
    meta = dff.groupby(dff["programa"].astype(str)).agg(fac=("facultad", "first"), seg=("programa_cluster", "first"),
                                                         nivel=("nivel", "first"))
    sem = sorted(s for s in dff["semestre"].dropna().astype(str).unique() if s.lower() != "nan")
    pv = _agg(X.assign(pr=dff["programa"].astype(str).to_numpy(), s=dff["semestre"].astype(str).to_numpy()), ["pr", "s"])
    pv = pv[pv["n"] >= 5].pivot(index="pr", columns="s", values="pct_pred").reindex(columns=sem)
    trend = {k: [None if _isnan(v) else round(float(v), 4) for v in row] for k, row in zip(pv.index, pv.to_numpy())}
    tbl = pd.DataFrame({
        "Programa": progs["pr"], "Facultad": progs["pr"].map(meta["fac"]), "Segmento": progs["pr"].map(meta["seg"]),
        "Nivel": progs["pr"].map(meta["nivel"]), "Créditos": progs["n"].astype(int),
        "% Alto predicho": progs["pct_pred"], "IC 95 %": [f"{fmt_pct(a)} – {fmt_pct(b)}" for a, b in
                                                          zip(progs["lo_pred"], progs["hi_pred"])],
        "Lift": progs["pct_pred"] / avg if avg else np.nan, "% Alto observado": progs["pct_obs"],
        "% mora": progs["pct_mora"], "Monto financiado": progs["monto"],
        "Tendencia (% Alto pred.)": progs["pr"].map(lambda k: trend.get(k, [])),
    })
    if not has_truth_all:
        tbl = tbl.drop(columns=["% Alto observado"])
    with st.container(border=True):
        st.dataframe(tbl, hide_index=True, width="stretch", height=420, column_config={
            "Programa": st.column_config.TextColumn(width="medium"),
            "Facultad": st.column_config.TextColumn(width="small"),
            "% Alto predicho": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                               max_value=float(max(progs["pct_pred"].max(), 0.01))),
            "Lift": st.column_config.NumberColumn(format="%.2f×"),
            "% Alto observado": st.column_config.NumberColumn(format="percent"),
            "% mora": st.column_config.NumberColumn(format="percent"),
            "Monto financiado": st.column_config.NumberColumn(format="accounting"),
            "Tendencia (% Alto pred.)": st.column_config.LineChartColumn(y_min=0, width="medium"),
        })
        download_bar(tbl.drop(columns=["Tendencia (% Alto pred.)"]), "programas_riesgo", key=f"{P}_dl_prog",
                     label="Programas")


# ======================================================================================
# Pestaña 3 · Financiero
# ======================================================================================
_FIN_VARS = {"valor_financiacion": ("Valor financiado", "money"), "vr_neto_matricula": ("Matrícula neta", "money"),
             "valor_primera_cuota": ("Primera cuota", "money"), "ratio_financiacion": ("Financiación / matrícula", "ratio"),
             "valor_cuota_inicial": ("Cuota inicial", "money")}


def _fmt_var(v, kind: str) -> str:
    return fmt_pct(v, 0) if kind == "ratio" else fmt_cop(v)


def _violin_fig(var: str, kind: str, cls: pd.Series, trim: bool) -> tuple[go.Figure, dict]:
    p = pal()
    v = pd.to_numeric(dff[var], errors="coerce")
    lo_q, hi_q = (v.quantile(0.01), v.quantile(0.99)) if trim else (v.min(), v.max())
    fig = go.Figure()
    stats = {}
    for r in RISK_ORDER:
        s = v[(cls.astype(str) == r).to_numpy()].dropna()
        stats[r] = {"n": len(s), "med": s.median() if len(s) else NAN, "q1": s.quantile(0.25) if len(s) else NAN,
                    "q3": s.quantile(0.75) if len(s) else NAN, "arr": s.to_numpy(dtype=float)}
        if len(s) < 3:
            continue
        sd = s.clip(lo_q, hi_q) if trim else s
        fig.add_trace(go.Violin(
            x=[f"{r}"] * len(sd), y=sd, name=r, line=dict(color=RISK_COLORS[r], width=1.6),
            fillcolor=RISK_COLORS[r], opacity=0.55, box=dict(visible=True, width=0.18, fillcolor=p["surface"],
                                                              line=dict(color=p["text"], width=1.2)),
            meanline=dict(visible=False), points=False, spanmode="hard", scalemode="width", width=0.8,
            hoveron="violins", hoverinfo="skip"))
        fig.add_annotation(x=r, y=stats[r]["med"], text=f"<b>{_fmt_var(stats[r]['med'], kind)}</b>", showarrow=False,
                           xshift=46, font=dict(size=11.5, color=p["text"]), xanchor="left",
                           bgcolor=p["surface"], bordercolor=p["border"], borderwidth=1, borderpad=3)
        fig.add_trace(go.Scatter(
            x=[r], y=[stats[r]["med"]], mode="markers", marker=dict(size=18, opacity=0), showlegend=False,
            hovertemplate=(f"<b>{r}</b> · n = {fmt_int(len(s))}<br>Mediana: {_fmt_var(stats[r]['med'], kind)}"
                           f"<br>P25–P75: {_fmt_var(stats[r]['q1'], kind)} – {_fmt_var(stats[r]['q3'], kind)}<extra></extra>")))
    if kind == "ratio":
        fig.add_hline(y=MAX_FINANCIACION_PCT, line=dict(color=p["muted"], width=1.1, dash="dot"))
        fig.add_annotation(x=1, xref="paper", y=MAX_FINANCIACION_PCT, text="Tope DA-02: 85 %", showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(size=11, color=p["muted"]))
    fig.update_yaxes(tickformat=".0%" if kind == "ratio" else "~s", tickprefix="" if kind == "ratio" else "$ ",
                     title_text=_FIN_VARS[var][0], hoverformat=",.0f")
    fig.update_xaxes(title_text="Clase de riesgo", tickfont=dict(size=12.5, color=p["text"]))
    fig.update_layout(violinmode="overlay", violingap=0.25, margin=dict(l=8, r=8, t=24, b=8), showlegend=False)
    return fig, stats


def _antig_fig(cls: pd.Series) -> go.Figure:
    p = pal()
    a = pd.to_numeric(dff["antiguedad_meses"], errors="coerce")
    mx = float(np.nanmax(a)) if a.notna().any() else 12
    bins = np.arange(0, mx + 4, 3)
    fig = go.Figure()
    for r in RISK_ORDER:
        s = a[(cls.astype(str) == r).to_numpy()].dropna()
        if len(s) < 5:
            continue
        h, e = np.histogram(s, bins=bins)
        share = h / h.sum()
        mids = (e[:-1] + e[1:]) / 2
        cd = np.c_[[f"{int(e0)}–{int(e1) - 1}" for e0, e1 in zip(e[:-1], e[1:])], [fmt_pct(v) for v in share],
                   [fmt_int(v) for v in h]]
        fig.add_trace(go.Scatter(x=mids, y=share, name=f"{r} (n = {fmt_int(len(s))})", mode="lines",
                                 line=dict(color=RISK_COLORS[r], width=2.2 if r != "Medio" else 1.6, shape="spline",
                                           smoothing=0.6),
                                 fill="tozeroy", fillcolor=_rgba(RISK_COLORS[r], 0.10), customdata=cd,
                                 hovertemplate=(f"<b>{r}</b> · %{{customdata[0]}} meses<br>%{{customdata[1]}} de la clase "
                                                "(%{customdata[2]} créditos)<extra></extra>")))
        med = float(s.median())
        fig.add_vline(x=med, line=dict(color=RISK_COLORS[r], width=1.3, dash="dot"))
        fig.add_annotation(x=med, y=1, yref="paper", text=f"Med. {r}: {fmt_num(med, 0)} m", showarrow=False,
                           xanchor="left", yanchor="top", xshift=3, yshift=-({"Alto": 0, "Medio": 16, "Bajo": 32}[r]),
                           font=dict(size=10.5, color=RISK_COLORS[r]))
    fig.update_xaxes(title_text="Antigüedad del crédito (meses)", rangemode="tozero")
    fig.update_yaxes(tickformat=".0%", title_text="% de la clase", rangemode="tozero")
    fig.update_layout(margin=dict(l=8, r=8, t=40, b=8), hovermode="closest")
    return fig


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _tab_financiero() -> None:
    section("Distribución financiera por clase de riesgo", "Violín = forma de la distribución; caja = mediana y "
            "rango intercuartílico. El δ de Cliff mide qué tan separadas están las clases Alto y Bajo.",
            kicker="Financiero")
    copts = ["Predicha", "Observada"] if has_truth_all else ["Predicha"]
    if st.session_state.get(f"{P}_fin_cls") not in copts:
        st.session_state.pop(f"{P}_fin_cls", None)
    avail = [v for v in _FIN_VARS if v in dff.columns and pd.to_numeric(dff[v], errors="coerce").notna().sum() >= 10]
    if not avail:
        empty_state("Sin variables financieras", "El dataset activo no trae montos suficientes para este análisis.", icon="💳")
        return
    if st.session_state.get(f"{P}_fin_var") not in avail:
        st.session_state.pop(f"{P}_fin_var", None)
    with st.container(border=True):
        c1, c2, c3 = st.columns([3.2, 1.3, 1.2], vertical_alignment="bottom")
        var = c1.segmented_control("Variable", avail, format_func=lambda v: _FIN_VARS[v][0], default=avail[0],
                                   key=f"{P}_fin_var", required=True) or avail[0]
        cb = c2.segmented_control("Clase", copts, default="Predicha", key=f"{P}_fin_cls", required=True)
        trim = c3.toggle("Recortar colas (p1–p99)", value=True, key=f"{P}_fin_trim")
        cls = dff["y_true"] if cb == "Observada" else dff["y_pred"]
        kind = _FIN_VARS[var][1]
        l, r = st.columns([2.1, 1], gap="medium")
        with l:
            fig, stats = _violin_fig(var, kind, cls, trim)
            show_fig(fig, key=f"{P}_violin", height=420, legend=False)
        with r:
            da, db = stats["Alto"], stats["Bajo"]
            delta = _cliffs_delta(da["arr"], db["arr"])
            rel = _div(da["med"] - db["med"], abs(db["med"])) if not (_isnan(da["med"]) or _isnan(db["med"])) else NAN
            tiles = [
                _tile("Mediana Alto", _fmt_var(da["med"], kind),
                      f"P25–P75: {_fmt_var(da['q1'], kind)} – {_fmt_var(da['q3'], kind)} · n = {fmt_int(da['n'])}", "alto"),
                _tile("Mediana Bajo", _fmt_var(db["med"], kind),
                      f"P25–P75: {_fmt_var(db['q1'], kind)} – {_fmt_var(db['q3'], kind)} · n = {fmt_int(db['n'])}", "bajo"),
                _tile("Diferencia de medianas", "—" if _isnan(rel) else f"{'+' if rel >= 0 else '−'}{fmt_num(abs(rel) * 100, 1)} %",
                      "Alto frente a Bajo", "accent"),
                _tile("δ de Cliff (Alto vs. Bajo)", fmt_num(delta, 2),
                      f"Efecto <b>{_cliff_txt(delta)}</b>. δ &gt; 0: los Alto tienden a valores mayores.", "medio"),
            ]
            st.markdown("<div class='sg-stack'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)
        _note(f"Clase {cb.lower()}. " + ("Colas recortadas al percentil 1–99 solo para la visualización; las medianas y "
                                         "el δ usan todos los datos. " if trim else "")
              + "δ de Cliff: |δ| &lt; 0,15 insignificante · &lt; 0,33 pequeño · &lt; 0,47 mediano · mayor, grande.")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        section("Antigüedad por clase", "Distribución de la antigüedad del crédito (bins de 3 meses) como % de cada clase.",
                kicker="Temporal")
        with st.container(border=True):
            show_fig(_antig_fig(cls), key=f"{P}_antig", height=380)
            ma = stats_med(cls, "Alto")
            mb = stats_med(cls, "Bajo")
            _note(f"Según el reporte, los créditos de riesgo Alto se agrupan en periodos iniciales. Aquí la mediana es "
                  f"<b>{fmt_num(ma, 0)} meses</b> en Alto vs. <b>{fmt_num(mb, 0)}</b> en Bajo (clase {cb.lower()}).")
    with c2:
        section("Riesgo por deciles", f"% Alto en cada decil de {_FIN_VARS[var][0].lower()} (D1 = 10 % más bajo).",
                kicker="Deciles")
        with st.container(border=True):
            v = pd.to_numeric(dff[var], errors="coerce")
            ok = v.notna().to_numpy()
            if ok.sum() < 30:
                _note("No hay suficientes valores para formar deciles.")
            else:
                dec = pd.qcut(v[ok].rank(method="first"), 10, labels=False).astype(int)
                xd = X[ok].assign(dec=dec.to_numpy())
                t = _agg(xd, "dec").sort_values("dec")
                rng = v[ok].groupby(dec.to_numpy()).agg(["min", "max"])
                t["hover_lbl"] = [f"D{int(k) + 1}: {_fmt_var(rng.loc[k, 'min'], kind)} – {_fmt_var(rng.loc[k, 'max'], kind)}"
                                  for k in t["dec"]]
                xl = [f"D{int(k) + 1}" for k in t["dec"]]
                fig = _pred_obs_bars(xl, t, TOT["pct_pred"], TOT["pct_obs"], has_truth_all)
                fig.update_xaxes(ticktext=[f"D{int(k) + 1}<br><span style='font-size:10px'>≤ {_fmt_var(rng.loc[k, 'max'], kind)}</span>"
                                           for k in t["dec"]], tickvals=xl)
                show_fig(fig, key=f"{P}_deciles", height=380)
                top = t.loc[t["pct_obs" if has_truth_all else "pct_pred"].idxmax()]
                _note(f"El decil con más riesgo es <b>{esc(top['hover_lbl'])}</b> "
                      f"({fmt_pct(top['pct_obs' if has_truth_all else 'pct_pred'])} de Alto "
                      f"{BASIS_TXT}). El reporte asocia financiaciones altas a mayor vulnerabilidad; los extremos de la "
                      "distribución son los que merecen revisión.")

    section("Regla DA-02 · financiación sobre el 85 % de la matrícula", "Créditos cuyo valor financiado supera el tope "
            "institucional frente al resto, con la diferencia en riesgo.", kicker="Política de crédito")
    ratio = pd.to_numeric(dff["ratio_financiacion"], errors="coerce")
    over = (ratio > MAX_FINANCIACION_PCT).to_numpy()
    if ratio.notna().sum() == 0:
        _note("El dataset activo no permite calcular la razón financiación / matrícula.")
        return
    xo = X.assign(g=np.where(over, "Sobre el tope", "Dentro del tope"))
    tg = _agg(xo, "g").set_index("g")
    a = tg.loc["Sobre el tope"] if "Sobre el tope" in tg.index else None
    b = tg.loc["Dentro del tope"] if "Dentro del tope" in tg.index else None
    vb = "pct_obs" if has_truth_all else "pct_pred"
    tiles = [_tile("Créditos sobre el 85 %", fmt_int(a["n"]) if a is not None else "0",
                   f"{fmt_pct(_div(a['n'] if a is not None else 0, TOT['n']))} de la cartera filtrada · "
                   f"{fmt_cop(a['monto']) if a is not None else '$ 0'} financiados", "accent")]
    if a is not None and b is not None:
        rr = _div(a[vb], b[vb])
        tiles.append(_tile(f"Alto {BASIS_TXT} · sobre el tope", fmt_pct(a[vb]),
                           f"IC 95 %: {fmt_pct(a['lo_' + BASIS])} – {fmt_pct(a['hi_' + BASIS])}", "alto"))
        tiles.append(_tile(f"Alto {BASIS_TXT} · dentro del tope", fmt_pct(b[vb]),
                           f"IC 95 %: {fmt_pct(b['lo_' + BASIS])} – {fmt_pct(b['hi_' + BASIS])}", "bajo"))
        tiles.append(_tile("Riesgo relativo", _x(rr, 2),
                           ("Los IC se traslapan: diferencia no concluyente." if (a["lo_" + BASIS] <= b["hi_" + BASIS]
                                                                                and b["lo_" + BASIS] <= a["hi_" + BASIS])
                            else "Diferencia significativa al 95 %."), "medio"))
    st.markdown("<div class='sg-tiles'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)
    _note("DA-02 exige que la financiación no supere el 85 % del valor neto de matrícula; el reporte identificó 812 "
          "registros CRq que incumplen. El listado para remediación está en <i>Monitoreo y calidad</i>.")


def stats_med(cls: pd.Series, r: str) -> float:
    a = pd.to_numeric(dff["antiguedad_meses"], errors="coerce")
    s = a[(cls.astype(str) == r).to_numpy()]
    return float(s.median()) if s.notna().any() else NAN


# ======================================================================================
# Pestaña 4 · Scoring externo
# ======================================================================================
def _tab_scoring() -> None:
    section("Rango de scoring externo y riesgo", "Rangos ordenados del peor al mejor puntaje de Datacrédito. Si el "
            "scoring discrimina, la curva baja de izquierda a derecha (relación monotónica).", kicker="Scoring externo")
    with st.container(border=True):
        c1, _ = st.columns([1.2, 3])
        smin = c1.select_slider("n mínimo por rango", [1, 10, 30, 50, 100], value=30, key=f"{P}_sc_min")
        t = _segments(dff, X, "score_corto")
        t = t[t["seg"].isin(_SCORE_LABELS) & (t["n"] >= smin)].copy()
        if len(t) < 2:
            _note("No hay suficientes rangos de scoring con el n mínimo en el filtro actual.")
            return
        t["rk"] = t["seg"].map(_SCORE_LABELS.index)
        t = t.sort_values("rk")
        t["hover_lbl"] = "Scoring " + t["seg"]
        l, r = st.columns([2.1, 1], gap="medium")
        with l:
            xl = t["seg"].tolist()
            fig = _pred_obs_bars(xl, t, TOT["pct_pred"], TOT["pct_obs"], has_truth_all)
            fig.update_xaxes(tickvals=xl, ticktext=[f"{s}<br><span style='font-size:10px'>n = {fmt_int(n)}</span>"
                                                     for s, n in zip(xl, t["n"])], title_text="Peor score  →  mejor score")
            show_fig(fig, key=f"{P}_score", height=420)
        with r:
            vcol = f"pct_{BASIS}"
            rho_o = _spearman(t["rk"], t["pct_obs"]) if has_truth_all else NAN
            rho_p = _spearman(t["rk"], t["pct_pred"])
            vals = t[vcol].to_numpy()
            inv = [f"{a} → {b}" for a, b, va, vb in zip(t["seg"][:-1], t["seg"][1:], vals[:-1], vals[1:]) if vb > va]
            worst = t.loc[t[vcol].idxmax()]
            best = t.iloc[-1]
            tiles = []
            if has_truth_all:
                tiles.append(_tile("ρ de Spearman · observado", fmt_num(rho_o, 2),
                                   "Correlación de rangos entre posición del score y % Alto (−1 = monotónica "
                                   "decreciente perfecta).", "accent"))
            tiles.append(_tile("ρ de Spearman · predicho", fmt_num(rho_p, 2),
                               "¿El modelo replica el gradiente del scoring?", "ink"))
            tiles.append(_tile(f"Riesgo relativo (Alto {BASIS_TXT})", _x(_div(worst[vcol], best[vcol]), 1),
                               f"<b>{esc(worst['seg'])}</b> {fmt_pct(worst[vcol])} vs. <b>{esc(best['seg'])}</b> "
                               f"{fmt_pct(best[vcol])}", "alto"))
            tiles.append(_tile("Inversiones del gradiente", str(len(inv)),
                               ("Pares consecutivos donde el riesgo sube al mejorar el score: " + esc(", ".join(inv)))
                               if inv else "El riesgo nunca sube al mejorar el score: gradiente monotónico.",
                               "medio" if inv else "bajo"))
            st.markdown("<div class='sg-stack'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)
        _note("Barras = % Alto predicho por el modelo; puntos = % Alto observado con su IC 95 % de Wilson. El reporte "
              "señala que los scores externos menos favorables coinciden con mayor incumplimiento; una inversión en "
              "«≤ 400» puede reflejar historiales de crédito escasos (score bajo por falta de información).")

    section("Valores de Datacrédito: máximo, medio y bajo", "Riesgo por quintiles de los valores reportados. La "
            "categoría «Sin dato» agrupa los ceros.", kicker="Variables de scoring")
    vv = pd.to_numeric(dff["valor_maximo"], errors="coerce")
    by_year = pd.DataFrame({"anio": dff["anio"], "ok": (vv > 0).to_numpy()}).dropna(subset=["anio"])
    cov = by_year.groupby("anio")["ok"].agg(["mean", "size"]).reset_index()
    st.markdown("<div class='sg-warn'><div class='ic'>⚠️</div><div class='tx'><b>0 = sin dato, no «valor cero».</b> "
                "Antes de 2024 la plataforma de scoring no reportaba estos valores, así que la mayoría de los créditos "
                "antiguos tienen 0. Compara el riesgo dentro de los quintiles con dato y lee «Sin dato» como un "
                "grupo aparte (sesgo de cohorte), no como el extremo inferior de la escala.</div></div>",
                unsafe_allow_html=True)
    with st.container(border=True):
        c1, _ = st.columns([2.2, 2])
        vopts = [c for c in ("valor_maximo", "valor_medio", "valor_bajo") if c in dff.columns]
        vname = c1.segmented_control("Variable", vopts, format_func=lambda c: {"valor_maximo": "Valor máximo",
                                     "valor_medio": "Valor medio", "valor_bajo": "Valor bajo"}[c],
                                     default=vopts[0], key=f"{P}_sc_var", required=True) or vopts[0]
        l, r = st.columns([2.1, 1], gap="medium")
        v = pd.to_numeric(dff[vname], errors="coerce")
        pos = (v > 0).to_numpy()
        with l:
            grp = np.array(["Sin dato"] * len(v), dtype=object)
            qlab = {}
            if pos.sum() >= 25:
                q = pd.qcut(v[pos].rank(method="first"), 5, labels=False).astype(int).to_numpy()
                rng = v[pos].groupby(q).agg(["min", "max"])
                for k in range(5):
                    qlab[k] = f"Q{k + 1}"
                grp[pos] = [f"Q{k + 1}" for k in q]
            t = _agg(X.assign(g=grp), "g")
            order = ["Sin dato"] + [f"Q{k + 1}" for k in range(5)]
            t = t.set_index("g").reindex([o for o in order if o in set(t["g"])]).reset_index()
            t["hover_lbl"] = [("Sin dato (0)" if g == "Sin dato" else
                               f"{g}: {fmt_cop(rng.loc[int(g[1:]) - 1, 'min'])} – {fmt_cop(rng.loc[int(g[1:]) - 1, 'max'])}")
                              for g in t["g"]]
            xl = t["g"].tolist()
            fig = _pred_obs_bars(xl, t, TOT["pct_pred"], TOT["pct_obs"], has_truth_all,
                                 hatch_first=(xl[:1] == ["Sin dato"]))
            fig.update_xaxes(tickvals=xl, ticktext=[("Sin dato<br><span style='font-size:10px'>n = " + fmt_int(n) + "</span>")
                                                     if g == "Sin dato" else
                                                     f"{g}<br><span style='font-size:10px'>≤ {fmt_cop(rng.loc[int(g[1:]) - 1, 'max'])}</span>"
                                                     for g, n in zip(xl, t["n"])])
            show_fig(fig, key=f"{P}_scval", height=380)
        with r:
            st.markdown("<div class='sg-cap'>Cobertura del dato por año de aprobación</div>", unsafe_allow_html=True)
            p = pal()
            fig2 = go.Figure(go.Bar(x=cov["anio"].astype(int).astype(str), y=cov["mean"],
                                    marker=dict(color=p["text"], line=dict(width=0)),
                                    text=[fmt_pct(v, 0) for v in cov["mean"]], textposition="outside", cliponaxis=False,
                                    textfont=dict(size=11.5, color=p["text"]),
                                    customdata=np.c_[[fmt_int(v) for v in cov["size"]]],
                                    hovertemplate="%{x}: %{text} con dato (%{customdata[0]} créditos)<extra></extra>"))
            fig2.update_yaxes(tickformat=".0%", range=[0, 1.15], title_text="% con valor > 0")
            fig2.update_xaxes(type="category")
            fig2.update_layout(barcornerradius=4, margin=dict(l=8, r=8, t=16, b=8), bargap=0.45)
            show_fig(fig2, key=f"{P}_sccov", height=300, legend=False)
            _note(f"{fmt_pct((vv > 0).mean())} de la cartera filtrada tiene dato de valor máximo.")

    section("Mora reportada y plataforma de scoring", "Riesgo Alto observado según la mora en Datacrédito y la versión "
            "de la plataforma (ROMBO V1/V2).", kicker="Señales complementarias")
    tiles = []
    for col, ttl in (("mora_txt", "Mora Datacrédito"), ("plataforma", "Plataforma")):
        tt = _segments(dff, X, col)
        tt = tt[tt["seg"] != "Sin dato"].sort_values("n", ascending=False)
        vcol = f"pct_{BASIS}"
        if len(tt) < 2:
            continue
        a, b = tt.iloc[0], tt.iloc[1]
        if col == "mora_txt":
            a, b = (tt[tt["seg"] == "Con mora"].iloc[0], tt[tt["seg"] == "Sin mora"].iloc[0]) if {"Con mora", "Sin mora"} <= set(tt["seg"]) else (a, b)
        overlap = a[f"lo_{BASIS}"] <= b[f"hi_{BASIS}"] and b[f"lo_{BASIS}"] <= a[f"hi_{BASIS}"]
        for rec in (a, b):
            tiles.append(_tile(f"{ttl} · {rec['seg']}", fmt_pct(rec[vcol]),
                               f"Alto {BASIS_TXT} · IC {fmt_pct(rec[f'lo_{BASIS}'])} – {fmt_pct(rec[f'hi_{BASIS}'])} · "
                               f"n = {fmt_int(rec['n'])}", "alto" if rec is a else "ink"))
        tiles[-1] = tiles[-1].replace("</div></div>", f" · RR {_x(_div(a[vcol], b[vcol]), 2)}"
                                      f"{' (no concluyente)' if overlap else ''}</div></div>", 1)
    if tiles:
        st.markdown("<div class='sg-tiles'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)
        _note("El reporte observa que la mora histórica «Sí» presenta más riesgo Alto que «No». RR = riesgo relativo "
              "(primera categoría ÷ segunda).")


# ======================================================================================
# Pestaña 5 · Perfil y equidad
# ======================================================================================
_FAIR_METRICS = {
    "Tasa de alerta": ("pct_pred", "Paridad demográfica: % de créditos marcados Alto por el modelo.", False),
    "Recall de Alto": ("recall", "Igualdad de oportunidad: % de los Alto reales que el modelo detecta.", True),
    "Precisión de Alto": ("precision", "% de las alertas Alto que resultan Alto observado.", True),
    "Falsos positivos": ("fpr", "% de créditos no Alto que el modelo marca como Alto.", True),
}


def _disparity_fig(t: pd.DataFrame, ref: str, mcol: str, mlabel: str) -> go.Figure:
    p = pal()
    t = t.copy()
    t["ratio"] = t[mcol] / float(t.loc[t["g"] == ref, mcol].iloc[0]) if float(t.loc[t["g"] == ref, mcol].iloc[0]) > 0 else np.nan
    t = t.sort_values("ratio", ascending=True, na_position="first")
    out = (t["ratio"] < BAND[0]) | (t["ratio"] > BAND[1])
    colors = [YELLOW if g == ref else (RISK_COLORS["Alto"] if o else RISK_COLORS["Bajo"]) for g, o in zip(t["g"], out)]
    ys = _dedupe([_short(g, 26) for g in t["g"]])
    rmax = float(np.nanmax(np.r_[t["ratio"].to_numpy(dtype=float), [1.4]]))
    rmin = float(np.nanmin(np.r_[t["ratio"].to_numpy(dtype=float), [0.6]]))
    fig = go.Figure()
    fig.add_vrect(x0=BAND[0], x1=BAND[1], fillcolor=_rgba(RISK_COLORS["Bajo"], 0.10), line_width=0, layer="below")
    fig.add_vline(x=1, line=dict(color=p["muted"], width=1.1, dash="dot"))
    for xv in BAND:
        fig.add_vline(x=xv, line=dict(color=_rgba(RISK_COLORS["Bajo"], 0.55), width=1))
    cd = np.c_[[esc(g) for g in t["g"]], [fmt_pct(v) for v in t[mcol]], [fmt_int(v) for v in t["n"]],
               [fmt_num(v, 2) for v in t["ratio"]]]
    fig.add_trace(go.Scatter(
        x=t["ratio"], y=ys, mode="markers+text", text=[f"  {fmt_num(v, 2)}" for v in t["ratio"]],
        textposition="middle right", textfont=dict(size=11.5, color=p["text"]), cliponaxis=False,
        marker=dict(size=14, color=colors, line=dict(color=p["bg"], width=2)), customdata=cd,
        hovertemplate=(f"<b>%{{customdata[0]}}</b><br>{esc(mlabel)}: %{{customdata[1]}}<br>Razón vs. {esc(ref)}: "
                       "<b>%{customdata[3]}</b><br>Créditos: %{customdata[2]}<extra></extra>")))
    fig.add_annotation(x=(BAND[0] + BAND[1]) / 2, y=1, yref="paper", text="Rango aceptable [0,8; 1,25]", showarrow=False,
                       yanchor="bottom", font=dict(size=11, color=p["muted"]))
    fig.update_xaxes(range=[max(0, rmin - 0.15), rmax + 0.25], title_text=f"Razón de disparidad ({mlabel.lower()} ÷ referencia)",
                     showgrid=True, gridcolor=p["grid"], zeroline=False)
    fig.update_yaxes(showgrid=False, ticksuffix="  ", tickfont=dict(size=12, color=p["text"]))
    fig.update_layout(margin=dict(l=8, r=16, t=30, b=8), showlegend=False)
    return fig


def _tab_equidad() -> None:
    sens_in_model = [f for f in ("genero", "estado_civil", "grupo_etnico", "fecha_nacimiento") if f in MODEL_FEATURES]
    st.markdown(
        "<div class='sg-ethic'><div class='ic'>🛡️</div><div><div class='t'>Uso ético de variables sensibles</div>"
        "<ul><li>Género, edad, estado civil y grupo étnico se muestran <b>solo para auditoría de sesgo</b>: verificar "
        "que las alertas del modelo no se distribuyan de forma desproporcionada entre grupos.</li>"
        "<li><b>No deben usarse para decidir sobre un estudiante individual</b>, priorizar el cobro ni diseñar campañas "
        "diferenciadas por estos atributos.</li>"
        "<li>Una razón de disparidad fuera de <b>[0,8; 1,25]</b> no prueba discriminación: obliga a revisar si la brecha "
        "también existe en el riesgo observado y a documentar la decisión.</li>"
        + (f"<li>Gobierno del modelo: el Random Forest empaquetado usa como entrada <b>{esc(', '.join(sens_in_model))}</b>. "
           "Se recomienda evaluar una versión sin estas variables y comparar desempeño y equidad.</li>" if sens_in_model else "")
        + "</ul></div></div>", unsafe_allow_html=True)

    sums = [f for f in (_fair_summary(dff, X, a) for a in SENSITIVE) if f]
    if sums:
        tiles = []
        for f in sums:
            ok = f["n_out"] == 0
            rng = f"{fmt_num(min(f['rmin'], 1), 2)} – {fmt_num(max(f['rmax'], 1), 2)}" if not _isnan(f["rmax"]) else "—"
            tiles.append(_tile(SENSITIVE[f["attr"]], ("✓ En rango" if ok else f"⚠ {f['n_out']} fuera"),
                               f"Razón de alertas {rng} · ref. <b>{esc(f['ref'])}</b> · {f['n_groups']} grupos",
                               "bajo" if ok else "alto"))
        st.markdown("<div class='sg-fair'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)
        _note("Resumen con grupos de al menos 100 créditos (los menores se agrupan en «Otros») y la categoría más "
              "numerosa como referencia. Ajusta los parámetros abajo para el detalle.")

    section("Tasa de alerta y riesgo observado por grupo", "Compara lo que el modelo marca (barras) con lo que se "
            "observa (puntos con IC 95 %) en cada grupo del atributo elegido.", kicker="Equidad")
    fopts = [m for m, (_, _, need) in _FAIR_METRICS.items() if has_truth_all or not need]
    if st.session_state.get(f"{P}_fair_metric") not in fopts:
        st.session_state.pop(f"{P}_fair_metric", None)
    with st.container(border=True):
        c1, c2, c3 = st.columns([2.1, 1.6, 1.1], vertical_alignment="bottom")
        attr = c1.segmented_control("Atributo", list(SENSITIVE), format_func=lambda a: SENSITIVE[a], default="genero_txt",
                                    key=f"{P}_fair_attr", required=True) or "genero_txt"
        fm = c2.segmented_control("Métrica de equidad", fopts, default="Tasa de alerta", key=f"{P}_fair_metric",
                                  required=True) or "Tasa de alerta"
        gmin = c3.select_slider("Agrupar en «Otros» si n <", [10, 30, 50, 100, 200, 500], value=100, key=f"{P}_fair_min")
        lab = _fold(_labels(dff, attr), gmin)
        t = _agg(X.assign(g=lab.to_numpy()), "g").sort_values("n", ascending=False)
        if len(t) < 2:
            _note("Solo hay un grupo con el filtro actual: no hay comparación posible.")
            return
        groups = t["g"].tolist()
        rkey = f"{P}_fair_ref_{attr}"
        if st.session_state.get(rkey) not in groups:
            st.session_state.pop(rkey, None)
        r1, r2 = st.columns([1.6, 3.1], vertical_alignment="bottom")
        ref = r1.selectbox("Grupo de referencia", groups, index=0, key=rkey)
        mcol, mdesc, _ = _FAIR_METRICS[fm]
        with r2:
            _note(f"<b>{esc(fm)}:</b> {esc(mdesc)} Razón = métrica del grupo ÷ métrica de <b>{esc(ref)}</b>.")
        l, r = st.columns([1.35, 1], gap="medium")
        with l:
            tt = t.set_index("g").loc[groups].reset_index()
            tt["hover_lbl"] = tt["g"]
            fig = _pred_obs_bars([_short(g, 18) for g in tt["g"]], tt, TOT["pct_pred"], TOT["pct_obs"], has_truth_all)
            show_fig(fig, key=f"{P}_fair_bars", height=400)
        with r:
            st.markdown("<div class='sg-cap'>Razón de disparidad</div>", unsafe_allow_html=True)
            st.markdown(_legend([(YELLOW, "Referencia", "dt"), (RISK_COLORS["Bajo"], "Dentro del rango", "dt"),
                                 (RISK_COLORS["Alto"], "Fuera del rango", "dt")]), unsafe_allow_html=True)
            ref_v = float(t.loc[t["g"] == ref, mcol].iloc[0])
            if _isnan(ref_v) or ref_v <= 0:
                _note(f"<b>{esc(ref)}</b> tiene {esc(fm.lower())} de 0 % (o sin dato) en el filtro actual: la razón "
                      "de disparidad no está definida para esta referencia.")
            else:
                show_fig(_disparity_fig(t, ref, mcol, fm), key=f"{P}_fair_disp", height=max(260, 46 * len(t) + 110),
                         legend=False)
        refrow = t[t["g"] == ref].iloc[0]
        tbl = pd.DataFrame({
            SENSITIVE[attr]: t["g"], "Créditos": t["n"].astype(int), "Tasa de alerta": t["pct_pred"],
            "Alto observado": t["pct_obs"], "Brecha pred. − obs.": t["gap"], "Recall Alto": t["recall"],
            "Precisión Alto": t["precision"], "Falsos positivos": t["fpr"],
            "Razón alerta": t["pct_pred"] / refrow["pct_pred"] if refrow["pct_pred"] > 0 else np.nan,
            "Razón observada": t["pct_obs"] / refrow["pct_obs"] if refrow["pct_obs"] > 0 else np.nan,
            "Razón recall": t["recall"] / refrow["recall"] if refrow["recall"] > 0 else np.nan,
        })
        tbl["Estado"] = np.where(tbl[SENSITIVE[attr]] == ref, "Referencia",
                                 np.where((tbl["Razón alerta"] < BAND[0]) | (tbl["Razón alerta"] > BAND[1]),
                                          "⚠ Fuera de rango", "✓ En rango"))
        if not has_truth_all:
            tbl = tbl.drop(columns=["Alto observado", "Brecha pred. − obs.", "Recall Alto", "Precisión Alto",
                                    "Falsos positivos", "Razón observada", "Razón recall"])
        pct_cols = [c for c in ("Tasa de alerta", "Alto observado", "Brecha pred. − obs.", "Recall Alto", "Precisión Alto",
                                "Falsos positivos") if c in tbl.columns]
        cfg = {c: st.column_config.NumberColumn(format="percent") for c in pct_cols}
        cfg.update({c: st.column_config.NumberColumn(format="%.2f") for c in ("Razón alerta", "Razón observada",
                                                                              "Razón recall") if c in tbl.columns})
        st.dataframe(tbl, hide_index=True, width="stretch", column_config=cfg)
        # lectura
        rr_obs = tbl.get("Razón observada")
        dev = (tbl["Razón alerta"] - 1).abs()
        if dev.notna().any():
            w = tbl.loc[dev.idxmax()]
            txt = (f"Mayor disparidad de alertas: <b>{esc(w[SENSITIVE[attr]])}</b> con razón "
                   f"<b>{fmt_num(w['Razón alerta'], 2)}</b>")
            if rr_obs is not None and not _isnan(w.get("Razón observada", NAN)):
                txt += (f" frente a una razón observada de <b>{fmt_num(w['Razón observada'], 2)}</b>: "
                        + ("el modelo amplifica la brecha real."
                           if abs(w["Razón alerta"] - 1) > abs(w["Razón observada"] - 1) + 0.05
                           else "la brecha del modelo es similar o menor a la observada."))
            _note(txt + " Los grupos con menos créditos tienen razones más inestables.")
        else:
            _note(f"El grupo de referencia <b>{esc(ref)}</b> no tiene alertas Alto en el filtro actual (tasa 0 %): la "
                  "razón de disparidad no está definida. Elige otra referencia o amplía los filtros.")
        download_bar(tbl, f"equidad_{attr}", key=f"{P}_dl_fair", label="Auditoría")


# ======================================================================================
# Pestaña 6 · Predicho vs observado
# ======================================================================================
def _pvo_scatter(t: pd.DataFrame, shift: float, nlab: int) -> go.Figure:
    p = pal()
    y = t["pct_pred"] - shift
    gap = y - t["pct_obs"]
    gmax = float(np.nanmax(np.abs(gap))) if len(t) else 0.05
    gmax = max(gmax, 0.01)
    mx = float(np.nanmax(np.r_[t["pct_obs"].to_numpy(dtype=float), y.to_numpy(dtype=float), [0.05]])) * 1.12
    score = np.abs(gap) * np.sqrt(t["n"])
    lab_idx = set(score.sort_values(ascending=False).index[:nlab])
    sizeref = 2.0 * float(t["n"].max()) / (48 ** 2)
    cd = np.c_[[esc(s) for s in t["seg"]], [fmt_int(v) for v in t["n"]], [fmt_pct(v) for v in t["pct_pred"]],
               [fmt_pct(v) for v in t["pct_obs"]], [f"{fmt_pct(a)} – {fmt_pct(b)}" for a, b in zip(t["lo_obs"], t["hi_obs"])],
               [_pp(v) for v in (t["pct_pred"] - t["pct_obs"])], [fmt_pct(v) for v in t["recall"]]]
    red, blue = RISK_COLORS["Alto"], _obs_color()
    mid = "#B5B3AA" if theme_mode() == "light" else "#5A606C"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=p["muted"], width=1.2, dash="dot"),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=t["pct_obs"], y=y, mode="markers+text", text=[_short(s, 22) if i in lab_idx else "" for i, s in zip(t.index, t["seg"])],
        textposition="top center", textfont=dict(size=11, color=p["text"]), cliponaxis=False,
        marker=dict(size=t["n"], sizemode="area", sizeref=sizeref, sizemin=6, color=gap, cmin=-gmax, cmax=gmax, cmid=0,
                    colorscale=[[0, red], [0.5, mid], [1, blue]], opacity=0.9, line=dict(color=p["bg"], width=1.5),
                    colorbar=dict(title=dict(text="Brecha", side="right", font=dict(size=11, color=p["muted"])),
                                  tickvals=[-gmax, 0, gmax], ticktext=[f"Subestima {_pp(-gmax, 0)}", "0", f"Sobreestima {_pp(gmax, 0)}"],
                                  thickness=10, len=0.8, outlinewidth=0, tickfont=dict(color=p["muted"], size=10))),
        customdata=cd, showlegend=False,
        hovertemplate=("<b>%{customdata[0]}</b> · n = %{customdata[1]}<br>Alto predicho: <b>%{customdata[2]}</b>"
                       "<br>Alto observado: <b>%{customdata[3]}</b> (IC %{customdata[4]})<br>Brecha: %{customdata[5]}"
                       " · recall Alto %{customdata[6]}<extra></extra>")))
    fig.add_trace(go.Scatter(x=[TOT["pct_obs"]], y=[TOT["pct_pred"] - shift], mode="markers+text", text=["  Cartera"],
                             textposition="middle right", textfont=dict(size=11.5, color=p["text"]),
                             marker=dict(symbol="star", size=17, color=YELLOW, line=dict(color=p["text"], width=1.2)),
                             hovertemplate=(f"<b>Cartera filtrada</b><br>Predicho {fmt_pct(TOT['pct_pred'])} · observado "
                                            f"{fmt_pct(TOT['pct_obs'])}<extra></extra>"), showlegend=False))
    fig.add_annotation(x=0.02 * mx, y=0.96 * mx, text="↖ El modelo <b>sobreestima</b>", showarrow=False, xanchor="left",
                       font=dict(size=11.5, color=blue))
    fig.add_annotation(x=0.98 * mx, y=0.04 * mx, text="El modelo <b>subestima</b> ↘", showarrow=False, xanchor="right",
                       font=dict(size=11.5, color=red))
    fig.update_xaxes(range=[0, mx], tickformat=".0%", title_text="% Alto observado (real)", showgrid=True,
                     gridcolor=p["grid"])
    fig.update_yaxes(range=[0, mx], tickformat=".0%",
                     title_text="% Alto predicho" + (" (descontado el sesgo global)" if shift else ""))
    fig.update_layout(margin=dict(l=8, r=8, t=24, b=8))
    return fig


def _dumbbell_fig(t: pd.DataFrame) -> go.Figure:
    p = pal()
    oc = _obs_color()
    t = t.iloc[::-1]
    ys = _dedupe([_short(s, 26) for s in t["seg"]])
    fig = go.Figure()
    for yv, a, b in zip(ys, t["pct_pred"], t["pct_obs"]):
        fig.add_trace(go.Scatter(x=[a, b], y=[yv, yv], mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=RISK_COLORS["Alto"] if b > a else oc, width=3)))
    cd = np.c_[[esc(s) for s in t["seg"]], [fmt_pct(v) for v in t["pct_pred"]], [fmt_pct(v) for v in t["pct_obs"]],
               [_pp(v) for v in t["gap"]], [fmt_int(v) for v in t["n"]]]
    ht = ("<b>%{customdata[0]}</b><br>Predicho %{customdata[1]} · observado %{customdata[2]}<br>Brecha %{customdata[3]}"
          " · n = %{customdata[4]}<extra></extra>")
    fig.add_trace(go.Scatter(x=t["pct_pred"], y=ys, mode="markers", name="Predicho",
                             marker=dict(size=11, color=p["text"], line=dict(color=p["bg"], width=2)), customdata=cd,
                             hovertemplate=ht))
    fig.add_trace(go.Scatter(x=t["pct_obs"], y=ys, mode="markers", name="Observado",
                             marker=dict(size=11, color=oc, line=dict(color=p["bg"], width=2)), customdata=cd,
                             hovertemplate=ht))
    fig.update_xaxes(tickformat=".0%", title_text="% riesgo Alto", showgrid=True, gridcolor=p["grid"], rangemode="tozero")
    fig.update_yaxes(showgrid=False, ticksuffix="  ", tickfont=dict(size=11.5, color=p["text"]))
    fig.update_layout(margin=dict(l=8, r=8, t=40, b=8))
    return fig


def _tab_pvo() -> None:
    if not has_truth_all:
        empty_state("Sin riesgo observado", "El dataset activo no trae la columna y_true: no es posible comparar lo "
                    "predicho con lo observado. Usa la base oficial o carga un archivo con riesgo real.", icon="🧪")
        return
    section("¿Dónde se equivoca el modelo?", "Cada burbuja es un segmento: sobre la diagonal el modelo marca más Alto "
            "de lo observado; bajo la diagonal, menos. Tamaño = créditos; color = brecha.", kicker="Predicho vs observado")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.5, 1.1, 1.1, 1.4], vertical_alignment="bottom")
        dim = c1.selectbox("Dimensión", EXPLORER_DIMS, index=EXPLORER_DIMS.index("programa"), format_func=_dim_title,
                           key=f"{P}_pvo_dim")
        nmin = c2.select_slider("n mínimo", [20, 30, 50, 100, 200], value=50, key=f"{P}_pvo_min")
        nlab = c3.select_slider("Etiquetas", [0, 4, 6, 8, 12], value=6, key=f"{P}_pvo_lab")
        adj = c4.toggle("Descontar el sesgo global", value=False, key=f"{P}_pvo_adj",
                        help="Resta a cada segmento la brecha de la cartera (predicho − observado) para aislar los "
                             "segmentos que se desvían más allá del sesgo general de la regla argmax.")
        t = _segments(dff, X, dim)
        t = t[(t["n"] >= nmin) & (t["n_obs"] > 0)].reset_index(drop=True)
        if len(t) < 2:
            _note("Se necesitan al menos dos segmentos con el n mínimo para comparar.")
            return
        bias = float(TOT["pct_pred"] - TOT["pct_obs"])
        shift = bias if adj else 0.0
        l, r = st.columns([1.55, 1], gap="medium")
        with l:
            show_fig(_pvo_scatter(t, shift, nlab), key=f"{P}_pvo_scatter", height=470, legend=False)
        with r:
            t2 = t.assign(gap=t["pct_pred"] - shift - t["pct_obs"])
            t2["w"] = np.abs(t2["gap"]) * np.sqrt(t2["n"])
            under = t2[t2["gap"] < 0].sort_values("w", ascending=False).head(5)
            over = t2[t2["gap"] > 0].sort_values("w", ascending=False).head(5)
            sel = pd.concat([under.sort_values("gap"), over.sort_values("gap")])
            st.markdown("<div class='sg-cap'>Mayores brechas (ponderadas por √n)</div>", unsafe_allow_html=True)
            st.markdown(_legend([(pal()["text"], "Predicho", "dt"), (_obs_color(), "Observado", "dt"),
                                 (RISK_COLORS["Alto"], "Subestima", "ln"), (_obs_color(), "Sobreestima", "ln")]),
                        unsafe_allow_html=True)
            dd = sel.assign(pct_pred=sel["pct_pred"] - shift)
            show_fig(_dumbbell_fig(dd), key=f"{P}_pvo_dumb", height=max(300, 38 * len(sel) + 90), legend=False)
        w = t["n"] / t["n"].sum()
        mae = float((np.abs(t["pct_pred"] - shift - t["pct_obs"]) * w).sum())
        corr = float(np.corrcoef(t["pct_pred"], t["pct_obs"])[0, 1]) if len(t) >= 3 else NAN
        share_under = float(((t["pct_pred"] - shift - t["pct_obs"]) < 0).mean())
        tiles = [
            _tile("Sesgo global (pred. − obs.)", _pp(bias), "Negativo: la regla argmax marca menos Alto de los reales.",
                  "alto" if bias < -0.005 else "bajo"),
            _tile("Error absoluto ponderado", _pp(mae).replace("+", ""), f"Promedio de |brecha| por segmento, ponderado "
                  f"por créditos{' (tras descontar el sesgo)' if adj else ''}.", "accent"),
            _tile("Correlación entre segmentos", fmt_num(corr, 2), "¿Ordena el modelo los segmentos como la realidad? "
                  "(r de Pearson entre tasas)", "ink"),
            _tile("Segmentos subestimados", fmt_pct(share_under, 0), f"{int(round(share_under * len(t)))} de {len(t)} "
                  "segmentos quedan bajo la diagonal.", "medio"),
        ]
        st.markdown("<div class='sg-tiles'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)
        _note("Para recuperar los Alto que se escapan, el reporte propone bajar el umbral de P(Alto) a 0,30 (DE-03: recall "
              "≥ 0,75); el simulador está en <i>Desempeño del modelo</i>.")
    tbl = pd.DataFrame({DIMS[dim]: t["seg"], "Créditos": t["n"].astype(int), "% Alto predicho": t["pct_pred"],
                        "% Alto observado": t["pct_obs"], "IC inf. obs.": t["lo_obs"], "IC sup. obs.": t["hi_obs"],
                        "Brecha (pred. − obs.)": t["pct_pred"] - t["pct_obs"], "Recall Alto": t["recall"],
                        "Precisión Alto": t["precision"]}).sort_values("Brecha (pred. − obs.)")
    with st.expander(f"Tabla de brechas · {len(tbl)} segmentos", expanded=False):
        st.dataframe(tbl, hide_index=True, width="stretch", height=min(38 * len(tbl) + 40, 420),
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in tbl.columns[2:]})
        download_bar(tbl, f"predicho_vs_observado_{dim}", key=f"{P}_dl_pvo", label="Brechas")
    if can_access("modelo"):
        st.page_link(PAGES["modelo"][0], label="Ir a Desempeño del modelo (umbral DE-03, curvas y matriz de confusión)",
                     icon=":material/model_training:")


# ======================================================================================
# Pestañas
# ======================================================================================
section("Análisis por pestañas", "Cada pestaña responde a una pregunta del enfoque descriptivo; todas respetan los "
        "filtros globales de la barra lateral.", kicker="Exploración")
_TABS = ["🧭 Explorador de segmentos", "🎓 Académico", "💳 Financiero", "📊 Scoring externo", "⚖️ Perfil y equidad",
         "🎯 Predicho vs observado"]
tabs = st.tabs(_TABS, key=f"{P}_tab", on_change="rerun")
_RENDER = [_tab_explorer, _tab_academico, _tab_financiero, _tab_scoring, _tab_equidad, _tab_pvo]
_any_open = any(t.open for t in tabs)
for i, (tb, fn) in enumerate(zip(tabs, _RENDER)):
    with tb:
        if tb.open or (not _any_open and i == 0):
            fn()

footer()
