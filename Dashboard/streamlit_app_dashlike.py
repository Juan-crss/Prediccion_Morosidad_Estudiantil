# Dashboard/streamlit_app_dashlike.py
from __future__ import annotations

import re
import unicodedata
import hashlib
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ================= Utilidades =================

def nrm(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9\s]", " ", s).strip()

def nombre_fake(seed, genero=None):
    m = ["Juan", "Carlos", "Andrés", "Diego", "Luis",
         "Mateo", "Jorge", "Felipe", "Daniel", "Santiago"]
    f = ["María", "Laura", "Ana", "Camila", "Valentina",
         "Carolina", "Paula", "Daniela", "Sara", "Gabriela"]
    ap = ["García", "Rodríguez", "López", "Martínez", "Hernández",
          "Gómez", "Díaz", "Ramírez", "Torres", "Vargas"]
    h = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16)
    base = f if str(genero).lower() in {"f", "femenino"} else m
    return f"{base[h % 10]} {ap[(h // 97) % 10]}".title()

# ================= Config Streamlit =================

DASH_TITLE = "Señales tempranas: Detectando riesgo de incumplimiento en créditos estudiantiles"
st.set_page_config(
    page_title=DASH_TITLE,
    layout="wide",
)

brand, bg = "#003366", "#f7f9fb"

st.markdown(
    f"""
    <style>
      /* Fuerza look claro (aunque el usuario esté en dark mode) */
      :root {{ color-scheme: light; }}
      html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: #ffffff !important;
        color: #111111 !important;
      }}
      [data-testid="stSidebar"] {{
        background: #ffffff !important;
      }}
      .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 1.5rem;
      }}
      .dash-header {{
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 15px 25px;
        background: {bg};
        border-bottom: 2px solid #ccc;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        border-radius: 8px;
        margin-bottom: 10px;
      }}
      .dash-title {{
        font-size: 26px;
        font-weight: 800;
        color: {brand};
        margin: 0;
        line-height: 1.15;
      }}
      .dash-updated {{
        background: {brand};
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12);
        font-size: 16px;
        font-weight: 600;
        white-space: nowrap;
      }}
      .dash-divider {{
        border-left: 2px solid #ccc;
        height: 58px;
      }}
      .kpi {{
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
      }}
      .kpi-title {{
        color: #667;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 2px;
      }}
      .kpi-value {{
        color: {brand};
        font-weight: 900;
        font-size: 1.55rem;
        margin: 0;
      }}
      .kpi-sub {{
        color: #999;
        font-size: 0.72rem;
        margin-top: 4px;
      }}
      .card {{
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
      }}
      .section-title {{
        color: {brand};
        font-weight: 800;
        margin: 0 0 6px 0;
      }}
      .section-sub {{
        color: #555;
        font-size: 0.85rem;
        margin: 0 0 10px 0;
      }}
      .story-subtitle {{
        color: #1f2a44;
        font-weight: 800;
        font-size: 1.05rem;
        margin: 0 0 6px 0;
      }}
      .story-desc {{
        color: #444;
        font-size: 0.90rem;
        margin: 0 0 6px 0;
      }}
      .story-bridge {{
        color: #3a3a3a;
        font-size: 0.90rem;
        margin: 10px 0 4px 0;
      }}
      .story-conclusion {{
        color: #2b2b2b;
        font-size: 0.90rem;
        margin: 10px 0 0 0;
      }}
      .subsection-title {{
        color: {brand};
        font-weight: 850;
        font-size: 1.05rem;
        margin: 6px 0 4px 0;
      }}
      .subsection-sub {{
        color: #5a5a5a;
        font-size: 0.85rem;
        margin: 0 0 8px 0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================= Datos =================

@st.cache_data
def load_base():
    # Ruta: ../Database/Data_model_predictions/f_dash_with_preds.csv (fallback al nombre anterior)
    BASE_DIR = Path(__file__).resolve().parents[1] / "Database" / "Data_model_predictions"
    candidates = [
        BASE_DIR / "f_dash_with_preds.csv",
        BASE_DIR / "df_dash_with_preds.csv",
    ]
    DATA = next((p for p in candidates if p.exists()), None)
    if DATA is None:
        raise FileNotFoundError(
            "No se encontró el archivo de datos. Se buscó: "
            + ", ".join(str(p) for p in candidates)
        )

    df = pd.read_csv(DATA)

    # Fecha de aprobación
    if "fecha_aprobacion" in df.columns:
        df["fecha_aprobacion"] = pd.to_datetime(df["fecha_aprobacion"], errors="coerce")
    else:
        df["fecha_aprobacion"] = pd.NaT

    # Riesgo predicho
    RIESGO = "y_pred"
    ORDEN = ["Alto", "Medio", "Bajo"]
    if RIESGO in df.columns:
        df[RIESGO] = df[RIESGO].astype(str).str.strip().str.capitalize()
        df[RIESGO] = pd.Categorical(df[RIESGO], categories=ORDEN, ordered=True)

    # Probabilidad predicha (si existe)
    if "proba_pred" not in df.columns:
        df["proba_pred"] = np.nan

    # Año y periodo
    df["anio"] = df["fecha_aprobacion"].dt.year
    df["periodo"] = df["fecha_aprobacion"].dt.to_period("M").astype(str)

    # Nombre anonimizado si no existe
    col_id = next((c for c in df.columns if c.lower() == "idbanner"), None)
    col_gen = next((c for c in df.columns if c.lower() in {"genero", "sexo"}), None)

    if "nombre" not in df.columns:
        if col_id is None:
            df["nombre"] = [nombre_fake(i) for i in range(len(df))]
        else:
            df["nombre"] = df.apply(
                lambda r: nombre_fake(f"{r[col_id]}-{r.get(col_gen, '')}", r.get(col_gen, "")),
                axis=1
            )

    # Programa y Facultad
    for c in ["programa", "facultad"]:
        if c not in df.columns:
            df[c] = "No definido"

    # Clusters
    def rule_cluster(txt):
        t = nrm(txt)
        if any(x in t for x in ["ingenier", "sistemas", "software", "datos"]):
            return "Software y TI"
        if any(x in t for x in ["medic", "salud", "enfermer", "odont"]):
            return "Medicina y Salud"
        if any(x in t for x in ["admin", "negoc", "finan", "conta", "mercad"]):
            return "Negocios y Administración"
        if any(x in t for x in ["derech", "jur"]):
            return "Derecho"
        return "Otros"

    df["programa_cluster"] = df["programa"].astype(str).map(rule_cluster)
    df["facultad_cluster"] = df["facultad"].astype(str).map(rule_cluster)

    # Flag mora
    pos_mora = [c for c in df.columns if c.lower() in {"en_mora_datacredito", "flag_mora_bureau", "mora"}]
    if pos_mora:
        c = pos_mora[0]
        df["mora_flag"] = (
            df[c].astype(str).str.lower().str.strip()
              .isin({"1", "si", "true", "yes", "y", "en mora", "mora"})
              .astype(int)
        )
    else:
        df["mora_flag"] = 0

    # Créditos activos
    if col_id:
        df["_credits_by_id"] = df.groupby(col_id)[col_id].transform("size")
    else:
        df["_credits_by_id"] = 1

    # Tipo cliente limpio (se conserva por si existe en datos; no se usa como filtro)
    if "cliente" in df.columns:
        cli = df["cliente"].astype(str).str.strip().str.lower()
        cli = cli.replace({"estudiante": "estudiante", "no estudiante": "no estudiante"})
        df["cliente_limpio"] = cli.map({"estudiante": "Estudiante", "no estudiante": "No estudiante"}).fillna("Otro")
    else:
        df["cliente_limpio"] = "Otro"

    # Coordenadas y valor exposición
    lat_col = next((c for c in df.columns if c.lower() == "latitud"), None)
    lon_col = next((c for c in df.columns if c.lower() == "longitud"), None)
    VAL_COL = next((c for c in df.columns if c.lower() in {"valor_financiacion", "vr_neto_matricula"}), None)

    # Orden por fecha desc
    df = df.sort_values("fecha_aprobacion", ascending=False).reset_index(drop=True)

    # Defaults fecha: desde el primer dato hasta el último dato
    if df["fecha_aprobacion"].notna().any():
        fecha_max = df["fecha_aprobacion"].max().date()
        fecha_min = df["fecha_aprobacion"].min().date()
        fecha_ini_default = fecha_min
        fecha_fin_default = fecha_max
    else:
        fecha_ini_default = None
        fecha_fin_default = None

    return df, col_id, col_gen, lat_col, lon_col, VAL_COL, RIESGO, ORDEN, fecha_ini_default, fecha_fin_default

df_base, col_id, col_gen, lat_col, lon_col, VAL_COL, RIESGO, ORDEN, fecha_ini_default, fecha_fin_default = load_base()
fecha_hoy = date.today().strftime("%Y-%m-%d")

# ================= Figuras =================

pal_riesgo = {"Alto": "#d9534f", "Medio": "#f0ad4e", "Bajo": "#5cb85c"}
PLOT_TEMPLATE = "plotly_white"

def fig_riesgo_resumen(dff):
    if RIESGO not in dff.columns or dff[RIESGO].dropna().empty:
        return px.bar(title="Distribución de riesgo (predicho)", template=PLOT_TEMPLATE)

    g = dff[RIESGO].value_counts(dropna=True).rename_axis("riesgo").reset_index(name="n")
    g["pct"] = g["n"] / g["n"].sum() * 100
    g["etq"] = g["n"].map("{:,.0f}".format) + " | " + g["pct"].map("{:,.1f}%".format)

    order_map = {k: i for i, k in enumerate(ORDEN)}
    g = g.sort_values("riesgo", key=lambda s: s.map(order_map))

    fig = px.bar(
        g, x="riesgo", y="n", text="etq", color="riesgo",
        color_discrete_map=pal_riesgo,
        title="Distribución de riesgo (predicho)",
        template=PLOT_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Nivel de riesgo",
        yaxis_title="Créditos",
        showlegend=False,
        title_font={"size": 16, "color": brand},
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

def fig_cuotas_tiempo(dff):
    if "cuotas" not in dff.columns or dff["fecha_aprobacion"].isna().all():
        return px.line(title="Cuotas promedio por mes y riesgo", template=PLOT_TEMPLATE)

    tmp = dff.dropna(subset=["fecha_aprobacion"])
    if tmp.empty:
        return px.line(title="Cuotas promedio por mes y riesgo", template=PLOT_TEMPLATE)

    g = tmp.groupby(["periodo", RIESGO])["cuotas"].mean().reset_index(name="cuotas_prom")
    order_map = {k: i for i, k in enumerate(ORDEN)}
    g = g.sort_values(
        ["periodo", RIESGO],
        key=lambda col: col.map(order_map) if col.name == RIESGO else col
    )

    fig = px.line(
        g, x="periodo", y="cuotas_prom", color=RIESGO,
        color_discrete_map=pal_riesgo,
        markers=True,
        title="Cuotas promedio por mes y riesgo",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title="Mes de aprobación",
        yaxis_title="Cuotas promedio",
        legend_title_text="Riesgo",
        title_font={"size": 16, "color": brand},
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

def fig_riesgo_tiempo(dff):
    if RIESGO not in dff.columns or dff["fecha_aprobacion"].isna().all():
        return px.area(title="Cómo cambia el riesgo en el tiempo", template=PLOT_TEMPLATE)

    tmp = dff.dropna(subset=["fecha_aprobacion"])
    if tmp.empty:
        return px.area(title="Cómo cambia el riesgo en el tiempo", template=PLOT_TEMPLATE)

    g = tmp.groupby(["periodo", RIESGO]).size().rename("n").reset_index()
    tot = g.groupby("periodo")["n"].transform("sum")
    g["pct"] = g["n"] / tot * 100

    order_map = {k: i for i, k in enumerate(ORDEN)}
    g = g.sort_values(
        ["periodo", RIESGO],
        key=lambda col: col.map(order_map) if col.name == RIESGO else col
    )

    fig = px.area(
        g, x="periodo", y="pct", color=RIESGO,
        color_discrete_map=pal_riesgo,
        title="Cómo cambia el riesgo en el tiempo",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title="Mes de aprobación",
        yaxis_title="% de créditos",
        legend_title_text="Riesgo",
        title_font={"size": 16, "color": brand},
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

def fig_mapa(dff):
    if dff is None or dff.empty or (lat_col not in dff.columns) or (lon_col not in dff.columns):
        fig = px.scatter_mapbox(
            lat=[], lon=[], zoom=4, height=420,
            title="Riesgo en el mapa (sin coordenadas)",
            template=PLOT_TEMPLATE,
        )
        fig.update_layout(mapbox_style="open-street-map")
        return fig

    dd = dff.dropna(subset=[lat_col, lon_col])
    if dd.empty:
        fig = px.scatter_mapbox(
            lat=[], lon=[], zoom=4, height=420,
            title="Riesgo en el mapa (sin coordenadas válidas)",
            template=PLOT_TEMPLATE,
        )
        fig.update_layout(mapbox_style="open-street-map")
        return fig

    fig = px.scatter_mapbox(
        dd,
        lat=lat_col,
        lon=lon_col,
        color=RIESGO,
        hover_name="nombre",
        color_discrete_map=pal_riesgo,
        zoom=4,
        height=420,
        title="Riesgo predicho por territorio",
        labels={RIESGO: "Riesgo"},
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=60, b=0),
        title_font={"size": 16, "color": brand},
        legend_title_text="Riesgo",
    )
    return fig

def fig_mora_tiempo(dff):
    if dff is None or dff.empty or dff["fecha_aprobacion"].isna().all():
        return px.line(title="Mora real (Datacrédito) en el tiempo", template=PLOT_TEMPLATE)

    tmp = dff.dropna(subset=["fecha_aprobacion"]).copy()
    if tmp.empty:
        return px.line(title="Mora real (Datacrédito) en el tiempo", template=PLOT_TEMPLATE)

    g = tmp.groupby("periodo")["mora_flag"].mean().reset_index(name="mora_pct")
    g["mora_pct"] = g["mora_pct"] * 100

    fig = px.line(
        g, x="periodo", y="mora_pct",
        markers=True,
        title="Mora real (Datacrédito) en el tiempo",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title="Mes de aprobación",
        yaxis_title="% en mora",
        title_font={"size": 16, "color": brand},
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

def fig_heat_cluster_anio(dff):
    tmp = dff.copy()
    if tmp["anio"].isna().all():
        return px.density_heatmap(title="Mora por segmento y año", template=PLOT_TEMPLATE)

    tmp = tmp.groupby(["programa_cluster", "anio"])["mora_flag"].mean().reset_index(name="mora_pct")
    tmp["mora_pct"] = tmp["mora_pct"] * 100

    fig = px.density_heatmap(
        tmp, x="anio", y="programa_cluster", z="mora_pct",
        color_continuous_scale="Blues",
        title="Mora por segmento y año",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title="Año",
        yaxis_title="Segmento de programa",
        coloraxis_colorbar_title="% en mora",
        title_font={"size": 16, "color": brand},
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

# ================= UI (Header) =================

# intenta usar el mismo logo: Dashboard/assets/logo_uni.png
logo_path = Path(__file__).resolve().parent / "assets" / "logo_uni.png"

c1, c2, c3, c4, c5 = st.columns([1.2, 0.12, 6, 0.12, 2.2], vertical_alignment="center")

with c1:
    if logo_path.exists():
        st.image(str(logo_path), width=180)
    else:
        st.write("")

with c2:
    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="dash-title">{DASH_TITLE}</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)

with c5:
    st.markdown(f'<div class="dash-updated">Actualizado al {fecha_hoy}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="padding: 6px 6px 0 6px;">
      <p style="margin-bottom: 8px; color:#333;">
        La morosidad en créditos estudiantiles es un indicador financiero que apoya a la universidad a identificar el riesgo directo para tomar decisiones frente a la sostenibilidad institucional y la permanencia estudiantil.
        Históricamente, su gestión ha sido reactiva, basada en edades de cartera o cuando el incumplimiento ya ocurrió.
        Este visualizador nace como una herramienta estratégica que traduce analítica predictiva en decisiones accionables, por los que permite anticipar el riesgo, segmentar perfiles y priorizar intervenciones de recaudo.
        Más que mostrar datos, integra un modelo de predicción que transforma información dispersa en alertas tempranas y visión preventiva para la gestión financiera.
      </p>
      <p style="font-size: 12px; color:#666; margin-top: 0;">
        Fuentes: <b>f_dash_with_preds.csv</b> (base resultante del modelo) · Detalles en GitHub:
        <a href="https://github.com/CarolinaFuentes13/Visualizaci-n-y-storytelling.git" target="_blank">Visualización y storytelling</a>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)



# ================= Tabs =================
tab_dashboard, tab_docs = st.tabs(["Dashboard", "Documentación"])

with tab_dashboard:
# ================= Filtros =================

    def filtrar(_df, f_ini, f_fin, fac_clu, prog_clu):
        dff = _df.copy()

        # fechas (date_input devuelve date, y df es datetime)
        if f_ini:
            dff = dff[dff["fecha_aprobacion"] >= pd.to_datetime(f_ini)]
        if f_fin:
            dff = dff[dff["fecha_aprobacion"] <= pd.to_datetime(f_fin)]

        if fac_clu:
            dff = dff[dff["facultad_cluster"].isin(fac_clu)]
        if prog_clu:
            dff = dff[dff["programa_cluster"].isin(prog_clu)]

        return dff

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Filtros</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Ajusta el rango de fechas y enfoca el análisis por segmentos de programa y facultad.</div>',
            unsafe_allow_html=True
        )

        f1, f2, f3 = st.columns(3)

        with f1:
            # Filtro amigable por Año–Mes (evita errores por fechas inválidas o rangos invertidos)
            if df_base["fecha_aprobacion"].notna().any():
                _min_dt = pd.to_datetime(df_base["fecha_aprobacion"].min())
                _max_dt = pd.to_datetime(df_base["fecha_aprobacion"].max())

                _periods = pd.period_range(_min_dt, _max_dt, freq="M")
                _labels = [p.strftime("%Y-%m") for p in _periods]

                _desde = st.selectbox("Desde (año-mes)", options=_labels, index=0)
                _hasta = st.selectbox("Hasta (año-mes)", options=_labels, index=len(_labels) - 1)

                _p_desde = pd.Period(_desde, freq="M")
                _p_hasta = pd.Period(_hasta, freq="M")

                # si el usuario elige al revés, lo corregimos automáticamente
                if _p_desde > _p_hasta:
                    _p_desde, _p_hasta = _p_hasta, _p_desde

                f_ini = _p_desde.to_timestamp(how="start")
                f_fin = _p_hasta.to_timestamp(how="end")
            else:
                f_ini, f_fin = None, None
                st.info("No hay fechas válidas en la base para filtrar.")

        with f2:

            fac_clu_sel = st.multiselect(
                "Facultad (segmento)",
                options=sorted(df_base["facultad_cluster"].dropna().unique()),
                default=[]
            )

        with f3:
            prog_clu_sel = st.multiselect(
                "Programa (segmento)",
                options=sorted(df_base["programa_cluster"].dropna().unique()),
                default=[]
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # aplicar filtros
    dff = filtrar(df_base, f_ini, f_fin, fac_clu_sel, prog_clu_sel)

    # ================= KPIs (Sección 1) =================

    st.markdown('<div style="margin-top: 14px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    st.markdown('<div class="story-subtitle">La foto rápida antes de actuar</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="story-desc">
          Los indicadores clave presentados a continuación resumen el estado actual de la cartera de créditos estudiantiles según los filtros aplicados.
          Los primeros cuatro métricas muestran el volumen y nivel de riesgo del portafolio: cuántos créditos están siendo analizados,
          cuántos se clasifican en riesgo alto, qué proporción representan y cuál es el porcentaje de mora real reportado en Datacrédito.
          Los cuatro indicadores inferiores complementan esta vista con la exposición económica, mostrando el monto total financiado y su distribución
          entre los niveles de riesgo alto, medio y bajo.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    n = len(dff)
    n_alto = int(dff[dff[RIESGO].eq("Alto")].shape[0]) if n > 0 else 0
    pct_alto = (n_alto / n * 100) if n > 0 else 0.0
    mora_pct = float(dff["mora_flag"].mean() * 100) if n > 0 else 0.0

    if VAL_COL and (VAL_COL in dff.columns) and n > 0:
        val = dff[VAL_COL].fillna(0).clip(lower=0)
        exp_tot = float(val.sum())
        exp_alto = float(val[dff[RIESGO].eq("Alto")].sum())
        exp_med = float(val[dff[RIESGO].eq("Medio")].sum())
        exp_baj = float(val[dff[RIESGO].eq("Bajo")].sum())
    else:
        exp_tot = exp_alto = exp_med = exp_baj = 0.0

    def kpi_html(title, value, subtitle):
        return f"""
        <div class="kpi">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{subtitle}</div>
        </div>
        """

    row1 = st.columns(4)
    row1[0].markdown(kpi_html("Créditos filtrados", f"{n:,}", "Número total de créditos según los filtros aplicados"), unsafe_allow_html=True)
    row1[1].markdown(kpi_html("Créditos en riesgo alto", f"{n_alto:,}", "Cantidad de créditos clasificados en nivel Alto"), unsafe_allow_html=True)
    row1[2].markdown(kpi_html("% en riesgo alto", f"{pct_alto:.1f}%", "Proporción de créditos en nivel Alto sobre el total filtrado"), unsafe_allow_html=True)
    row1[3].markdown(kpi_html("% en mora (Datacrédito)", f"{mora_pct:.1f}%", "Porcentaje de créditos reportados en mora"), unsafe_allow_html=True)

    row2 = st.columns(4)
    row2[0].markdown(kpi_html("Monto total financiado (COP)", f"{exp_tot:,.0f}", "Suma del valor financiado de los créditos filtrados"), unsafe_allow_html=True)
    row2[1].markdown(kpi_html("Monto en riesgo alto (COP)", f"{exp_alto:,.0f}", "Valor financiado asociado a créditos en nivel Alto"), unsafe_allow_html=True)
    row2[2].markdown(kpi_html("Monto en riesgo medio (COP)", f"{exp_med:,.0f}", "Valor financiado asociado a créditos en nivel Medio"), unsafe_allow_html=True)
    row2[3].markdown(kpi_html("Monto en riesgo bajo (COP)", f"{exp_baj:,.0f}", "Valor financiado asociado a créditos en nivel Bajo"), unsafe_allow_html=True)

    # Conclusión KPIs
    if n == 0:
        concl_kpi = "Con los filtros actuales no hay créditos visibles. Ajusta el rango de fechas o los segmentos para volver a tener señal."
    else:
        concl_kpi = (
            f"En esta vista, el riesgo alto representa <b>{pct_alto:.1f}%</b> de los créditos "
            f"({n_alto:,} casos). Lo importante: su peso real se entiende mejor cuando se mira la exposición "
            f"(COP <b>{exp_alto:,.0f}</b>) y se contrasta con la mora efectiva (Datacrédito: <b>{mora_pct:.1f}%</b>)."
        )

    st.markdown(f'<div class="story-conclusion">{concl_kpi}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="story-bridge">Ahora pasamos de la foto a los patrones: en los gráficos se ve cómo se distribuye el riesgo y dónde se concentra la mora real.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # ================= Gráficos (Sección 2) =================

    st.markdown('<div style="margin-top: 6px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Gráficos</div>', unsafe_allow_html=True)
    st.markdown('<div class="story-subtitle">De los números a la historia</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="story-desc">
          A continuación se presentan las visualizaciones que permiten explorar el comportamiento de la cartera desde distintas perspectivas.
          El gráfico de barras muestra la distribución de créditos por nivel de riesgo predicho. La línea de cuotas promedio permite identificar
          patrones de financiación según el mes de aprobación y el nivel de riesgo. El mapa de calor revela en qué segmentos y años se concentra la mora real.
          También se muestra cómo ha evolucionado la distribución de riesgo en el tiempo y, finalmente, el mapa georreferenciado ubica territorialmente los créditos
          según su categoría de riesgo predicha.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Sub-sección: Riesgo predicho ---
    st.markdown('<div class="subsection-title">Riesgo predicho</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-sub">Aquí vemos la alerta temprana: cuánto riesgo hay, cómo cambia y dónde se está ubicando.</div>', unsafe_allow_html=True)

    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(fig_riesgo_resumen(dff), use_container_width=True)
    with cB:
        st.plotly_chart(fig_riesgo_tiempo(dff), use_container_width=True)

    cC, cD = st.columns(2)
    with cC:
        st.plotly_chart(fig_cuotas_tiempo(dff), use_container_width=True)
    with cD:
        st.plotly_chart(fig_mapa(dff), use_container_width=True)

    st.write("")

    # --- Sub-sección: Mora real (Datacrédito) ---
    st.markdown('<div class="subsection-title">Mora real (Datacrédito)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-sub">Esta es la confirmación: dónde ya hay incumplimiento reportado y cómo se comporta en el tiempo.</div>', unsafe_allow_html=True)

    cE, cF = st.columns(2)
    with cE:
        st.plotly_chart(fig_mora_tiempo(dff), use_container_width=True)
    with cF:
        st.plotly_chart(fig_heat_cluster_anio(dff), use_container_width=True)

    # Conclusión Gráficos
    if n == 0:
        concl_graf = "Sin datos filtrados no es posible leer patrones. Ajusta filtros para ver la narrativa en las curvas y el mapa."
    else:
        # Insight ligero y robusto (sin volverse “muy IA”)
        last_period = None
        try:
            last_period = sorted(dff["periodo"].dropna().unique())[-1] if dff["periodo"].notna().any() else None
        except Exception:
            last_period = None

        mora_cases = int(dff["mora_flag"].sum())
        mora_share = (mora_cases / n * 100) if n else 0.0

        concl_graf = (
            f"Los gráficos confirman dos ideas: el riesgo predicho se mueve por cohorte (mes de aprobación) y no se reparte igual en el tiempo. "
            f"Además, la mora real ya aparece en <b>{mora_cases:,}</b> créditos (<b>{mora_share:.1f}%</b>), "
            f"lo que ayuda a contrastar la alerta del modelo con el comportamiento observado."
        )
        if last_period:
            concl_graf += f" En el cierre de la serie (<b>{last_period}</b>) puedes comparar si el riesgo se está calentando o estabilizando."

    st.markdown(f'<div class="story-conclusion">{concl_graf}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="story-bridge">Después de ver los patrones, la tabla permite aterrizar el análisis: quiénes son los casos en mora y cómo los clasifica el modelo.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # ================= Tabla (Sección 3) =================

    st.markdown('<div style="margin-top: 6px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tabla</div>', unsafe_allow_html=True)
    st.markdown('<div class="story-subtitle">Del mapa al caso puntual</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="story-desc">
          Aquí se listan los créditos con reporte de mora según Datacrédito. La idea es pasar del patrón al detalle:
          ver el programa, la fecha de aprobación, el nivel de riesgo predicho y la probabilidad estimada.
          Esta vista es útil para priorizar gestión y revisar casos recientes.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Casos en mora (Datacrédito)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Se listan los créditos con reporte de mora según Datacrédito, junto con su nivel de riesgo predicho y la probabilidad estimada.</div>',
        unsafe_allow_html=True
    )

    top = dff[dff["mora_flag"] == 1].copy()
    if not top.empty:
        top["proba_str"] = top["proba_pred"].mul(100).map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")
        cols = ["nombre", "programa", "fecha_aprobacion", RIESGO, "proba_str", "_credits_by_id"]
        tbl = top.sort_values("fecha_aprobacion", ascending=False)[cols].head(10)
    else:
        tbl = pd.DataFrame(columns=["nombre", "programa", "fecha_aprobacion", RIESGO, "proba_str", "_credits_by_id"])

    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Conclusión Tabla
    if n == 0:
        concl_tab = "Con los filtros actuales no hay casos para mostrar en la tabla."
    else:
        concl_tab = (
            f"La tabla resume los casos en mora dentro de esta vista. Úsala como lista de acción: "
            f"prioriza por fecha, revisa el riesgo predicho y cruza con la exposición para decidir a quién llamar primero."
        )

    st.markdown(f'<div class="story-conclusion">{concl_tab}</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("<hr/>", unsafe_allow_html=True)

    # ================= Conclusión final del dashboard =================

    # Métricas sobre el total (sin filtros) para un cierre más “executive”
    n_all = len(df_base)
    n_alto_all = int(df_base[df_base[RIESGO].eq("Alto")].shape[0]) if n_all > 0 else 0
    pct_alto_all = (n_alto_all / n_all * 100) if n_all > 0 else 0.0

    if VAL_COL and (VAL_COL in df_base.columns) and n_all > 0:
        val_all = df_base[VAL_COL].fillna(0).clip(lower=0)
        exp_alto_all = float(val_all[df_base[RIESGO].eq("Alto")].sum())
    else:
        exp_alto_all = 0.0

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cierre</div>', unsafe_allow_html=True)
    st.markdown('<div class="story-subtitle">Lo que nos deja la historia</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="story-desc">
          Al analizar <b>{n_all:,.0f}</b> créditos estudiantiles logramos identificar que, aunque la mayor parte del portafolio se ubica en nivel de riesgo bajo,
          existe un <b>{pct_alto_all:.1f}%</b> clasificado en riesgo alto que concentra una exposición relevante (COP <b>{exp_alto_all:,.0f}</b>).
          Esto recuerda que el foco no debe estar solo en la cantidad de casos, sino en el impacto económico que pueden concentrar ciertos segmentos.
          <br/><br/>
          Además, al observar la evolución en el tiempo y las diferencias por segmento, queda claro que el comportamiento no es uniforme:
          hay cohortes y grupos donde la señal se intensifica y otros donde se estabiliza. En conclusión, el riesgo se entiende mejor cuando se mira
          desde varias dimensiones: monto, tiempo, segmento y territorio, y este tablero convierte esa complejidad en alertas tempranas para priorizar acciones.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

with tab_docs:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Documentación</div>', unsafe_allow_html=True)
    st.markdown('<div class="story-subtitle">De datos dispersos a señales tempranas</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="story-desc">
          Este proyecto integra información histórica de crédito estudiantil (2021–2025) proveniente de fuentes institucionales,
          la somete a un proceso formal de calidad de datos y luego aplica un modelo predictivo para clasificar el riesgo de mora por crédito.
          El resultado final se consolida en un archivo (<b>f_dash_with_preds.csv</b>) que alimenta este visualizador en Streamlit.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="story-bridge">Resumen del proceso paso a paso:</div>', unsafe_allow_html=True)

    st.markdown("**1) Extracción y preparación de fuentes**")
    st.markdown(
        """
        - Se extraen de forma manual archivos desde plataformas institucionales (principalmente **Banner** e **Iceberg**).
        - Antes de analizarlos, se realiza **anonimización** en Excel para proteger datos sensibles.
        - Se estandarizan formatos y se alistan las llaves necesarias para integrar las fuentes.
        """
    )
    st.markdown("---")

    st.markdown("**2) Calidad de datos: perfilamiento y remediación**")
    st.markdown(
        """
        - Se construye un **catálogo/diccionario** para entender variables y reglas de negocio.
        - Se ejecutan dos etapas en Python: **perfilamiento** (evaluación de calidad) y **remediación** (correcciones).
        - Salida: CSVs limpios y consistentes, listos para el modelo de datos.
        """
    )
    st.markdown("---")

    st.markdown("**3) Modelo de datos: dataset final para el modelo**")
    st.markdown(
        """
        - Se integran los datasets limpios mediante llaves definidas.
        - Se eliminan registros con nulos en campos críticos y se filtran variables relevantes.
        - Se enriquece la geografía con **DIVIPOLA** para normalizar y habilitar análisis territorial.
        - Salida típica: un CSV final de modelado (por ejemplo, *df_modelo.csv*).
        """
    )
    st.markdown("---")

    st.markdown("**4) Modelo predictivo: riesgo de mora por crédito**")
    st.markdown(
        """
        - Se implementa un pipeline en **scikit-learn** y un clasificador **LightGBM** (multiclase).
        - Se usa partición 80/20 con estratificación para evaluar en datos no vistos.
        - Se generan clases predichas y probabilidades por nivel de riesgo.
        - Salida: el CSV consolidado con predicciones que consume el dashboard.
        """
    )
    st.markdown("---")

    st.markdown("**5) Visualización en Streamlit: del modelo a la acción**")
    st.markdown(
        """
        - La app carga el dataset final y aplica transformaciones ligeras para visualización.
        - Incluye **filtros**, **KPIs**, **gráficos** y una **tabla** para priorizar intervención.
        - Objetivo: convertir el scoring en una lectura clara de “dónde está el foco” (segmento, tiempo, monto y territorio).
        """
    )

    st.markdown('<div class="story-bridge">Cómo replicar:</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. Asegura el archivo **f_dash_with_preds.csv** en la ruta esperada por la app.
        2. Instala dependencias (Streamlit, pandas, numpy, plotly, etc.).
        3. Ejecuta:
        """
    )
    st.code('streamlit run streamlit_app_dashlike.py')

    st.markdown(
        """
        <div class="story-desc">
          Para ver el detalle completo y el paso a paso visual, consulta la presentación en Canva:
          <a href="https://www.canva.com/design/DAHCMba91Y4/p_1FZFzisR8IJTOCd0dTjA/view?utm_content=DAHCMba91Y4&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h300aab9c8b" target="_blank">Documentación (Canva)</a>.
          <br/><br/>
          Repositorio con scripts y documentación técnica:
          <a href="https://github.com/CarolinaFuentes13/Visualizaci-n-y-storytelling.git" target="_blank">Visualización y storytelling</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)
