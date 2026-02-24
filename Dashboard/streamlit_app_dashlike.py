# Dashboard/streamlit_app_dashlike.py
from __future__ import annotations

import re
import unicodedata
import hashlib
from pathlib import Path
from datetime import date, timedelta

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
    m = ["juan", "carlos", "andres", "diego", "luis",
         "mateo", "jorge", "felipe", "daniel", "santiago"]
    f = ["maria", "laura", "ana", "camila", "valentina",
         "carolina", "paula", "daniela", "sara", "gabriela"]
    ap = ["garcia", "rodriguez", "lopez", "martinez", "hernandez",
          "gomez", "diaz", "ramirez", "torres", "vargas"]
    h = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16)
    base = f if str(genero).lower() in {"f", "femenino"} else m
    return f"{base[h % 10]} {ap[(h // 97) % 10]}".title()

# ================= Config Streamlit =================

st.set_page_config(
    page_title="Riesgo de morosidad en créditos estudiantiles",
    layout="wide",
)

brand, bg = "#003366", "#f7f9fb"

st.markdown(
    f"""
    <style>
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ================= Datos =================

@st.cache_data
def load_base():
    # Ruta: ../Database/Data_model_predictions/df_dash_with_preds.csv
    DATA = Path(__file__).resolve().parents[1] / "Database" / "Data_model_predictions" / "df_dash_with_preds.csv"
    if not DATA.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {DATA}")

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

    # Tipo cliente limpio
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

    # Defaults fecha: último mes desde la fecha máxima disponible
    if df["fecha_aprobacion"].notna().any():
        fecha_max = df["fecha_aprobacion"].max().date()
        fecha_min = df["fecha_aprobacion"].min().date()
        fecha_ini_default = max(fecha_min, fecha_max - timedelta(days=30))
        fecha_fin_default = fecha_max
    else:
        fecha_ini_default = None
        fecha_fin_default = None

    return df, col_id, col_gen, lat_col, lon_col, VAL_COL, RIESGO, ORDEN, fecha_ini_default, fecha_fin_default

df_base, col_id, col_gen, lat_col, lon_col, VAL_COL, RIESGO, ORDEN, fecha_ini_default, fecha_fin_default = load_base()
fecha_hoy = date.today().strftime("%Y-%m-%d")

# ================= Figuras =================

pal_riesgo = {"Alto": "#d9534f", "Medio": "#f0ad4e", "Bajo": "#5cb85c"}

def fig_riesgo_resumen(dff):
    if RIESGO not in dff.columns or dff[RIESGO].dropna().empty:
        return px.bar(title="Créditos por nivel de riesgo")
    g = dff[RIESGO].value_counts(dropna=True).rename_axis("riesgo").reset_index(name="n")
    g["pct"] = g["n"] / g["n"].sum() * 100
    g["etq"] = g["n"].map("{:,.0f}".format) + " | " + g["pct"].map("{:,.1f}%".format)
    order_map = {k: i for i, k in enumerate(ORDEN)}
    g = g.sort_values("riesgo", key=lambda s: s.map(order_map))
    fig = px.bar(
        g, x="riesgo", y="n", text="etq", color="riesgo",
        color_discrete_map=pal_riesgo,
        title="Créditos por nivel de riesgo"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Nivel de riesgo",
        yaxis_title="Número de créditos",
        showlegend=False,
        title_font={"size": 16, "color": brand}
    )
    return fig

def fig_cuotas_tiempo(dff):
    if "cuotas" not in dff.columns or dff["fecha_aprobacion"].isna().all():
        return px.line(title="Cuotas promedio por riesgo y mes de aprobación")
    tmp = dff.dropna(subset=["fecha_aprobacion"])
    if tmp.empty:
        return px.line(title="Cuotas promedio por riesgo y mes de aprobación")
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
        title="Cuotas promedio por riesgo y mes de aprobación"
    )
    fig.update_layout(
        xaxis_title="Año-mes de aprobación",
        yaxis_title="Cuotas promedio",
        legend_title_text="Nivel de riesgo",
        title_font={"size": 16, "color": brand}
    )
    return fig

def fig_heat_cluster_anio(dff):
    tmp = dff.copy()
    if tmp["anio"].isna().all():
        return px.density_heatmap(title="Porcentaje de mora por segmento de programa y año")
    tmp = tmp.groupby(["programa_cluster", "anio"])["mora_flag"].mean().reset_index(name="mora_pct")
    tmp["mora_pct"] = tmp["mora_pct"] * 100
    fig = px.density_heatmap(
        tmp, x="anio", y="programa_cluster", z="mora_pct",
        color_continuous_scale="Blues",
        title="Porcentaje de mora por segmento de programa y año"
    )
    fig.update_layout(
        xaxis_title="Año",
        yaxis_title="Segmento de programa",
        coloraxis_colorbar_title="% en mora",
        title_font={"size": 16, "color": brand}
    )
    return fig

def fig_riesgo_tiempo(dff):
    if RIESGO not in dff.columns or dff["fecha_aprobacion"].isna().all():
        return px.area(title="Distribución de niveles de riesgo en el tiempo")
    tmp = dff.dropna(subset=["fecha_aprobacion"])
    if tmp.empty:
        return px.area(title="Distribución de niveles de riesgo en el tiempo")
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
        title="Distribución de niveles de riesgo en el tiempo"
    )
    fig.update_layout(
        xaxis_title="Año-mes de aprobación",
        yaxis_title="% de créditos",
        legend_title_text="Nivel de riesgo",
        title_font={"size": 16, "color": brand}
    )
    return fig

def fig_mapa(dff):
    if dff is None or dff.empty or (lat_col not in dff.columns) or (lon_col not in dff.columns):
        fig = px.scatter_mapbox(lat=[], lon=[], zoom=4, height=420, title="Mapa de créditos (sin datos geográficos)")
        fig.update_layout(mapbox_style="open-street-map")
        return fig

    dd = dff.dropna(subset=[lat_col, lon_col])
    if dd.empty:
        fig = px.scatter_mapbox(lat=[], lon=[], zoom=4, height=420, title="Mapa de créditos (sin datos geográficos válidos)")
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
        title="Mapa de créditos por categoría de riesgo predicha",
        labels={RIESGO: "Categoría predicha"},
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=60, b=0),
        title_font={"size": 16, "color": brand},
        legend_title_text="Categoría predicha",
    )
    return fig

# ================= UI (Header) =================

# intenta usar el mismo logo: Dashboard/assets/logo_uni.png o Dashboard/assets/logo_uni.png (Dash lo usa /assets/..)
logo_path = Path(__file__).resolve().parent / "assets" / "logo_uni.png"
logo_html = ""
if logo_path.exists():
    # Streamlit permite <img src="data:..."> pero lo simple es st.image en columna.
    pass

c1, c2, c3, c4, c5 = st.columns([1.2, 0.12, 6, 0.12, 2.2], vertical_alignment="center")

with c1:
    if logo_path.exists():
        st.image(str(logo_path), width=180)
    else:
        st.write("")  # si no hay logo, no rompe

with c2:
    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="dash-title">Riesgo de morosidad en créditos estudiantiles</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)

with c5:
    st.markdown(f'<div class="dash-updated">Actualizado al {fecha_hoy}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="padding: 6px 6px 0 6px;">
      <p style="margin-bottom: 4px; color:#333;">
        Este tablero muestra las predicciones de nivel de riesgo de morosidad (Alto, Medio y Bajo) para los créditos estudiantiles internos.
        Use los filtros para explorar por fecha de aprobación, programa, facultad y tipo de cliente.
      </p>
      <p style="font-size: 12px; color:#666; margin-top: 0;">
        Los montos corresponden al valor financiado de los créditos y se utilizan para dimensionar la exposición de la cartera por nivel de riesgo.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================= Filtros =================

def filtrar(_df, nombres, riesgo, f_ini, f_fin, fac, prog, fac_clu, prog_clu, cli):
    dff = _df.copy()

    if nombres:
        dff = dff[dff["nombre"].isin(nombres)]
    if riesgo:
        dff = dff[dff[RIESGO].isin(riesgo)]

    # fechas (date_input devuelve date, y df es datetime)
    if f_ini:
        dff = dff[dff["fecha_aprobacion"] >= pd.to_datetime(f_ini)]
    if f_fin:
        dff = dff[dff["fecha_aprobacion"] <= pd.to_datetime(f_fin)]

    if fac:
        dff = dff[dff["facultad"].astype(str).isin(fac)]
    if prog:
        dff = dff[dff["programa"].astype(str).isin(prog)]

    if fac_clu:
        dff = dff[dff["facultad_cluster"].isin(fac_clu)]
    if prog_clu:
        dff = dff[dff["programa_cluster"].isin(prog_clu)]

    if cli:
        dff = dff[dff["cliente_limpio"].isin(cli)]

    return dff

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Filtros</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        nombres_sel = st.multiselect(
            "Buscar nombre",
            options=sorted(df_base["nombre"].dropna().unique().tolist()),
            default=[],
        )
    with f2:
        riesgos_disponibles = [c for c in ORDEN if c in df_base.get(RIESGO, pd.Series(dtype=object)).astype(str).unique()]
        riesgo_sel = st.multiselect("Nivel de riesgo predicho", options=riesgos_disponibles, default=[])
    with f3:
        if fecha_ini_default and fecha_fin_default:
            fecha_rango = st.date_input(
                "Fecha de aprobación",
                value=(fecha_ini_default, fecha_fin_default),
                min_value=df_base["fecha_aprobacion"].min().date() if df_base["fecha_aprobacion"].notna().any() else None,
                max_value=df_base["fecha_aprobacion"].max().date() if df_base["fecha_aprobacion"].notna().any() else None,
            )
            # date_input puede devolver date o tuple(date,date)
            if isinstance(fecha_rango, tuple) and len(fecha_rango) == 2:
                f_ini, f_fin = fecha_rango
            else:
                f_ini, f_fin = fecha_rango, fecha_rango
        else:
            f_ini, f_fin = None, None
            st.date_input("Fecha de aprobación", value=None)
    with f4:
        cli_sel = st.multiselect("Tipo de cliente", options=sorted(df_base["cliente_limpio"].dropna().unique()), default=[])

    g1, g2 = st.columns(2)
    with g1:
        fac_sel = st.multiselect("Facultad", options=sorted(df_base["facultad"].astype(str).unique()), default=[])
    with g2:
        prog_sel = st.multiselect("Programa", options=sorted(df_base["programa"].astype(str).unique()), default=[])

    g3, g4 = st.columns(2)
    with g3:
        fac_clu_sel = st.multiselect("Facultad (segmento)", options=sorted(df_base["facultad_cluster"].dropna().unique()), default=[])
    with g4:
        prog_clu_sel = st.multiselect("Programa (segmento)", options=sorted(df_base["programa_cluster"].dropna().unique()), default=[])

    st.markdown('</div>', unsafe_allow_html=True)

# aplicar filtros
dff = filtrar(df_base, nombres_sel, riesgo_sel, f_ini, f_fin, fac_sel, prog_sel, fac_clu_sel, prog_clu_sel, cli_sel)

# ================= KPIs =================

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

st.write("")

# ================= Gráficas =================

cA, cB, cC = st.columns(3)
with cA:
    st.plotly_chart(fig_riesgo_resumen(dff), use_container_width=True)
with cB:
    st.plotly_chart(fig_cuotas_tiempo(dff), use_container_width=True)
with cC:
    st.plotly_chart(fig_heat_cluster_anio(dff), use_container_width=True)

st.write("")

cD, cE = st.columns(2)
with cD:
    st.plotly_chart(fig_riesgo_tiempo(dff), use_container_width=True)
with cE:
    st.plotly_chart(fig_mapa(dff), use_container_width=True)

st.write("")

# ================= Tabla Mora =================

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