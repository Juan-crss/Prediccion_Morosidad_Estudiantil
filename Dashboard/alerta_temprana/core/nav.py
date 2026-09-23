"""Registro único de páginas (clave → archivo, título, icono, grupo) y utilidades de navegación."""
from __future__ import annotations

import streamlit as st

PAGES = {
    "resumen": ("views/resumen.py", "Resumen ejecutivo", ":material/space_dashboard:", "Visión general"),
    "cola": ("views/cola_gestion.py", "Cola de gestión", ":material/format_list_numbered:", "Gestión de cartera"),
    "ficha": ("views/ficha_360.py", "Ficha 360° del estudiante", ":material/person_search:", "Gestión de cartera"),
    "segmentos": ("views/segmentos.py", "Segmentos y perfiles", ":material/donut_small:", "Análisis de riesgo"),
    "geografia": ("views/geografia.py", "Mapa de riesgo", ":material/map:", "Análisis de riesgo"),
    "modelo": ("views/modelo.py", "Desempeño del modelo", ":material/model_training:", "Modelo y gobierno"),
    "monitoreo": ("views/monitoreo.py", "Monitoreo y calidad", ":material/monitor_heart:", "Modelo y gobierno"),
    "metodologia": ("views/metodologia.py", "Metodología y trazabilidad", ":material/menu_book:", "Modelo y gobierno"),
    "carga": ("views/carga_prediccion.py", "Cargar y predecir", ":material/upload_file:", "Operación"),
    "simulador": ("views/simulador.py", "Simulador what-if", ":material/tune:", "Operación"),
}
GROUP_ORDER = ["Visión general", "Gestión de cartera", "Análisis de riesgo", "Modelo y gobierno", "Operación"]
# Páginas que usan los filtros globales de cartera en la barra lateral.
FILTER_PAGES = {"resumen", "cola", "ficha", "segmentos", "geografia", "monitoreo"}

FICHA_KEY = "ficha_estudiante"  # id_estudiante que debe abrir la Ficha 360°


def goto(key: str) -> None:
    """Navega a otra página del tablero (si el rol tiene acceso)."""
    st.switch_page(PAGES[key][0])


def open_ficha(id_estudiante: str) -> None:
    st.session_state[FICHA_KEY] = str(id_estudiante)
    goto("ficha")


def can_access(key: str) -> bool:
    from core.config import PAGE_ACCESS

    u = st.session_state.get("sat_user") or {}
    return u.get("role") in PAGE_ACCESS.get(key, set())
