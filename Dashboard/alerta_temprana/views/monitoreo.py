import streamlit as st
from core.components import page_header, footer
from core.data import get_active_df
from core.filters import apply_filters

df = get_active_df()
dff = apply_filters(df)
page_header("monitoreo", "Página en construcción")
st.write(len(dff))
footer()
