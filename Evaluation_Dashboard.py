import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from sklearn import linear_model

if "graph_mode" not in st.session_state:
    st.session_state["graph_mode"] = "Mean"


def mode_change():
    if st.session_state["graph_mode"] == "Mean":
        st.session_state["graph_mode"] = "Median"
    else:
        st.session_state["graph_mode"] = "Mean"


def read_csv():
    try:
        data = pd.read_csv("files/evaluation.csv", header=0, index_col=0)
    except OSError:
        data = pd.DataFrame()
    return data


df = read_csv()

st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="✅",
    layout="wide",
)

# page title
st.title("LLM Prototype Evaluation Dashboard")

# single-element container
placeholder = st.empty()

# top-level filters
concise_df = df[['theme', 'coherency', 'fluency']]

with placeholder.container():
    # create three columns
    st.dataframe(concise_df, use_container_width=True)

    column_content, column_rules, column_combat, column_other = st.columns(4)

    with column_content:
        st.markdown("")
    with column_rules:
        st.markdown("")
    with column_combat:
        st.markdown("")
    with column_other:
        st.markdown("")

    hyperparameter = st.selectbox("Select a hyperparameter", df.columns[1:6])

