import streamlit as st 
import json 

with open("config.json","r") as f:
    config_file = json.load(f)

st.set_page_config(layout="wide", page_title="WC 2026 Predictor")

with st.sidebar:
    st.markdown(f"*Version Number: {config_file['version_num']}*")

st.title("📊 Model Analysis")


