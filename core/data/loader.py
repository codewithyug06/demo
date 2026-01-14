import streamlit as st
import pandas as pd

# ADD THIS DECORATOR
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df