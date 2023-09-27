import pandas as pd
import streamlit as st

from .config import get_config
from .vectorstores import get_vectorstore


@st.cache_data
def load_config(config_path=st.session_state.get("config_path", None)):
    return get_config(config_path)


@st.cache_resource
def load_db(config):
    return get_vectorstore(config)


@st.cache_data
def load_db_data(config, include=["metadatas", "documents", "embeddings"]):
    db = load_db(config)
    data = db.get(include=include)
    df = pd.DataFrame.from_dict(data)
    return df.set_index("ids")