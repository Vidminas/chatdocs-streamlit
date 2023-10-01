import glob
from collections import defaultdict

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


@st.cache_data
def best_columns_for(config, query: str, files: list, dtypes: tuple):
    columns = {}
    db = load_db(config)

    for file in files:
        # See Chroma docs for filter syntax: https://docs.trychroma.com/usage-guide
        # although currently the langchain vectorstore doesn't work with Chroma v4 properly
        # while the docs include some new features
        candidates = db.similarity_search(query, filter={"source": {"$eq": file}})
        candidates = [
            c.page_content for c in candidates if c.metadata["dtype"] in dtypes
        ]
        if len(candidates) > 0:
            columns[file] = candidates[0]

    return columns


@st.cache_data
def reorganise_headers(data: pd.DataFrame):
    sheets_to_columns = defaultdict(list)

    for row in data.itertuples(index=False):
        source = row.metadatas["source"]
        if not source.endswith(".csv"):
            continue

        header = row.documents
        dtype = row.metadatas["dtype"]
        sheets_to_columns[source].append((header, dtype))

    return sheets_to_columns


@st.cache_data
def load_csv_files(pattern: str, sheets_to_columns: dict[str, list]):
    data_files = glob.glob(pattern, recursive=True)
    loaded_data = {}

    for file in data_files:
        columns = sheets_to_columns[file]
        csv_data = pd.read_csv(file, encoding="utf-8")

        for cname, ctype in columns:
            try:
                csv_data[cname] = csv_data[cname].astype(ctype)
            except (pd.errors.ParserError, TypeError, ValueError) as e:
                print(f"Couldn't convert column {cname} to expected type {ctype}: {e}")

        loaded_data[file] = csv_data

    return loaded_data
