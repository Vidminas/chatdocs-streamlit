import os

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

import streamlit as st
from streamlit import runtime

# allow relative imports when running with streamlit
if runtime.exists() and not __package__:
    from pathlib import Path

    __package__ = Path(__file__).parent.name

from chatdocs.st_utils import (
    load_config,
    load_db_data,
    reorganise_headers,
    load_csv_files,
)


@st.cache_resource
def gen_profile_report(*report_args, **report_kwargs):
    return ProfileReport(*report_args, **report_kwargs)


def main():
    config = load_config()
    data = load_db_data(config)
    sheets_to_columns = reorganise_headers(data)
    files = list(sheets_to_columns.keys())

    dir = st.sidebar.text_input("Data directory", value="Frankenstein Drivers")
    loaded_data = load_csv_files(os.path.join(dir, "**/*.csv"), sheets_to_columns)

    selected_file = st.selectbox(label="Select data file to profile", options=files, format_func=lambda file: os.path.basename(file))
    limit = st.sidebar.number_input("Row limit per-file", value=1000)
    dataset = loaded_data[selected_file].head(limit)
    report = gen_profile_report(dataset, title="Profiling Report", explorative=True)
    st_profile_report(report)


if __name__ == "__main__":
    main()