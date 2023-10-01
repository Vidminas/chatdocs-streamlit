import glob
import os

import pandas as pd
import streamlit as st
from streamlit import runtime

# allow relative imports when running with streamlit
if runtime.exists() and not __package__:
    from pathlib import Path

    __package__ = Path(__file__).parent.name

from chatdocs.st_utils import load_config, load_db_data, best_columns_for
from .data_merging import reorganise_headers


@st.cache_data
def load_csv_files(dir: str, sheets_to_columns: dict[str, list]):
    data_files = glob.glob(os.path.join(dir, "**/*.csv"), recursive=True)
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


@st.cache_data
def make_lat_lon_df(
    files, lat_columns, lon_columns, data: dict[str, pd.DataFrame]
):
    latlonlist = []

    for file in files:
        if file in lat_columns and file in lon_columns:
            latlons = data[file][[lat_columns[file], lon_columns[file]]].dropna()
            latlons.columns = ["lat", "lon"]
            latlonlist.append(latlons)

    return pd.concat(latlonlist, axis=0, copy=False, ignore_index=True, sort=False)


def main():
    config = load_config()
    data = load_db_data(config)
    sheets_to_columns = reorganise_headers(data)
    files = list(sheets_to_columns.keys())

    dir = st.sidebar.text_input("Data directory", value="Frankenstein Drivers")
    loaded_data = load_csv_files(dir, sheets_to_columns)

    lat_query = st.sidebar.text_input("Latitude header query", value="lat")
    lon_query = st.sidebar.text_input("Longitude header query", value="lon")
    lat_columns = best_columns_for(config, lat_query, files, ("Float64"))
    lon_columns = best_columns_for(config, lon_query, files, ("Float64"))

    st.dataframe((lat_columns, lon_columns))

    st.map(make_lat_lon_df(files, lat_columns, lon_columns, loaded_data))


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
