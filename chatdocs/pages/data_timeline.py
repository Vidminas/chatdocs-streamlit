import os
import pprint

import pandas as pd
import streamlit as st
from streamlit import runtime
from streamlit_timeline import timeline

# allow relative imports when running with streamlit
if runtime.exists() and not __package__:
    from pathlib import Path

    __package__ = Path(__file__).parent.name

from chatdocs.st_utils import load_config, load_db_data, best_columns_for, load_csv_files
from .data_merging import reorganise_headers


@st.cache_data
def make_timeline_json(
    sheets_to_columns, datetime_columns, data: dict[str, pd.DataFrame], limit
):
    items = []
    for file in sheets_to_columns:
        if file in datetime_columns:
            for row in data[file].itertuples():
                content = pprint.pformat(row._asdict()).replace("\n", "<br>")

                date = getattr(row, datetime_columns[file])
                if not isinstance(date, pd.Timestamp):
                    try:
                        date = pd.to_datetime(date, infer_datetime_format=True, errors="raise")
                    except (pd.errors.ParserError, TypeError, ValueError):
                        continue

                # https://timeline.knightlab.com/docs/json-format.html
                items.append(
                    {
                        "unique_id": f"{file}-{row.Index}",
                        "text": {
                            "headline": f"Event ({os.path.basename(file)})",
                            "text": content,
                        },
                        "start_date": {
                            "year": date.year,
                            "month": date.month,
                            "day": date.day,
                        },
                    }
                )

                # Limit to avoid running out of memory or crashing browser
                if row.Index > limit:
                    break

    return {"events": items}


def main():
    config = load_config()
    data = load_db_data(config)
    sheets_to_columns = reorganise_headers(data)

    dir = st.sidebar.text_input("Data directory", value="Frankenstein Drivers")
    loaded_data = load_csv_files(os.path.join(dir, "**/*.csv"), sheets_to_columns)

    datetime_query = st.sidebar.text_input("Datetime header query", value="datetime")
    datetime_columns = best_columns_for(config, datetime_query, list(sheets_to_columns.keys()), ("datetime64[ns]", "datetime64[ns, UTC]"))

    st.dataframe(datetime_columns)

    limit = st.sidebar.number_input("Item limit per-file", value=100)
    json = make_timeline_json(sheets_to_columns, datetime_columns, loaded_data, limit)
    timeline(json, height=600)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
