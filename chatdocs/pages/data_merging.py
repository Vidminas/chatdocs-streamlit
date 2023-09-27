from collections import defaultdict

import pandas as pd
import streamlit as st

from uuid import uuid4
from abc import ABC, abstractmethod
from streamlit_elements import dashboard, mui, elements
from contextlib import contextmanager
from types import SimpleNamespace

from streamlit import runtime

# allow relative imports when running with streamlit
if runtime.exists() and not __package__:
    from pathlib import Path

    __package__ = Path(__file__).parent.name

from chatdocs.st_utils import load_config, load_db, load_db_data


class Dashboard:
    DRAGGABLE_CLASS = "draggable"

    def __init__(self):
        self._layout = []

    def _register(self, item):
        self._layout.append(item)

    @contextmanager
    def __call__(self, **props):
        # Draggable classname query selector.
        props["draggableHandle"] = f".{Dashboard.DRAGGABLE_CLASS}"

        with dashboard.Grid(self._layout, **props):
            yield

    class Item(ABC):
        def __init__(self, board, x, y, w, h, **item_props):
            self._key = str(uuid4())
            self._draggable_class = Dashboard.DRAGGABLE_CLASS
            self._dark_mode = True
            board._register(dashboard.Item(self._key, x, y, w, h, **item_props))

        def _switch_theme(self):
            self._dark_mode = not self._dark_mode

        @contextmanager
        def title_bar(self, padding="5px 15px 5px 15px", dark_switcher=True):
            with mui.Stack(
                className=self._draggable_class,
                alignItems="center",
                direction="row",
                spacing=1,
                sx={
                    "padding": padding,
                    "borderBottom": 1,
                    "borderColor": "divider",
                },
            ):
                yield

                if dark_switcher:
                    if self._dark_mode:
                        mui.IconButton(mui.icon.DarkMode, onClick=self._switch_theme)
                    else:
                        mui.IconButton(mui.icon.LightMode, sx={"color": "#ffc107"}, onClick=self._switch_theme)

        @abstractmethod
        def __call__(self):
            """Show elements."""
            raise NotImplementedError
        

class DataGrid(Dashboard.Item):
    def _handle_edit(self, params):
        print(params)

    def __call__(self, title: str, data: list):
        with mui.Paper(key=self._key, sx={"display": "flex", "flexDirection": "column", "borderRadius": 3, "overflow": "hidden"}, elevation=1):
            with self.title_bar(padding="10px 15px 10px 15px", dark_switcher=False):
                mui.icon.ViewCompact()
                mui.Typography(title)

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                mui.DataGrid(
                    columns=[{ "field": "columns", "flex": 1 }],
                    rows=[{ "id": idx, "columns": elem } for idx, elem in enumerate(data)],
                    checkboxSelection=True,
                    disableSelectionOnClick=True,
                    onCellEditCommit=self._handle_edit,
                )


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


def main():
    config = load_config()
    data = load_db_data(config, include=["metadatas", "documents"])
    sheets_to_columns = reorganise_headers(data)

    st.dataframe(pd.DataFrame.from_dict(sheets_to_columns, orient="index").T)

    if "w" not in st.session_state:
        GRID_COLUMNS = 12
        ELEM_HEIGHT = 6
        ELEM_WIDTH = 4

        board = Dashboard()
        w = SimpleNamespace(
            dashboard=board,
            **{
                f"data_grid_{sheet}": DataGrid(board, (idx * ELEM_WIDTH) % GRID_COLUMNS, (idx * ELEM_HEIGHT) / GRID_COLUMNS, ELEM_WIDTH, ELEM_HEIGHT, minH=4)
                for idx, sheet in enumerate(sheets_to_columns)
            }
        )
        st.session_state.w = w
    else:
        w = st.session_state.w

    with elements("dashboard"):
        with w.dashboard(rowHeight=57):
            for sheet, sheet_data in sheets_to_columns.items():
                getattr(w, f"data_grid_{sheet}")(sheet, sheet_data)
            

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()