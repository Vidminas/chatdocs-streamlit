from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_chunked

from uuid import uuid4
from abc import ABC, abstractmethod
from streamlit_elements import dashboard, mui, elements, sync
from contextlib import contextmanager
from types import SimpleNamespace

import streamlit as st
from streamlit import runtime

# allow relative imports when running with streamlit
if runtime.exists() and not __package__:
    from pathlib import Path

    __package__ = Path(__file__).parent.name

from chatdocs.st_utils import load_config, load_db_data


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
                        mui.IconButton(
                            mui.icon.LightMode,
                            sx={"color": "#ffc107"},
                            onClick=self._switch_theme,
                        )

        @abstractmethod
        def __call__(self):
            """Show elements."""
            raise NotImplementedError


class DataGrid(Dashboard.Item):
    def _handle_edit(self, params):
        print(params)

    def __call__(
        self, title: str, data: list, *, checkboxSelection=True, selectionStateKey=None
    ):
        with mui.Paper(
            key=self._key,
            sx={
                "display": "flex",
                "flexDirection": "column",
                "borderRadius": 3,
                "overflow": "hidden",
            },
            elevation=1,
        ):
            with self.title_bar(padding="10px 15px 10px 15px", dark_switcher=False):
                mui.icon.ViewCompact()
                mui.Typography(title)

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                self.columns = [
                    {"field": f"column {column}", "flex": 1}
                    for column in range(len(data[0]))
                ]
                self.rows = [
                    {
                        "id": idx,
                        **{
                            f"column {column}": value
                            for (column, value) in enumerate(elem)
                        },
                    }
                    for (idx, elem) in enumerate(data)
                ]

                selection_kwargs = (
                    {}
                    if selectionStateKey is None
                    else {
                        "selectionModel": st.session_state.get(selectionStateKey, []),
                        "onSelectionModelChange": sync(selectionStateKey),
                    }
                )

                mui.DataGrid(
                    columns=self.columns,
                    rows=self.rows,
                    checkboxSelection=checkboxSelection,
                    disableSelectionOnClick=True,
                    onCellEditCommit=self._handle_edit,
                    **selection_kwargs,
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


@st.cache_data
def suggested_merges(threshold: float, data: pd.DataFrame):
    suggestions = []

    for distances in pairwise_distances_chunked(
        np.vstack(data["embeddings"]), metric="cosine", n_jobs=-1
    ):
        candidate_pairs = np.argwhere(distances < threshold)

        for c1_idx, c2_idx in candidate_pairs:
            if c1_idx == c2_idx:
                continue

            c1 = data.iloc[c1_idx]
            c2 = data.iloc[c2_idx]
            if c1["metadatas"]["source"] == c2["metadatas"]["source"]:
                continue
            if not c1["metadatas"]["source"].endswith(".csv") or not c2["metadatas"][
                "source"
            ].endswith(".csv"):
                continue
            if c1["metadatas"]["dtype"] != c2["metadatas"]["dtype"]:
                continue
            suggestions.append((c1, c2))
        break

    return suggestions


@dataclass
class CSVHeader:
    name: str
    source: str


def save_merges(datagrid):
    selected_rows = st.session_state.get("selected_rows", [])
    merges = []

    for row in selected_rows:
        data = datagrid.rows[row]
        merges.append(
            (
                CSVHeader(data["column 0"], data["column 1"]),
                CSVHeader(data["column 2"], data["column 3"]),
            )
        )

    st.session_state["merges"] = merges


def main():
    config = load_config()
    data = load_db_data(config)
    sheets_to_columns = reorganise_headers(data)

    st.header("All spreadsheet columns")
    st.dataframe(pd.DataFrame.from_dict(sheets_to_columns, orient="index").T)

    if "w1" not in st.session_state:
        GRID_COLUMNS = 12
        ELEM_HEIGHT = 6
        ELEM_WIDTH = 4

        board1 = Dashboard()
        w1 = SimpleNamespace(
            dashboard=board1,
            suggested_merges=DataGrid(
                board1, 0, 0, GRID_COLUMNS, ELEM_HEIGHT, isDraggable=False
            ),
        )
        st.session_state.w1 = w1

        board2 = Dashboard()
        w2 = SimpleNamespace(
            dashboard=board2,
            **{
                f"data_grid_{sheet}": DataGrid(
                    board2,
                    (idx * ELEM_WIDTH) % GRID_COLUMNS,
                    (idx * ELEM_HEIGHT) / GRID_COLUMNS,
                    ELEM_WIDTH,
                    ELEM_HEIGHT,
                    minH=4,
                )
                for idx, sheet in enumerate(sheets_to_columns)
            },
        )
        st.session_state.w2 = w2

    else:
        w1 = st.session_state.w1
        w2 = st.session_state.w2

    with elements("dashboard"):
        with w1.dashboard(rowHeight=57):
            threshold = st.sidebar.slider(
                "Header similarity threshold", value=0.02, min_value=0.0, max_value=0.1
            )
            merge_rows = [
                (
                    c1["documents"],
                    c1["metadatas"]["source"],
                    c2["documents"],
                    c2["metadatas"]["source"],
                )
                for c1, c2 in suggested_merges(threshold, data)
            ]
            w1.suggested_merges(
                "Suggested column merges", merge_rows, selectionStateKey="selected_rows"
            )
            mui.Button(
                "Save merges",
                isDraggable=False,
                onClick=lambda: save_merges(w1.suggested_merges),
            )

        with w2.dashboard(rowHeight=57):
            for sheet, sheet_data in sheets_to_columns.items():
                getattr(w2, f"data_grid_{sheet}")(
                    sheet, sheet_data, checkboxSelection=False
                )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
