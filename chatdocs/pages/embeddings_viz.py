from operator import itemgetter

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import plotly.graph_objects as go

from streamlit import runtime

# allow relative imports when running with streamlit
if runtime.exists() and not __package__:
    from pathlib import Path

    __package__ = Path(__file__).parent.name

from chatdocs.st_utils import load_config, load_db, load_db_data


# Color scheme and sizing for plot markers
COLOR_PAPER_BG = "rgba(0, 0, 0, 0)"
COLOR_PLOT_BG = "rgba(0, 0, 0, 0)"
COLOR_DEFAULT = "rgb(42, 157, 143)"
COLOR_RELEVANT = "rgb(233, 196, 106)"
COLOR_HIGHLIGHT = "rgb(231, 111, 81)"
SIZE_DEFAULT = 5
SIZE_RELEVANT = 10
SIZE_HIGHLIGHT = 25


@st.cache_data
def transform_labels(labels: pd.Series):
    return labels.apply(lambda label: label.replace("\n", "<br>"))


@st.cache_data
def transform_embeddings(embeddings: pd.Series, dim_reduction: str, n_components: int):
    embedding_size = len(embeddings[0])
    data = embeddings.transform(
        {f"dim{i+1}": itemgetter(i) for i in range(embedding_size)}
    )
    data = StandardScaler().fit_transform(data)
    if dim_reduction == "PCA":
        return PCA(n_components).fit_transform(data)
    elif dim_reduction == "TruncatedSVD":
        return TruncatedSVD(n_components).fit_transform(data)
    elif dim_reduction == "t-SNE":
        # Suggestion from sklearn TSNE docs to reduce noise and speed up computation
        if embedding_size > 50:
            data = PCA(n_components=50).fit_transform(data)
        return TSNE(n_components).fit_transform(data)
    return None


@st.cache_data
def process_query(config: dict, query: str):
    db = load_db(config)
    embedded_query = db.embeddings.embed_query(query)

    results = db._collection.query(
        query_embeddings=embedded_query,
        n_results=4,
        include=["distances"],
    )

    return embedded_query, results["ids"][0], results["distances"][0]


def main():
    config = load_config()

    st.sidebar.title("Options")
    view_db = st.sidebar.checkbox("View DB", value=True)
    viz_embeds = st.sidebar.checkbox("Visualize embeddings", value=False)

    if view_db:
        st.dataframe(load_db_data(config))

    if viz_embeds:
        dim_reduction = st.sidebar.selectbox(
            "Dimensionality reduction algorithm", ("PCA", "TruncatedSVD", "t-SNE")
        )
        n_components = st.sidebar.radio(
            "Dimensions", (2, 3), format_func=lambda dim: f"{dim}D"
        )
        show_labels = st.sidebar.checkbox("Show labels", value=False)
        query = st.sidebar.text_input("Query (optional)")

        db_data = load_db_data(config)
        num_docs = len(db_data["documents"])
        sizes = [SIZE_DEFAULT] * num_docs
        colors = [COLOR_DEFAULT] * num_docs

        if query:
            embedded_query, relevant_ids, distances = process_query(
                config, query
            )
            query_data_row = pd.DataFrame.from_dict(
                {
                    "ids": "query",
                    "documents": query,
                    "embeddings": [embedded_query],
                    "metadatas": None,
                }
            ).set_index("ids")
            db_data = pd.concat((db_data, query_data_row))
            sizes.append(SIZE_HIGHLIGHT)
            colors.append(COLOR_HIGHLIGHT)

            for id, distance in zip(relevant_ids, distances):
                db_data["documents"][id] += f"\n\nDistance: {distance}"
                idx = db_data.index.get_loc(id)
                sizes[idx] = SIZE_RELEVANT
                colors[idx] = COLOR_RELEVANT

        labels = transform_labels(db_data["documents"])
        data = transform_embeddings(db_data["embeddings"], dim_reduction, n_components)

        scatter_kwargs = dict(
            x=data[:, 0],
            y=data[:, 1],
            mode="markers" if not show_labels else "markers+text",
            text=labels,
            marker=dict(color=colors, size=sizes),
        )
        scatter = None
        if n_components == 2:
            scatter = go.Scatter(**scatter_kwargs)
        elif n_components == 3:
            scatter = go.Scatter3d(z=data[:, 2], **scatter_kwargs)

        fig = go.Figure(
            data=[scatter],
            layout=go.Layout(paper_bgcolor=COLOR_PAPER_BG, plot_bgcolor=COLOR_PLOT_BG),
        )
        fig.update_layout(
            margin={"r": 50, "t": 100, "l": 0, "b": 0}, height=750, width=850
        )
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
