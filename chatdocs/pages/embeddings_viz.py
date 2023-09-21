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

from chatdocs.config import get_config
from chatdocs.vectorstores import get_vectorstore


# Color scheme and sizing for plot markers
COLOR_PAPER_BG = "rgba(0, 0, 0, 0)"
COLOR_PLOT_BG = "rgba(0, 0, 0, 0)"
COLOR_DEFAULT = "rgb(42, 157, 143)"
COLOR_RELEVANT = "rgb(233, 196, 106)"
COLOR_HIGHLIGHT = "rgb(231, 111, 81)"
SIZE_DEFAULT = 5
SIZE_RELEVANT = 10
SIZE_HIGHLIGHT = 25


@st.cache_resource
def load_db(config):
    return get_vectorstore(config)


@st.cache_data
def load_db_data(config):
    db = load_db(config)
    data = db.get(include=["metadatas", "documents", "embeddings"])
    df = pd.DataFrame.from_dict(data)
    return df.set_index("ids")


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
    config_path = st.session_state.get("config_path", None)
    config = get_config(config_path)

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
        query = st.sidebar.text_input("Query (optional)")

        db_data = load_db_data(config)
        num_docs = len(db_data["documents"])
        sizes = [SIZE_DEFAULT] * num_docs
        colors = [COLOR_DEFAULT] * num_docs

        if query:
            embedded_query, relevant_ids, relevance_scores = process_query(
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

            for id, relevance_score in zip(relevant_ids, relevance_scores):
                db_data["documents"][id] += f"\n\nRelevance score: {relevance_score}"
                idx = db_data.index.get_loc(id)
                sizes[idx] = SIZE_RELEVANT
                colors[idx] = COLOR_RELEVANT

        labels = transform_labels(db_data["documents"])
        data = transform_embeddings(db_data["embeddings"], dim_reduction, n_components)

        scatter = go.Scatter3d if n_components == 3 else go.Scatter
        scatter_kwargs = dict(
            x=data[:, 0],
            y=data[:, 1],
            mode="markers",
            text=labels,
            marker=dict(color=colors, size=sizes),
        )
        if n_components == 3:
            scatter_kwargs["z"] = data[:, 2]

        fig = go.Figure(
            data=[scatter(**scatter_kwargs)],
            layout=go.Layout(paper_bgcolor=COLOR_PAPER_BG, plot_bgcolor=COLOR_PLOT_BG),
        )
        fig.update_layout(
            margin={"r": 50, "t": 100, "l": 0, "b": 0}, height=750, width=850
        )
        st.plotly_chart(fig)


main()