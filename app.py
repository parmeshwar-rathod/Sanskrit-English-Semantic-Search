
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sanskrit → English Semantic Search",
    page_icon="📜",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "parmeshwar-rathod/sanskrit-e5-finetuned"
DATASET_PATH = "./dataset_export.json"
METRICS_PATH = "./metrics_report.json"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["raw_corpus"]


# ============================================================
# LOAD METRICS
# ============================================================

@st.cache_data
def load_metrics():

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return {}


# ============================================================
# CREATE CORPUS EMBEDDINGS
# ============================================================

@st.cache_data
def create_corpus_embeddings(_model, corpus):

    texts = []

    for item in corpus:

        text = (
            f"Sanskrit: {item['sanskrit']} "
            f"IAST: {item['iast']} "
            f"English: {item['gloss']} "
            f"Topics: {' '.join(item['topics'])}"
        )

        texts.append("passage: " + text)

    embeddings = _model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    model,
    corpus,
    corpus_embeddings,
    top_k
):

    query_embedding = model.encode(
        "query: " + query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores = np.dot(
        corpus_embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append({
            "score": float(scores[idx]),
            "data": corpus[idx]
        })

    return results


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model = load_model()
    corpus = load_dataset()
    metrics = load_metrics()

    corpus_embeddings = create_corpus_embeddings(
        model,
        corpus
    )

except Exception as e:

    st.error(
        "There was a problem loading the model or project files."
    )

    st.exception(e)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📜 Sanskrit → English Semantic Search")

st.markdown(
    """
### Explore Sanskrit verses using semantic search

Search using **Sanskrit, IAST transliteration, or English**.
The application uses a fine-tuned multilingual E5
Sentence Transformer model to find semantically relevant verses.
"""
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📖 About")

    st.write(
        """
        This application performs semantic retrieval
        between Sanskrit verses and English meanings.
        """
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("Fine-tuned multilingual E5")

    st.subheader("📚 Corpus")

    st.write(f"{len(corpus)} Sanskrit verses")

    st.divider()

    st.subheader("📊 Model Performance")

    if metrics:

        finetuned_metrics = metrics.get(
            "finetuned_test",
            {}
        )

        st.metric(
            "Recall@1",
            f"{finetuned_metrics.get('recall@1', 0):.2%}"
        )

        st.metric(
            "Recall@3",
            f"{finetuned_metrics.get('recall@3', 0):.2%}"
        )

        st.metric(
            "Recall@5",
            f"{finetuned_metrics.get('recall@5', 0):.2%}"
        )

        st.metric(
            "NDCG@1",
            f"{finetuned_metrics.get('ndcg@1', 0):.2%}"
        )

        st.metric(
            "NDCG@3",
            f"{finetuned_metrics.get('ndcg@3', 0):.2%}"
        )

        st.metric(
            "MRR",
            f"{finetuned_metrics.get('mrr', 0):.2%}"
        )

    else:

        st.info(
            "Evaluation metrics are available in "
            "metrics_report.json."
        )


# ============================================================
# EXAMPLE QUERIES
# ============================================================

st.subheader("💡 Example Queries")

example_queries = [
    "How can a person perform their duty without worrying about results?",
    "यदा यदा हि धर्मस्य",
    "karmany-evadhikaras te ma phalesu kadacana",
    "How can I control a restless mind?",
    "What are the three gates to self destruction?"
]

selected_example = st.selectbox(
    "Choose an example:",
    ["-- Select an example --"] + example_queries
)


# ============================================================
# QUERY INPUT
# ============================================================

default_query = ""

if selected_example != "-- Select an example --":
    default_query = selected_example

query = st.text_area(
    "🔍 Enter your query",
    value=default_query,
    height=100,
    placeholder=(
        "Enter Sanskrit, IAST, or English query..."
    )
)


# ============================================================
# TOP-K
# ============================================================

top_k = st.selectbox(
    "Number of results",
    options=[1, 3, 5],
    index=1
)


# ============================================================
# SEARCH BUTTON
# ============================================================

search_clicked = st.button(
    "🔍 Search",
    type="primary",
    use_container_width=True
)


# ============================================================
# SEARCH RESULTS
# ============================================================

if search_clicked:

    if not query.strip():

        st.warning(
            "Please enter a query before searching."
        )

    else:

        with st.spinner(
            "Searching Sanskrit verses..."
        ):

            results = semantic_search(
                query,
                model,
                corpus,
                corpus_embeddings,
                top_k
            )

        st.success(
            f"Found {len(results)} relevant result(s)."
        )

        st.subheader("🔎 Search Results")

        for i, result in enumerate(results, start=1):

            item = result["data"]
            score = result["score"]

            with st.container(border=True):

                st.markdown(
                    f"### {i}. {item['id']}"
                )

                st.metric(
                    "Similarity Score",
                    f"{score:.4f}"
                )

                st.markdown("**Sanskrit**")

                st.write(
                    item["sanskrit"]
                )

                st.markdown("**IAST**")

                st.write(
                    item["iast"]
                )

                st.markdown("**English Meaning**")

                st.write(
                    item["gloss"]
                )

                st.markdown("**Related Topics**")

                st.write(
                    ", ".join(item["topics"])
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sanskrit → English Semantic Retrieval | "
    "Fine-tuned Multilingual E5 Model"
)