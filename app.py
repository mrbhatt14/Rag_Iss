import os
import re
from pathlib import Path
from typing import Any

# Keep local development output focused on the app instead of ChromaDB telemetry.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
import pandas as pd
from chromadb.config import Settings
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "uploads" / "data.xlsx"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "excel_rows"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CANDIDATE_LIMIT = 5
RESULT_LIMIT = 3
MIN_RELEVANCE_SCORE = 0.45
MIN_TOKEN_LENGTH = 3

STOP_WORDS = {
    "all",
    "and",
    "are",
    "data",
    "did",
    "does",
    "doing",
    "excel",
    "file",
    "for",
    "from",
    "get",
    "give",
    "how",
    "in",
    "into",
    "is",
    "me",
    "only",
    "record",
    "records",
    "result",
    "results",
    "row",
    "rows",
    "show",
    "tell",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


app = Flask(__name__)
excel_rows_cache: list[dict[str, Any]] = []

# Load shared services once when Flask starts. The first run may download the
# sentence-transformers model if it is not already cached on your machine.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)


def row_to_searchable_text(row: pd.Series) -> str:
    """Convert one Excel row into readable text for embedding and display."""
    parts = []

    for column_name, value in row.items():
        if pd.notna(value):
            parts.append(f"{column_name}: {value}")

    return " | ".join(parts)


def extract_search_tokens(text: str) -> set[str]:
    """Extract meaningful tokens for exact matching against Excel row values."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())

    return {
        token
        for token in tokens
        if len(token) >= MIN_TOKEN_LENGTH and token not in STOP_WORDS
    }


def metadata_contains_query_token(metadata: dict[str, Any], query_tokens: set[str]) -> bool:
    """Return True when a row contains a meaningful query token exactly."""
    return bool(get_matching_tokens(metadata, query_tokens))


def get_matching_tokens(metadata: dict[str, Any], query_tokens: set[str]) -> set[str]:
    """Find meaningful query tokens that appear in one Excel row."""
    row_text = " ".join(str(value) for value in metadata.values()).lower()
    row_tokens = set(re.findall(r"[a-z0-9]+", row_text))

    return query_tokens.intersection(row_tokens)


def search_exact_rows(query_tokens: set[str]) -> list[dict[str, Any]]:
    """Return rows matching exact spreadsheet values before semantic fallback."""
    if not query_tokens:
        return []

    exact_results = []

    for row in excel_rows_cache:
        matched_tokens = get_matching_tokens(row["metadata"], query_tokens)

        if not matched_tokens:
            continue

        exact_results.append(
            {
                "text": row["text"],
                "row": row["metadata"],
                "score": 1.0,
                "matched_token_count": len(matched_tokens),
            }
        )

    exact_results.sort(key=lambda result: result["matched_token_count"], reverse=True)

    for result in exact_results:
        result.pop("matched_token_count", None)

    return exact_results


def load_excel_rows() -> list[dict[str, Any]]:
    """Read the Excel file and convert rows into ChromaDB-ready records."""
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel file was not found at {EXCEL_PATH}. Add your file as uploads/data.xlsx."
        )

    dataframe = pd.read_excel(EXCEL_PATH)

    if dataframe.empty:
        return []

    dataframe = dataframe.fillna("")
    records: list[dict[str, Any]] = []

    for index, row in dataframe.iterrows():
        searchable_text = row_to_searchable_text(row)
        row_data = {str(key): str(value) for key, value in row.to_dict().items()}

        records.append(
            {
                "id": f"row-{index}",
                "text": searchable_text,
                "metadata": row_data,
            }
        )

    return records


def rebuild_collection(rows: list[dict[str, Any]]) -> None:
    """Create a fresh ChromaDB collection from the current Excel rows."""
    existing_collections = {
        collection.name for collection in chroma_client.list_collections()
    }

    if COLLECTION_NAME in existing_collections:
        chroma_client.delete_collection(name=COLLECTION_NAME)

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    if not rows:
        return

    documents = [row["text"] for row in rows]
    ids = [row["id"] for row in rows]
    metadatas = [row["metadata"] for row in rows]
    embeddings = embedding_model.encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def initialize_vector_store() -> tuple[bool, str]:
    """Load Excel data and prepare ChromaDB before handling searches."""
    global excel_rows_cache

    try:
        rows = load_excel_rows()
        excel_rows_cache = rows
        rebuild_collection(rows)

        if not rows:
            return False, "The Excel file exists, but it does not contain any rows."

        return True, f"The Excel file has {len(rows)} rows."
    except FileNotFoundError as error:
        return False, str(error)
    except Exception as error:
        return False, f"Could not initialize the vector store: {error}"


vector_store_ready, startup_message = initialize_vector_store()
last_excel_modified_at = EXCEL_PATH.stat().st_mtime if EXCEL_PATH.exists() else None


def refresh_vector_store_if_needed() -> tuple[bool, str]:
    """Rebuild ChromaDB when uploads/data.xlsx changes while Flask is running."""
    global excel_rows_cache, last_excel_modified_at, startup_message, vector_store_ready

    if not EXCEL_PATH.exists():
        excel_rows_cache = []
        vector_store_ready = False
        startup_message = (
            f"Excel file was not found at {EXCEL_PATH}. "
            "Add your file as uploads/data.xlsx."
        )
        last_excel_modified_at = None
        return vector_store_ready, startup_message

    current_modified_at = EXCEL_PATH.stat().st_mtime

    if current_modified_at == last_excel_modified_at:
        return vector_store_ready, startup_message

    rows = load_excel_rows()
    excel_rows_cache = rows
    rebuild_collection(rows)
    last_excel_modified_at = current_modified_at

    if not rows:
        vector_store_ready = False
        startup_message = "The Excel file exists, but it does not contain any rows."
    else:
        vector_store_ready = True
        startup_message = f"The Excel file has {len(rows)} rows."

    return vector_store_ready, startup_message


@app.route("/")
def homepage():
    """Render the retrieval search page."""
    return render_template(
        "index.html",
        vector_store_ready=vector_store_ready,
        startup_message=startup_message,
    )


@app.route("/search", methods=["POST"])
def search():
    """Receive a query, run semantic search, and return relevant Excel rows."""
    ready, message = refresh_vector_store_if_needed()

    if not ready:
        return (
            jsonify(
                {
                    "error": message,
                    "results": [],
                    "status": message,
                    "status_ready": False,
                }
            ),
            500,
        )

    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    query_tokens = extract_search_tokens(query)

    if not query:
        return (
            jsonify(
                {
                    "error": "Please enter a search query.",
                    "results": [],
                    "status": message,
                    "status_ready": ready,
                }
            ),
            400,
        )

    try:
        exact_results = search_exact_rows(query_tokens)

        if exact_results:
            return jsonify(
                {
                    "results": exact_results,
                    "status": message,
                    "status_ready": ready,
                }
            )

        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        query_embedding = embedding_model.encode([query]).tolist()[0]

        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=CANDIDATE_LIMIT,
            include=["documents", "metadatas", "distances"],
        )

        documents = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        distances = search_results.get("distances", [[]])[0]

        if not documents:
            return jsonify(
                {
                    "message": "No matching results were found.",
                    "results": [],
                    "status": message,
                    "status_ready": ready,
                }
            )

        results = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            # ChromaDB returns the nearest rows even when they are weak matches.
            # This score keeps the UI focused on rows that are actually relevant.
            score = round(1 / (1 + float(distance)), 4)
            has_exact_match = metadata_contains_query_token(metadata, query_tokens)

            if score < MIN_RELEVANCE_SCORE and not has_exact_match:
                continue

            results.append(
                {
                    "text": document,
                    "row": metadata,
                    "score": score,
                    "has_exact_match": has_exact_match,
                }
            )

        exact_results = [result for result in results if result["has_exact_match"]]

        if exact_results:
            results = exact_results

        results = results[:RESULT_LIMIT]

        for result in results:
            result.pop("has_exact_match", None)

        if not results:
            return jsonify(
                {
                    "message": "No matching results were found.",
                    "results": [],
                    "status": message,
                    "status_ready": ready,
                }
            )

        return jsonify(
            {
                "results": results,
                "status": message,
                "status_ready": ready,
            }
        )
    except Exception as error:
        return (
            jsonify(
                {
                    "error": f"Search failed: {error}",
                    "results": [],
                    "status": startup_message,
                    "status_ready": vector_store_ready,
                }
            ),
            500,
        )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=True)
