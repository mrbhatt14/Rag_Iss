import os
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

# Keep local development output focused on the app instead of ChromaDB telemetry.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "uploads" / "data.xlsx"
DEFAULT_GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1nwSzIkL-8Dmatx-dIS5zETsgZ4GPzaOJXguVDRqNbgQ/edit?usp=sharing"
)
GOOGLE_SHEET_CSV_URL = os.environ.get(
    "GOOGLE_SHEET_CSV_URL",
    DEFAULT_GOOGLE_SHEET_URL,
).strip()
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION_NAME = "excel_rows"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CANDIDATE_LIMIT = 5
RESULT_LIMIT = 5
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
data_source_signature: Optional[str] = None
embedding_model: Any = None
chroma_client: Any = None


def get_embedding_model() -> Any:
    """Load the embedding model lazily to reduce server startup memory usage."""
    global embedding_model

    if embedding_model is None:
        from sentence_transformers import SentenceTransformer

        try:
            import torch

            torch.set_num_threads(1)
        except Exception:
            pass

        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    return embedding_model


def get_chroma_client() -> Any:
    """Create the ChromaDB client lazily so Gunicorn can bind its port first."""
    global chroma_client

    if chroma_client is None:
        import chromadb
        from chromadb.config import Settings

        chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

    return chroma_client


def row_to_searchable_text(row: pd.Series) -> str:
    """Convert one Excel row into readable text for embedding and display."""
    parts = []

    for column_name, value in row.items():
        if pd.notna(value):
            parts.append(f"{column_name}: {value}")

    return " | ".join(parts)


def get_data_source_name() -> str:
    """Return the configured spreadsheet source name for status messages."""
    return "Google Sheet" if GOOGLE_SHEET_CSV_URL else "Excel file"


def google_sheet_url_to_csv_url(url: str) -> str:
    """Convert a normal Google Sheets URL into a CSV export URL when possible."""
    parsed_url = urlparse(url)

    if "docs.google.com" not in parsed_url.netloc:
        return url

    match = re.search(r"/spreadsheets/d/([^/]+)", parsed_url.path)

    if not match:
        return url

    spreadsheet_id = match.group(1)
    query_params = parse_qs(parsed_url.query)
    gid = query_params.get("gid", ["0"])[0]

    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        f"/export?format=csv&gid={gid}"
    )


def dataframe_signature(dataframe: pd.DataFrame) -> str:
    """Build a lightweight signature to detect spreadsheet content changes."""
    normalized = dataframe.fillna("").astype(str)
    row_hash_total = pd.util.hash_pandas_object(normalized, index=True).sum()
    columns = "|".join(str(column) for column in normalized.columns)

    return f"{columns}:{row_hash_total}:{len(normalized)}"


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


def load_spreadsheet_dataframe() -> pd.DataFrame:
    """Read data from Google Sheets CSV when configured, otherwise Excel."""
    if GOOGLE_SHEET_CSV_URL:
        csv_url = google_sheet_url_to_csv_url(GOOGLE_SHEET_CSV_URL)
        return pd.read_csv(csv_url)

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel file was not found at {EXCEL_PATH}. Add your file as uploads/data.xlsx."
        )

    return pd.read_excel(EXCEL_PATH)


def dataframe_to_rows(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a Pandas DataFrame into ChromaDB-ready records."""

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


def load_spreadsheet_rows() -> tuple[list[dict[str, Any]], str]:
    """Load spreadsheet rows and return rows plus a content signature."""
    dataframe = load_spreadsheet_dataframe()

    return dataframe_to_rows(dataframe), dataframe_signature(dataframe)


def rebuild_collection(rows: list[dict[str, Any]]) -> None:
    """Create a fresh ChromaDB collection from the current Excel rows."""
    client = get_chroma_client()

    existing_collections = {
        collection.name for collection in client.list_collections()
    }

    if COLLECTION_NAME in existing_collections:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    if not rows:
        return

    documents = [row["text"] for row in rows]
    ids = [row["id"] for row in rows]
    metadatas = [row["metadata"] for row in rows]
    embeddings = get_embedding_model().encode(
        documents,
        batch_size=16,
        show_progress_bar=False,
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


vector_store_ready = True
startup_message = "Spreadsheet data will load on first search."
last_excel_modified_at = EXCEL_PATH.stat().st_mtime if EXCEL_PATH.exists() else None


def refresh_vector_store_if_needed() -> tuple[bool, str]:
    """Rebuild ChromaDB when the spreadsheet changes while Flask is running."""
    global data_source_signature, excel_rows_cache
    global last_excel_modified_at, startup_message, vector_store_ready

    if GOOGLE_SHEET_CSV_URL:
        rows, signature = load_spreadsheet_rows()

        if signature == data_source_signature:
            return vector_store_ready, startup_message

        data_source_signature = signature
        excel_rows_cache = rows
        rebuild_collection(rows)

        if not rows:
            vector_store_ready = False
            startup_message = "The Google Sheet exists, but it does not contain any rows."
        else:
            vector_store_ready = True
            startup_message = f"The Google Sheet has {len(rows)} rows."

        return vector_store_ready, startup_message

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

    if current_modified_at == last_excel_modified_at and data_source_signature is not None:
        return vector_store_ready, startup_message

    rows, signature = load_spreadsheet_rows()
    data_source_signature = signature
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
    try:
        ready, message = refresh_vector_store_if_needed()
    except Exception as error:
        message = f"Could not refresh the spreadsheet data: {error}"

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

        collection = get_chroma_client().get_collection(name=COLLECTION_NAME)
        query_embedding = get_embedding_model().encode(
            [query],
            show_progress_bar=False,
        ).tolist()[0]

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
