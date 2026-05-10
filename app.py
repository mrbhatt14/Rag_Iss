import os
import re
import time
import math
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
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "uploads" / "data.xlsx"
DOCUMENT_UPLOAD_DIR = Path(os.environ.get("DOCUMENT_UPLOAD_DIR", BASE_DIR / "uploaded_documents"))
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
DOCUMENT_RESULT_LIMIT = 3
MIN_RELEVANCE_SCORE = 0.45
DOCUMENT_MIN_RELEVANCE_SCORE = 0.0
MIN_TOKEN_LENGTH = 3
ALLOWED_DOCUMENT_EXTENSIONS = {".csv", ".xlsx", ".txt", ".md", ".pdf"}

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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
excel_rows_cache: list[dict[str, Any]] = []
data_source_signature: Optional[str] = None
uploaded_document_signature: Optional[str] = None
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
    if has_uploaded_documents():
        return "uploaded documents"

    return "Google Sheet" if GOOGLE_SHEET_CSV_URL else "Excel file"


def has_uploaded_documents() -> bool:
    """Return True when users have uploaded supported documents."""
    return bool(get_uploaded_document_paths())


def get_uploaded_document_paths() -> list[Path]:
    """List supported uploaded documents."""
    if not DOCUMENT_UPLOAD_DIR.exists():
        return []

    return sorted(
        path
        for path in DOCUMENT_UPLOAD_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_DOCUMENT_EXTENSIONS
    )


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


def uploaded_documents_signature() -> str:
    """Build a signature from uploaded document names, sizes, and timestamps."""
    parts = []

    for path in get_uploaded_document_paths():
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")

    return "|".join(parts)


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


def split_answer_candidates(text: str) -> list[str]:
    """Split document text into short answer-like candidates."""
    heading_pattern = (
        r"\b(PROFILE|SUMMARY|EXPERIENCE|EDUCATION|PROJECTS|SKILLS|"
        r"CERTIFICATIONS|ACHIEVEMENTS|LEADERSHIP|ACTIVITIES)\b"
    )
    text_with_breaks = re.sub(heading_pattern, r". \1 ", text)
    pieces = re.split(r"(?<=[.!?])\s+", text_with_breaks)

    return [
        re.sub(r"\s+", " ", piece).strip(" .")
        for piece in pieces
        if len(piece.strip()) >= 25
    ]


def shorten_answer_text(text: str, query_tokens: set[str], limit: int = 320) -> str:
    """Keep direct answers readable when a matched resume section is long."""
    clean_text = re.sub(r"\s+", " ", text).strip()

    if len(clean_text) <= limit:
        return clean_text

    lower_text = clean_text.lower()
    token_positions = [
        lower_text.find(token)
        for token in query_tokens
        if lower_text.find(token) >= 0
    ]
    first_match = min(token_positions) if token_positions else 0
    start = max(0, first_match - 80)
    end = min(len(clean_text), start + limit)
    shortened = clean_text[start:end].strip()

    if start > 0:
        shortened = f"...{shortened}"

    if end < len(clean_text):
        shortened = f"{shortened}..."

    return shortened


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compare two embedding vectors without adding another dependency."""
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if not left_norm or not right_norm:
        return 0.0

    return dot_product / (left_norm * right_norm)


def extract_best_text_answer(
    query: str,
    results: list[dict[str, Any]],
    query_embedding: Optional[list[float]] = None,
) -> Optional[dict[str, str]]:
    """Return the best sentence-level answer using semantic similarity."""
    query_tokens = extract_search_tokens(query)

    if not query_tokens:
        return None

    candidates = []

    for result_index, result in enumerate(results):
        for candidate in split_answer_candidates(result["text"]):
            candidates.append(
                {
                    "text": candidate,
                    "source": str(result.get("row", {}).get("Source", "")),
                    "result_index": result_index,
                }
            )

    if not candidates:
        return None

    if query_embedding is None:
        query_embedding = get_embedding_model().encode(
            [query],
            show_progress_bar=False,
        ).tolist()[0]

    candidate_embeddings = get_embedding_model().encode(
        [candidate["text"] for candidate in candidates],
        show_progress_bar=False,
    ).tolist()

    best_candidate = None

    for candidate, candidate_embedding in zip(candidates, candidate_embeddings):
        score = cosine_similarity(query_embedding, candidate_embedding)
        score -= candidate["result_index"] * 0.01

        if not best_candidate or score > best_candidate["score"]:
            best_candidate = {
                "score": score,
                "text": candidate["text"],
                "source": candidate["source"],
            }

    if not best_candidate:
        return None

    return {
        "label": "Answer",
        "value": shorten_answer_text(best_candidate["text"], query_tokens),
        "source": best_candidate["source"],
    }


def extract_direct_answer(
    query: str,
    results: list[dict[str, Any]],
    query_embedding: Optional[list[float]] = None,
) -> Optional[dict[str, str]]:
    """Extract common direct answers from retrieved document chunks."""
    query_text = query.lower()
    extractors = [
        {
            "terms": {"email", "mail", "e-mail"},
            "label": "Email",
            "patterns": [r"[\w.\-+]+@[\w.\-]+\.\w+"],
        },
        {
            "terms": {"phone", "mobile", "contact", "number"},
            "label": "Phone",
            "patterns": [r"(?:\+?\d[\d\s().-]{8,}\d)"],
        },
        {
            "terms": {"name", "candidate"},
            "label": "Name",
            "patterns": [r"\b[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){1,3}\b"],
        },
        {
            "terms": {"university", "college", "education", "study", "studies", "studying", "school"},
            "label": "University",
            "patterns": [
                r"\bat\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+University)\b",
                r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+University)\b",
                r"\b(University of [A-Z][A-Za-z]+(?:\s+(?:and|And|&|[A-Z][A-Za-z]+)){0,6})\b",
            ],
        },
    ]

    for extractor in extractors:
        if not any(term in query_text for term in extractor["terms"]):
            continue

        for result in results:
            match = None

            for pattern in extractor["patterns"]:
                match = re.search(pattern, result["text"])

                if match:
                    break

            if not match:
                continue

            metadata = result.get("row", {})
            value = match.group(1) if match.lastindex else match.group(0)

            return {
                "label": extractor["label"],
                "value": value.strip(),
                "source": str(metadata.get("Source", "")),
            }

    return extract_best_text_answer(query, results, query_embedding)


def load_spreadsheet_dataframe() -> pd.DataFrame:
    """Read data from Google Sheets CSV when configured, otherwise Excel."""
    if GOOGLE_SHEET_CSV_URL:
        csv_url = google_sheet_url_to_csv_url(GOOGLE_SHEET_CSV_URL)
        separator = "&" if "?" in csv_url else "?"
        fresh_csv_url = f"{csv_url}{separator}_cache_bust={time.time_ns()}"

        return pd.read_csv(fresh_csv_url)

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel file was not found at {EXCEL_PATH}. Add your file as uploads/data.xlsx."
        )

    return pd.read_excel(EXCEL_PATH)


def text_to_document_rows(text: str, source_name: str) -> list[dict[str, Any]]:
    """Split plain document text into searchable chunks."""
    cleaned_text = re.sub(r"\s+", " ", text).strip()

    if not cleaned_text:
        return []

    chunk_size = 900
    overlap_words = 25
    words = cleaned_text.split()
    rows = []
    chunk_words = []
    chunk_index = 0

    for word in words:
        next_chunk = " ".join([*chunk_words, word])

        if chunk_words and len(next_chunk) > chunk_size:
            chunk = " ".join(chunk_words)
            rows.append(
                build_document_row(source_name, chunk_index, chunk)
            )

            chunk_index += 1
            chunk_words = chunk_words[-overlap_words:]

        chunk_words.append(word)

    if chunk_words:
        chunk = " ".join(chunk_words)
        rows.append(build_document_row(source_name, chunk_index, chunk))

    return rows


def build_document_row(source_name: str, chunk_index: int, chunk: str) -> dict[str, Any]:
    """Create one searchable document chunk with display metadata."""
    return {
        "id": f"{source_name}-{chunk_index}",
        "text": chunk,
        "metadata": {
            "Source": source_name,
            "Type": "document",
            "Chunk": str(chunk_index + 1),
            "Content": chunk,
        },
    }


def dataframe_to_document_rows(dataframe: pd.DataFrame, source_name: str) -> list[dict[str, Any]]:
    """Convert table-like uploaded documents into searchable rows."""
    rows = dataframe_to_rows(dataframe)

    for index, row in enumerate(rows):
        row["id"] = f"{source_name}-{index}"
        row["metadata"] = {
            "Source": source_name,
            **row["metadata"],
        }

    return rows


def read_pdf_text(path: Path) -> str:
    """Extract text from a PDF using pypdf when available."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF support requires pypdf to be installed.") from error

    reader = PdfReader(str(path))
    page_text = [page.extract_text() or "" for page in reader.pages]

    return "\n".join(page_text)


def load_uploaded_document_rows() -> list[dict[str, Any]]:
    """Load supported uploaded documents into ChromaDB-ready records."""
    rows = []

    for path in get_uploaded_document_paths():
        suffix = path.suffix.lower()

        if suffix == ".csv":
            rows.extend(dataframe_to_document_rows(pd.read_csv(path), path.name))
        elif suffix == ".xlsx":
            rows.extend(dataframe_to_document_rows(pd.read_excel(path), path.name))
        elif suffix in {".txt", ".md"}:
            rows.extend(text_to_document_rows(path.read_text(encoding="utf-8"), path.name))
        elif suffix == ".pdf":
            rows.extend(text_to_document_rows(read_pdf_text(path), path.name))

    return rows


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
    global uploaded_document_signature
    global last_excel_modified_at, startup_message, vector_store_ready

    if has_uploaded_documents():
        signature = uploaded_documents_signature()

        if signature == uploaded_document_signature:
            return vector_store_ready, startup_message

        rows = load_uploaded_document_rows()
        uploaded_document_signature = signature
        excel_rows_cache = rows
        rebuild_collection(rows)

        if not rows:
            vector_store_ready = False
            startup_message = "Uploaded documents do not contain searchable text."
        else:
            vector_store_ready = True
            startup_message = f"Loaded {len(get_uploaded_document_paths())} uploaded document(s)."

        return vector_store_ready, startup_message

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


@app.route("/upload", methods=["POST"])
def upload_documents():
    """Upload documents and rebuild the vector store from uploaded content."""
    files = request.files.getlist("documents")

    if not files or all(not file.filename for file in files):
        return jsonify({"error": "Please choose at least one document to upload."}), 400

    DOCUMENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    uploaded_files = []

    for file in files:
        if not file.filename:
            continue

        original_filename = secure_filename(file.filename)
        suffix = Path(original_filename).suffix.lower()

        if suffix not in ALLOWED_DOCUMENT_EXTENSIONS:
            return (
                jsonify(
                    {
                        "error": (
                            f"Unsupported file type: {original_filename}. "
                            "Upload CSV, XLSX, TXT, Markdown, or PDF files."
                        )
                    }
                ),
                400,
            )

        file_path = DOCUMENT_UPLOAD_DIR / original_filename
        file.save(file_path)
        uploaded_files.append(original_filename)

    try:
        ready, message = refresh_vector_store_if_needed()
    except Exception as error:
        message = f"Upload succeeded, but indexing failed: {error}"

        return (
            jsonify(
                {
                    "error": message,
                    "status": message,
                    "status_ready": False,
                }
            ),
            500,
        )

    return jsonify(
        {
            "message": f"Uploaded {len(uploaded_files)} file(s): {', '.join(uploaded_files)}",
            "status": message,
            "status_ready": ready,
        }
    )


@app.route("/uploads/clear", methods=["POST"])
def clear_uploaded_documents():
    """Remove uploaded documents and return to the spreadsheet data source."""
    global uploaded_document_signature

    removed_count = 0

    for path in get_uploaded_document_paths():
        path.unlink()
        removed_count += 1

    uploaded_document_signature = None

    try:
        ready, message = refresh_vector_store_if_needed()
    except Exception as error:
        message = f"Uploaded documents were removed, but re-indexing failed: {error}"

        return (
            jsonify(
                {
                    "error": message,
                    "status": message,
                    "status_ready": False,
                }
            ),
            500,
        )

    return jsonify(
        {
            "message": f"Removed {removed_count} uploaded document(s).",
            "status": message,
            "status_ready": ready,
        }
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
    using_uploaded_documents = has_uploaded_documents()

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
        exact_results = [] if using_uploaded_documents else search_exact_rows(query_tokens)

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
        min_relevance_score = (
            DOCUMENT_MIN_RELEVANCE_SCORE
            if using_uploaded_documents
            else MIN_RELEVANCE_SCORE
        )

        for document, metadata, distance in zip(documents, metadatas, distances):
            # Spreadsheet rows use a stricter cutoff because exact column matches are available.
            # Uploaded documents return the top semantic chunks from ChromaDB directly.
            score = round(1 / (1 + float(distance)), 4)
            has_exact_match = (
                False
                if using_uploaded_documents
                else metadata_contains_query_token(metadata, query_tokens)
            )

            if score < min_relevance_score and not has_exact_match:
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

        if exact_results and not using_uploaded_documents:
            results = exact_results

        result_limit = DOCUMENT_RESULT_LIMIT if using_uploaded_documents else RESULT_LIMIT
        results = results[:result_limit]
        direct_answer = (
            extract_direct_answer(query, results, query_embedding)
            if using_uploaded_documents
            else None
        )

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
                "direct_answer": direct_answer,
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
