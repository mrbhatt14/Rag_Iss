# Excel RAG Semantic Search

A small Flask web app that reads rows from `uploads/data.xlsx` or a Google Sheet, embeds them with `sentence-transformers`, stores them in ChromaDB, and retrieves the most relevant rows for a user search query.

## Folder Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── uploads/
    └── data.xlsx
```

## Local Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Use either a local Excel file or Google Sheets.

For local Excel, put your file at:

```text
uploads/data.xlsx
```

The included file is sample data. Replace it with your own `.xlsx` file when needed.

For Google Sheets, share the sheet so anyone with the link can view it, then set:

```bash
export GOOGLE_SHEET_CSV_URL="https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit?gid=0"
```

The app currently uses this Google Sheet by default:

```text
https://docs.google.com/spreadsheets/d/1nwSzIkL-8Dmatx-dIS5zETsgZ4GPzaOJXguVDRqNbgQ/edit?usp=sharing
```

The app converts normal Google Sheets links into CSV export links automatically. If `GOOGLE_SHEET_CSV_URL` is set, it overrides the default Google Sheet. To go back to local Excel, clear `DEFAULT_GOOGLE_SHEET_URL` in `app.py`.

4. Start the Flask app:

```bash
python app.py
```

5. Open the app in your browser:

```text
http://127.0.0.1:5000
```

## How It Works

1. Pandas reads Google Sheets CSV when `GOOGLE_SHEET_CSV_URL` is set; otherwise it reads `uploads/data.xlsx`.
2. Each row is converted into searchable text like `Column: value | Column: value`.
3. `all-MiniLM-L6-v2` creates embeddings for every row.
4. ChromaDB stores the row text, metadata, and embeddings in `chroma_db/`.
5. The `/search` route embeds the user query and retrieves the closest rows.
6. The frontend displays matching rows without a chatbot interface.

## Error Handling

The app handles:

- missing `uploads/data.xlsx`
- invalid or unreachable Google Sheets URL
- an empty search query
- an empty Excel file or Google Sheet
- no matching results
- backend search failures

## Notes

- The first startup can take longer because the embedding model may need to download.
- Local Excel changes and Google Sheets changes are checked before each search. ChromaDB is rebuilt only when the spreadsheet content changes.

## Deploy On Render

This project is better suited to Render than static hosts like GitHub Pages because it needs a Python Flask backend, sentence-transformers, and ChromaDB.

1. Push this repo to GitHub.
2. In Render, click **New > Web Service**.
3. Connect the GitHub repo.
4. Use these settings:

```text
Language: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --timeout 120
```

5. Add these environment variables:

```text
GOOGLE_SHEET_CSV_URL=https://docs.google.com/spreadsheets/d/1nwSzIkL-8Dmatx-dIS5zETsgZ4GPzaOJXguVDRqNbgQ/edit?usp=sharing
CHROMA_DIR=/opt/render/project/src/chroma_db
ANONYMIZED_TELEMETRY=False
PYTHON_VERSION=3.11.9
```

The included `render.yaml` can also be used as a Render Blueprint. The first deploy may be slow because PyTorch and the embedding model are large.
