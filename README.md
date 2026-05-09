# Excel RAG Semantic Search

A small Flask web app that reads rows from `uploads/data.xlsx`, embeds them with `sentence-transformers`, stores them in ChromaDB, and retrieves the most relevant rows for a user search query.

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

3. Put your Excel file at:

```text
uploads/data.xlsx
```

The included file is sample data. Replace it with your own `.xlsx` file when needed.

4. Start the Flask app:

```bash
python app.py
```

5. Open the app in your browser:

```text
http://127.0.0.1:5000
```

## How It Works

1. Pandas reads `uploads/data.xlsx`.
2. Each row is converted into searchable text like `Column: value | Column: value`.
3. `all-MiniLM-L6-v2` creates embeddings for every row.
4. ChromaDB stores the row text, metadata, and embeddings in `chroma_db/`.
5. The `/search` route embeds the user query and retrieves the closest rows.
6. The frontend displays matching rows without a chatbot interface.

## Error Handling

The app handles:

- missing `uploads/data.xlsx`
- an empty search query
- an empty Excel file
- no matching results
- backend search failures

## Notes

- The first startup can take longer because the embedding model may need to download.
- Restart Flask after replacing `uploads/data.xlsx` so ChromaDB is rebuilt from the new file.
