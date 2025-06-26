# Data Analysis Chatbot (RAG & SQL Agents)

A powerful Streamlit-based chatbot that allows you to analyze your data using two advanced approaches:
- **RAG Tool**: Retrieval-Augmented Generation for document-based Q&A.
- **SQL Agent**: Natural language to SQL for structured database queries.

---

## Features

- **Chatbot UI**: Modern, interactive chat interface with avatars and conversation history.
- **Tool Selection**: Choose between SQL Agent and RAG Tool from the sidebar.
- **Data Analysis**: Query your data using natural language—either via SQL or document search.
- **Table Output**: SQL results are shown as formatted tables with column headers.
- **Clear Chat**: Instantly reset the conversation.
- **Custom Sidebar**: Settings, tool info, and credits.
- **Extensible**: Easily add new data, retrain vector DBs, or expand with new tools.

---

## Directory Structure

```
Final/
├── configs/
│   ├── project_config.yml
│   └── tools_config.yml
├── data/
│   ├── csv_vectordb/         # Chroma vector DB for CSV docs (RAG)
│   ├── json_vectordb/        # Chroma vector DB for JSON docs (RAG)
│   ├── csv_sql.db            # SQLite DB for CSV data (SQL Agent)
│   ├── docs/
│   │   ├── csv_files/        # Place your CSV files here
│   │   └── json_files/       # Place your JSON files here
│   └── .txt                  # Data directory info
├── src/
│   ├── chatbot_app.py        # Main Streamlit chatbot app
│   ├── prepare_sqlite_db.py  # Script to build SQLite DB from CSVs
│   └── prepare_vector_db.py  # Script to build vector DB from docs
└── Tools/
    ├── RAG agents/
    │   ├── rag_tool.py       # RAG tool callable
    │   └── rag_tool.ipynb    # RAG tool notebook
    └── Sql agents/
        ├── sql_agent.py      # SQL agent callable
        └── test.ipynb        # SQL agent notebook
```

---

## Setup Instructions
### 1. **Clone the Repository**

```bash
git clone https://github.com/DevanshSharma867/Smart-QnA-Chatbot.git
cd Smart-QnA-Chatbot
```
### 2. **Install Dependencies**
Optionally you can set up a venv
```bash
python -v venv venv
venv\scripts\activate
```
```bash
pip install -r requirements.txt
```

### 2. **Set Up Environment Variables**

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

### 3. **Prepare Your Data**

- **CSV/JSON Files**: Place your CSV files in `data/docs/csv_files/` and JSON files in `data/docs/json_files/`.
- **(Optional) Example files**: `cluster_data.csv` and `eda_output.json` are provided.

### 4. **Build Databases**

#### a. **Prepare SQLite DB for SQL Agent**

```bash
python src/prepare_sqlite_db.py
```
- This will scan `data/docs/csv_files/` and create/update `data/csv_sql.db`.

#### b. **Prepare Vector DB for RAG Tool**

```bash
python src/prepare_vector_db.py
```
- This will process your CSV/JSON files and build the Chroma vector DBs in `data/csv_vectordb/` and `data/json_vectordb/`.

---

## Running the Chatbot

```bash
streamlit run src/chatbot_app.py
```

- Use the sidebar to select between **SQL Agent** and **RAG Tool**.
- Type your question in the chat input.
- SQL results will be shown as tables with column headers.
- Use "Clear Chat" to reset the conversation.
- Sidebar includes tool info and credits.

---

## Tool Details

### RAG Tool

- Uses Chroma vector DB and OpenAI embeddings.
- Searches your CSV/JSON documents for relevant content and answers your questions.

### SQL Agent

- Uses a SQLite database built from your CSV files.
- Converts natural language questions to SQL and executes them.
- Returns results as tables with column names.

---

## Customization

- **Change number of rows in SQL sample**: Edit the `limit` parameter in `get_sample_data` in `Tools/Sql agents/sql_agent.py`.
- **Add new data**: Place new files in `data/docs/csv_files/` or `data/docs/json_files/` and rerun the preparation scripts.
- **Tune vector DB or LLM settings**: Edit `configs/tools_config.yml`.

---

## Credits

**Conceptualized and developed by Devansh**

---

## Troubleshooting

- **OPENAI_API_KEY not found**: Ensure your `.env` file is present and correct.
- **No data found**: Make sure your CSV/JSON files are in the correct folders.
- **Vector DB/SQL DB not updating**: Rerun the preparation scripts after adding new data.

---
