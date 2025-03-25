# Streamlit Examples

This project demonstrates how to use AutoFlow as a Retrieval-Augmented Generation (RAG) framework and Streamlit as the web framework for building a knowledge search application.

## Prerequisites
- **Python 3.12.4** (Check the version specified in `.python-version`). You can use `pyenv` to manage your Python versions.
- **macOS users:** Ensure `mysqlclient` is installed.

## Installation and Usage

**Step 1: Install Dependencies**

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r reqs.txt
```

**Step 2: Run the Streamlit App**
Start the application with:

```bash
streamlit run build-knowledge-search-with-autoflow-and-streamlit.py
```

**Step 3: Open in Browser**

Once the app is running, open http://localhost:8501 in your browser and follow these steps:


1. Enter your [OpenAI API key](https://platform.openai.com/api-keys) in left sidebar
2. Enter your TiDB Cloud connection string in the left sidebar. Use the SQLAlchemy format ( `mysql+pymysql://root@localhost:4000/test`) To obtain your connection string, go to the TiDB Cloud console, click **Connect**, and select `SQLAlchemy` from the **Connect With** dropdown.
3. Press **Enter** to start initialization (this may take a few minutes).
4. Once initialization completes, upload a file to build the knowledge base. Then enter your queries to retrieve answers from the newly generated knowledge base.
