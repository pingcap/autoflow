#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pip install autoflow-ai==0.0.1.dev10 streamlit
import os
from uuid import UUID
from pathlib import Path

import streamlit as st # type: ignore
from sqlalchemy import create_engine
from autoflow import Autoflow
from autoflow.schema import IndexMethod
from autoflow.llms.chat_models import ChatModel
from autoflow.llms.embeddings import EmbeddingModel

st.set_page_config(page_title="SearchGPT", page_icon="📖", layout="wide")
st.header("📖 SearchGPT")

with st.sidebar:
    st.markdown(
        "## How to use\n"
        "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below 🔑\n"  # noqa: E501
        "2. Enter your [TiDB Cloud](https://tidbcloud.com) database connection URL below 🔗\n"
        "3. Upload a pdf, docx, or txt file 📄\n"
        "4. Ask a question about the document 💬\n"
    )
    openai_api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Paste your OpenAI API key here (sk-...)",
        help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
        value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
    )

    database_url_input = st.text_input(
        "Database URL",
        type="password",
        placeholder="e.g. mysql+pymysql://root@localhost:4000/test",
        help="You can get your database URL from https://tidbcloud.com",
        value=os.environ.get("DATABASE_URL", None)
            or st.session_state.get("DATABASE_URL", "")
    )

    st.session_state["OPENAI_API_KEY"] = openai_api_key_input
    st.session_state["DATABASE_URL"] = database_url_input

openai_api_key = st.session_state.get("OPENAI_API_KEY")
database_url = st.session_state.get("DATABASE_URL")

if not openai_api_key or not database_url:
    st.error("Please enter your OpenAI API key and TiDB Cloud connection string.")
    st.stop()

af = Autoflow(create_engine(database_url))
chat_model = ChatModel("gpt-4o-mini", api_key=openai_api_key)
embed_model = EmbeddingModel(
    model_name="text-embedding-3-small",
    dimensions=1536,
    api_key=openai_api_key,
)

kb = af.create_knowledge_base(
    id=UUID("655b6cf3-8b30-4839-ba8b-5ed3c502f30e"),
    name="New KB",
    description="This is a knowledge base for testing",
    index_methods=[IndexMethod.VECTOR_SEARCH, IndexMethod.KNOWLEDGE_GRAPH],
    chat_model=chat_model,
    embedding_model=embed_model,
)

with st.form(key="file_upload_form"):
    uploaded_file = st.file_uploader(
        "Upload a .pdf, .docx, .md or .txt file",
        type=["pdf", "docx", "txt", "md"],
        help="Scanned documents are not supported yet!",
    )
    upload = st.form_submit_button("Upload")
    if not uploaded_file:
        st.error("Please upload a valid file.")
        st.stop()
    if upload:
        file_path = f"/tmp/{uploaded_file.name}"
        with st.spinner("Indexing document... This may take a while ⏳"):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            kb.import_documents_from_files(files=[Path(file_path),])
            import time; time.sleep(3)

with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")

if submit:
    if not query:
        st.error("Please enter a valid query.")
        st.stop()
    vector_search_col, graph_search_col = st.columns(2)
    result = kb.search_documents(query=query, similarity_top_k=3)
    kg = kb.search_knowledge_graph(query="What is TiDB?")

    with vector_search_col:
        st.markdown("#### Vector Search Results")
        [(c.score, c.chunk.text) for c in result.chunks]

    with graph_search_col:
        st.markdown("#### Graph Search Results")
        [(r.rag_description) for r in kg.relationships]