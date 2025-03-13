#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from uuid import UUID
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine
from autoflow import Autoflow
from autoflow.schema import IndexMethod
from autoflow.llms.chat_models import ChatModel
from autoflow.llms.embeddings import EmbeddingModel
from llama_index.core.llms import ChatMessage

st.set_page_config(
    page_title="Demo of Autoflow and Streamlit", page_icon="📖", layout="wide"
)
st.header("📖 Knowledge base app built with Autoflow and Streamlit")

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
        autocomplete="off",
        help="You can get your database URL from https://tidbcloud.com",
        value=os.environ.get("DATABASE_URL", None)
        or "mysql+pymysql://root@localhost:4000/test"
        or st.session_state.get("DATABASE_URL", ""),
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
embedding_model = EmbeddingModel(
    model_name="text-embedding-3-small",
    dimensions=1536,
    api_key=openai_api_key,
)
kb = af.create_knowledge_base(
    id=UUID(
        "655b6cf3-8b30-4839-ba8b-5ed3c502f30e"
    ),  # For not creating a new KB every time
    name="New KB",
    description="This is a knowledge base for testing",
    index_methods=[IndexMethod.VECTOR_SEARCH, IndexMethod.KNOWLEDGE_GRAPH],
    chat_model=chat_model,
    embedding_model=embedding_model,
)

with st.form(key="file_upload_form"):
    uploaded_file = st.file_uploader(
        "Upload a .pdf, .docx, .md or .txt file",
        type=["pdf", "docx", "txt", "md"],
        help="Scanned documents are not supported yet!",
    )
    upload = st.form_submit_button("Upload")
    if upload:
        if not uploaded_file:
            st.error("Please upload a valid file.")
            st.stop()
        file_path = f"/tmp/{uploaded_file.name}"
        with st.spinner(
            "Indexing document... This may take a while ⏳(import time; time.sleep(3))"
        ):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            kb.import_documents_from_files(
                files=[
                    Path(file_path),
                ]
            )
            import time

            time.sleep(3)

for line in ["generated", "past", "corpus"]:
    if line not in st.session_state:
        st.session_state[line] = []

for o in ["kg"]:
    if o not in st.session_state:
        st.session_state[o] = None


def on_submit():
    user_input = st.session_state.user_input
    if user_input:
        result = kb.search_documents(query=user_input, similarity_top_k=3)
        st.session_state["corpus"] = result.chunks
        kg = kb.search_knowledge_graph(query=user_input)
        st.session_state["kg"] = kg
        messages = [
            ChatMessage(
                role="system",
                content="Here are some relevant documents about your query:\n\n"
                + "\n".join(c.chunk.text for c in result.chunks),
            ),
            ChatMessage(
                role="user",
                content=user_input + "\n(in markdown, removed unused breaklines)",
            ),
        ]
        resp = chat_model.chat(messages)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(str(resp.message))


chat_section, corpus_section = st.columns(2)
with chat_section:
    st.markdown("##### Chats")
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for i in range(len(st.session_state["generated"])):
            with st.chat_message("user"):
                st.write(st.session_state["past"][i])
            with st.chat_message("assistant"):
                st.write(st.session_state["generated"][i])

    with st.container():
        st.chat_input(
            "Input your question about this document here.",
            key="user_input",
            on_submit=on_submit,
        )

with corpus_section:
    st.markdown("##### Vector Search Results")
    corpus_placeholder = st.empty()
    with corpus_placeholder.container():
        [c.chunk for c in st.session_state["corpus"]] if st.session_state[
            "corpus"
        ] else "Please input a query left."

    st.markdown("##### Knowledge Graph Search Results")
    kg_placeholder = st.empty()
    with kg_placeholder.container():
        kg = st.session_state["kg"]
        [
            r.rag_description for r in kg.relationships
        ] if kg else "Please input a query left."
