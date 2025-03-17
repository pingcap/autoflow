#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from openai import OpenAI
import streamlit as st
from sqlalchemy import text
from autoflow.storage.tidb import TiDBClient
from pydantic import BaseModel


class QuestionSQLResponse(BaseModel):
    question: str
    sql: str
    markdown: str


st.set_page_config(page_title="Text2SQL", page_icon="📖", layout="wide")
with st.sidebar:
    st.markdown("# Text2SQL")
    st.markdown(
        "## How to use\n"
        "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below 🔑\n"  # noqa: E501
        "2. Enter your [TiDB Cloud](https://tidbcloud.com) database connection URL below 🔗\n"
        "3. Ask a question in the right chat boxgh 🤖\n"
    )
    st.warning(
        "Please double check the generated SQL before running it on your database."
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

db = TiDBClient.connect(database_url)
oai = OpenAI(api_key=openai_api_key)

for item in ["generated", "past"]:
    if item not in st.session_state:
        st.session_state[item] = []

table_definitions = []
current_database = db._db_engine.url.database
with db._db_engine.connect() as conn:
    for table_name in db.table_names():
        table_definitions.append(
            conn.execute(text(f"SHOW CREATE TABLE `{table_name}`")).first()
        )


def on_submit():
    user_input = st.session_state.user_input
    if user_input:
        response = (
            oai.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                        You are a very senior database administrator who can write SQL very well,
                        please write MySQL SQL to answer user question,
                        Use backticks to quote table names and column names,
                        here are some table definitions in database,
                        the database name is {current_database}\n\n"""
                        + "\n".join("|".join(t) for t in table_definitions),
                    },
                    {"role": "user", "content": f"Question: {user_input}\n"},
                ],
                response_format=QuestionSQLResponse,
            )
            .choices[0]
            .message.parsed
        )
        st.session_state.past.append(user_input)

        # Execute the SQL query and set the result
        answer = None
        with db._db_engine.connect() as conn:
            try:
                result = conn.execute(text(response.sql))
                rows = [tuple(result.keys())]
                rows.extend(row for row in result.fetchall())
                sql_result = "\n".join(str(row) for row in rows)

                answer = (
                    oai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a markdown formatter, format the user input to markdown, format the data row into markdown tables.",
                            },
                            {
                                "role": "user",
                                "content": f"""
                            Question: {response.question}\n\n
                            SQL: {response.sql}\n\n
                            Markdown: {response.markdown}\n\n
                            Result: {sql_result}""",
                            },
                        ],
                    )
                    .choices[0]
                    .message.content
                )
                st.session_state.generated.append(answer)
            except Exception as e:
                st.session_state.generated.append(f"Error: {e}")


st.markdown("##### User Query")
with st.container():
    st.chat_input(
        "Input your question, e.g. how many tables?",
        key="user_input",
        on_submit=on_submit,
    )

    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            with st.chat_message("user"):
                st.write(st.session_state["past"][i])
            with st.chat_message("assistant"):
                st.write(st.session_state["generated"][i])
