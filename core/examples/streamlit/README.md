# Streamlit Examples

* Use autoflow==0.0.1.dev12 as RAG framework
* Use Streamlit as web framework


## How to run

**Step1**: Install the required packages

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r reqs.txt
```

**Step2**: Run the Streamlit app

```bash
streamlit run core/examples/streamlit/build-knowledge-search-with-autoflow-and-streamlit.py
```

**Step3**: Open the browser and visit `http://localhost:8501`

* Input OpenAI API key in left sidebar
* Input the TiDB Cloud connection string in left sidebar