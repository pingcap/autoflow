import logging
import os

import requests
import typing
import uuid
import json
from tqdm import tqdm
from datetime import datetime
from langfuse import Langfuse
from langfuse.client import DatasetItemClient
from langfuse.model import DatasetStatus
from tenacity import retry, stop_after_attempt, wait_fixed
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI

from app.core.config import settings
from app.evaluation.evaluators import (
    LanguageEvaluator,
    ToxicityEvaluator,
    E2ERagEvaluator,
)
import pandas as pd
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    SemanticSimilarity,
)
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

DEFAULT_METRICS = ["toxicity", "language"]
DEFAULT_TIDB_AI_CHAT_ENGINE = "default"


class Evaluation:
    """
    Evaluate a dataset using TiDB AI and Langfuse.

    Args:
        dataset_name: "customize" or the name of the dataset in langfuse to evaluate
        run_name: The name of the run to create. If not provided, a random name will be generated.
        llm_provider: The LLM provider to use. Can be "openai" or "google".

    Examples:

    ```python
    evaluation = Evaluation(dataset_name="my_dataset")
    evaluation.run()
    ```
    """

    def __init__(
        self,
        dataset_name: str,
        run_name: typing.Optional[str] = None,
        llm_provider: typing.Literal["openai", "gemini"] = "openai",
        tidb_ai_chat_engine: typing.Optional[str] = DEFAULT_TIDB_AI_CHAT_ENGINE,
    ) -> None:
        self.langfuse = Langfuse()
        self.dataset_name = dataset_name
        self.is_customize_dataset = dataset_name == "customize"
        if not self.is_customize_dataset:
            self.dataset = self.langfuse.get_dataset(dataset_name)

        self.tidb_ai_chat_engine = tidb_ai_chat_engine

        if run_name is None:
            random_str = uuid.uuid4().hex[:6]
            self.run_name = datetime.now().strftime(f"%Y-%m-%d-%H-{random_str}")
        else:
            self.run_name = run_name

        llm_provider = llm_provider.lower()
        if llm_provider == "openai":
            self._llama_llm = OpenAI(model="gpt-4o")
        elif llm_provider == "gemini":
            self._llama_llm = Gemini(model="models/gemini-1.5-flash")
        else:
            raise ValueError(f"Invalid LLM provider: {llm_provider}")

        self._metrics = {
            "language": LanguageEvaluator(llm=self._llama_llm),
            "toxicity": ToxicityEvaluator(llm=self._llama_llm),
            "e2e_rag": E2ERagEvaluator(model="gpt-4o"),
        }

    def runeval_dataset(
        self,
        csv_dataset: str,
        run_size: int = 30,
        checkpoint_file: str = "checkpoint.json",
        error_file: str = "eval_error.csv",
    ) -> None:
        if not os.path.exists(csv_dataset):
            raise FileNotFoundError(f"File not found: {csv_dataset}")

        df = pd.read_csv(csv_dataset)
        eval_list = df.to_dict(orient="records")
        eval_list = eval_list[:run_size]

        # checkpoint info
        ragas_list = []
        completed_queries = set()
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
                completed_queries = set(checkpoint_data["completed_queries"])
                ragas_list = checkpoint_data["ragas_list"]

        # error info
        error_list = []
        errored_queries = set()
        if os.path.exists(error_file):
            error_df = pd.read_csv(error_file)
            error_list = error_df.to_dict(orient="records")
            errored_queries = set(item["query"] for item in error_list)

        for item in tqdm(eval_list):
            if item["query"] in completed_queries or item["query"] in errored_queries:
                continue  # skip completed or errored queries

            messages = [{"role": "user", "content": item["query"]}]
            try:
                response, _ = self._generate_answer_by_tidb_ai(messages)
                user_input = json.dumps(messages)

                ragas_list.append(
                    {
                        "user_input": user_input,
                        "reference": item["reference"],
                        "response": response,
                        # TODO: we cannot get retrieved_contexts now, due to the external engine
                        # "retrieved_contexts": [],
                        # Add rest fields from raw data
                        **{
                            k: v
                            for k, v in item.items()
                            if k not in ["query", "reference"]
                        },
                    }
                )

                # save the checkpoint file
                completed_queries.add(item["query"])
                checkpoint_data = {
                    "completed_queries": list(completed_queries),
                    "ragas_list": ragas_list,
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f)
            except Exception as e:
                print(f"Error processing query: {item['query']}, error: {e}")
                item["error_message"] = str(e)
                error_list.append(item)  # Add the item to the error list

                # Save the errors to the error file
                pd.DataFrame(error_list).to_csv(error_file, index=False)

        ragas_dataset = EvaluationDataset.from_list(ragas_list)
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-large")
        )
        metrics = [
            # LLMContextRecall(llm=evaluator_llm),  # retrieved_contexts required
            FactualCorrectness(llm=evaluator_llm),
            # Faithfulness(llm=evaluator_llm),  # retrieved_contexts required
            SemanticSimilarity(embeddings=evaluator_embeddings),
        ]
        results = evaluate(dataset=ragas_dataset, metrics=metrics)
        df_results = results.to_pandas()

        df_raw_data = pd.DataFrame(ragas_list)
        additional_columns = df_raw_data.drop(
            columns=["user_input", "reference", "response"]
        )
        df_results_combined = pd.concat([df_results, additional_columns], axis=1)

        df_results_combined = df_results_combined.applymap(
            lambda x: x.replace("\n", "\\n").replace("\r", "\\r")
            if isinstance(x, str)
            else x
        )
        df_results_combined.to_csv(f"results_{self.run_name}.csv", index=False)

        print(f"Saved results to results_{self.run_name}.csv")

    def run(self, metrics: list = DEFAULT_METRICS) -> None:
        for item in tqdm(self.dataset.items):
            if item.status != DatasetStatus.ACTIVE:
                continue

            sample_data = self.parse_sample(item)
            output, trace_id = self._generate_answer_by_tidb_ai(sample_data["messages"])
            trace_data = fetch_rag_data(self.langfuse, trace_id)
            question = json.dumps(sample_data["messages"])
            item.link(
                trace_or_observation=None,
                trace_id=trace_id,
                run_name=self.run_name,
            )

            for metric in metrics:
                evaluator = self._metrics[metric]
                result = evaluator.evaluate(
                    query=question,
                    response=output,
                    contexts=trace_data.get("retrieval_context", []),
                    reference=sample_data.get("expected_output", None),
                )
                if isinstance(result, dict):
                    for eval_name, eval_res in result.items():
                        self.langfuse.score(
                            trace_id=trace_id,
                            name=eval_name,
                            value=eval_res.score,
                            comment=eval_res.feedback,
                        )
                else:
                    self.langfuse.score(
                        trace_id=trace_id,
                        name=metric,
                        value=result.score,
                        comment=result.feedback,
                    )

    def parse_sample(self, item: DatasetItemClient):
        expected_output = item.expected_output
        messages = []
        if "history" in item.input:
            messages = [
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in item.input["history"]
            ]

        if "userInput" in item.input:
            messages.append({"role": "user", "content": item.input["userInput"]})
        elif "input" in item.input:
            messages.append({"role": "user", "content": item.input["input"]})

        sample_data = {
            "messages": messages,
            "expected_output": expected_output,
        }

        if "retrieval_context" in item.input:
            sample_data["retrieval_context"] = item.input["retrieval_context"]
        if "graph_context" in item.input:
            sample_data["graph_context"] = item.input["graph_context"]
        if "refined_question" in item.input:
            sample_data["refined_question"] = item.input["refined_question"]

        return sample_data

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(5))
    def _generate_answer_by_tidb_ai(self, messages: list) -> (str, str):
        response = requests.post(
            settings.TIDB_AI_CHAT_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.TIDB_AI_API_KEY}",
                "Origin": "evaluation",
            },
            json={
                "messages": messages,
                "index": "default",
                "chat_engine": self.tidb_ai_chat_engine,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        if data["trace"] is None:
            trace_id = None
        else:
            trace_url = data["trace"]["langfuse_url"]
            trace_id = parse_langfuse_trace_id_from_url(trace_url)

        answer = data["content"]
        return answer, trace_id

    def generate_answer_by_tidb_ai(self, messages: list) -> str:
        response = requests.post(
            settings.TIDB_AI_CHAT_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.TIDB_AI_API_KEY}",
            },
            json={
                "messages": messages,
                "index": "default",
                "chat_engine": self.tidb_ai_chat_engine,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.text

        return data


def parse_langfuse_trace_id_from_url(trace_url: str) -> str:
    # Example trace_url: https://us.cloud.langfuse.com/trace/87e7eb2e-b789-4b23-af60-fbcf0fd517a1
    return trace_url.split("/")[-1]


def fetch_rag_data(langfuse_client: Langfuse, tracing_id: str):
    graph_context_key = "retrieve_from_graph"
    reranking_key = "reranking"
    refined_question_key = "condense_question"

    tracing_data = langfuse_client.fetch_trace(tracing_id)

    data = {
        "history": tracing_data.data.input["chat_history"],
        "input": tracing_data.data.input["user_question"],
        "graph_context": None,
        "refined_question": None,
        "retrieval_context": None,
        "output": (
            tracing_data.data.output["content"]
            if tracing_data.data.output is not None
            and "content" in tracing_data.data.output
            else None
        ),
        "source_tracing_id": tracing_id,
    }

    for ob in tracing_data.data.observations:
        if graph_context_key == ob.name:
            graph_context = {query: sg for query, sg in ob.output["graph"].items()}
            for _, sg in graph_context.items():
                for entity in sg["entities"]:
                    entity.pop("meta", None)
            data["graph_context"] = graph_context
        if reranking_key == ob.name:
            retrieval_context = []
            for node in ob.output["nodes"]:
                retrieval_context.append(node["node"]["text"])
            data["retrieval_context"] = retrieval_context
        if refined_question_key == ob.name:
            refined_question = ob.output
            data["refined_question"] = refined_question

    return data
