import warnings

import click
import uvicorn

# Every server entry point must use the same authentication, cookie and CORS
# middleware. Keep the CLI commands here, not a second FastAPI application.
from app.api_server import app  # noqa: F401


@click.group(context_settings={"max_content_width": 150})
def cli():
    pass


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host, default=127.0.0.1")
@click.option("--port", default=3000, help="Port, default=3000")
@click.option("--workers", default=4)
def runserver(host, port, workers):
    warnings.warn(
        "This command will start the server in development mode, do not use it in production."
    )
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="debug",
        workers=workers,
    )


@cli.command()
@click.option(
    "--dataset", default="regression", help="Dataset name, default=regression"
)
@click.option(
    "--llm-provider",
    default="openai",
    help="LLM provider, default=openai, options=[openai, gemini]",
)
@click.option("--run-name", default=None, help="Run name, default=None")
@click.option(
    "--tidb-ai-chat-engine",
    default="default",
    help="TiDB AI chat engine, default=default",
)
def runeval(dataset, llm_provider, run_name, tidb_ai_chat_engine):
    from app.evaluation.evals import Evaluation

    eval = Evaluation(
        dataset_name=dataset,
        llm_provider=llm_provider,
        run_name=run_name,
        tidb_ai_chat_engine=tidb_ai_chat_engine,
    )
    eval.run()


@cli.command()
@click.option(
    "--csv",
    default="autoflow_dataset.csv",
    help="Dataset CSV file name that contains two columns `query` and `reference`, default='autoflow_dataset.csv'",
)
@click.option(
    "--llm-provider",
    default="openai",
    help="LLM provider, default=openai, options=[openai, gemini]",
)
@click.option("--run-name", default=None, help="Run name, default=None")
@click.option(
    "--tidb-ai-chat-engine",
    default="default",
    help="TiDB AI chat engine, default=default",
)
@click.option("--run-size", default=30, help="Run size, default=30")
def runeval_dataset(csv, llm_provider, run_name, tidb_ai_chat_engine, run_size):
    from app.evaluation.evals import Evaluation

    evaluation = Evaluation(
        dataset_name="customize",
        llm_provider=llm_provider,
        run_name=run_name,
        tidb_ai_chat_engine=tidb_ai_chat_engine,
    )
    evaluation.runeval_dataset(csv_dataset=csv, run_size=run_size)


@cli.command()
@click.option("--query", default=None, help="query")
def generate_answer_by_tidb_ai(query: str):
    from app.evaluation.evals import Evaluation

    evaluation = Evaluation(
        dataset_name="customize",
        llm_provider="openai",
        run_name=None,
        tidb_ai_chat_engine="default",
    )

    print(
        evaluation.generate_answer_by_tidb_ai(
            messages=[{"role": "user", "content": query}]
        )
    )


if __name__ == "__main__":
    cli()
