import os

import dspy
from app.rag.question_gen.query_decomposer import DecomposeQueryModule


def save_decompose_query_program():
    dspy_lm = dspy.LM(model="gpt-4.1-2025-04-14-mini", api_key=os.getenv("OPENAI_API_KEY"))
    module = DecomposeQueryModule(dspy_lm)
    module.save("dspy_compiled_program/decompose_query/program.json")


if __name__ == "__main__":
    save_decompose_query_program()
