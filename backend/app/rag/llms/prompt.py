from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.rich import RichPromptTemplate


def resolve_prompt_template(
    template_str: str, llm: LLM, **kwargs
) -> RichPromptTemplate:
    if llm.class_name() == "Ollama_llm":
        template_str = template_str + "\n/no_think\n"

    return RichPromptTemplate(template_str, **kwargs)
