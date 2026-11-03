"""Data-only, bounded Jinja prompts implementing LlamaIndex's prompt interface."""

from datetime import date, datetime
from enum import Enum
from functools import lru_cache
import math
from uuid import UUID

from jinja2 import StrictUndefined, TemplateError, meta, nodes
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.visitor import NodeTransformer
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.prompts.base import BasePromptTemplate
from pydantic import BaseModel, Field

MAX_TEMPLATE_BYTES = 32768
MAX_CONTEXT_BYTES = 512 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_ITERATIONS = 10000
FILTERS = {
    "tojson",
    "default",
    "lower",
    "upper",
    "trim",
    "length",
    "join",
    "first",
    "last",
    "capitalize",
    "title",
    "string",
}
TESTS = {
    "defined",
    "undefined",
    "none",
    "string",
    "number",
    "mapping",
    "iterable",
    "boolean",
    "true",
    "false",
    "equalto",
}
ALLOWED_NODES = {
    nodes.Template,
    nodes.Output,
    nodes.TemplateData,
    nodes.Name,
    nodes.Const,
    nodes.Getattr,
    nodes.Getitem,
    nodes.For,
    nodes.If,
    nodes.Compare,
    nodes.Operand,
    nodes.And,
    nodes.Or,
    nodes.Not,
    nodes.Test,
    nodes.Filter,
    nodes.Call,
    nodes.Keyword,
    nodes.Tuple,
    nodes.List,
    nodes.Dict,
    nodes.Pair,
    nodes.Assign,
    nodes.CondExpr,
}


class PromptSafetyError(ValueError):
    pass


class RenderBudget:
    def __init__(self):
        self.iterations = 0
        self.context_bytes = 0
        self.values = 0

    def iterate(self, iterable):
        for value in iterable:
            self.iterations += 1
            if self.iterations > MAX_ITERATIONS:
                raise PromptSafetyError("Prompt iteration budget exceeded")
            yield value

    def plain(self, value, depth=0):
        self.values += 1
        if depth > 16 or self.values > 20000:
            raise PromptSafetyError("Prompt context is too complex")
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="python")
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, (date, datetime, UUID)):
            value = str(value)
        if value is None or type(value) is bool:
            return value
        if type(value) is int:
            if value.bit_length() > 128:
                raise PromptSafetyError("Prompt number is too large")
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if type(value) is str:
            self.context_bytes += len(value.encode())
            if self.context_bytes > MAX_CONTEXT_BYTES:
                raise PromptSafetyError("Prompt context is too large")
            return value
        if type(value) in (list, tuple):
            if len(value) > 2000:
                raise PromptSafetyError("Prompt collection is too large")
            return [self.plain(item, depth + 1) for item in value]
        if type(value) is dict:
            if len(value) > 2000 or any(type(key) is not str for key in value):
                raise PromptSafetyError("Invalid prompt mapping")
            return {
                self.plain(key, depth + 1): self.plain(item, depth + 1)
                for key, item in value.items()
            }
        raise PromptSafetyError("Prompt context must contain only data values")


class PromptEnvironment(ImmutableSandboxedEnvironment):
    def is_safe_callable(self, value):
        owner = getattr(value, "__self__", None)
        name = getattr(value, "__name__", "")
        return (type(owner) is dict and name in {"items", "keys", "values"}) or (
            isinstance(owner, RenderBudget) and name == "iterate"
        )


ENV = PromptEnvironment(
    undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True
)
ENV.globals.clear()
ENV.filters = {key: value for key, value in ENV.filters.items() if key in FILTERS}
ENV.tests = {key: value for key, value in ENV.tests.items() if key in TESTS}


class BoundLoops(NodeTransformer):
    def visit_For(self, node, *args, **kwargs):
        node = self.generic_visit(node, *args, **kwargs)
        node.iter = nodes.Call(
            nodes.Name("_bounded_iter", "load"), [node.iter], [], None, None
        ).set_lineno(node.lineno)
        return node


@lru_cache(maxsize=128)
def compile_prompt(template):
    if not isinstance(template, str) or len(template.encode()) > MAX_TEMPLATE_BYTES:
        raise PromptSafetyError("Prompt template is too large")
    try:
        tree = ENV.parse(template)
        all_nodes = list(tree.find_all(nodes.Node))
        if len(all_nodes) > 1024:
            raise PromptSafetyError("Prompt template is too complex")
        for node in all_nodes:
            if type(node) not in ALLOWED_NODES:
                raise PromptSafetyError("Unsupported prompt operation")
            if isinstance(node, nodes.Name) and node.name.startswith("_"):
                raise PromptSafetyError("Private prompt names are forbidden")
            if isinstance(node, nodes.Getattr) and node.attr.startswith("_"):
                raise PromptSafetyError("Private prompt attributes are forbidden")
            if isinstance(node, nodes.Getitem) and (
                not isinstance(node.arg, nodes.Const)
                or type(node.arg.value) not in (str, int)
                or isinstance(node.arg.value, str)
                and node.arg.value.startswith("_")
            ):
                raise PromptSafetyError(
                    "Only public literal prompt lookups are supported"
                )
            if isinstance(node, nodes.For) and node.recursive:
                raise PromptSafetyError("Recursive prompt loops are forbidden")
            if isinstance(node, nodes.Call) and (
                not isinstance(node.node, nodes.Getattr)
                or node.node.attr not in {"items", "keys", "values"}
                or node.args
                or node.kwargs
                or node.dyn_args is not None
                or node.dyn_kwargs is not None
            ):
                raise PromptSafetyError("Prompt function calls are forbidden")
            if isinstance(node, (nodes.Filter, nodes.Test)):
                allowed = FILTERS if isinstance(node, nodes.Filter) else TESTS
                if (
                    node.name not in allowed
                    or node.dyn_args is not None
                    or node.dyn_kwargs is not None
                ):
                    raise PromptSafetyError("Unsupported prompt filter or test")
                for argument in list(node.args) + [kw.value for kw in node.kwargs]:
                    if (
                        not isinstance(argument, nodes.Const)
                        or isinstance(argument.value, str)
                        and len(argument.value) > 128
                        or type(argument.value) is int
                        and not -10 <= argument.value <= 10
                    ):
                        raise PromptSafetyError(
                            "Prompt filter arguments must be small constants"
                        )
                if node.name == "tojson" and (
                    node.args
                    or any(
                        kw.key != "indent"
                        or type(kw.value.value) is not int
                        or not 0 <= kw.value.value <= 4
                        for kw in node.kwargs
                    )
                ):
                    raise PromptSafetyError(
                        "JSON prompt formatting accepts only indent 0 to 4"
                    )
        variables = sorted(meta.find_undeclared_variables(tree))
        tree = BoundLoops().visit(tree)
        compiled = ENV.template_class.from_code(
            ENV, ENV.compile(tree), ENV.globals, None
        )
        return compiled, variables
    except (TemplateError, RecursionError, SyntaxError):
        raise PromptSafetyError("Invalid or unsupported prompt syntax") from None


def validate_prompt(template):
    compile_prompt(template)
    return template


class SafePromptTemplate(BasePromptTemplate):
    template_str: str
    kwargs: dict = Field(default_factory=dict)

    def __init__(
        self,
        template_str,
        metadata=None,
        output_parser=None,
        template_var_mappings=None,
        function_mappings=None,
        **kwargs,
    ):
        if function_mappings:
            raise PromptSafetyError("Prompt function mappings are forbidden")
        _, variables = compile_prompt(template_str)
        super().__init__(
            template_str=template_str,
            template_vars=variables,
            metadata=metadata or {"prompt_type": "custom"},
            kwargs=kwargs,
            output_parser=output_parser,
            template_var_mappings=template_var_mappings or {},
            function_mappings={},
        )

    def partial_format(self, **kwargs):
        result = self.model_copy()
        result.kwargs = {**self.kwargs, **kwargs}
        return result

    def format(
        self, llm=None, messages_to_prompt=None, completion_to_prompt=None, **kwargs
    ):
        template, _ = compile_prompt(self.template_str)
        budget = RenderBudget()
        mapped = self._map_template_vars({**self.kwargs, **kwargs})
        data = budget.plain(
            {key: mapped[key] for key in self.template_vars if key in mapped}
        )
        data["_bounded_iter"] = budget.iterate
        chunks, size = [], 0
        try:
            for chunk in template.generate(**data):
                size += len(chunk.encode())
                if size > MAX_OUTPUT_BYTES:
                    raise PromptSafetyError("Prompt output budget exceeded")
                chunks.append(chunk)
        except TemplateError:
            raise PromptSafetyError(
                "Prompt could not be rendered with the supplied data"
            ) from None
        return "".join(chunks)

    def format_messages(self, llm=None, **kwargs):
        messages = [ChatMessage(role="user", content=self.format(llm=llm, **kwargs))]
        return (
            self.output_parser.format_messages(messages)
            if self.output_parser
            else messages
        )

    def get_template(self, llm=None):
        return self.template_str
