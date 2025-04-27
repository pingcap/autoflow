from typing import Optional


def format_namespace(namespace: Optional[str] = None) -> str:
    return namespace.replace("-", "_") if namespace else ""
