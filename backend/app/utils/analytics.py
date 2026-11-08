import re

_ANALYTICS_ID = re.compile(r"(?:G-[A-Z0-9]{6,20}|UA-[0-9]{4,12}-[0-9]{1,4})\Z")


def safe_analytics_id(value: object) -> str:
    return value if isinstance(value, str) and _ANALYTICS_ID.fullmatch(value) else ""
