import re
from typing import Dict, Iterable, Iterator, Tuple

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_START_RE = re.compile(r"<think\b[^>]*>.*", re.IGNORECASE | re.DOTALL)
_GENERIC_ASSISTANT_QUESTION_RE = re.compile(
    r"\b("
    r"how (can|may) i (assist|help) you|"
    r"what specific topic would you like|"
    r"do you prefer the questions|"
    r"how many follow[-‑ ]?up questions|"
    r"should the questions be concise|"
    r"is there a particular audience"
    r")\b",
    re.IGNORECASE,
)


def strip_hidden_reasoning(text: str | None) -> str:
    if not text:
        return ""

    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_START_RE.sub("", cleaned)
    return cleaned.strip()


def iter_without_hidden_reasoning(chunks: Iterable[str]) -> Iterator[str]:
    """Yield text chunks while suppressing <think>...</think> blocks."""
    buffer = ""
    in_think_block = False
    start_tag_overlap = len("<think") - 1
    end_tag_overlap = len("</think>") - 1

    for chunk in chunks:
        if not chunk:
            continue

        buffer += chunk

        while buffer:
            lower_buffer = buffer.lower()

            if in_think_block:
                end_index = lower_buffer.find("</think>")
                if end_index < 0:
                    buffer = buffer[-end_tag_overlap:]
                    break

                buffer = buffer[end_index + len("</think>") :]
                in_think_block = False
                continue

            start_index = lower_buffer.find("<think")
            if start_index < 0:
                emit_length = max(0, len(buffer) - start_tag_overlap)
                if emit_length == 0:
                    break

                yield buffer[:emit_length]
                buffer = buffer[emit_length:]
                break

            if start_index > 0:
                yield buffer[:start_index]

            tag_end_index = buffer.find(">", start_index)
            if tag_end_index < 0:
                buffer = buffer[start_index:]
                break

            buffer = buffer[tag_end_index + 1 :]
            in_think_block = True

    if buffer and not in_think_block:
        yield strip_hidden_reasoning(buffer)


def parse_question_lines(text: str | None, limit: int = 5) -> list[str]:
    cleaned = strip_hidden_reasoning(text)
    questions: list[str] = []

    for line in cleaned.splitlines():
        line = line.strip()
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[\.)、]\s*", "", line)
        if not line or line.startswith("<"):
            continue
        if not line.endswith(("?", "？")):
            continue
        if _GENERIC_ASSISTANT_QUESTION_RE.search(line):
            continue
        questions.append(line)
        if len(questions) >= limit:
            break

    return questions


def _parse_response_format(response_format_str: str) -> Dict[str, str]:
    """
    Parses the requirements string into a dictionary.

    Args:
        req_str (str): The requirements string.

    Returns:
        Dict[str, str]: A dictionary of parsed requirements.
    """
    requirements = {}
    parts = re.split(r",\s*(?=\w[\w\s]*:\s*[^,()]+)", response_format_str)
    for part in parts:
        if ":" in part:
            key, value = part.split(":", 1)
            requirements[key.strip()] = value.strip()
        else:
            requirements[part.strip()] = None
    return requirements


def parse_goal_response_format(goal: str) -> Tuple[str, Dict[str, str]]:
    """
    Extracts the main goal and its requirements from the input string.

    Args:
        goal (str): The input question string with optional requirements.

    Returns:
        Tuple[str, Dict[str, str]]: A tuple containing the main goal and a dictionary of requirements.
    """
    # Initialize
    clean_goal = goal.strip()
    response_format = None

    # Remove starting quote if present
    if clean_goal.startswith('"'):
        clean_goal = clean_goal[1:].strip()

    # Remove ending quote if present
    if clean_goal.endswith('"'):
        clean_goal = clean_goal[:-1].strip()

    # Function to find the last balanced parentheses by reverse traversal
    def extract_last_parentheses(s: str) -> Tuple[str, str]:
        """
        Extracts the last balanced parentheses content from the string by traversing from the end.

        Args:
            s (str): The input string.

        Returns:
            Tuple[str, str]: A tuple containing the string without the last parentheses
                             and the content within the last parentheses.
        """
        stack = []
        last_close = s.rfind(")")
        if last_close == -1:
            return s, ""  # No closing parenthesis found

        for i in range(last_close, -1, -1):
            if s[i] == ")":
                stack.append(i)
            elif s[i] == "(":
                if stack:
                    stack.pop()
                    if not stack:
                        # Found the matching opening parenthesis
                        return s[:i].strip(), s[i + 1 : last_close].strip()
        return s, ""  # No matching opening parenthesis found

    # Extract the last parentheses content
    clean_goal, req_str = extract_last_parentheses(clean_goal)

    if req_str:
        response_format = _parse_response_format(req_str)
    else:
        response_format = {}

    return clean_goal, response_format
