import re
from typing import Tuple, Dict


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
        question_str (str): The input question string with optional requirements.

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
