"""Extract fixed model-test classifications; never echo arbitrary container logs."""

import re
import sys

seen = set()
for line in sys.stdin:
    match = re.search(
        r"Model test failed \((configuration|llm|dspy), ([A-Za-z_][A-Za-z0-9_]{0,79})\)",
        line,
    )
    if match:
        seen.add("model-test: " + "/".join(match.groups()))
    status = re.search(r'"POST /api/v1/admin/llms/test HTTP/1\.[01]" ([1-5][0-9]{2})\b', line)
    if status:
        seen.add("model-test HTTP status: " + status.group(1))
for classification in sorted(seen):
    print("[E2E server diagnostics] " + classification)
if not seen:
    print("[E2E server diagnostics] No classified model-test result available")
