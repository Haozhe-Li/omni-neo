"""Python code execution tool powered by E2B.

Pure computation sandbox — no plotting, no image output, no GUI.
Use the charting skill for visualisations instead.
"""

from __future__ import annotations

import logging
import os

from e2b_code_interpreter import Sandbox
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Block any attempt to open a display or spawn a GUI inside the sandbox.
_HEADLESS_PRELUDE = """\
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""
"""


@tool
def run_python(filename: str, code: str) -> str:
    """Execute Python code and return stdout, the final expression value, and any errors.

    Use this for computation, data analysis, math, simulations, string processing,
    or anything that requires actual execution rather than approximation.

    This tool is text-only. It cannot produce charts, plots, images, or any visual
    output — do NOT attempt to use matplotlib, PIL, or similar. For visualisations
    use the charting skill instead.

    The sandbox has numpy, pandas, scipy, scikit-learn, sympy, and requests
    pre-installed. For other packages, prepend:
        import subprocess; subprocess.run(["pip", "install", "pkg"], check=True)

    Each call is isolated — do not rely on variables from previous calls.

    Args:
        filename: A short, descriptive name for this snippet, e.g.
            "binary_tree_demo.py" — shown to the user as the label for this
            code. Always end it in ".py".
        code: Complete, self-contained Python code to execute.

    Returns:
        stdout, the value of the last expression (if any), and error details.
    """
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        return "Error: E2B_API_KEY is not set."

    try:
        with Sandbox.create() as sbx:
            execution = sbx.run_code(_HEADLESS_PRELUDE + code)
    except Exception as exc:
        return f"Sandbox error: {exc}"

    parts: list[str] = []

    if execution.logs.stdout:
        stdout = "".join(execution.logs.stdout).strip()
        if stdout:
            parts.append(f"Output:\n{stdout}")

    if execution.logs.stderr:
        stderr = "".join(execution.logs.stderr).strip()
        if stderr:
            parts.append(f"Stderr:\n{stderr}")

    if execution.error:
        err = execution.error
        parts.append(f"Error ({err.name}):\n{err.value}")
        if getattr(err, "traceback", None):
            parts.append(f"Traceback:\n{err.traceback}")

    for result in execution.results:
        if result.text:
            parts.append(f"Result: {result.text}")

    return "\n\n".join(parts) if parts else "Code executed successfully with no output."


if __name__ == "__main__":
    print(run_python.invoke({"filename": "power_of_two.py", "code": "print(2 ** 32)"}))
