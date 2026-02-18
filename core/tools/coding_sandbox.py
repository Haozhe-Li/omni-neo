from __future__ import annotations
import json
import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional, Dict


def check_compile(code_string: str) -> tuple[bool, str]:
    """
    Test if a Python code string can be compiled.

    Args:
        code_string (str): Python code to test

    Returns:
        tuple: (bool, str) - (True if compiles, error message if any)
    """
    try:
        ast.parse(code_string)
        compile(code_string, "<string>", "exec")
        return True, "Code compiles successfully"
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except (ValueError, TypeError) as e:
        return False, f"Compilation error: {str(e)}"


@dataclass
class RunResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def run_python_code(
    code: str,
    *,
    timeout_s: float = 2.0,
    python_executable: Optional[str] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> RunResult:
    py = python_executable or sys.executable

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/snippet.py"
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            cp = subprocess.run(
                [py, path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=cwd,
                env=env,
            )
            return RunResult(
                ok=(cp.returncode == 0),
                returncode=cp.returncode,
                stdout=cp.stdout,
                stderr=cp.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as e:
            return RunResult(
                ok=False,
                returncode=-1,
                stdout=(e.stdout or "")
                if isinstance(e.stdout, str)
                else (e.stdout or b"").decode("utf-8", "replace"),
                stderr=(e.stderr or "")
                if isinstance(e.stderr, str)
                else (e.stderr or b"").decode("utf-8", "replace"),
                timed_out=True,
            )


def run_python_tool(code: str) -> str:
    """Executes Python code.

    Args:
        code (str): The Python code to execute. You must explicitly print the output, otherwise it will be empty.

    Returns:
        str: The result of the executed code.
    """

    run_res = run_python_code(code)
    return json.dumps(
        {
            "ok": run_res.ok,
            "returncode": run_res.returncode,
            "stdout": run_res.stdout,
            "stderr": run_res.stderr,
            "timed_out": run_res.timed_out,
        }
    )
