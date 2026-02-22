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


def is_safe_code(code: str) -> tuple[bool, str]:
    """Very basic AST-level security check to prevent obvious harmful operations.

    Args:
        code (str): The code to check.

    Returns:
        tuple[bool, str]: (True if safe, error message if any)
    """
    forbidden_modules = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "urllib",
        "requests",
        "pathlib",
        "pty",
        "glob",
        "pickle",
        "shelve",
        "marshal",
        "dbm",
        "sqlite3",
    }
    # Block common file-saving methods used in data science libraries
    forbidden_funcs = {
        "eval",
        "exec",
        "open",
        "__import__",
        "compile",
        "save",
        "savefig",
        "to_csv",
        "to_json",
        "to_excel",
        "to_parquet",
        "to_pickle",
        "to_feather",
        "to_stata",
        "to_hdf",
        "to_sql",
        "dump",
    }
    forbidden_attrs = {
        "__builtins__",
        "__class__",
        "__bases__",
        "__subclasses__",
        "__getattribute__",
        "__globals__",
    }

    try:
        tree = ast.parse(code)
    except Exception as e:
        return False, f"Parse error: {str(e)}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module in forbidden_modules:
                    return False, f"Importing '{base_module}' is strictly forbidden."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split(".")[0]
                if base_module in forbidden_modules:
                    return (
                        False,
                        f"Importing from '{base_module}' is strictly forbidden.",
                    )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_funcs:
                return False, f"Calling '{node.func.id}()' is strictly forbidden."
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in forbidden_funcs
            ):
                return (
                    False,
                    f"Calling '{node.func.attr}()' via attribute is forbidden.",
                )
        elif isinstance(node, ast.Name):
            if node.id in forbidden_attrs:
                return False, f"Accessing magic attribute '{node.id}' is forbidden."
        elif isinstance(node, ast.Attribute):
            if node.attr in forbidden_attrs:
                return False, f"Accessing magic attribute '{node.attr}' is forbidden."

    return True, "Safe"


def _run_python_code(
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
                cwd=cwd
                or td,  # Force CWD to temp dir if not specified to prevent file persistence
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
    is_safe, reason = is_safe_code(code)
    if not is_safe:
        return json.dumps(
            {
                "ok": False,
                "returncode": 1,
                "stdout": "",
                "stderr": f"Security Error: {reason}\\nThis operation was blocked by the security sandbox.",
                "timed_out": False,
            }
        )

    run_res = _run_python_code(code)
    return json.dumps(
        {
            "ok": run_res.ok,
            "returncode": run_res.returncode,
            "stdout": run_res.stdout,
            "stderr": run_res.stderr,
            "timed_out": run_res.timed_out,
        }
    )


if __name__ == "__main__":
    print(run_python_tool("import os"))
    print(run_python_tool("print('hello')"))
