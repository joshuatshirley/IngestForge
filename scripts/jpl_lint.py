"""
JPL Rule #4 Linter Script.

Task 115: Automated compliance check for NASA JPL Power of Ten (Rule #4).
Ensures all functions remain under 60 lines.
"""

import ast
import os
from pathlib import Path

# NASA JPL Rule #4: Maximum function length (lines of code)
MAX_FUNCTION_LENGTH = 60


def check_file(file_path):
    violations = []
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_line = node.lineno
            end_line = node.end_lineno
            length = end_line - start_line + 1

            if length > MAX_FUNCTION_LENGTH:
                violations.append((node.name, length, start_line))

    return violations


def main():
    target_dir = Path("ingestforge")
    total_violations = 0

    print(f"--- JPL Rule #4 Lint: Max {MAX_FUNCTION_LENGTH} lines per function ---")

    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                try:
                    violations = check_file(path)
                    if violations:
                        print(f"\n[VIOLATION] {path}")
                        for name, length, line in violations:
                            print(
                                f"  - Function '{name}' at line {line}: {length} lines"
                            )
                            total_violations += 1
                except Exception as e:
                    print(f"[ERROR] Could not parse {path}: {e}")

    if total_violations > 0:
        print(f"\nTotal violations: {total_violations}")
    else:
        print("\nAll functions compliant with JPL Rule #4.")


if __name__ == "__main__":
    main()
