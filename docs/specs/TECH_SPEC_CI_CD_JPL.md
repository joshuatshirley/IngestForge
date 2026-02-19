# Technical Spec: CI/CD Automation (JPL Compliance)

## Goal
Automate the 10 Commandments audit so that no PR can be merged if it violates modularity, nesting, or safety rules.

---

## 1. Static Analysis Suite (Rule #10)
**Workflow**: `.github/workflows/quality_gate.yml`

| Rule | Tool | Threshold |
|------|------|-----------|
| **Rule #1 (Nesting)** | `ruff` (C901) | Max Complexity: 10 |
| **Rule #4 (Modularity)**| `ruff` | Max function lines: 60 |
| **Rule #9 (Types)** | `mypy --strict`| 100% Coverage |
| **Rule #8 (Cleverness)**| `vulture` | Remove 100% of dead code |

---

## 2. Custom Rule Auditors (The "Commandment Script")
**Script**: `scripts/validate_commandments.py`

### 2.1 Assertion Density Checker (Rule #5)
*   Count lines of functional code (LOC).
*   Count `assert` statements.
*   **Fail if**: `assertions / functional_LOC < 0.1`.

### 2.2 Recursion Hunter (Rule #1)
*   Check for functions calling themselves.
*   **Fail if**: Recursion detected.

### 2.3 Global Scope Guard (Rule #6)
*   Scan for module-level variables that are not Constants (UPPER_CASE).
*   **Fail if**: Any module-level stateful objects found.

---

## 3. The "Golden" Integration Test
*   Executes the `quickstart` flow in a fresh virtualenv.
*   Asserts that `.ingestforge/` and `.data/` are correctly populated.
*   Runs `ingestforge doctor` and fails if any core check is not `ok`.
