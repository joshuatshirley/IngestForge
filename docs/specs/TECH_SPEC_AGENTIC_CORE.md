# Technical Spec: Agentic ReAct Engine

## Goal
Implement a robust, finite-step reasoning agent that uses tools (search, ingest, query) to answer complex topics.

---

## 1. Interaction Protocol (The "Thought-Action" Loop)
The agent follows a strict state-machine loop. Recursive calls are FORBIDDEN (JPL Rule #1).

**State Transitions**:
1.  **Input**: Goal string (e.g., "Summarize recent papers on AI alignment").
2.  **Prompt**: System prompt + History + Current Observation.
3.  **LLM Call**: Returns `Thought` and `Action`.
4.  **Parse**: Regex-parse the action name and JSON arguments.
5.  **Dispatch**: Execute tool (Rule #7: Validate tool params before call).
6.  **Observe**: Return tool output to history.
7.  **Increment**: `step_count += 1`.

---

## 2. Tool Registry Specification
Tools must be defined as simple functions with typed inputs.

```python
class ToolRegistry:
    def execute(self, tool_name: str, args: dict) -> str:
        """Rule #7: Parameter validation."""
        assert tool_name in self.tools, f"Unknown tool: {tool_name}"
        # Validate args against tool's JSON schema
        return self.tools[tool_name](**args)
```

---

## 3. Safety & Bounds (JPL Rule #2)
*   **Max Steps**: Hard-coded limit of 10 steps.
*   **Loop Condition**: `while step < max_steps and not finalized:`.
*   **Cost Guard**: Assert that the cumulative token usage is within the project limit (default 100k tokens per goal).

---

## 4. Error Recovery (Rule #1)
Do NOT use deep try-except blocks.
*   **Invalid Action**: If LLM returns a hallucinated tool, provide a "User: Error - tool X does not exist" observation and let the LLM retry.
*   **Empty Results**: If search returns nothing, the observation should suggest "Try a broader query."
