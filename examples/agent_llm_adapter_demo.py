"""Demo script showing how to use the LLM Adapter with ReActEngine.

This example demonstrates:
1. Creating an LLM adapter
2. Connecting it to a ReAct engine
3. Registering tools
4. Running an autonomous research task"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for tests import
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingestforge.agent.llm_adapter import create_llm_think_adapter
from ingestforge.agent.react_engine import ReActEngine, SimpleTool
from ingestforge.llm.base import GenerationConfig

# For this demo, we'll use a simple mock LLM
# In production, you'd use OllamaClient, ClaudeClient, etc.
from tests.fixtures.agents import MockLLM


def create_demo_tools() -> list[SimpleTool]:
    """Create demo tools for the agent."""
    search_tool = SimpleTool(
        name="search_web",
        description="Search the web for information",
        _fn=lambda query: f"Found articles about: {query}",
    )

    calculate_tool = SimpleTool(
        name="calculate",
        description="Perform mathematical calculations",
        _fn=lambda expression: f"Result: {eval(expression)}",
    )

    summarize_tool = SimpleTool(
        name="summarize",
        description="Summarize a piece of text",
        _fn=lambda text: f"Summary of {len(text)} chars",
    )

    return [search_tool, calculate_tool, summarize_tool]


def run_demo() -> None:
    """Run the demo agent."""
    print("=== LLM Adapter Demo ===\n")

    # 1. Create a mock LLM (in production, use real LLM client)
    llm = MockLLM()
    llm.set_responses(
        [
            # First iteration: search for info
            "Thought: I need to search for information about solar panels\n"
            "Action: search_web\n"
            'Action Input: {"query": "solar panel efficiency 2024"}',
            # Second iteration: calculate something
            "Thought: Now I'll calculate the energy output\n"
            "Action: calculate\n"
            'Action Input: {"expression": "350 * 0.20"}',
            # Final iteration: provide answer
            "Thought: I have all the information needed\n"
            "Action: FINISH\n"
            "Action Input: {}",
        ]
    )

    # 2. Create LLM adapter with custom config
    config = GenerationConfig(
        max_tokens=1000,
        temperature=0.7,
        stop_sequences=["Observation:"],
    )
    adapter = create_llm_think_adapter(llm_client=llm, config=config)

    # 3. Create ReAct engine with the adapter
    engine = ReActEngine(think_fn=adapter, max_iterations=10)

    # 4. Register tools
    for tool in create_demo_tools():
        engine.register_tool(tool)

    print(
        f"Registered {len(engine.tool_names)} tools: {', '.join(engine.tool_names)}\n"
    )

    # 5. Run the agent
    task = "Research solar panel efficiency and calculate potential output"
    print(f"Task: {task}\n")

    result = engine.run(task)

    # 6. Display results
    print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Iterations: {result.iterations}")
    print(f"Final State: {result.state.value}")
    print("\nStep-by-step execution:")

    for step in result.steps:
        print(f"\n--- Step {step.iteration + 1} ---")
        print(f"Thought: {step.thought}")
        if step.action:
            print(f"Action: {step.action}")
            print(f"Input: {step.action_input}")
            print(f"Observation: {step.observation}")

    if result.final_answer:
        print("\n=== Final Answer ===")
        print(result.final_answer)


if __name__ == "__main__":
    run_demo()
