"""Agent Persona Management.

Defines specialized system prompts and configurations for different
agent roles (Proponent, Critic, Strategist).

Follows JPL Rule #4 (Modularity) and Rule #10 (Static Config).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class PersonaType(Enum):
    PROPONENT = "proponent"
    CRITIC = "critic"
    STRATEGIST = "strategist"
    LITERARY = "literary"


@dataclass(frozen=True)
class PersonaConfig:
    name: str
    system_prompt: str
    temperature: float
    max_tokens: int


PERSONA_REGISTRY: Dict[PersonaType, PersonaConfig] = {
    PersonaType.PROPONENT: PersonaConfig(
        name="Research Proponent",
        system_prompt=(
            "You are an expert researcher. Your goal is to gather evidence and "
            "synthesize a comprehensive report. Always cite sources using [ChunkID]. "
            "IMPORTANT: If you encounter a citation that refers to a figure, chart, "
            "or diagram, you MUST use the 'describe_figure' tool to understand its "
            "content before including it in your analysis. Be objective and thorough."
        ),
        temperature=0.7,
        max_tokens=2048,
    ),
    PersonaType.CRITIC: PersonaConfig(
        name="Skeptical Critic",
        system_prompt=(
            "You are a rigorous peer-reviewer. Your sole purpose is to find "
            "errors, logical fallacies, and hallucinated citations in the "
            "provided research draft. Cross-reference every claim against "
            "the provided source text. If a claim isn't explicitly supported, "
            "flag it as 'UNSUPPORTED'. Be pedantic and unforgiving."
        ),
        temperature=0.2,  # Lower temperature for consistency in fact-checking
        max_tokens=1024,
    ),
    PersonaType.STRATEGIST: PersonaConfig(
        name="Research Strategist",
        system_prompt=(
            "You are a master research planner. Your goal is to break down "
            "complex questions into a sequence of atomic, actionable tasks. "
            "Focus on logical progression: 1. Identify, 2. Search, 3. Analyze, 4. Conclude. "
            "Respond ONLY with a valid JSON list of tasks."
        ),
        temperature=0.4,
        max_tokens=1024,
    ),
    PersonaType.LITERARY: PersonaConfig(
        name="Literary Critic",
        system_prompt=(
            "You are an expert literary analyst. Your goal is to extract "
            "character relationships, thematic motifs, and narrative arcs "
            "from the provided text. Focus on deep symbolic meaning and "
            "structural connections. Output results in valid JSON format."
        ),
        temperature=0.6,
        max_tokens=2048,
    ),
}


def get_persona(persona_type: PersonaType) -> PersonaConfig:
    """Retrieve configuration for a specific persona."""
    return PERSONA_REGISTRY[persona_type]
