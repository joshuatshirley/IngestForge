"""Debate command - Generate pro/con analysis.

Analyzes both sides of an argument with evidence-based reasoning
using multi-agent adversarial debate.

Follows Commandments #4 (Small Functions), #7 (Check Parameters), and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.argument.base import ArgumentCommand
from ingestforge.agent.fact_checker import Claim, DebateOrchestrator
from ingestforge.agent.debate_adapter import (
    create_proponent_function,
    create_critic_function,
)
from ingestforge.agent.verification_ui import VerificationDisplay


class DebateCommand(ArgumentCommand):
    """Generate pro/con debate analysis."""

    def execute(
        self,
        claim: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """Generate debate analysis for a claim using adversarial debate."""
        try:
            self.validate_claim(claim)
            ctx = self.initialize_context(project, require_storage=False)

            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            # Create debate orchestrator with LLM-powered agents
            proponent_fn = create_proponent_function(llm_client)
            critic_fn = create_critic_function(llm_client)
            orchestrator = DebateOrchestrator(
                proponent_fn=proponent_fn,
                critic_fn=critic_fn,
                max_rounds=3,
            )

            # Create claim object
            claim_obj = Claim(
                content=claim,
                source="user_input",
                context="",
            )

            # Run verification
            self.console.print("\n[cyan]Running adversarial verification...[/cyan]\n")
            result = orchestrator.verify(claim_obj)

            # Display results using rich UI
            display = VerificationDisplay(console=self.console)
            display.display(result)

            # Save if requested
            if output:
                result_dict = result.to_dict()
                self.save_json_output(
                    output,
                    result_dict,
                    f"Verification result saved to: {output}",
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Debate generation failed")


def command(
    claim: str = typer.Argument(..., help="Claim to analyze"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (JSON)"
    ),
) -> None:
    """Generate pro/con debate analysis for a claim.

    Examples:
        ingestforge argument debate "AI will replace human workers"
        ingestforge argument debate "Climate change is urgent" -o debate.json
    """
    cmd = DebateCommand()
    exit_code = cmd.execute(claim, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
