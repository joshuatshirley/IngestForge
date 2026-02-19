"""
Code Intelligence Assistant Demo.

Task 313: Demonstrates AST-aware code understanding with precise citations.
Processes an IngestForge file and queries it using the Code vertical.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ingestforge.ingest.code_extractor import extract_to_artifact
from ingestforge.verticals.code.extractor import CodeIntelligenceExtractor
from ingestforge.verticals.code.generator import CodeIntelligenceGenerator
from ingestforge.verticals.code.models import CodeIntelligenceModel


def run_demo():
    print("🚀 Starting Code Intelligence Demo...")

    # 1. Select a file to analyze (using the extractor itself as a meta-demo)
    file_to_analyze = Path("ingestforge/ingest/code_extractor.py")
    if not file_to_analyze.exists():
        print(f"❌ Error: {file_to_analyze} not found.")
        return

    # 2. Perform raw AST extraction to IFCodeArtifact
    print(f"📄 Analyzing {file_to_analyze}...")
    artifact = extract_to_artifact(file_to_analyze)
    print(
        f"✅ Extracted {len(artifact.symbols)} symbols and {len(artifact.imports)} imports."
    )

    # 3. Use the Code Intelligence Vertical to structure the data
    v_extractor = CodeIntelligenceExtractor()
    file_analysis = v_extractor.extract_from_artifact(artifact)

    # 4. Aggregate into a Repository Model
    repo_model = CodeIntelligenceModel(
        module_name="ingestforge.ingest",
        files=[file_analysis],
        total_lines=artifact.line_count,
        architecture_summary="Tree-sitter based code extraction pipeline with JPL compliance.",
        entry_points=["code_extractor.py"],
    )

    # 5. Use the Generator to answer a query
    generator = CodeIntelligenceGenerator()

    query = "How is TreeSitterExtractor implemented and what does it extract?"
    print(f"\n🔍 Query: '{query}'")

    response = generator.synthesize_response(query, repo_model)
    print("\n🤖 Assistant Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)

    # 6. Show Quality Notes (JPL Audit)
    print("\n🛡️ Quality Audit (JPL Power of Ten):")
    for note in file_analysis.quality_notes:
        print(f"  - {note}")


if __name__ == "__main__":
    run_demo()
