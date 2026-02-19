"""
ADO-specific enricher for Azure DevOps work items.

Migrated to IFProcessor interface.
Extracts ADO-specific entities like work item IDs, package references,
Apex class references, and classifies work items.

NASA JPL Power of Ten compliant.
"""

import re
import warnings
from typing import Any, Dict, List

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.logging import get_logger
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFFailureArtifact

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_ENTITIES_PER_CHUNK = 50
MAX_CONCEPTS_PER_CHUNK = 20


class ADOEnricher(IFProcessor):
    """
    Enrich ADO work item chunks with specialized metadata.

    Implements IFProcessor interface.

    Extracts:
    - ADO work item ID references (#12345)
    - Package name references (aie-base-code)
    - Apex class references (AccountsSelector, OpportunitiesService)
    - Work item type classification
    - Parent-child relationship hints

    Rule #9: Complete type hints.
    """

    # Patterns for entity extraction
    ADO_ID_PATTERN = re.compile(r"#(\d{4,6})\b")
    PACKAGE_PATTERN = re.compile(r"\b(aie-[a-z0-9-]+)\b", re.IGNORECASE)
    APEX_CLASS_PATTERN = re.compile(
        r"\b([A-Z][a-zA-Z0-9]*(?:Selector|Service|Handler|Controller|Domain|Batch|Queueable|TriggerHandler))\b"
    )
    LWC_COMPONENT_PATTERN = re.compile(
        r"\b([a-z][a-zA-Z0-9]*(?:Component|Card|Form|List|Table|Modal|Panel))\b"
    )

    # Salesforce object patterns
    SOBJECT_PATTERN = re.compile(
        r"\b(Account|Contact|Lead|Opportunity|Case|Campaign|"
        r"Territory2|User|Task|Event|ContentDocument|"
        r"Party_Profile__c|Person_Employment__c|"
        r"Applicant_Screening__c|Examination__c|Waiver__c|"
        r"Person_Life_Event__c|Received_Document__c)\b",
        re.IGNORECASE,
    )

    # Integration patterns
    INTEGRATION_PATTERN = re.compile(
        r"\b(MIRS|DISS|ARISS|CIMT|MEPS|EMM|SDI|PSI|ECMA|RCN)\b", re.IGNORECASE
    )

    # Keyword patterns for classification
    FEATURE_KEYWORDS = {"feature", "capability", "enhancement", "improvement", "new"}
    BUG_KEYWORDS = {"bug", "fix", "issue", "error", "defect", "broken", "incorrect"}
    SECURITY_KEYWORDS = {
        "security",
        "permission",
        "access",
        "stig",
        "ato",
        "vulnerability",
    }
    INTEGRATION_KEYWORDS = {
        "integration",
        "api",
        "mule",
        "interface",
        "sync",
        "data transfer",
    }
    UI_KEYWORDS = {
        "ui",
        "ux",
        "page layout",
        "component",
        "lwc",
        "flexcard",
        "omnistudio",
    }
    TESTING_KEYWORDS = {"test", "qa", "validation", "verification", "unit test"}

    def __init__(self) -> None:
        """Initialize ADO enricher."""
        self._version = "2.0.0"

    # -------------------------------------------------------------------------
    # IFProcessor Interface Implementation
    # -------------------------------------------------------------------------

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "ado-enricher"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Capabilities provided by this processor."""
        return ["ado-enrichment", "entity-extraction", "domain-classification"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 30  # Regex-based, lightweight

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process an artifact to extract ADO-specific entities.

        Implements IFProcessor.process().
        Rule #4: Function < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact (must be IFChunkArtifact).

        Returns:
            Derived IFChunkArtifact with ADO entities in metadata.
        """
        # Validate input type
        if not isinstance(artifact, IFChunkArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-ado-failure",
                error_message=(
                    f"ADOEnricher requires IFChunkArtifact, "
                    f"got {type(artifact).__name__}"
                ),
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
            )

        # Extract entities and concepts
        entities, concepts = self._extract_ado_entities(artifact.content)

        # Build updated metadata
        new_metadata = dict(artifact.metadata)

        # Merge with existing entities/concepts
        existing_entities = new_metadata.get("entities", [])
        existing_concepts = new_metadata.get("concepts", [])

        new_metadata["entities"] = list(set(existing_entities + entities))[
            :MAX_ENTITIES_PER_CHUNK
        ]
        new_metadata["concepts"] = list(set(existing_concepts + concepts))[
            :MAX_CONCEPTS_PER_CHUNK
        ]
        new_metadata["ado_enricher_version"] = self.version

        # Return derived artifact
        return artifact.derive(
            self.processor_id,
            artifact_id=f"{artifact.artifact_id}-ado",
            metadata=new_metadata,
        )

    def is_available(self) -> bool:
        """
        ADO enricher is always available (pattern-based).

        Implements IFProcessor.is_available().
        """
        return True

    def teardown(self) -> bool:
        """
        Clean up resources.

        Implements IFProcessor.teardown().
        """
        return True

    # -------------------------------------------------------------------------
    # Entity Extraction Logic
    # -------------------------------------------------------------------------

    def _extract_ado_entities(self, content: str) -> tuple:
        """
        Extract ADO-specific entities and concepts from content.

        Rule #4: Function < 60 lines.
        """
        entities: List[str] = []
        concepts: List[str] = []

        # Extract ADO IDs
        ado_ids = set(self.ADO_ID_PATTERN.findall(content))
        for ado_id in ado_ids:
            entities.append(f"ado_id:#{ado_id}")

        # Extract package references
        packages = set(self.PACKAGE_PATTERN.findall(content.lower()))
        for pkg in packages:
            entities.append(f"package:{pkg}")
            concepts.append(pkg)

        # Extract Apex class references
        apex_classes = set(self.APEX_CLASS_PATTERN.findall(content))
        for cls in apex_classes:
            entities.append(f"apex_class:{cls}")
            concepts.append(cls)

        # Extract LWC component references
        lwc_components = set(self.LWC_COMPONENT_PATTERN.findall(content))
        for comp in lwc_components:
            entities.append(f"lwc:{comp}")

        # Extract SObject references
        sobjects = set(match.lower() for match in self.SOBJECT_PATTERN.findall(content))
        for obj in sobjects:
            entities.append(f"sobject:{obj}")

        # Extract integration references
        integrations = set(
            match.upper() for match in self.INTEGRATION_PATTERN.findall(content)
        )
        for integ in integrations:
            entities.append(f"integration:{integ}")
            concepts.append(integ)

        # Classify work item domain
        domain = self._classify_domain(content)
        if domain:
            entities.append(f"domain:{domain}")

        return entities, concepts

    def _classify_domain(self, content: str) -> str:
        """
        Classify the work item into a domain category.

        Rule #4: Function < 60 lines.
        """
        content_lower = content.lower()

        # Count keyword matches for each domain
        scores = {
            "security": sum(1 for kw in self.SECURITY_KEYWORDS if kw in content_lower),
            "integration": sum(
                1 for kw in self.INTEGRATION_KEYWORDS if kw in content_lower
            ),
            "ui": sum(1 for kw in self.UI_KEYWORDS if kw in content_lower),
            "testing": sum(1 for kw in self.TESTING_KEYWORDS if kw in content_lower),
        }

        # Return the domain with highest score (if any)
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)

        return ""

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def extract_references(self, content: str) -> Dict[str, List[str]]:
        """
        Extract all references from content.

        Useful for building cross-references between work items and code.

        Args:
            content: Text content to analyze

        Returns:
            Dictionary with reference types and their values
        """
        return {
            "ado_ids": list(set(self.ADO_ID_PATTERN.findall(content))),
            "packages": list(set(self.PACKAGE_PATTERN.findall(content.lower()))),
            "apex_classes": list(set(self.APEX_CLASS_PATTERN.findall(content))),
            "lwc_components": list(set(self.LWC_COMPONENT_PATTERN.findall(content))),
            "sobjects": list(
                set(m.lower() for m in self.SOBJECT_PATTERN.findall(content))
            ),
            "integrations": list(
                set(m.upper() for m in self.INTEGRATION_PATTERN.findall(content))
            ),
        }

    def get_linked_work_items(self, content: str) -> List[int]:
        """
        Extract all linked work item IDs from content.

        Args:
            content: Text content to analyze

        Returns:
            List of work item IDs referenced in the content
        """
        return [int(id) for id in self.ADO_ID_PATTERN.findall(content)]

    def get_linked_code_entities(self, content: str) -> List[str]:
        """
        Extract all code entity references (classes, packages) from content.

        Args:
            content: Text content to analyze

        Returns:
            List of code entity names
        """
        entities: List[str] = []
        entities.extend(self.APEX_CLASS_PATTERN.findall(content))
        entities.extend(self.PACKAGE_PATTERN.findall(content.lower()))
        entities.extend(self.LWC_COMPONENT_PATTERN.findall(content))
        return list(set(entities))

    # -------------------------------------------------------------------------
    # Legacy API (Backward Compatibility)
    # -------------------------------------------------------------------------

    def enrich_chunk(self, chunk: ChunkRecord) -> ChunkRecord:
        """
        Add ADO-specific entities to chunk.

        .. deprecated:: 2.0.0
            Use :meth:`process` with IFChunkArtifact instead.

        Args:
            chunk: ChunkRecord to enrich

        Returns:
            Chunk with entities populated
        """
        warnings.warn(
            "enrich_chunk() is deprecated. Use process() with IFChunkArtifact instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        content = chunk.content
        entities, concepts = self._extract_ado_entities(content)
        self._merge_entities_and_concepts(chunk, entities, concepts)
        return chunk

    def _merge_entities_and_concepts(
        self,
        chunk: ChunkRecord,
        entities: List[str],
        concepts: List[str],
    ) -> None:
        """Merge extracted entities and concepts into chunk."""
        if chunk.entities:
            chunk.entities.extend(entities)
        else:
            chunk.entities = entities

        if chunk.concepts:
            chunk.concepts.extend(concepts)
        else:
            chunk.concepts = concepts

        # Deduplicate with bounds
        chunk.entities = list(set(chunk.entities))[:MAX_ENTITIES_PER_CHUNK]
        chunk.concepts = list(set(chunk.concepts))[:MAX_CONCEPTS_PER_CHUNK]

    def enrich_batch(
        self, chunks: List[ChunkRecord], **kwargs: Any
    ) -> List[ChunkRecord]:
        """
        Enrich multiple chunks.

        .. deprecated:: 2.0.0
            Use :meth:`process` with IFChunkArtifact instead.
        """
        warnings.warn(
            "enrich_batch() is deprecated. Use process() with IFChunkArtifact instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [self._enrich_chunk_internal(chunk) for chunk in chunks]

    def _enrich_chunk_internal(self, chunk: ChunkRecord) -> ChunkRecord:
        """Internal method without deprecation warning."""
        content = chunk.content
        entities, concepts = self._extract_ado_entities(content)
        self._merge_entities_and_concepts(chunk, entities, concepts)
        return chunk
