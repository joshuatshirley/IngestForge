"""
Cross-Vertical Lateral Connection Engine.

Identifies connections between documents in different domains
(e.g., linking a Cyber vulnerability to an Urban Planning risk).
"""

import logging
from typing import List, Dict, Any
from collections import defaultdict


logger = logging.getLogger(__name__)


class LateralLinker:
    """
    Finds lateral connections between domain-specific metadata.
    """

    def find_connections(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """
        Scans a list of chunks and returns a list of discovered connections.

        Connection Types:
        1. Shared Entity: Common person/org/loc across domains.
        2. Shared Identifier: CVE, Part Number, or ID appearing in mixed contexts.
        3. Semantic Overlap: Domain-specific keys matching across verticals.
        """
        connections = []

        # Index chunks by entities and domain-specific IDs
        entity_map = defaultdict(list)
        id_map = defaultdict(list)

        for chunk in chunks:
            metadata = chunk.metadata or {}
            domains = metadata.get("detected_domains", [metadata.get("primary_domain")])
            domains = [d for d in domains if d]

            # Index by NER entities
            entities = getattr(chunk, "entities", []) or []
            for ent in entities:
                entity_map[str(ent).lower()].append((chunk, domains))

            # Index by Domain-Specific IDs (CVEs, Part Numbers, etc.)
            id_keys = [
                "cyber_cve_id",
                "cyber_all_cves",
                "auto_part_number",
                "bio_experiment_id",
                "mfg_part_number",
                "urban_zoning_code",
            ]
            for key in id_keys:
                if key in metadata:
                    val = metadata[key]
                    vals = val if isinstance(val, list) else [val]
                    for v in vals:
                        id_map[str(v).upper()].append((chunk, domains))

        # Identify "Cross-Domain" Entity Links
        for entity, occurrences in entity_map.items():
            if len(occurrences) < 2:
                continue

            # Check if occurrences span different domains
            domain_set = set()
            for _, doms in occurrences:
                domain_set.update(doms)

            if len(domain_set) > 1:
                connections.append(
                    {
                        "type": "lateral_entity_link",
                        "entity": entity,
                        "domains": list(domain_set),
                        "chunk_ids": [c[0].chunk_id for c in occurrences],
                    }
                )

        # Identify "Cross-Domain" ID Links (Very Strong)
        for identifier, occurrences in id_map.items():
            if len(occurrences) < 2:
                continue

            domain_set = set()
            for _, doms in occurrences:
                domain_set.update(doms)

            if len(domain_set) > 1:
                connections.append(
                    {
                        "type": "cross_domain_id_collision",
                        "id": identifier,
                        "domains": list(domain_set),
                        "chunk_ids": [c[0].chunk_id for c in occurrences],
                    }
                )

        return connections
