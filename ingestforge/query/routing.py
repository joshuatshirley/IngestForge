"""
Intelligent Query Routing and Domain-Specific Optimization.

Maps classified query domains to optimized retrieval strategies,
weight modifiers, and tool activation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DomainStrategy:
    """Strategy for domain-specific retrieval optimization."""

    name: str
    bm25_modifier: float = 1.0
    semantic_modifier: float = 1.0
    required_metadata: List[str] = field(default_factory=list)
    boost_fields: Dict[str, float] = field(default_factory=dict)
    active_tools: List[str] = field(default_factory=list)


DOMAIN_STRATEGIES: Dict[str, DomainStrategy] = {
    "legal": DomainStrategy(
        name="legal",
        bm25_modifier=1.2,  # Boost precise terminology
        required_metadata=["docket_number", "jurisdiction", "citation"],
        boost_fields={"citation": 2.0, "court": 1.5},
        active_tools=["legal_citation_resolver"],
    ),
    "tech": DomainStrategy(
        name="tech",
        bm25_modifier=1.1,
        semantic_modifier=0.9,  # Code can be semantically noisy
        required_metadata=["language", "imports"],
        boost_fields={"function_name": 2.0, "class_name": 2.0},
        active_tools=["tree_sitter_lookup"],
    ),
    "pkm": DomainStrategy(
        name="pkm",
        bm25_modifier=0.8,
        semantic_modifier=1.2,  # PKM relies on interconnected ideas
        required_metadata=["aliases", "tags"],
        boost_fields={"aliases": 1.5},
        active_tools=["wikilink_resolver"],
    ),
    "cyber": DomainStrategy(
        name="cyber",
        bm25_modifier=1.3,  # IP addresses and CVEs need exact match
        required_metadata=["src_ip", "event_id", "cve_id"],
        boost_fields={"cve_id": 3.0, "event_id": 2.0},
        active_tools=["attack_mapper", "timeline_builder"],
    ),
    "genealogy": DomainStrategy(
        name="genealogy",
        bm25_modifier=1.1,
        required_metadata=["birth_date", "parent_of"],
        boost_fields={"name": 2.0},
        active_tools=["family_tree_walker"],
    ),
    "rpg": DomainStrategy(
        name="rpg",
        semantic_modifier=1.1,
        required_metadata=["rpg_type", "rpg_stats"],
        boost_fields={"rpg_type": 1.5},
        active_tools=["stat_block_viewer"],
    ),
    "urban": DomainStrategy(
        name="urban",
        bm25_modifier=1.2,  # Zoning codes are exact
        required_metadata=["urban_zoning_code", "urban_far_ratio"],
        boost_fields={"urban_zoning_code": 2.5},
    ),
    "ai_safety": DomainStrategy(
        name="ai_safety",
        semantic_modifier=1.2,  # Highly conceptual
        required_metadata=["ai_model_name", "ai_param_count"],
        boost_fields={"ai_model_name": 2.0},
    ),
    "gaming": DomainStrategy(
        name="gaming",
        bm25_modifier=1.1,
        required_metadata=["game_stat_changes", "game_patch_ver"],
        boost_fields={"game_stat_changes": 1.5},
    ),
    "auto": DomainStrategy(
        name="auto",
        bm25_modifier=1.4,  # Part numbers must be exact
        required_metadata=["auto_vin", "auto_part_number"],
        boost_fields={"auto_part_number": 3.0, "auto_vin": 2.0},
    ),
    "bio": DomainStrategy(
        name="bio",
        bm25_modifier=1.2,
        required_metadata=["bio_experiment_id", "bio_lab_ref"],
        boost_fields={"bio_experiment_id": 2.0},
    ),
    "museum": DomainStrategy(
        name="museum",
        semantic_modifier=1.1,
        required_metadata=["museum_artist", "museum_era"],
        boost_fields={"museum_artist": 2.0},
    ),
    "spiritual": DomainStrategy(
        name="spiritual",
        bm25_modifier=1.2,  # Verse citations
        required_metadata=["spiritual_scripture", "spiritual_verse_ref"],
        boost_fields={"spiritual_verse_ref": 2.0},
    ),
    "wellness": DomainStrategy(
        name="wellness",
        bm25_modifier=1.1,
        required_metadata=["wellness_calories", "wellness_macros"],
        boost_fields={"wellness_macros": 1.5},
    ),
}


def get_domain_strategy(domain: str) -> Optional[DomainStrategy]:
    """Get the strategy for a specific domain."""
    return DOMAIN_STRATEGIES.get(domain.lower())


def get_merged_strategy(domains: List[str]) -> DomainStrategy:
    """Merge multiple domain strategies (e.g. for ambiguous queries)."""
    if not domains:
        return DomainStrategy(name="default")

    if len(domains) == 1:
        return get_domain_strategy(domains[0]) or DomainStrategy(name="default")

    # Merge logic: average modifiers, union of required metadata and tools
    merged = DomainStrategy(name="merged")
    bm25_mods = []
    sem_mods = []

    for d in domains:
        strat = get_domain_strategy(d)
        if strat:
            bm25_mods.append(strat.bm25_modifier)
            sem_mods.append(strat.semantic_modifier)
            merged.required_metadata.extend(strat.required_metadata)
            merged.active_tools.extend(strat.active_tools)
            # Merge boost fields (take max boost)
            for field, boost in strat.boost_fields.items():
                merged.boost_fields[field] = max(
                    merged.boost_fields.get(field, 0), boost
                )

    if bm25_mods:
        merged.bm25_modifier = sum(bm25_mods) / len(bm25_mods)
    if sem_mods:
        merged.semantic_modifier = sum(sem_mods) / len(sem_mods)

    # Unique lists
    merged.required_metadata = list(set(merged.required_metadata))
    merged.active_tools = list(set(merged.active_tools))

    return merged
