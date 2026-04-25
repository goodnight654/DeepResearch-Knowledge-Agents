"""Canonical DeepResearch agent exports.

The old `deepsearch_agent` module remains as a compatibility layer because the
project previously exposed `/deep-search` and `DeepSearchAgent`.
"""

from agents.deepsearch_agent import (
    DeepResearchAgent,
    DeepResearchEvidence,
    DeepResearchResult,
    DeepResearchStep,
    DeepSearchAgent,
    DeepSearchEvidence,
    DeepSearchResult,
    DeepSearchStep,
)

__all__ = [
    "DeepResearchAgent",
    "DeepResearchEvidence",
    "DeepResearchResult",
    "DeepResearchStep",
    "DeepSearchAgent",
    "DeepSearchEvidence",
    "DeepSearchResult",
    "DeepSearchStep",
]
