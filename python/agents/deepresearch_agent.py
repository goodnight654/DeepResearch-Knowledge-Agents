"""Canonical DeepResearch public imports.

The implementation stays in ``deepsearch_agent`` so existing import paths keep
working; new callers should import the DeepResearch names from this module.
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
