"""Fusion strategies for combining multiple ranked lists into a single ranking."""
from retrieval.fusion.base import BaseFusion
from retrieval.fusion.rrf import RRFFusion

__all__ = ["BaseFusion", "RRFFusion"]
