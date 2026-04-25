import hashlib
from dataclasses import dataclass, field
from typing import Literal


def _compute_hash(text: str) -> str:
    """Return SHA-256 hex digest of text — used as content fingerprint for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ParsedSection:
    section_name: str                                      # e.g. "Item 1A Risk Factors"
    content_type: Literal["narrative", "table"]
    text: str
    section_order: int                                     # 0-based position among all sections in the filing


@dataclass
class ParentChunk:
    section_name: str
    content_type: Literal["narrative", "table"]
    text: str
    token_count: int
    filing_chunk_index: int                                # 0-based position among all parent chunks in the filing
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.content_hash = _compute_hash(self.text)


@dataclass
class ChildChunk:
    section_name: str
    content_type: Literal["narrative", "table"]
    text: str                                              # prefix + raw content — ready to embed
    token_count: int
    filing_chunk_index: int                                # 0-based position among all child chunks in the filing
    parent_chunk_index: int                                # filing_chunk_index of the parent this child belongs to
    content_hash: str = field(init=False)
    embedding: list[float] | None = field(default=None)        # filled by Embedder; None until then
    embedding_model: str | None = field(default=None)          # set by pipeline alongside embedding

    def __post_init__(self) -> None:
        self.content_hash = _compute_hash(self.text)
