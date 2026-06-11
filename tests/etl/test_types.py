from etl.types import ChildChunk, ParentChunk, _compute_hash


def test_compute_hash_is_deterministic():
    assert _compute_hash("hello") == _compute_hash("hello")


def test_compute_hash_differs_by_input():
    assert _compute_hash("hello") != _compute_hash("world")


def test_parent_chunk_hash_auto_computed():
    chunk = ParentChunk(
        section_name="Item 1",
        content_type="narrative",
        text="hello world",
        token_count=2,
        filing_chunk_index=0,
    )
    assert chunk.content_hash == _compute_hash("hello world")


def test_child_chunk_hash_auto_computed():
    chunk = ChildChunk(
        section_name="Item 1",
        content_type="narrative",
        text="hello world",
        token_count=2,
        filing_chunk_index=0,
        parent_chunk_index=0,
    )
    assert chunk.content_hash == _compute_hash("hello world")


def test_child_chunk_embedding_defaults_to_none():
    chunk = ChildChunk(
        section_name="Item 1",
        content_type="narrative",
        text="hello",
        token_count=1,
        filing_chunk_index=0,
        parent_chunk_index=0,
    )
    assert chunk.embedding is None
    assert chunk.embedding_model is None


def test_different_texts_produce_different_hashes():
    c1 = ParentChunk(section_name="s", content_type="narrative", text="aaa", token_count=1, filing_chunk_index=0)
    c2 = ParentChunk(section_name="s", content_type="narrative", text="bbb", token_count=1, filing_chunk_index=0)
    assert c1.content_hash != c2.content_hash
