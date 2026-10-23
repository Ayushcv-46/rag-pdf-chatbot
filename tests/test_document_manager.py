import os
import shutil
import pytest
import fitz
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding

from core_rag import (
    RAGIndexManager,
    compute_file_hash,
    load_pdf_file,
    build_fusion_query_engine
)


def create_sample_pdf(filepath: str, text: str):
    """Create a minimal real PDF containing the given text using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    doc.save(filepath)
    doc.close()


@pytest.fixture
def mock_rag_env(tmp_path):
    """Fixture to set up isolated data and chroma directories with mock embeddings."""
    data_dir = str(tmp_path / "data")
    chroma_dir = str(tmp_path / "chroma_db")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)

    # Use lightweight mock embedding to avoid external model downloads or API keys
    Settings.embed_model = MockEmbedding(embed_dim=16)
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    manager = RAGIndexManager(data_dir=data_dir, chroma_dir=chroma_dir)
    return manager, data_dir, chroma_dir


def test_empty_knowledge_base(mock_rag_env):
    """Ensure an empty data directory returns None without crashing."""
    manager, data_dir, _ = mock_rag_env
    index = manager.sync_index()
    assert index is None
    assert manager.get_indexed_files() == []


def test_add_document(mock_rag_env):
    """Test adding a new PDF document into the index."""
    manager, data_dir, _ = mock_rag_env
    pdf_path = os.path.join(data_dir, "doc1.pdf")
    create_sample_pdf(pdf_path, "This is the first document about artificial intelligence.")

    index = manager.sync_index()
    assert index is not None
    assert "doc1.pdf" in manager.get_indexed_files()
    assert manager.chroma_collection.count() > 0
    assert len(index.docstore.docs) > 0


def test_replace_modified_document(mock_rag_env):
    """Test replacing an existing document with modified content."""
    manager, data_dir, _ = mock_rag_env
    pdf_path = os.path.join(data_dir, "doc1.pdf")
    create_sample_pdf(pdf_path, "Original content before modification.")

    manager.sync_index()
    orig_hash = manager.manifest["doc1.pdf"]["hash"]
    orig_node_ids = set(manager.manifest["doc1.pdf"]["node_ids"])

    # Overwrite doc1.pdf with new content
    create_sample_pdf(pdf_path, "Updated content with completely different information.")
    new_hash = compute_file_hash(pdf_path)
    assert orig_hash != new_hash

    index = manager.sync_index()
    assert index is not None
    assert manager.manifest["doc1.pdf"]["hash"] == new_hash

    # Ensure old nodes are not in docstore
    new_node_ids = set(manager.manifest["doc1.pdf"]["node_ids"])
    for old_id in orig_node_ids:
        assert old_id not in index.docstore.docs


def test_remove_document(mock_rag_env):
    """Test removing a document and ensuring chunks are purged."""
    manager, data_dir, _ = mock_rag_env
    pdf_path = os.path.join(data_dir, "doc_to_delete.pdf")
    create_sample_pdf(pdf_path, "Content that will soon be deleted.")

    manager.sync_index()
    assert "doc_to_delete.pdf" in manager.get_indexed_files()

    # Remove the document
    manager.remove_document("doc_to_delete.pdf")

    assert not os.path.exists(pdf_path)
    assert "doc_to_delete.pdf" not in manager.get_indexed_files()
    assert manager.chroma_collection.count() == 0
    assert len(manager.docstore.docs) == 0

    # Sync again should reflect empty state
    index = manager.sync_index()
    assert index is None


def test_invalid_pdf_handling(mock_rag_env):
    """Test that a corrupt PDF raises an understandable error and leaves valid files safe."""
    manager, data_dir, _ = mock_rag_env
    valid_pdf = os.path.join(data_dir, "valid.pdf")
    create_sample_pdf(valid_pdf, "Valid readable document content.")
    manager.sync_index()
    assert "valid.pdf" in manager.get_indexed_files()

    # Create an invalid corrupt PDF
    corrupt_pdf = os.path.join(data_dir, "corrupt.pdf")
    with open(corrupt_pdf, "wb") as f:
        f.write(b"NOT A REAL PDF HEADER JUST JUNK DATA")

    with pytest.raises(ValueError, match="Could not read PDF"):
        manager.sync_index()

    # Valid file must remain intact in index
    assert "valid.pdf" in manager.get_indexed_files()


def test_multi_document_retrieval_and_removal(mock_rag_env):
    """Test indexing multiple documents and removing one while preserving the other."""
    manager, data_dir, _ = mock_rag_env
    pdf1 = os.path.join(data_dir, "doc1.pdf")
    pdf2 = os.path.join(data_dir, "doc2.pdf")
    create_sample_pdf(pdf1, "First document content.")
    create_sample_pdf(pdf2, "Second document content.")

    index = manager.sync_index()
    assert index is not None
    assert len(manager.get_indexed_files()) == 2

    # Remove doc1
    manager.remove_document("doc1.pdf")
    assert "doc1.pdf" not in manager.get_indexed_files()
    assert "doc2.pdf" in manager.get_indexed_files()

    # Ensure doc2 content is still present in docstore
    doc2_node_ids = manager.manifest["doc2.pdf"]["node_ids"]
    for nid in doc2_node_ids:
        assert nid in manager.docstore.docs


def test_removed_content_not_retrieved_by_search(mock_rag_env):
    """Test that removed document content cannot be retrieved by vector search."""
    manager, data_dir, _ = mock_rag_env
    pdf1 = os.path.join(data_dir, "doc1.pdf")
    pdf2 = os.path.join(data_dir, "doc2.pdf")
    create_sample_pdf(pdf1, "UniqueSecretAlpha999 is only found in doc one.")
    create_sample_pdf(pdf2, "UniqueSecretBeta888 is only found in doc two.")

    index = manager.sync_index()
    assert index is not None

    retriever = index.as_retriever(similarity_top_k=5)
    results = retriever.retrieve("UniqueSecretAlpha999")
    assert any("UniqueSecretAlpha999" in r.node.text for r in results)

    # Now remove doc1
    manager.remove_document("doc1.pdf")
    index = manager.sync_index()
    assert index is not None

    # Query again
    retriever = index.as_retriever(similarity_top_k=5)
    results_after = retriever.retrieve("UniqueSecretAlpha999")
    assert not any("UniqueSecretAlpha999" in r.node.text for r in results_after)

