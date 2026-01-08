import base64
import io
import os
from pathlib import Path

import pytest

from practorflow.llm.document.document_loader import DocumentLoader


@pytest.fixture
def loader():
    return DocumentLoader(
        use_docling=False,
        retrieval_chunk_size=50,
        retrieval_chunk_overlap=10,
        context_chunk_size=100,
        context_chunk_overlap=20,
    )


@pytest.fixture
def temp_text_file(tmp_path: Path):
    content = (
        "This is sentence one. "
        "This is sentence two. "
        "This is sentence three. "
        "This is sentence four."
    )
    path = tmp_path / "test.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_supported_extensions_contains_text_and_code(loader):
    assert ".txt" in loader.supported_extensions
    assert ".py" in loader.supported_extensions


def test_is_supported_with_supported_extension(loader):
    assert loader.is_supported("file.txt") is True
    assert loader.is_supported("script.py") is True


def test_is_supported_with_unsupported_extension(loader):
    assert loader.is_supported("archive.zip") is False


def test_generate_unique_id_uniqueness(loader):
    id1 = loader._generate_unique_id("file")
    id2 = loader._generate_unique_id("file")

    assert id1 != id2
    assert id1.startswith("file_")
    assert id2.startswith("file_")


def test_load_file_basic_text(temp_text_file, loader):
    result = loader.load_file(str(temp_text_file))

    assert result["filename"] == "test.txt"
    assert result["file_type"] == ".txt"
    assert isinstance(result["content"], str)

    assert result["metadata"]["retrieval_chunk_count"] > 0
    assert result["metadata"]["context_chunk_count"] > 0

    assert len(result["retrieval_chunks"]) > 0
    assert len(result["context_chunks"]) > 0


def test_load_file_nonexistent_raises(loader):
    with pytest.raises(FileNotFoundError):
        loader.load_file("does_not_exist.txt")


def test_load_file_directory_raises(tmp_path, loader):
    with pytest.raises(ValueError):
        loader.load_file(str(tmp_path))


def test_load_from_bytes_text(loader):
    text = b"Hello world. This is a test document."
    result = loader.load_from_bytes(text, filename="bytes.txt")

    assert result["file_type"] == ".txt"
    assert "Hello world" in result["content"]
    assert len(result["retrieval_chunks"]) > 0
    assert len(result["context_chunks"]) > 0


def test_load_from_stream(loader):
    stream = io.BytesIO(b"Streamed content for testing.")
    result = loader.load_from_stream(stream, filename="stream.txt")

    assert result["filename"] == "stream.txt"
    assert "Streamed content" in result["content"]


def test_load_from_base64_plain(loader):
    text = b"Base64 encoded text."
    encoded = base64.b64encode(text).decode("utf-8")

    result = loader.load_from_base64(encoded, filename="base64.txt")

    assert result["filename"] == "base64.txt"
    assert "Base64 encoded text" in result["content"]


def test_load_from_base64_with_data_uri(loader):
    text = b"Data URI content."
    encoded = base64.b64encode(text).decode("utf-8")
    data_uri = f"data:text/plain;base64,{encoded}"

    result = loader.load_from_base64(data_uri, filename="data.txt")

    assert "Data URI content" in result["content"]


def test_chunk_parent_child_relationship(loader):
    content = (
        "Sentence one. Sentence two. Sentence three. " "Sentence four. Sentence five."
    )

    retrieval, context = loader._create_small_to_big_chunks(
        content=content,
        filename="test.txt",
        extension=".txt",
    )

    assert len(context) > 0
    assert len(retrieval) > 0

    context_ids = {c["id"] for c in context}

    for chunk in retrieval:
        assert chunk["parent_id"] in context_ids
        assert chunk["metadata"]["chunk_type"] == "retrieval"


def test_determine_extension_from_filename(loader):
    ext = loader._determine_extension(b"abc", "file.md", None)
    assert ext == ".md"


def test_determine_extension_from_mime(loader):
    ext = loader._determine_extension(b"abc", None, "text/plain")
    assert ext == ".txt"


def test_determine_extension_pdf_magic_bytes(loader):
    ext = loader._determine_extension(b"%PDF-1.4 something", None, None)
    assert ext == ".pdf"


def test_load_file_unsupported_extension_raises(tmp_path):
    from practorflow.llm.document.document_loader import DocumentLoader

    loader = DocumentLoader(use_docling=False)

    # Create a file with an unsupported extension
    bad_file = tmp_path / "file.unsupported"
    bad_file.write_text("some content", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        loader.load_file(str(bad_file))

    assert "Unsupported file type" in str(exc_info.value)


def test_load_from_bytes_cannot_determine_file_type_raises(monkeypatch):
    from practorflow.llm.document.document_loader import DocumentLoader

    loader = DocumentLoader(use_docling=False)

    # Force _determine_extension to return None
    monkeypatch.setattr(
        loader,
        "_determine_extension",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError) as exc_info:
        loader.load_from_bytes(
            file_bytes=b"some random bytes",
            filename="unknown",
            mime_type=None,
        )

    assert str(exc_info.value) == "Could not determine file type"


def test_load_from_bytes_generates_filename_when_missing():
    from practorflow.llm.document.document_loader import DocumentLoader

    loader = DocumentLoader(use_docling=False)

    result = loader.load_from_bytes(
        file_bytes=b"Hello world",
        filename=None,
        mime_type="text/plain",
    )

    # Filename should be auto-generated
    assert result["filename"] == "upload.txt"
    assert result["file_type"] == ".txt"
    assert "Hello world" in result["content"]


def test_split_text_with_empty_text_returns_empty_list():
    from practorflow.llm.document.document_loader import DocumentLoader

    loader = DocumentLoader(use_docling=False)

    chunks = loader._split_text(
        text="",
        chunk_size=100,
        overlap=10,
    )

    assert chunks == []


def test_split_text_overlap_sentences_are_carried_over():
    from practorflow.llm.document.document_loader import DocumentLoader

    loader = DocumentLoader(use_docling=False)

    # Carefully crafted sentences with predictable lengths
    text = (
        "AAAAA. "  # len ~= 6
        "BBBBB. "  # len ~= 6
        "CCCCC. "  # len ~= 6
        "DDDDD."  # len ~= 6
    )

    # Force:
    # - chunk_size small enough to split after ~2 sentences
    # - overlap large enough to include exactly 1 sentence
    chunks = loader._split_text(
        text=text,
        chunk_size=15,
        overlap=7,  # enough to include one sentence ("BBBBB.")
    )

    # We expect multiple chunks
    assert len(chunks) >= 2

    first_chunk = chunks[0]["text"]
    second_chunk = chunks[1]["text"]

    # The overlap sentence ("BBBBB.") should appear in BOTH chunks
    assert "BBBBB." in first_chunk
    assert "BBBBB." in second_chunk


def test_read_text_file_unicode_decode_error_fallback(tmp_path, monkeypatch):
    from pathlib import Path
    from practorflow.llm.document.document_loader import DocumentLoader

    loader = DocumentLoader(use_docling=False)

    # Create a file with bytes that will be read via read_bytes()
    file_path = tmp_path / "binary.txt"
    raw_bytes = b"\xff\xfe\xfa\xfb"
    file_path.write_bytes(raw_bytes)

    # Force Path.read_text to raise UnicodeDecodeError
    def raise_unicode_error(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")

    monkeypatch.setattr(Path, "read_text", raise_unicode_error)

    content = loader._read_text_file(file_path)

    # Assertions
    assert isinstance(content, str)
    assert len(content) > 0


def test_decode_text_unicode_decode_error_fallback():
    from practorflow.llm.document.document_loader import DocumentLoader

    loader = DocumentLoader(use_docling=False)

    # Invalid UTF-8 byte sequence (guaranteed to raise UnicodeDecodeError)
    raw_bytes = b"\xff\xfe\xfa\xfb"

    content = loader._decode_text(raw_bytes, filename="binary.txt")

    assert isinstance(content, str)
    assert len(content) > 0

def test_determine_extension_docx_magic_bytes():
    from practorflow.llm.document.document_loader import DocumentLoader

    loader = DocumentLoader(use_docling=False)

    file_bytes = b"PK\x03\x04" + b"word/document.xml"
    ext = loader._determine_extension(file_bytes, filename=None, mime_type=None)

    assert ext == ".docx"


def test_determine_extension_xlsx_magic_bytes():
    from practorflow.llm.document.document_loader import DocumentLoader

    loader = DocumentLoader(use_docling=False)

    file_bytes = b"PK\x03\x04" + b"xl/workbook.xml"
    ext = loader._determine_extension(file_bytes, filename=None, mime_type=None)

    assert ext == ".xlsx"


def test_determine_extension_pptx_magic_bytes():
    from practorflow.llm.document.document_loader import DocumentLoader

    loader = DocumentLoader(use_docling=False)

    file_bytes = b"PK\x03\x04" + b"ppt/presentation.xml"
    ext = loader._determine_extension(file_bytes, filename=None, mime_type=None)

    assert ext == ".pptx"

def test_determine_extension_defaults_to_txt_when_no_match():
    from practorflow.llm.document.document_loader import DocumentLoader

    loader = DocumentLoader(use_docling=False)

    # Bytes that do NOT match:
    # - filename extension
    # - MIME mapping
    # - PDF magic bytes
    # - ZIP/PK Office formats
    file_bytes = b"RANDOM_BINARY_DATA_NOT_MATCHING_ANY_MAGIC"

    ext = loader._determine_extension(
        file_bytes=file_bytes,
        filename=None,
        mime_type=None,
    )

    assert ext == ".txt"