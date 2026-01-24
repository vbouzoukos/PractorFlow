import sys
import io
import base64
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from practorflow.llm.document.document_loader import DocumentLoader


@pytest.fixture
def loader_with_docling():
    """
    Provides a DocumentLoader with a fully mocked Docling environment.
    This correctly mocks the *import path* used inside DocumentLoader.__init__.
    """

    # -------------------------------
    # Fake docling.document_converter
    # -------------------------------
    fake_docling = ModuleType("docling")
    fake_document_converter = ModuleType("docling.document_converter")

    # Mock table -> dataframe -> dict
    mock_table = MagicMock()
    mock_table.export_to_dataframe.return_value.to_dict.return_value = [
        {"col1": "value1"}
    ]

    # Mock document
    mock_document = MagicMock()
    mock_document.export_to_markdown.return_value = "Docling parsed content."
    mock_document.tables = [mock_table]

    # Mock conversion result
    mock_result = MagicMock()
    mock_result.document = mock_document

    class FakeDocumentConverter:
        def __init__(self):
            pass

        def convert(self, *args, **kwargs):
            return mock_result

    fake_document_converter.DocumentConverter = FakeDocumentConverter

    # Inject fake modules
    sys.modules["docling"] = fake_docling
    sys.modules["docling.document_converter"] = fake_document_converter

    try:
        loader = DocumentLoader(use_docling=True)
        yield loader
    finally:
        # Cleanup to avoid cross-test pollution
        sys.modules.pop("docling.document_converter", None)
        sys.modules.pop("docling", None)


def test_docling_enabled_when_available(loader_with_docling):
    assert loader_with_docling.use_docling is True
    assert loader_with_docling.docling_parser is not None


def test_parse_docling_file(tmp_path, loader_with_docling):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF fake pdf content")

    content, tables = loader_with_docling._parse_docling(pdf_path)

    assert content == "Docling parsed content."
    assert len(tables) == 1
    assert tables[0]["data"] == [{"col1": "value1"}]


def test_load_file_with_docling(tmp_path, loader_with_docling):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF fake content")

    result = loader_with_docling.load_file(str(pdf_path))

    assert result["file_type"] == ".pdf"
    assert result["content"] == "Docling parsed content."
    assert result["tables"]
    assert len(result["retrieval_chunks"]) > 0
    assert len(result["context_chunks"]) > 0


def test_parse_docling_bytes(loader_with_docling):
    content, tables = loader_with_docling._parse_docling_bytes(
        b"%PDF fake bytes", ".pdf"
    )

    assert content == "Docling parsed content."
    assert len(tables) == 1
    assert tables[0]["data"] == [{"col1": "value1"}]


def test_temp_file_cleanup_on_parse_docling_bytes(loader_with_docling, tmp_path):
    with patch("tempfile.NamedTemporaryFile") as mock_tmp:
        fake_tmp = MagicMock()
        fake_tmp.name = str(tmp_path / "temp.pdf")
        mock_tmp.return_value.__enter__.return_value = fake_tmp

        with patch("os.path.exists", return_value=True) as mock_exists:
            with patch("os.remove") as mock_remove:
                loader_with_docling._parse_docling_bytes(
                    b"%PDF fake bytes", ".pdf"
                )

                mock_exists.assert_called()
                mock_remove.assert_called_with(fake_tmp.name)


def test_load_from_bytes_with_docling(loader_with_docling):
    result = loader_with_docling.load_from_bytes(
        b"%PDF fake bytes",
        filename="upload.pdf",
        mime_type="application/pdf",
    )

    assert result["file_type"] == ".pdf"
    assert result["content"] == "Docling parsed content."
    assert result["tables"]


def test_load_from_stream_with_docling(loader_with_docling):
    stream = io.BytesIO(b"%PDF streamed fake content")

    result = loader_with_docling.load_from_stream(
        stream,
        filename="stream.pdf",
        mime_type="application/pdf",
    )

    assert result["file_type"] == ".pdf"
    assert result["content"] == "Docling parsed content."


def test_load_from_base64_with_docling(loader_with_docling):
    raw = b"%PDF base64 fake content"
    encoded = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:application/pdf;base64,{encoded}"

    result = loader_with_docling.load_from_base64(
        data_uri,
        filename="base64.pdf",
    )

    assert result["file_type"] == ".pdf"
    assert result["content"] == "Docling parsed content."

def test_docling_importerror_fallback(monkeypatch):
    """
    Covers the ImportError branch in DocumentLoader.__init__ when
    Docling is not available.
    """

    # Ensure docling is NOT importable
    monkeypatch.delitem(sys.modules, "docling", raising=False)
    monkeypatch.delitem(sys.modules, "docling.document_converter", raising=False)

    # Force ImportError on import
    def fake_import(*args, **kwargs):
        raise ImportError("Docling not installed")

    monkeypatch.setattr(
        "builtins.__import__",
        fake_import,
    )

    loader = DocumentLoader(use_docling=True)

    assert loader.use_docling is False
    assert loader.docling_parser is None

def test_parse_docling_table_extraction_exception_logged(
    tmp_path, loader_with_docling, monkeypatch
):
    from unittest.mock import MagicMock
    from practorflow.llm.document import document_loader

    # Sentinel exception to prove logger.warning was executed
    class LoggerHit(Exception):
        pass

    def raise_on_warning(*args, **kwargs):
        raise LoggerHit()

    # HARD REPLACE logger.warning
    monkeypatch.setattr(
        document_loader.logger,
        "warning",
        raise_on_warning,
    )

    # Build docling result with failing table
    failing_table = MagicMock()
    failing_table.export_to_dataframe.side_effect = RuntimeError("boom")

    mock_document = MagicMock()
    mock_document.export_to_markdown.return_value = "Docling parsed content."
    mock_document.tables = [failing_table]

    mock_result = MagicMock()
    mock_result.document = mock_document

    # Ensure convert() always returns THIS result
    loader_with_docling.docling_parser.convert = MagicMock(
        return_value=mock_result
    )

    pdf_path = tmp_path / "tables.pdf"
    pdf_path.write_bytes(b"%PDF fake content")

    # If logger.warning is reached → LoggerHit is raised
    try:
        loader_with_docling._parse_docling(pdf_path)
    except LoggerHit:
        return  # ✅ TEST PASSES

    # If we get here, the except branch never executed
    pytest.fail("logger.warning was never called")

def test_parse_docling_bytes_table_extraction_exception_logged(
    loader_with_docling, monkeypatch
):
    from unittest.mock import MagicMock
    from practorflow.llm.document import document_loader

    # Sentinel exception to prove logger.warning was executed
    class LoggerHit(Exception):
        pass

    def raise_on_warning(*args, **kwargs):
        raise LoggerHit()

    # HARD REPLACE logger.warning
    monkeypatch.setattr(
        document_loader.logger,
        "warning",
        raise_on_warning,
    )

    # Build docling result with failing table
    failing_table = MagicMock()
    failing_table.export_to_dataframe.side_effect = RuntimeError("boom")

    mock_document = MagicMock()
    mock_document.export_to_markdown.return_value = "Docling parsed content."
    mock_document.tables = [failing_table]

    mock_result = MagicMock()
    mock_result.document = mock_document

    # Ensure convert() always returns THIS result
    loader_with_docling.docling_parser.convert = MagicMock(
        return_value=mock_result
    )

    # If logger.warning is reached → LoggerHit is raised
    try:
        loader_with_docling._parse_docling_bytes(
            b"%PDF fake bytes", ".pdf"
        )
    except LoggerHit:
        return  # ✅ TEST PASSES

    # If we get here, the except branch never executed
    pytest.fail("logger.warning was never called")
