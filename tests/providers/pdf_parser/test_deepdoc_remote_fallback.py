"""Tests for DeepDoc remote routing, local fallback, and the element translator.

Every test here is a pure unit test: no network, no DeepDoc model cache, and no
writes outside ``tmp_path``. The remote client
(``deepdoc_remote.parse_document_remote``) and the local parser factory
(``DeepDocParser._get_parser_for_ext``) are the two seams that get monkeypatched,
which lets the routing decision be asserted without either side actually running.

The central guarantee under test is that remote mode never instantiates a local
parser. ``DeepDocPdfParser()`` eagerly loads ONNX models and may download them
from ModelScope, so the remote path must not touch it at all. That is asserted by
arming ``_get_parser_for_ext`` as a tripwire that raises ``AssertionError``: if the
production code ever moves local-parser construction back above the remote
dispatch, these tests fail immediately.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from xagent.providers.pdf_parser import deepdoc as deepdoc_module
from xagent.providers.pdf_parser import deepdoc_remote
from xagent.providers.pdf_parser.deepdoc import (
    DeepDocParser,
    _translate_pdf_bboxes,
    _translate_remote_elements,
)
from xagent.providers.pdf_parser.deepdoc_remote import DeepDocRemoteError

REMOTE_URL_ENV = "XAGENT_DEEPDOC_XINFERENCE_URL"


# ==========================================
# HELPERS AND FIXTURES
# ==========================================


class RecordingProgressCallback:
    """Minimal ``ProgressCallback`` implementation that records status updates."""

    def __init__(self) -> None:
        self.statuses: List[str] = []

    def on_status_update(
        self, status: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.statuses.append(status)


class BrokenProgressCallback:
    """A progress sink that always raises.

    ``DeepDocProgressAdapter.get_callback()`` calls ``on_status_update`` again
    from inside its own ``except`` handler, so a sink that raises unconditionally
    makes the adapter's callback re-raise rather than swallow. Reporting the
    fallback must therefore be guarded by the caller, or a recoverable remote
    failure would turn into a hard parse failure.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def on_status_update(
        self, status: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.call_count += 1
        raise RuntimeError("progress sink is broken")


class FakeLocalParser:
    """Stand-in for ``DeepDocPdfParser`` that returns canned bboxes."""

    def __init__(self, bboxes: List[Dict[str, Any]]) -> None:
        self.bboxes = bboxes
        self.calls: List[Dict[str, Any]] = []

    def parse_into_bboxes(self, file_path: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        self.calls.append({"file_path": file_path, **kwargs})
        return self.bboxes


@pytest.fixture(autouse=True)
def isolate_artifacts_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect saved images into ``tmp_path``.

    ``_save_bytes_to_disk`` resolves against the module-level ``ARTIFACTS_DIR``
    (which defaults under the user's real ``~/.xagent``), so every test that
    exercises image bytes must have it pointed somewhere disposable.
    """
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(deepdoc_module, "ARTIFACTS_DIR", artifacts_dir)
    return artifacts_dir


@pytest.fixture(autouse=True)
def remote_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from "remote not configured".

    A developer ``.env`` (loaded by the root conftest with ``override=True``)
    could otherwise set the remote URL and silently flip the routing decision
    these tests are asserting.
    """
    monkeypatch.delenv(REMOTE_URL_ENV, raising=False)


@pytest.fixture
def remote_configured(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure a remote DeepDoc URL that is never actually contacted."""
    url = "http://deepdoc.invalid:9997"
    monkeypatch.setenv(REMOTE_URL_ENV, url)
    return url


def arm_local_parser_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any local-parser construction an immediate test failure."""

    def _tripwire(self: DeepDocParser, ext: str) -> Any:
        raise AssertionError(
            f"_get_parser_for_ext({ext!r}) was called on the remote path; "
            "remote mode must never instantiate a local DeepDoc parser"
        )

    monkeypatch.setattr(DeepDocParser, "_get_parser_for_ext", _tripwire)


def arm_remote_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any remote call an immediate test failure."""

    def _tripwire(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "parse_document_remote was called while remote mode is unconfigured"
        )

    monkeypatch.setattr(deepdoc_remote, "parse_document_remote", _tripwire)


def remote_pdf_elements() -> List[Dict[str, Any]]:
    """Canned remote response for a PDF: one text element, one table, one figure."""
    return [
        {
            "type": "text",
            "text": "Remote parsed paragraph",
            "image": None,
            "metadata": {
                "layout_type": "text",
                "page_number": 1,
                "col_id": 0,
                "positions": [[1, 10, 20, 30, 40]],
            },
        },
        {
            "type": "table",
            "text": "<table><tr><td>Remote</td></tr></table>",
            "image": None,
            "metadata": {
                "layout_type": "table",
                "page_number": 2,
                "col_id": 1,
                "positions": [[2, 11, 21, 31, 41]],
            },
        },
        {
            "type": "figure",
            "text": "Remote figure caption",
            "image": None,
            "metadata": {
                "layout_type": "figure",
                "page_number": 3,
                "col_id": 0,
                "positions": [[3, 12, 22, 32, 42]],
            },
        },
    ]


def remote_docx_elements() -> List[Dict[str, Any]]:
    """Canned remote response for a DOCX: styled paragraphs plus a table."""
    return [
        {
            "type": "text",
            "text": "Remote DOCX heading",
            "image": None,
            "metadata": {"style": "Heading 1"},
        },
        {
            "type": "text",
            "text": "Remote DOCX body",
            "image": None,
            "metadata": {"style": "Normal"},
        },
        {
            "type": "table",
            "text": "<table><tr><td>docx</td></tr></table>",
            "image": None,
            "metadata": {},
        },
    ]


# ==========================================
# ROUTING: REMOTE SUCCESS
# ==========================================


class TestRemoteRoutingSuccess:
    """Remote mode must succeed without ever touching the local ONNX path."""

    @pytest.mark.asyncio
    async def test_pdf_goes_remote_without_local_parser(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_configured: str
    ) -> None:
        """A configured remote server handles .pdf; the local parser stays untouched."""
        pdf_file = tmp_path / "remote.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        arm_local_parser_tripwire(monkeypatch)

        recorded: Dict[str, Any] = {}

        def fake_parse_document_remote(
            file_path: Any, **kwargs: Any
        ) -> List[Dict[str, Any]]:
            recorded["file_path"] = file_path
            recorded.update(kwargs)
            return remote_pdf_elements()

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", fake_parse_document_remote
        )

        parser = DeepDocParser()
        result = await parser.parse(str(pdf_file), doc_id="remote_pdf_doc")

        # The remote client was reached with the routing arguments it needs.
        assert recorded["file_path"] == str(pdf_file)
        assert recorded["ext"] == ".pdf"
        assert recorded["zoomin"] == 3
        assert callable(recorded["save_image"])

        # The parse succeeded purely from remote elements.
        assert [segment.text for segment in result.text_segments] == [
            "Remote parsed paragraph"
        ]
        assert len(result.tables) == 1
        assert result.tables[0].html == "<table><tr><td>Remote</td></tr></table>"
        assert len(result.figures) == 1
        assert result.figures[0].text == "Remote figure caption"

        # And the backend marker says remote, not local.
        assert result.metadata["deepdoc_backend"] == "remote"
        assert result.metadata["file_type"] == ".pdf"
        assert result.metadata["parse_method"] == "deepdoc"

    @pytest.mark.asyncio
    async def test_docx_goes_remote_without_local_parser(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_configured: str
    ) -> None:
        """Remote routing is not PDF-only: .docx goes remote too."""
        docx_file = tmp_path / "remote.docx"
        # A ZIP magic header, so the cheap Open XML pre-check passes without
        # needing a real DOCX on disk.
        docx_file.write_bytes(b"PK\x03\x04 not really a docx")

        arm_local_parser_tripwire(monkeypatch)

        recorded: Dict[str, Any] = {}

        def fake_parse_document_remote(
            file_path: Any, **kwargs: Any
        ) -> List[Dict[str, Any]]:
            recorded["file_path"] = file_path
            recorded.update(kwargs)
            return remote_docx_elements()

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", fake_parse_document_remote
        )

        parser = DeepDocParser()
        result = await parser.parse(str(docx_file), doc_id="remote_docx_doc")

        assert recorded["ext"] == ".docx"
        assert [segment.text for segment in result.text_segments] == [
            "Remote DOCX heading",
            "Remote DOCX body",
        ]
        # The DOCX-specific metadata key survives the remote translation.
        assert result.text_segments[0].metadata["style"] == "Heading 1"
        assert result.text_segments[1].metadata["style"] == "Normal"
        assert len(result.tables) == 1
        assert result.metadata["deepdoc_backend"] == "remote"
        assert result.metadata["file_type"] == ".docx"

    @pytest.mark.asyncio
    async def test_remote_success_reports_progress(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_configured: str
    ) -> None:
        """Remote mode wires the progress adapter for every format, not just PDF."""
        txt_file = tmp_path / "remote.txt"
        txt_file.write_text("local text that must never be read", encoding="utf-8")

        arm_local_parser_tripwire(monkeypatch)

        def fake_parse_document_remote(
            file_path: Any, callback: Any = None, **kwargs: Any
        ) -> List[Dict[str, Any]]:
            assert callback is not None, (
                "remote mode should forward a progress callback"
            )
            callback(0.05, "Uploading document to remote DeepDoc server")
            callback(1.0, "Remote DeepDoc parse finished (0.12s)")
            return [{"type": "text", "text": "remote txt", "image": None}]

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", fake_parse_document_remote
        )

        progress = RecordingProgressCallback()
        parser = DeepDocParser()
        result = await parser.parse(
            str(txt_file), progress_callback=progress, doc_id="remote_txt_doc"
        )

        assert result.metadata["deepdoc_backend"] == "remote"
        assert [segment.text for segment in result.text_segments] == ["remote txt"]
        # The adapter strips the timing suffix, matching local DeepDoc's shape.
        assert progress.statuses == [
            "Uploading document to remote DeepDoc server",
            "Remote DeepDoc parse finished",
        ]

    @pytest.mark.asyncio
    async def test_remote_bytesio_goes_remote_without_local_parser(
        self, monkeypatch: pytest.MonkeyPatch, remote_configured: str
    ) -> None:
        """An in-memory spreadsheet also routes remote, with no local parse."""
        arm_local_parser_tripwire(monkeypatch)

        def fake_parse_document_remote(
            file_path: Any, **kwargs: Any
        ) -> List[Dict[str, Any]]:
            assert isinstance(file_path, BytesIO)
            assert kwargs["ext"] == ".xlsx"
            return [
                {
                    "type": "text",
                    "text": "Name: Ada | Role: Engineer",
                    "image": None,
                    "metadata": {
                        "sheet_name": "Sheet1",
                        "row_number": 2,
                        "row_type": "data",
                    },
                }
            ]

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", fake_parse_document_remote
        )

        parser = DeepDocParser()
        result = await parser._parse_impl(
            BytesIO(b"not really an xlsx"),
            file_ext=".xlsx",
            doc_id="remote_xlsx_doc",
        )

        assert result.metadata["deepdoc_backend"] == "remote"
        assert result.metadata["source"] == "memory_buffer"
        assert len(result.text_segments) == 1
        assert result.text_segments[0].metadata["sheet_name"] == "Sheet1"
        assert result.text_segments[0].metadata["row_number"] == 2
        assert result.text_segments[0].metadata["row_type"] == "data"


# ==========================================
# ROUTING: REMOTE FAILURE FALLS BACK TO LOCAL
# ==========================================


class TestRemoteFailureFallsBackToLocal:
    """Any remote failure must degrade to local parsing with a warning."""

    @pytest.mark.asyncio
    async def test_remote_error_falls_back_to_local_pdf(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        remote_configured: str,
    ) -> None:
        """DeepDocRemoteError yields the local result, a warning, and backend=local."""
        pdf_file = tmp_path / "fallback.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        def failing_parse_document_remote(*args: Any, **kwargs: Any) -> Any:
            raise DeepDocRemoteError("Remote DeepDoc request failed: boom")

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", failing_parse_document_remote
        )

        local_bboxes = [
            {
                "layout_type": "text",
                "text": "Locally parsed paragraph",
                "positions": [[1, 10, 20, 30, 40]],
            },
            {
                "layout_type": "table",
                "text": "<table><tr><td>Local</td></tr></table>",
                "image": None,
            },
        ]
        fake_parser = FakeLocalParser(local_bboxes)
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        parser = DeepDocParser()
        with caplog.at_level("WARNING", logger="xagent.providers.pdf_parser.deepdoc"):
            result = await parser.parse(str(pdf_file), doc_id="fallback_pdf_doc")

        # The local result is what comes back.
        assert [segment.text for segment in result.text_segments] == [
            "Locally parsed paragraph"
        ]
        assert len(result.tables) == 1
        assert result.tables[0].html == "<table><tr><td>Local</td></tr></table>"
        assert result.metadata["deepdoc_backend"] == "local"

        # The local parser really ran, with the local call signature.
        assert len(fake_parser.calls) == 1
        assert fake_parser.calls[0]["file_path"] == str(pdf_file)
        assert fake_parser.calls[0]["zoomin"] == 3

        # And the failure was logged rather than swallowed.
        assert "Remote DeepDoc parse failed" in caplog.text
        assert "falling back to local" in caplog.text

    @pytest.mark.asyncio
    async def test_broken_progress_sink_does_not_abort_the_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        remote_configured: str,
    ) -> None:
        """Reporting the fallback must never be able to prevent the fallback.

        ``DeepDocProgressAdapter.get_callback()`` re-raises out of its own
        ``except`` handler when the sink misbehaves, so without the caller's
        try/except around the fallback notification this test fails with
        ``RuntimeError`` instead of returning the local result.
        """
        pdf_file = tmp_path / "broken_sink.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        def failing_parse_document_remote(*args: Any, **kwargs: Any) -> Any:
            raise DeepDocRemoteError("Remote DeepDoc request failed: boom")

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", failing_parse_document_remote
        )

        fake_parser = FakeLocalParser(
            [{"layout_type": "text", "text": "Local survived the broken sink"}]
        )
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        progress = BrokenProgressCallback()
        parser = DeepDocParser()
        with caplog.at_level("WARNING"):
            result = await parser.parse(
                str(pdf_file),
                progress_callback=progress,
                doc_id="broken_sink_doc",
            )

        assert [segment.text for segment in result.text_segments] == [
            "Local survived the broken sink"
        ]
        assert result.metadata["deepdoc_backend"] == "local"
        # The sink was genuinely exercised and genuinely raised.
        assert progress.call_count > 0
        assert "Progress callback failed while reporting the DeepDoc fallback" in (
            caplog.text
        )

    @pytest.mark.asyncio
    async def test_remote_error_falls_back_to_local_xlsx(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        remote_configured: str,
    ) -> None:
        """The .xlsx fallback reaches the openpyxl row reader, not a DeepDoc parser."""
        arm_local_parser_tripwire(monkeypatch)

        def failing_parse_document_remote(*args: Any, **kwargs: Any) -> Any:
            raise DeepDocRemoteError("Remote DeepDoc returned an unusable response")

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", failing_parse_document_remote
        )

        recorded: Dict[str, Any] = {}

        def fake_parse_xlsx_rows(file_path: Any, **kwargs: Any) -> Any:
            recorded.update(kwargs)
            from xagent.providers.pdf_parser.base import ParsedTextSegment, ParseResult

            return ParseResult(
                text_segments=[
                    ParsedTextSegment(text="local xlsx row", metadata=dict(kwargs))
                ],
                metadata=dict(kwargs),
            )

        monkeypatch.setattr(deepdoc_module, "_parse_xlsx_rows", fake_parse_xlsx_rows)

        parser = DeepDocParser()
        result = await parser._parse_impl(
            BytesIO(b"not really an xlsx"),
            file_ext=".xlsx",
            doc_id="fallback_xlsx_doc",
        )

        assert recorded["deepdoc_backend"] == "local"
        assert [segment.text for segment in result.text_segments] == ["local xlsx row"]


# ==========================================
# ROUTING: ENV UNSET STAYS LOCAL
# ==========================================


class TestEnvUnsetStaysLocal:
    """With no remote URL configured, nothing may reach the remote client."""

    @pytest.mark.asyncio
    async def test_pure_local_path_never_calls_remote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The default configuration parses locally with no remote attempt."""
        pdf_file = tmp_path / "local_only.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        arm_remote_tripwire(monkeypatch)

        fake_parser = FakeLocalParser(
            [{"layout_type": "text", "text": "Local only paragraph"}]
        )
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        parser = DeepDocParser()
        result = await parser.parse(str(pdf_file), doc_id="local_only_doc")

        assert [segment.text for segment in result.text_segments] == [
            "Local only paragraph"
        ]
        assert result.metadata["deepdoc_backend"] == "local"
        assert len(fake_parser.calls) == 1

    @pytest.mark.asyncio
    async def test_malformed_remote_url_degrades_to_local(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A typo in the URL must not break every parse; it degrades to local."""
        pdf_file = tmp_path / "malformed_url.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        monkeypatch.setenv(REMOTE_URL_ENV, "ftp://not-http")
        arm_remote_tripwire(monkeypatch)

        fake_parser = FakeLocalParser(
            [{"layout_type": "text", "text": "Local despite a bad URL"}]
        )
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        parser = DeepDocParser()
        with caplog.at_level("WARNING"):
            result = await parser.parse(str(pdf_file), doc_id="malformed_url_doc")

        assert [segment.text for segment in result.text_segments] == [
            "Local despite a bad URL"
        ]
        assert result.metadata["deepdoc_backend"] == "local"
        assert "parsing locally" in caplog.text


# ==========================================
# TRANSLATOR
# ==========================================


class TestTranslateRemoteElements:
    """``_translate_remote_elements`` must match the local translators' output."""

    @staticmethod
    def base_kwargs() -> Dict[str, Any]:
        return {
            "source": "remote.pdf",
            "file_type": ".pdf",
            "parse_method": "deepdoc",
            "deepdoc_backend": "remote",
        }

    def test_pdf_shaped_elements_match_local_translation(self) -> None:
        """A PDF-shaped element set translates exactly like the local bbox path."""
        doc_id = "translator_pdf_doc"
        kwargs = self.base_kwargs()

        elements = [
            {
                "type": "text",
                "text": "Paragraph one",
                "image": None,
                "metadata": {
                    "layout_type": "text",
                    "page_number": 4,
                    "col_id": 1,
                    "positions": [[4, 10, 20, 30, 40]],
                },
            },
            {
                "type": "table",
                "text": "<table><tr><td>T</td></tr></table>",
                "image": None,
                "metadata": {
                    "layout_type": "table",
                    "page_number": 5,
                    "col_id": 0,
                    "positions": [[5, 11, 21, 31, 41]],
                },
            },
            {
                "type": "figure",
                "text": "A caption",
                "image": None,
                "metadata": {
                    "layout_type": "figure",
                    "page_number": 6,
                    "col_id": 0,
                    "positions": [[6, 12, 22, 32, 42]],
                },
            },
        ]

        remote_result = _translate_remote_elements(doc_id, elements, **kwargs)

        # The equivalent local input: the same content, already bbox-shaped.
        local_bboxes = [
            {**element["metadata"], "text": element["text"], "image": element["image"]}
            for element in elements
        ]
        local_result = _translate_pdf_bboxes(doc_id, local_bboxes, **kwargs)

        assert len(remote_result.text_segments) == len(local_result.text_segments) == 1
        assert len(remote_result.tables) == len(local_result.tables) == 1
        assert len(remote_result.figures) == len(local_result.figures) == 1

        assert remote_result.text_segments[0].text == local_result.text_segments[0].text
        assert (
            remote_result.text_segments[0].metadata
            == local_result.text_segments[0].metadata
        )
        assert remote_result.tables[0].html == local_result.tables[0].html
        assert remote_result.tables[0].metadata == local_result.tables[0].metadata
        assert remote_result.figures[0].text == local_result.figures[0].text
        assert remote_result.figures[0].metadata == local_result.figures[0].metadata
        assert remote_result.metadata == local_result.metadata == kwargs

    def test_positions_are_enriched_with_col_id(self) -> None:
        """``positions`` gain ``col_id`` at index 1 and float coordinates."""
        result = _translate_remote_elements(
            "positions_doc",
            [
                {
                    "type": "text",
                    "text": "Positioned text",
                    "image": None,
                    "metadata": {
                        "layout_type": "text",
                        "page_number": 1,
                        "col_id": 0,
                        "positions": [[1, 10, 20, 30, 40]],
                    },
                }
            ],
            **self.base_kwargs(),
        )

        metadata = result.text_segments[0].metadata
        assert metadata["positions"] == [[1, 0, 10.0, 20.0, 30.0, 40.0]]
        assert metadata["col_id"] == 0
        assert metadata["page_number"] == 1
        assert metadata["layout_type"] == "text"
        assert metadata["doc_id"] == "positions_doc"

    def test_non_zero_col_id_is_inserted_into_positions(self) -> None:
        """A two-column element carries its own ``col_id`` into every position."""
        result = _translate_remote_elements(
            "two_column_doc",
            [
                {
                    "type": "text",
                    "text": "Right column",
                    "image": None,
                    "metadata": {
                        "layout_type": "text",
                        "page_number": 2,
                        "col_id": 1,
                        "positions": [[2, 300, 590, 100, 140], [3, 300, 590, 0, 60]],
                    },
                }
            ],
            **self.base_kwargs(),
        )

        assert result.text_segments[0].metadata["positions"] == [
            [2, 1, 300.0, 590.0, 100.0, 140.0],
            [3, 1, 300.0, 590.0, 0.0, 60.0],
        ]

    def test_xlsx_shaped_elements_carry_row_metadata(self) -> None:
        """Spreadsheet metadata keys survive translation, matching local xlsx rows."""
        kwargs = {
            "source": "remote.xlsx",
            "file_type": ".xlsx",
            "parse_method": "deepdoc",
            "deepdoc_backend": "remote",
        }
        elements = [
            {
                "type": "text",
                "text": "[Sheet1] Quarterly report",
                "image": None,
                "metadata": {
                    "sheet_name": "Sheet1",
                    "row_number": 1,
                    "row_type": "title",
                },
            },
            {
                "type": "text",
                "text": "Name | Role",
                "image": None,
                "metadata": {
                    "sheet_name": "Sheet1",
                    "row_number": 2,
                    "row_type": "header",
                },
            },
            {
                "type": "text",
                "text": "Name: Ada | Role: Engineer",
                "image": None,
                "metadata": {
                    "sheet_name": "Sheet1",
                    "row_number": 3,
                    "row_type": "data",
                },
            },
        ]

        result = _translate_remote_elements("xlsx_doc", elements, **kwargs)

        assert result.tables == []
        assert result.figures == []
        assert len(result.text_segments) == 3
        assert [segment.metadata["row_type"] for segment in result.text_segments] == [
            "title",
            "header",
            "data",
        ]
        assert [segment.metadata["row_number"] for segment in result.text_segments] == [
            1,
            2,
            3,
        ]
        for segment in result.text_segments:
            assert segment.metadata["sheet_name"] == "Sheet1"
            # The local xlsx translator emits no positions, and neither does this.
            assert "positions" not in segment.metadata
            # Shared metadata from kwargs is merged into every segment.
            assert segment.metadata["source"] == "remote.xlsx"
            assert segment.metadata["deepdoc_backend"] == "remote"
            assert segment.metadata["doc_id"] == "xlsx_doc"

    def test_unknown_element_type_degrades_to_text(self) -> None:
        """A future element type must become a text segment, never be dropped."""
        result = _translate_remote_elements(
            "unknown_type_doc",
            [
                {
                    "type": "equation",
                    "text": "E = mc^2",
                    "image": None,
                    "metadata": {"page_number": 7},
                },
                {"type": "text", "text": "Ordinary text", "image": None},
            ],
            **self.base_kwargs(),
        )

        assert [segment.text for segment in result.text_segments] == [
            "E = mc^2",
            "Ordinary text",
        ]
        # The unknown type is preserved in metadata rather than rewritten.
        assert result.text_segments[0].metadata["layout_type"] == "equation"
        assert result.text_segments[0].metadata["page_number"] == 7
        assert result.tables == []
        assert result.figures == []

    def test_missing_and_none_metadata_do_not_blow_up(self) -> None:
        """Elements with absent, ``None``, or non-dict metadata still translate."""
        result = _translate_remote_elements(
            "sparse_metadata_doc",
            [
                {"type": "text", "text": "No metadata key"},
                {"type": "text", "text": "None metadata", "metadata": None},
                {"type": "text", "text": "List metadata", "metadata": ["nope"]},
                {"type": "table", "text": "<table></table>", "metadata": None},
                {"type": "figure", "text": "", "metadata": None},
            ],
            **self.base_kwargs(),
        )

        assert [segment.text for segment in result.text_segments] == [
            "No metadata key",
            "None metadata",
            "List metadata",
        ]
        for segment in result.text_segments:
            # _build_element_metadata supplies the defaults.
            assert segment.metadata["layout_type"] == "text"
            assert segment.metadata["page_number"] == 1
            assert segment.metadata["col_id"] == 0
            assert "positions" not in segment.metadata

        assert len(result.tables) == 1
        assert result.tables[0].metadata["image_path"] is None
        assert result.tables[0].metadata["type"] == "table"

        assert len(result.figures) == 1
        # An empty caption is backfilled so downstream processing has text.
        assert result.figures[0].text == "Figure"
        assert result.figures[0].metadata["image_path"] is None
        assert result.figures[0].metadata["type"] == "figure"

    def test_empty_element_list_yields_empty_result(self) -> None:
        """No elements means empty lists, with the shared metadata still set."""
        kwargs = self.base_kwargs()
        result = _translate_remote_elements("empty_doc", [], **kwargs)

        assert result.text_segments == []
        assert result.tables == []
        assert result.figures == []
        assert result.metadata == kwargs

    def test_image_path_strings_are_carried_onto_table_and_figure(
        self, tmp_path: Path
    ) -> None:
        """The remote client's saved-image paths flow through unchanged.

        The client rewrites ``image_base64`` into an on-disk path string, which is
        exactly what the local ``_handle_image`` string branch already accepts.
        """
        table_image = tmp_path / "table.png"
        table_image.write_bytes(b"fake png bytes")
        figure_image = tmp_path / "figure.png"
        figure_image.write_bytes(b"fake png bytes")

        result = _translate_remote_elements(
            "image_doc",
            [
                {
                    "type": "table",
                    "text": "<table><tr><td>with image</td></tr></table>",
                    "image": str(table_image),
                    "metadata": {"page_number": 1},
                },
                {
                    "type": "figure",
                    "text": "Figure with image",
                    "image": str(figure_image),
                    "metadata": {"page_number": 2},
                },
            ],
            **self.base_kwargs(),
        )

        assert result.tables[0].metadata["image_path"] == str(table_image)
        assert result.figures[0].metadata["image_path"] == str(figure_image)

    def test_real_image_bytes_are_saved_under_the_patched_artifacts_dir(
        self, isolate_artifacts_dir: Path
    ) -> None:
        """``_save_bytes_to_disk`` writes only inside the patched artifacts dir."""
        image_path = Path(
            deepdoc_module._save_bytes_to_disk("bytes_doc", b"fake png bytes", ".png")
        )

        assert image_path.is_file()
        assert image_path.read_bytes() == b"fake png bytes"
        assert isolate_artifacts_dir in image_path.parents


class TestBackendMarkerOnEveryFormat:
    """`deepdoc_backend` must survive every format's translator.

    Only the PDF translator used to forward the parse metadata onto the
    ParseResult, so a DOCX/XLSX/MD/TXT/CSV parse returned `deepdoc_backend=None`
    and the remote-versus-local outcome was unobservable for those formats.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("suffix", "payload"),
        [
            (".txt", b"plain text body"),
            (".md", b"# heading\n\nbody text\n"),
        ],
    )
    async def test_local_parse_marks_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        suffix: str,
        payload: bytes,
    ) -> None:
        """Formats that bypass the PDF branch still report the local backend."""
        source = tmp_path / f"sample{suffix}"
        source.write_bytes(payload)

        arm_remote_tripwire(monkeypatch)

        parser = DeepDocParser()
        result = await parser.parse(str(source), doc_id="backend-marker")

        assert result.metadata.get("deepdoc_backend") == "local"
