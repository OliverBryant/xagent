"""Tests for the remote DeepDoc parse client.

Every test here is a pure unit test: an ``httpx.MockTransport`` is injected
through the ``_transport`` seam of :func:`parse_document_remote`, so nothing
touches the network, and ``save_image`` is a plain callable, so no DeepDoc
parser is imported or constructed and no ONNX model cache is needed.

The failure cases matter more than the happy path. The client's contract is
that *every* remote problem surfaces as :class:`DeepDocRemoteError` and nothing
else, because the caller's single ``except DeepDocRemoteError`` clause is what
makes the fallback to local parsing unconditional. A leaked
``httpx.ConnectError`` or ``binascii.Error`` would escape that clause and break
parsing outright, so each failure test asserts the type is not merely "an
exception" but exactly that one.
"""

import base64
import binascii
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import pytest

from xagent.providers.pdf_parser.deepdoc_remote import (
    PARSE_ENDPOINT,
    DeepDocRemoteError,
    _build_headers,
    _mime_type_for,
    _normalize_elements,
    is_remote_configured,
    parse_document_remote,
)

BASE_URL = "http://gpu-host.internal:9997"
API_KEY_ENV = "XAGENT_DEEPDOC_XINFERENCE_API_KEY"
SHARED_API_KEY_ENV = "XINFERENCE_API_KEY"
URL_ENV = "XAGENT_DEEPDOC_XINFERENCE_URL"
TIMEOUT_ENV = "XAGENT_DEEPDOC_XINFERENCE_TIMEOUT_SECONDS"

# Smallest possible real PNG, so the decoded bytes are a plausible image rather
# than arbitrary base64 padding.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB"
    "/wFa1z9CAAAAAElFTkSuQmCC"
)
PNG_1X1_BASE64 = base64.b64encode(PNG_1X1).decode("ascii")


@pytest.fixture(autouse=True)
def remote_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure remote parsing and drop any ambient Xinference credentials.

    ``tests/conftest.py`` loads ``.env``/``example.env``, so a developer key in
    the environment would otherwise decide whether an ``Authorization`` header
    is sent. Both key variables are cleared here and set explicitly by the
    tests that care.
    """
    monkeypatch.setenv(URL_ENV, BASE_URL)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(SHARED_API_KEY_ENV, raising=False)
    monkeypatch.delenv(TIMEOUT_ENV, raising=False)


class RecordingSaveImage:
    """Stand-in for the parser's image writer, recording what it was handed."""

    def __init__(self, path: str = "/artifacts/table_0.png") -> None:
        self.path = path
        self.calls: list[bytes] = []

    def __call__(self, image_bytes: bytes) -> str:
        self.calls.append(image_bytes)
        return self.path


def json_transport(
    payload: Any,
    *,
    status_code: int = 200,
    sink: Optional[list[httpx.Request]] = None,
) -> httpx.MockTransport:
    """Return a transport answering every request with ``payload`` as JSON."""

    def handler(request: httpx.Request) -> httpx.Response:
        if sink is not None:
            sink.append(request)
        return httpx.Response(status_code, json=payload, request=request)

    return httpx.MockTransport(handler)


def sample_payload() -> dict[str, Any]:
    """A two-element response: one image-less text block, one table with a PNG."""
    return {
        "filename": "report.pdf",
        "file_type": ".pdf",
        "elapsed_ms": 4521,
        "elements": [
            {
                "type": "text",
                "text": "Quarterly summary",
                "image_base64": None,
                "metadata": {"page_number": 1, "layout_type": "text"},
            },
            {
                "type": "table",
                "text": "<table><tr><td>Cell</td></tr></table>",
                "image_base64": PNG_1X1_BASE64,
                "metadata": {"page_number": 2, "layout_type": "table"},
            },
        ],
    }


class TestParseDocumentRemoteSuccess:
    """Happy-path element shape and request shape."""

    def test_elements_are_translated_to_local_shape(self, tmp_path: Path) -> None:
        """image_base64 becomes an ``image`` path, or None, and metadata survives."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        save_image = RecordingSaveImage()

        elements = parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=save_image,
            _transport=json_transport(sample_payload()),
        )

        assert len(elements) == 2
        text, table = elements

        # image_base64 is consumed, never forwarded.
        assert "image_base64" not in text
        assert "image_base64" not in table

        assert text["type"] == "text"
        assert text["text"] == "Quarterly summary"
        assert text["image"] is None
        assert text["metadata"] == {"page_number": 1, "layout_type": "text"}

        assert table["type"] == "table"
        assert table["text"] == "<table><tr><td>Cell</td></tr></table>"
        assert table["image"] == save_image.path
        assert table["metadata"] == {"page_number": 2, "layout_type": "table"}

        # Only the table carried an image, so exactly one write happened.
        assert save_image.calls == [PNG_1X1]

    def test_callback_reports_start_and_completion(self, tmp_path: Path) -> None:
        """The completion notice keeps the ``message (1.23s)`` shape local DeepDoc emits."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        progress: list[tuple[float, str]] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            callback=lambda fraction, message: progress.append((fraction, message)),
            _transport=json_transport(sample_payload()),
        )

        assert [fraction for fraction, _ in progress] == [0.05, 1.0]
        assert progress[-1][1].startswith("Remote DeepDoc parse finished (")
        assert progress[-1][1].endswith("s)")

    def test_request_carries_the_file_zoomin_and_auth_header(
        self, tmp_path: Path
    ) -> None:
        """Multipart body holds the real filename; zoomin and the bearer token ride along."""
        source = tmp_path / "quarterly report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            zoomin=5,
            _transport=json_transport({"elements": []}, sink=requests),
        )

        assert len(requests) == 1
        request = requests[0]
        assert request.method == "POST"
        assert str(request.url) == f"{BASE_URL}{PARSE_ENDPOINT}"

        body = request.content.decode("utf-8", errors="replace")
        assert 'name="file"' in body
        assert 'filename="quarterly report.pdf"' in body
        assert "application/pdf" in body
        assert "%PDF-1.7 fake" in body
        assert 'name="zoomin"' in body
        assert "\r\n\r\n5\r\n" in body

    def test_zoomin_defaults_to_three(self, tmp_path: Path) -> None:
        """The default matches the local parser's parse_into_bboxes(zoomin=3)."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        assert "\r\n\r\n3\r\n" in requests[0].content.decode("utf-8", errors="replace")

    def test_authorization_header_present_when_a_key_is_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        monkeypatch.setenv(API_KEY_ENV, "secret-key")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        assert requests[0].headers["authorization"] == "Bearer secret-key"

    def test_authorization_header_absent_when_no_key_is_configured(
        self, tmp_path: Path
    ) -> None:
        """An unauthenticated self-hosted Xinference must not receive a bogus header."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        assert "authorization" not in requests[0].headers

    def test_image_scope_is_not_sent(self, tmp_path: Path) -> None:
        """The client relies on the server's ``table_figure`` default."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        assert "image_scope" not in requests[0].content.decode(
            "utf-8", errors="replace"
        )

    @pytest.mark.parametrize(
        "configured_url",
        [
            BASE_URL,
            f"{BASE_URL}/",
            f"{BASE_URL}///",
            f"{BASE_URL}  ",
        ],
        ids=["bare", "trailing-slash", "many-slashes", "trailing-space"],
    )
    def test_url_has_no_double_slash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured_url: str
    ) -> None:
        """The config getter strips the trailing slash, so the path joins cleanly."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        monkeypatch.setenv(URL_ENV, configured_url)
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        url = str(requests[0].url)
        assert url == f"{BASE_URL}{PARSE_ENDPOINT}"
        assert "//v1" not in url

    def test_bytesio_filename_falls_back_and_position_is_restored(self) -> None:
        """An in-memory document uploads as ``document{ext}`` and is left rewound as found."""
        stream = BytesIO(b"col1,col2\nval1,val2\n")
        stream.seek(4)
        requests: list[httpx.Request] = []

        parse_document_remote(
            stream,
            ext=".csv",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        body = requests[0].content.decode("utf-8", errors="replace")
        assert 'filename="document.csv"' in body
        assert "text/csv" in body
        # The whole buffer went up, not just the tail after the cursor.
        assert "col1,col2" in body
        # The caller may still read from its own buffer after a fallback.
        assert stream.tell() == 4

    @pytest.mark.parametrize(
        ("ext", "expected"),
        [
            (".pdf", "application/pdf"),
            (".csv", "text/csv"),
            (".PDF", "application/pdf"),
            (".unknown", "application/octet-stream"),
        ],
    )
    def test_mime_type_lookup(self, ext: str, expected: str) -> None:
        assert _mime_type_for(ext) == expected

    def test_empty_element_list_is_a_valid_response(self, tmp_path: Path) -> None:
        """A document with no extractable content is a success, not a failure."""
        source = tmp_path / "empty.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        save_image = RecordingSaveImage()

        elements = parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=save_image,
            _transport=json_transport({"elements": []}),
        )

        assert elements == []
        assert save_image.calls == []


def error_transport(exc: Exception) -> httpx.MockTransport:
    """Return a transport whose handler raises ``exc`` instead of responding."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.MockTransport(handler)


def raw_transport(
    body: bytes, *, status_code: int = 200, content_type: str = "application/json"
) -> httpx.MockTransport:
    """Return a transport answering with an exact byte body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=body,
            headers={"content-type": content_type},
            request=request,
        )

    return httpx.MockTransport(handler)


def failing_save_image(exc: Exception) -> Callable[[bytes], str]:
    """Return a ``save_image`` that raises ``exc`` when the client writes an image."""

    def save_image(image_bytes: bytes) -> str:
        raise exc

    return save_image


class TestParseDocumentRemoteFailures:
    """Every remote problem must surface as DeepDocRemoteError and nothing else."""

    @pytest.mark.parametrize(
        "transport_factory",
        [
            pytest.param(
                lambda: json_transport({"detail": "inference failed"}, status_code=500),
                id="http-500",
            ),
            pytest.param(
                lambda: json_transport({"detail": "bad token"}, status_code=401),
                id="http-401",
            ),
            pytest.param(
                lambda: error_transport(
                    httpx.ConnectError("connection refused"),
                ),
                id="connection-error",
            ),
            pytest.param(
                lambda: error_transport(httpx.ReadTimeout("read timed out")),
                id="read-timeout",
            ),
            pytest.param(
                lambda: raw_transport(b"<html>502 Bad Gateway</html>"),
                id="non-json-body",
            ),
            pytest.param(lambda: raw_transport(b""), id="empty-body"),
            pytest.param(
                lambda: json_transport([{"type": "text", "text": "x"}]),
                id="top-level-list",
            ),
            pytest.param(
                lambda: json_transport({"filename": "report.pdf"}),
                id="elements-missing",
            ),
            pytest.param(
                lambda: json_transport({"elements": None}),
                id="elements-none",
            ),
            pytest.param(
                lambda: json_transport({"elements": {"type": "text"}}),
                id="elements-not-a-list",
            ),
            pytest.param(
                lambda: json_transport({"elements": ["just a string"]}),
                id="element-not-a-dict",
            ),
            pytest.param(
                lambda: json_transport({"elements": [None]}),
                id="element-is-none",
            ),
            pytest.param(
                lambda: json_transport({"elements": [{"type": "text"}]}),
                id="element-missing-text",
            ),
            pytest.param(
                lambda: json_transport({"elements": [{"text": "x"}]}),
                id="element-missing-type",
            ),
            pytest.param(
                lambda: json_transport(
                    {
                        "elements": [
                            {
                                "type": "table",
                                "text": "<table></table>",
                                "image_base64": "a",
                            }
                        ]
                    }
                ),
                id="undecodable-image",
            ),
        ],
    )
    def test_failure_raises_deepdoc_remote_error(
        self, tmp_path: Path, transport_factory: Callable[[], httpx.MockTransport]
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=transport_factory(),
            )

    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(RuntimeError("artifact dir vanished"), id="runtime-error"),
            pytest.param(OSError("disk full"), id="os-error"),
            pytest.param(
                binascii.Error("re-raised decode failure"), id="binascii-error"
            ),
        ],
    )
    def test_save_image_failure_raises_deepdoc_remote_error(
        self, tmp_path: Path, exc: Exception
    ) -> None:
        """save_image is caller-supplied, so any exception from it must be wrapped."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=failing_save_image(exc),
                _transport=json_transport(sample_payload()),
            )

    @pytest.mark.parametrize(
        ("transport_factory", "leaked_type"),
        [
            pytest.param(
                lambda: error_transport(httpx.ConnectError("connection refused")),
                httpx.ConnectError,
                id="connection-error",
            ),
            pytest.param(
                lambda: error_transport(httpx.ReadTimeout("read timed out")),
                httpx.ReadTimeout,
                id="read-timeout",
            ),
            pytest.param(
                lambda: json_transport({"detail": "boom"}, status_code=500),
                httpx.HTTPStatusError,
                id="http-500",
            ),
            pytest.param(
                lambda: raw_transport(b"<html>oops</html>"),
                ValueError,
                id="non-json-body",
            ),
            pytest.param(
                lambda: json_transport({"elements": [None]}),
                ValueError,
                id="element-is-none",
            ),
        ],
    )
    def test_underlying_exception_types_do_not_escape(
        self,
        tmp_path: Path,
        transport_factory: Callable[[], httpx.MockTransport],
        leaked_type: type[Exception],
    ) -> None:
        """Guards the parametrized suite above against a passing-for-the-wrong-reason bug.

        ``DeepDocRemoteError`` derives straight from ``Exception``, so it is not
        an instance of any of these. Asserting that keeps the failure suite
        honest: were the client to stop wrapping, the raised type would satisfy
        neither this check nor ``pytest.raises(DeepDocRemoteError)``.
        """
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        with pytest.raises(DeepDocRemoteError) as excinfo:
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=transport_factory(),
            )

        assert not isinstance(excinfo.value, leaked_type)
        assert isinstance(excinfo.value.__cause__, leaked_type)

    def test_missing_source_file_raises_deepdoc_remote_error(
        self, tmp_path: Path
    ) -> None:
        """An unreadable local file must fall back rather than raise OSError."""
        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(tmp_path / "does-not-exist.pdf"),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=json_transport(sample_payload()),
            )

    def test_unconfigured_url_raises_deepdoc_remote_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Calling the client without configuration is a caller bug, still wrapped."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        monkeypatch.delenv(URL_ENV, raising=False)

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=json_transport(sample_payload()),
            )

    def test_failure_does_not_invoke_the_completion_callback(
        self, tmp_path: Path
    ) -> None:
        """A failed parse must not report progress it did not make."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        progress: list[tuple[float, str]] = []

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                callback=lambda fraction, message: progress.append((fraction, message)),
                _transport=json_transport({"detail": "boom"}, status_code=500),
            )

        assert [fraction for fraction, _ in progress] == [0.05]

    def test_bytesio_position_is_restored_after_a_failure(self) -> None:
        """The buffer must be reusable by the local fallback path."""
        stream = BytesIO(b"col1,col2\nval1,val2\n")
        stream.seek(7)

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                stream,
                ext=".csv",
                save_image=RecordingSaveImage(),
                _transport=json_transport({"detail": "boom"}, status_code=500),
            )

        assert stream.tell() == 7


class TestNormalizeElements:
    """Direct coverage of the translator, where wrapping is not in the way."""

    def test_absent_image_base64_still_yields_an_image_key(self) -> None:
        """The caller always reads ``image``, so the key must exist even unset."""
        save_image = RecordingSaveImage()

        elements = _normalize_elements(
            {"elements": [{"type": "text", "text": "hello"}]}, save_image
        )

        assert elements == [{"type": "text", "text": "hello", "image": None}]
        assert save_image.calls == []

    @pytest.mark.parametrize("empty", [None, "", 0], ids=["none", "empty-str", "zero"])
    def test_falsy_image_base64_is_treated_as_no_image(self, empty: Any) -> None:
        save_image = RecordingSaveImage()

        elements = _normalize_elements(
            {"elements": [{"type": "text", "text": "hi", "image_base64": empty}]},
            save_image,
        )

        assert elements[0]["image"] is None
        assert save_image.calls == []

    def test_source_payload_is_not_mutated(self) -> None:
        """Elements are copied, so a caller retrying locally sees its own data intact."""
        payload = {
            "elements": [
                {
                    "type": "table",
                    "text": "<table></table>",
                    "image_base64": PNG_1X1_BASE64,
                }
            ]
        }

        _normalize_elements(payload, RecordingSaveImage())

        assert payload["elements"][0]["image_base64"] == PNG_1X1_BASE64
        assert "image" not in payload["elements"][0]

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({}, id="elements-missing"),
            pytest.param({"elements": None}, id="elements-none"),
            pytest.param({"elements": "text"}, id="elements-str"),
            pytest.param({"elements": [42]}, id="element-int"),
            pytest.param({"elements": [{"type": "text"}]}, id="missing-text"),
            pytest.param({"elements": [{"text": "x"}]}, id="missing-type"),
        ],
    )
    def test_malformed_payloads_raise_value_error(self, payload: Any) -> None:
        """ValueError is what parse_document_remote translates into its own error."""
        with pytest.raises(ValueError):
            _normalize_elements(payload, RecordingSaveImage())

    def test_undecodable_image_raises_binascii_error(self) -> None:
        with pytest.raises(binascii.Error):
            _normalize_elements(
                {"elements": [{"type": "table", "text": "t", "image_base64": "abcde"}]},
                RecordingSaveImage(),
            )


class TestBuildHeaders:
    """Auth header construction, including the shared-key fallback."""

    def test_no_key_configured_yields_no_headers(self) -> None:
        assert _build_headers() == {}

    def test_dedicated_key_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(API_KEY_ENV, "dedicated")
        assert _build_headers() == {"Authorization": "Bearer dedicated"}

    def test_shared_key_is_the_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SHARED_API_KEY_ENV, "shared")
        assert _build_headers() == {"Authorization": "Bearer shared"}

    def test_dedicated_key_wins_over_the_shared_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_KEY_ENV, "dedicated")
        monkeypatch.setenv(SHARED_API_KEY_ENV, "shared")
        assert _build_headers() == {"Authorization": "Bearer dedicated"}


class TestIsRemoteConfigured:
    """Configuration detection must never raise; a typo means local parsing."""

    def test_unset_url_is_not_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(URL_ENV, raising=False)
        assert is_remote_configured() is False

    @pytest.mark.parametrize(
        "value",
        ["", "   "],
        ids=["empty", "whitespace"],
    )
    def test_blank_url_is_not_configured(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(URL_ENV, value)
        assert is_remote_configured() is False

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("ftp://host", id="wrong-scheme"),
            pytest.param("gpu-host.internal:9997", id="no-scheme"),
            pytest.param("http://host:9997?token=x", id="query-string"),
            pytest.param("http://host:9997#frag", id="fragment"),
            pytest.param("http://", id="no-netloc"),
        ],
    )
    def test_malformed_url_degrades_to_local_without_raising(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """The config getter raises ValueError; is_remote_configured must swallow it."""
        monkeypatch.setenv(URL_ENV, value)
        assert is_remote_configured() is False

    def test_malformed_url_logs_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(URL_ENV, "ftp://host")

        with caplog.at_level("WARNING"):
            assert is_remote_configured() is False

        assert "parsing locally" in caplog.text

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("http://host:9997", id="http"),
            pytest.param("https://host", id="https"),
            pytest.param("http://host:9997/", id="trailing-slash"),
            pytest.param("http://host:9997/base/path", id="with-path"),
        ],
    )
    def test_valid_url_is_configured(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(URL_ENV, value)
        assert is_remote_configured() is True
