"""Tests for the remote DeepDoc OCR client.

Every test here is a pure unit test: an ``httpx.MockTransport`` is injected
through the ``_transport`` seam of :func:`parse_document_remote`, so nothing
touches the network and no DeepDoc parser is imported or constructed, meaning
no ONNX model cache is needed.

Two concerns dominate.

First, *every* remote problem must surface as :class:`DeepDocRemoteError` and
nothing else, because the caller's single ``except DeepDocRemoteError`` clause
is what makes the fallback to local parsing unconditional. A leaked
``httpx.ConnectError`` or ``ValueError`` would escape that clause and break
parsing outright, so each failure test asserts the type is exactly that one.

Second, the spatial join must never lose recognized text. The server returns
OCR lines and layout blocks from two separate calls, and joining them is the
only place this client can silently drop content, so the join tests assert on
which lines land where rather than merely on element counts.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest

from xagent.providers.pdf_parser.deepdoc_remote import (
    OCR_ENDPOINT,
    TOKEN_ENDPOINT,
    DeepDocRemoteError,
    _join_ocr_and_layout,
    _mime_type_for,
    is_remote_configured,
    parse_document_remote,
)

BASE_URL = "http://gpu-host.internal:9997"
API_KEY_ENV = "XAGENT_DEEPDOC_XINFERENCE_API_KEY"
SHARED_API_KEY_ENV = "XINFERENCE_API_KEY"
URL_ENV = "XAGENT_DEEPDOC_XINFERENCE_URL"
TIMEOUT_ENV = "XAGENT_DEEPDOC_XINFERENCE_TIMEOUT_SECONDS"
MODEL_UID_ENV = "XAGENT_DEEPDOC_XINFERENCE_MODEL_UID"
USERNAME_ENV = "XAGENT_DEEPDOC_XINFERENCE_USERNAME"
PASSWORD_ENV = "XAGENT_DEEPDOC_XINFERENCE_PASSWORD"


@pytest.fixture(autouse=True)
def remote_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure remote parsing and drop any ambient Xinference credentials.

    ``tests/conftest.py`` loads ``.env``/``example.env``, so a developer key in
    the environment would otherwise decide whether an ``Authorization`` header
    is sent. Every credential variable is cleared here and set explicitly by the
    tests that care.
    """
    monkeypatch.setenv(URL_ENV, BASE_URL)
    for name in (
        API_KEY_ENV,
        SHARED_API_KEY_ENV,
        TIMEOUT_ENV,
        MODEL_UID_ENV,
        USERNAME_ENV,
        PASSWORD_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def quad(x0: float, top: float, x1: float, bottom: float) -> list[list[float]]:
    """Return the four-corner OCR box the server reports for a rectangle."""
    return [[x0, top], [x1, top], [x1, bottom], [x0, bottom]]


def ocr_line(text: str, x0: float, top: float, x1: float, bottom: float) -> dict:
    """Return one ``lines`` entry in the server's measured shape."""
    return {"box": quad(x0, top, x1, bottom), "text": text, "score": 0.99}


def ocr_payload(*pages: list[dict]) -> dict[str, Any]:
    """Wrap per-page line lists in the ``task=ocr`` response envelope."""
    return {
        "pages": [
            {"page": index, "result": {"task": "ocr", "lines": lines}}
            for index, lines in enumerate(pages, start=1)
        ]
    }


def layout_block(
    layout_type: str, x0: float, top: float, x1: float, bottom: float
) -> dict:
    """Return one ``layouts`` entry in the server's measured shape."""
    return {"type": layout_type, "bbox": [x0, top, x1, bottom], "score": 0.9}


def layout_payload(*pages: list[dict]) -> dict[str, Any]:
    """Wrap per-page block lists in the ``task=layout`` response envelope."""
    return {
        "pages": [
            {"page": index, "result": {"task": "layout", "layouts": layouts}}
            for index, layouts in enumerate(pages, start=1)
        ]
    }


def pages_of(payload: dict[str, Any], task: str) -> list[tuple[int, dict[str, Any]]]:
    """Reshape a response envelope into what :func:`_join_ocr_and_layout` takes."""
    del task
    return [(page["page"], page["result"]) for page in payload["pages"]]


def task_of(request: httpx.Request) -> Optional[str]:
    """Return the ``task`` named in a request's ``kwargs`` form field."""
    body = request.content.decode("utf-8", errors="replace")
    for candidate in ("ocr", "layout", "table"):
        if f'"task": "{candidate}"' in body or f'"task":"{candidate}"' in body:
            return candidate
    return None


# A distinct "not supplied" marker, so a test can serve a literal JSON ``null``
# body without it being mistaken for "use the default".
UNSET = object()


def deepdoc_transport(
    *,
    ocr: Any = UNSET,
    layout: Any = UNSET,
    token: Any = UNSET,
    ocr_status: int = 200,
    layout_status: int = 200,
    token_status: int = 200,
    sink: Optional[list[httpx.Request]] = None,
) -> httpx.MockTransport:
    """Return a transport that answers the token and both OCR tasks.

    Routing on the ``task`` form field is what lets a test give the two calls
    different answers, which is how the ocr-fails-but-layout-succeeds cases are
    expressed.
    """
    if ocr is UNSET:
        ocr = ocr_payload([])
    if layout is UNSET:
        layout = layout_payload([])
    if token is UNSET:
        token = {"access_token": "jwt-token"}

    def handler(request: httpx.Request) -> httpx.Response:
        if sink is not None:
            sink.append(request)
        if request.url.path == TOKEN_ENDPOINT:
            return httpx.Response(token_status, json=token, request=request)
        if task_of(request) == "layout":
            return httpx.Response(layout_status, json=layout, request=request)
        return httpx.Response(ocr_status, json=ocr, request=request)

    return httpx.MockTransport(handler)


def raising_transport(exc: Exception) -> httpx.MockTransport:
    """Return a transport whose every request raises ``exc``."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.MockTransport(handler)


def pdf_file(tmp_path: Path, name: str = "report.pdf") -> str:
    """Write a stand-in PDF and return its path."""
    source = tmp_path / name
    source.write_bytes(b"%PDF-1.7 fake")
    return str(source)


def unused_save_image(image_bytes: bytes) -> str:
    """Fail loudly if the client ever tries to save an image.

    The OCR endpoint returns no image bytes, so the caller's writer must stay
    untouched on this path.
    """
    raise AssertionError("save_image must not be called on the OCR path")


class TestRequestShape:
    """The two calls, their form fields, and their auth header."""

    def test_two_ocr_calls_are_made_with_the_right_tasks(self, tmp_path: Path) -> None:
        """One ocr call and one layout call, in that order, to /v1/images/ocr."""
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        parse_calls = [r for r in requests if r.url.path == OCR_ENDPOINT]
        assert len(parse_calls) == 2
        assert [task_of(r) for r in parse_calls] == ["ocr", "layout"]
        assert all(str(r.url) == f"{BASE_URL}{OCR_ENDPOINT}" for r in parse_calls)
        assert all(r.method == "POST" for r in parse_calls)

    def test_ocr_call_requests_dict_results(self, tmp_path: Path) -> None:
        """Without return_dict the server answers with positional tuples."""
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        ocr_call = next(r for r in requests if task_of(r) == "ocr")
        body = ocr_call.content.decode("utf-8", errors="replace")
        assert '"return_dict": true' in body

        layout_call = next(r for r in requests if task_of(r) == "layout")
        assert "return_dict" not in layout_call.content.decode(
            "utf-8", errors="replace"
        )

    def test_request_carries_model_uid_and_the_file(self, tmp_path: Path) -> None:
        """``model`` and ``image`` are the two required form fields."""
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path, "quarterly report.pdf"),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        body = next(r for r in requests if task_of(r) == "ocr").content.decode(
            "utf-8", errors="replace"
        )
        assert 'name="model"' in body
        assert "\r\n\r\nDeepDoc\r\n" in body
        # The field is "image", not "file": the endpoint is an image model's.
        assert 'name="image"' in body
        assert 'filename="quarterly report.pdf"' in body
        assert "application/pdf" in body
        assert "%PDF-1.7 fake" in body
        assert 'name="kwargs"' in body

    def test_model_uid_is_configurable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A model launched under a custom UID must still be addressable."""
        monkeypatch.setenv(MODEL_UID_ENV, "deepdoc-gpu1")
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        body = requests[0].content.decode("utf-8", errors="replace")
        assert "\r\n\r\ndeepdoc-gpu1\r\n" in body

    def test_second_call_re_sends_the_file(self, tmp_path: Path) -> None:
        """A consumed upload handle would make the layout call send nothing."""
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        for request in (r for r in requests if r.url.path == OCR_ENDPOINT):
            assert "%PDF-1.7 fake" in request.content.decode("utf-8", errors="replace")

    def test_bytesio_is_sent_twice_and_left_where_it_was_found(self) -> None:
        """The caller may still read the buffer after a fallback."""
        buffer = BytesIO(b"%PDF-1.7 in memory")
        buffer.seek(4)
        requests: list[httpx.Request] = []

        parse_document_remote(
            buffer,
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        uploads = [r for r in requests if r.url.path == OCR_ENDPOINT]
        assert len(uploads) == 2
        for request in uploads:
            body = request.content.decode("utf-8", errors="replace")
            assert "%PDF-1.7 in memory" in body
            assert 'filename="document.pdf"' in body
        assert buffer.tell() == 4

    @pytest.mark.parametrize(
        ("ext", "expected"),
        [
            (".pdf", "application/pdf"),
            (".PDF", "application/pdf"),
            (".png", "image/png"),
            (".jpg", "image/jpeg"),
            (".jpeg", "image/jpeg"),
            (".tiff", "image/tiff"),
            (".unknown", "application/octet-stream"),
        ],
    )
    def test_mime_type_for(self, ext: str, expected: str) -> None:
        assert _mime_type_for(ext) == expected

    @pytest.mark.parametrize(
        "configured_url",
        [BASE_URL, f"{BASE_URL}/", f"{BASE_URL}///", f"{BASE_URL}  "],
        ids=["bare", "trailing-slash", "many-slashes", "trailing-space"],
    )
    def test_url_has_no_double_slash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured_url: str
    ) -> None:
        """The config getter strips the trailing slash, so the path joins cleanly."""
        monkeypatch.setenv(URL_ENV, configured_url)
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        assert str(requests[0].url) == f"{BASE_URL}{OCR_ENDPOINT}"


class TestAuthentication:
    """Static key, JWT exchange, and the unauthenticated deployment."""

    def test_configured_key_is_used_verbatim_without_a_token_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A deployment issuing API keys must not be forced through /token."""
        monkeypatch.setenv(API_KEY_ENV, "secret-key")
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        assert all(r.url.path != TOKEN_ENDPOINT for r in requests)
        assert all(
            r.headers["authorization"] == "Bearer secret-key"
            for r in requests
            if r.url.path == OCR_ENDPOINT
        )

    def test_shared_xinference_key_is_honored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(SHARED_API_KEY_ENV, "shared-key")
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        assert requests[0].headers["authorization"] == "Bearer shared-key"

    def test_username_and_password_are_exchanged_for_a_jwt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The measured deployment authenticates this way, not with a static key."""
        monkeypatch.setenv(USERNAME_ENV, "admin")
        monkeypatch.setenv(PASSWORD_ENV, "admin123")
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(
                token={"access_token": "issued-jwt"}, sink=requests
            ),
        )

        token_calls = [r for r in requests if r.url.path == TOKEN_ENDPOINT]
        assert len(token_calls) == 1
        assert token_calls[0].method == "POST"
        body = token_calls[0].content.decode("utf-8")
        assert '"username": "admin"' in body or '"username":"admin"' in body
        assert '"password": "admin123"' in body or '"password":"admin123"' in body

        assert all(
            r.headers["authorization"] == "Bearer issued-jwt"
            for r in requests
            if r.url.path == OCR_ENDPOINT
        )

    def test_key_wins_over_username_and_password(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(API_KEY_ENV, "secret-key")
        monkeypatch.setenv(USERNAME_ENV, "admin")
        monkeypatch.setenv(PASSWORD_ENV, "admin123")
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        assert all(r.url.path != TOKEN_ENDPOINT for r in requests)
        assert requests[0].headers["authorization"] == "Bearer secret-key"

    def test_no_auth_header_when_nothing_is_configured(self, tmp_path: Path) -> None:
        """An unauthenticated self-hosted Xinference must not get a bogus header."""
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        assert all(r.url.path != TOKEN_ENDPOINT for r in requests)
        assert all("authorization" not in r.headers for r in requests)

    @pytest.mark.parametrize(
        "only_set", [USERNAME_ENV, PASSWORD_ENV], ids=["username", "password"]
    )
    def test_half_configured_credentials_do_not_trigger_an_exchange(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, only_set: str
    ) -> None:
        """Half a credential pair cannot succeed, so it must not be attempted."""
        monkeypatch.setenv(only_set, "value")
        requests: list[httpx.Request] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(sink=requests),
        )

        assert all(r.url.path != TOKEN_ENDPOINT for r in requests)

    def test_token_endpoint_failure_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(USERNAME_ENV, "admin")
        monkeypatch.setenv(PASSWORD_ENV, "wrong")

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=deepdoc_transport(
                    token={"detail": "bad credentials"}, token_status=401
                ),
            )

    @pytest.mark.parametrize(
        "token_body",
        [{}, {"access_token": ""}, {"access_token": 42}, []],
        ids=["missing", "empty", "non-string", "not-an-object"],
    )
    def test_unusable_token_response_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, token_body: Any
    ) -> None:
        monkeypatch.setenv(USERNAME_ENV, "admin")
        monkeypatch.setenv(PASSWORD_ENV, "admin123")

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=deepdoc_transport(token=token_body),
            )


class TestSpatialJoin:
    """Grouping OCR lines into layout blocks, the core of this client."""

    def test_lines_inside_a_block_merge_in_reading_order(self) -> None:
        payload_ocr = ocr_payload(
            [
                # Deliberately out of order, so the sort is what fixes it.
                ocr_line("second line", 10, 60, 200, 80),
                ocr_line("first line", 10, 10, 200, 30),
            ]
        )
        payload_layout = layout_payload([layout_block("text", 0, 0, 300, 100)])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert len(elements) == 1
        assert elements[0]["text"] == "first line\nsecond line"
        assert elements[0]["type"] == "text"

    def test_lines_on_the_same_row_order_left_to_right(self) -> None:
        payload_ocr = ocr_payload(
            [
                ocr_line("right", 200, 10, 300, 30),
                ocr_line("left", 10, 10, 100, 30),
            ]
        )
        payload_layout = layout_payload([layout_block("text", 0, 0, 400, 100)])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert elements[0]["text"] == "left\nright"

    def test_unclaimed_lines_become_their_own_elements(self) -> None:
        """Losing recognized text is worse than emitting it unstructured."""
        payload_ocr = ocr_payload(
            [
                ocr_line("inside", 10, 10, 200, 30),
                ocr_line("orphan", 10, 500, 200, 520),
            ]
        )
        payload_layout = layout_payload([layout_block("text", 0, 0, 300, 100)])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert [element["text"] for element in elements] == ["inside", "orphan"]
        orphan = elements[1]
        assert orphan["type"] == "text"
        assert orphan["metadata"]["layout_type"] == "text"
        assert orphan["metadata"]["top"] == 500

    def test_membership_is_decided_by_the_line_center(self) -> None:
        """A line poking out of its block still belongs to it."""
        payload_ocr = ocr_payload([ocr_line("overhanging", 10, 10, 400, 30)])
        # The block's right edge cuts the line at x=300; its center is x=205.
        payload_layout = layout_payload([layout_block("text", 0, 0, 300, 100)])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert len(elements) == 1
        assert elements[0]["metadata"]["layout_type"] == "text"

    def test_a_line_is_claimed_by_only_one_block(self) -> None:
        """Overlapping layout blocks must not duplicate the text between them."""
        payload_ocr = ocr_payload([ocr_line("once", 10, 10, 200, 30)])
        payload_layout = layout_payload(
            [
                layout_block("text", 0, 0, 300, 100),
                layout_block("title", 0, 0, 300, 100),
            ]
        )

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert [element["text"] for element in elements] == ["once"]

    def test_captions_lose_to_the_block_they_duplicate(self) -> None:
        """Measured behavior: the layout model mislabels titles as table captions."""
        payload_ocr = ocr_payload([ocr_line("Complex Tables", 10, 10, 200, 30)])
        payload_layout = layout_payload(
            [
                layout_block("table caption", 0, 0, 300, 100),
                layout_block("title", 0, 0, 300, 100),
            ]
        )

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert len(elements) == 1
        assert elements[0]["metadata"]["layout_type"] == "title"

    def test_tables_and_figures_outrank_the_text_block_over_them(self) -> None:
        payload_ocr = ocr_payload([ocr_line("cell", 10, 10, 200, 30)])
        payload_layout = layout_payload(
            [
                layout_block("text", 0, 0, 300, 100),
                layout_block("table", 0, 0, 300, 100),
            ]
        )

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert len(elements) == 1
        assert elements[0]["type"] == "table"

    @pytest.mark.parametrize(
        ("layout_type", "expected"),
        [
            ("table", "table"),
            ("figure", "figure"),
            ("figure caption", "figure"),
            ("text", "text"),
            ("title", "text"),
            ("equation", "text"),
            ("header", "text"),
            ("footer", "text"),
            ("reference", "text"),
            ("table caption", "text"),
            ("_background_", "text"),
            ("something-new", "text"),
        ],
    )
    def test_layout_type_maps_to_element_type(
        self, layout_type: str, expected: str
    ) -> None:
        payload_ocr = ocr_payload([ocr_line("content", 10, 10, 200, 30)])
        payload_layout = layout_payload([layout_block(layout_type, 0, 0, 300, 100)])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert elements[0]["type"] == expected
        # The raw label survives regardless of how it was mapped.
        assert elements[0]["metadata"]["layout_type"] == layout_type

    def test_layout_type_is_normalized(self) -> None:
        payload_ocr = ocr_payload([ocr_line("content", 10, 10, 200, 30)])
        payload_layout = layout_payload([layout_block("  TABLE ", 0, 0, 300, 100)])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert elements[0]["type"] == "table"
        assert elements[0]["metadata"]["layout_type"] == "table"

    def test_elements_are_sorted_by_page_then_position(self) -> None:
        payload_ocr = ocr_payload(
            [ocr_line("page one", 10, 500, 200, 520)],
            [ocr_line("page two", 10, 10, 200, 30)],
        )
        payload_layout = layout_payload([], [])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert [element["text"] for element in elements] == ["page one", "page two"]
        assert [element["metadata"]["page_number"] for element in elements] == [1, 2]

    def test_empty_blocks_are_not_emitted(self) -> None:
        """A bare figure carries no text, so it would only add noise."""
        payload_ocr = ocr_payload([ocr_line("elsewhere", 10, 500, 200, 520)])
        payload_layout = layout_payload([layout_block("figure", 0, 0, 300, 100)])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert [element["text"] for element in elements] == ["elsewhere"]

    def test_blank_recognitions_are_dropped(self) -> None:
        payload_ocr = ocr_payload(
            [
                ocr_line("kept", 10, 10, 200, 30),
                ocr_line("   ", 10, 40, 200, 60),
            ]
        )
        payload_layout = layout_payload([layout_block("text", 0, 0, 300, 100)])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert [element["text"] for element in elements] == ["kept"]

    def test_pages_with_no_layout_still_yield_their_text(self) -> None:
        """A layout call returning fewer pages must not silently drop text."""
        payload_ocr = ocr_payload(
            [ocr_line("page one", 10, 10, 200, 30)],
            [ocr_line("page two", 10, 10, 200, 30)],
        )
        payload_layout = layout_payload([layout_block("text", 0, 0, 300, 100)])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert [element["text"] for element in elements] == ["page one", "page two"]

    def test_a_page_split_across_response_entries_is_one_reading_order(self) -> None:
        """Per-batch sorting would concatenate two orders instead of merging them."""
        ocr_entries = [
            (1, {"task": "ocr", "lines": [ocr_line("second", 10, 100, 200, 120)]}),
            (1, {"task": "ocr", "lines": [ocr_line("first", 10, 10, 200, 30)]}),
        ]
        layout_entries = [(1, {"task": "layout", "layouts": []})]

        elements = _join_ocr_and_layout(ocr_entries, layout_entries)

        assert [element["text"] for element in elements] == ["first", "second"]

    def test_a_block_split_across_response_entries_is_ranked_as_one(self) -> None:
        """A table arriving in a later entry must still outrank an earlier text block."""
        ocr_entries = [
            (1, {"task": "ocr", "lines": [ocr_line("cell", 10, 10, 200, 30)]})
        ]
        layout_entries = [
            (1, {"task": "layout", "layouts": [layout_block("text", 0, 0, 300, 100)]}),
            (1, {"task": "layout", "layouts": [layout_block("table", 0, 0, 300, 100)]}),
        ]

        elements = _join_ocr_and_layout(ocr_entries, layout_entries)

        assert len(elements) == 1
        assert elements[0]["type"] == "table"

    def test_no_text_at_all_yields_no_elements(self) -> None:
        elements = _join_ocr_and_layout(
            pages_of(ocr_payload([]), "ocr"), pages_of(layout_payload([]), "layout")
        )

        assert elements == []

    def test_inverted_layout_bounds_are_normalized(self) -> None:
        """A bbox reported bottom-up would otherwise contain nothing."""
        payload_ocr = ocr_payload([ocr_line("content", 10, 10, 200, 30)])
        payload_layout = {
            "pages": [
                {
                    "page": 1,
                    "result": {
                        "task": "layout",
                        "layouts": [
                            {"type": "text", "bbox": [300, 100, 0, 0], "score": 0.9}
                        ],
                    },
                }
            ]
        }

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert len(elements) == 1
        assert elements[0]["metadata"]["x0"] == 10


class TestElementShape:
    """The contract ``_translate_remote_elements`` in deepdoc.py consumes."""

    def test_element_keys_and_metadata(self) -> None:
        payload_ocr = ocr_payload(
            [
                ocr_line("first", 10, 10, 200, 30),
                ocr_line("second", 20, 60, 250, 80),
            ]
        )
        payload_layout = layout_payload([layout_block("title", 0, 0, 300, 100)])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        assert len(elements) == 1
        element = elements[0]
        assert set(element) == {"type", "text", "image", "metadata"}
        # No image bytes come back from this endpoint, ever.
        assert element["image"] is None
        assert element["metadata"] == {
            "page_number": 1,
            # Bounds are the union of the claimed lines, not the block's bbox,
            # so they describe the text rather than the detector's guess.
            "x0": 10,
            "x1": 250,
            "top": 10,
            "bottom": 80,
            "layout_type": "title",
            "col_id": 0,
            "positions": [[1, 10, 250, 10, 80]],
        }

    def test_positions_survive_the_translator_enrichment(self) -> None:
        """_build_element_metadata inserts col_id at index 1 of each position."""
        payload_ocr = ocr_payload([ocr_line("content", 10, 20, 200, 30)])
        payload_layout = layout_payload([layout_block("text", 0, 0, 300, 100)])

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(payload_layout, "layout")
        )

        position = elements[0]["metadata"]["positions"][0]
        assert len(position) == 5
        page, x0, x1, top, bottom = position
        assert page == 1
        assert (x0, x1, top, bottom) == (10, 200, 20, 30)

    def test_skewed_boxes_are_reduced_to_their_bounds(self) -> None:
        """OCR quads are not axis-aligned; the metadata is rectangular."""
        payload_ocr = {
            "pages": [
                {
                    "page": 1,
                    "result": {
                        "task": "ocr",
                        "lines": [
                            {
                                "box": [[10, 12], [200, 10], [202, 30], [12, 32]],
                                "text": "skewed",
                                "score": 0.9,
                            }
                        ],
                    },
                }
            ]
        }

        elements = _join_ocr_and_layout(
            pages_of(payload_ocr, "ocr"), pages_of(layout_payload([]), "layout")
        )

        metadata = elements[0]["metadata"]
        assert (metadata["x0"], metadata["x1"]) == (10, 202)
        assert (metadata["top"], metadata["bottom"]) == (10, 32)

    def test_end_to_end_shape_through_the_transport(self, tmp_path: Path) -> None:
        elements = parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(
                ocr=ocr_payload([ocr_line("Quarterly summary", 10, 10, 200, 30)]),
                layout=layout_payload([layout_block("title", 0, 0, 300, 100)]),
            ),
        )

        assert elements == [
            {
                "type": "text",
                "text": "Quarterly summary",
                "image": None,
                "metadata": {
                    "page_number": 1,
                    "x0": 10,
                    "x1": 200,
                    "top": 10,
                    "bottom": 30,
                    "layout_type": "title",
                    "col_id": 0,
                    "positions": [[1, 10, 200, 10, 30]],
                },
            }
        ]


class TestProgressCallback:
    """Progress notices, whose shape the existing adapter depends on."""

    def test_callback_reports_start_layout_and_completion(self, tmp_path: Path) -> None:
        """The completion notice keeps the ``message (1.23s)`` shape local DeepDoc emits."""
        progress: list[tuple[float, str]] = []

        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            callback=lambda fraction, message: progress.append((fraction, message)),
            _transport=deepdoc_transport(),
        )

        assert [fraction for fraction, _ in progress] == [0.05, 0.5, 1.0]
        assert progress[-1][1].startswith("Remote DeepDoc parse finished (")
        assert progress[-1][1].endswith("s)")

    def test_no_callback_is_fine(self, tmp_path: Path) -> None:
        parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(),
        )


class TestFailuresBecomeDeepDocRemoteError:
    """Every failure mode must surface as the one type the caller catches."""

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 500, 503])
    def test_error_status_on_the_ocr_call(self, tmp_path: Path, status: int) -> None:
        with pytest.raises(DeepDocRemoteError) as excinfo:
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=deepdoc_transport(ocr={"detail": "nope"}, ocr_status=status),
            )

        assert "Remote DeepDoc request failed" in str(excinfo.value)

    @pytest.mark.parametrize("status", [401, 500])
    def test_error_status_on_the_layout_call(self, tmp_path: Path, status: int) -> None:
        """A half-succeeded parse must fall back, not return partial results."""
        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=deepdoc_transport(
                    ocr=ocr_payload([ocr_line("text", 10, 10, 200, 30)]),
                    layout={"detail": "nope"},
                    layout_status=status,
                ),
            )

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("connection refused"),
            httpx.ConnectTimeout("connect timed out"),
            httpx.ReadTimeout("read timed out"),
            httpx.WriteTimeout("write timed out"),
            httpx.PoolTimeout("pool timed out"),
            httpx.RemoteProtocolError("peer closed connection"),
        ],
    )
    def test_transport_failures(self, tmp_path: Path, exc: Exception) -> None:
        with pytest.raises(DeepDocRemoteError) as excinfo:
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=raising_transport(exc),
            )

        assert "Remote DeepDoc request failed" in str(excinfo.value)

    def test_non_json_body(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway</html>", request=request)

        with pytest.raises(DeepDocRemoteError) as excinfo:
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=httpx.MockTransport(handler),
            )

        assert "non-JSON" in str(excinfo.value)

    @pytest.mark.parametrize(
        "body",
        [[], "a string", 42],
        ids=["list", "string", "int"],
    )
    def test_body_is_not_an_object(self, tmp_path: Path, body: Any) -> None:
        with pytest.raises(DeepDocRemoteError) as excinfo:
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=deepdoc_transport(ocr=body),
            )

        assert "unusable response" in str(excinfo.value)

    def test_empty_body(self, tmp_path: Path) -> None:
        """A 200 with no body at all is still a fallback, not a crash."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"", request=request)

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=httpx.MockTransport(handler),
            )

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"pages": None},
            {"pages": "not-a-list"},
            {"pages": [None]},
            {"pages": [{"page": 1}]},
            {"pages": [{"page": 1, "result": "not-an-object"}]},
            {"pages": [{"page": "one", "result": {"lines": []}}]},
        ],
        ids=[
            "missing-pages",
            "null-pages",
            "pages-not-a-list",
            "page-not-an-object",
            "missing-result",
            "result-not-an-object",
            "non-integer-page",
        ],
    )
    def test_malformed_pages(self, tmp_path: Path, body: Any) -> None:
        with pytest.raises(DeepDocRemoteError) as excinfo:
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=deepdoc_transport(ocr=body),
            )

        assert "unusable response" in str(excinfo.value)

    @pytest.mark.parametrize(
        "lines",
        [
            None,
            "not-a-list",
            [None],
            ["a string"],
            [{"box": quad(0, 0, 1, 1)}],
            [{"box": quad(0, 0, 1, 1), "text": 42}],
            [{"text": "no box"}],
            [{"box": "not-a-list", "text": "x"}],
            [{"box": [[0, 0], [1, 1]], "text": "too few points"}],
            [{"box": [[0, 0], [1, 1], ["a", "b"]], "text": "non-numeric"}],
            [{"box": [[0, 0], [1, 1], [2]], "text": "short point"}],
        ],
        ids=[
            "null-lines",
            "lines-not-a-list",
            "line-not-an-object",
            "line-is-a-string",
            "missing-text",
            "non-string-text",
            "missing-box",
            "box-not-a-list",
            "too-few-points",
            "non-numeric-point",
            "short-point",
        ],
    )
    def test_malformed_ocr_lines(self, tmp_path: Path, lines: Any) -> None:
        body = {"pages": [{"page": 1, "result": {"task": "ocr", "lines": lines}}]}

        with pytest.raises(DeepDocRemoteError) as excinfo:
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=deepdoc_transport(ocr=body),
            )

        assert "unusable response" in str(excinfo.value)

    @pytest.mark.parametrize(
        "layouts",
        [
            None,
            "not-a-list",
            [None],
            [{"type": "text"}],
            [{"type": "text", "bbox": "not-a-list"}],
            [{"type": "text", "bbox": [0, 0, 1]}],
            [{"type": "text", "bbox": [0, 0, 1, "x"]}],
        ],
        ids=[
            "null-layouts",
            "layouts-not-a-list",
            "block-not-an-object",
            "missing-bbox",
            "bbox-not-a-list",
            "too-short-bbox",
            "non-numeric-bbox",
        ],
    )
    def test_malformed_layout_blocks(self, tmp_path: Path, layouts: Any) -> None:
        body = {
            "pages": [{"page": 1, "result": {"task": "layout", "layouts": layouts}}]
        }

        with pytest.raises(DeepDocRemoteError) as excinfo:
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=deepdoc_transport(
                    ocr=ocr_payload([ocr_line("text", 10, 10, 200, 30)]), layout=body
                ),
            )

        assert "unusable response" in str(excinfo.value)

    def test_missing_layout_type_defaults_to_text(self, tmp_path: Path) -> None:
        """A block without a usable type is still text worth keeping."""
        body = {
            "pages": [
                {
                    "page": 1,
                    "result": {
                        "task": "layout",
                        "layouts": [{"bbox": [0, 0, 300, 100], "score": 0.9}],
                    },
                }
            ]
        }

        elements = parse_document_remote(
            pdf_file(tmp_path),
            ext=".pdf",
            save_image=unused_save_image,
            _transport=deepdoc_transport(
                ocr=ocr_payload([ocr_line("content", 10, 10, 200, 30)]), layout=body
            ),
        )

        assert elements[0]["type"] == "text"
        assert elements[0]["metadata"]["layout_type"] == "text"

    def test_unreadable_source_file(self, tmp_path: Path) -> None:
        with pytest.raises(DeepDocRemoteError) as excinfo:
            parse_document_remote(
                str(tmp_path / "does-not-exist.pdf"),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=deepdoc_transport(),
            )

        assert "Remote DeepDoc parse failed" in str(excinfo.value)

    def test_unconfigured_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv(URL_ENV, raising=False)

        with pytest.raises(DeepDocRemoteError) as excinfo:
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                _transport=deepdoc_transport(),
            )

        assert "not configured" in str(excinfo.value)

    def test_a_raising_callback_does_not_escape_as_itself(self, tmp_path: Path) -> None:
        """Even a caller-supplied sink blowing up must end in the fallback."""

        def exploding_callback(fraction: float, message: str) -> None:
            if fraction == 0.5:
                raise RuntimeError("sink exploded")

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                pdf_file(tmp_path),
                ext=".pdf",
                save_image=unused_save_image,
                callback=exploding_callback,
                _transport=deepdoc_transport(),
            )


class TestIsRemoteConfigured:
    def test_true_when_url_is_set(self) -> None:
        assert is_remote_configured() is True

    def test_false_when_url_is_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(URL_ENV, raising=False)
        assert is_remote_configured() is False

    @pytest.mark.parametrize(
        "bad_url",
        ["not-a-url", "ftp://host:9997", "http://", "http://host:9997?q=1"],
    )
    def test_malformed_url_degrades_to_local(
        self, monkeypatch: pytest.MonkeyPatch, bad_url: str
    ) -> None:
        """A typo in one variable must not break every document parse."""
        monkeypatch.setenv(URL_ENV, bad_url)
        assert is_remote_configured() is False
