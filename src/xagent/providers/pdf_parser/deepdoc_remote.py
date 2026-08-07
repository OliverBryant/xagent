"""HTTP client for offloading DeepDoc parsing to Xinference's OCR endpoint.

Set ``XAGENT_DEEPDOC_XINFERENCE_URL`` to route parsing to a GPU host instead of
running local ONNX inference. Any failure raises :class:`DeepDocRemoteError`,
which the caller treats as "fall back to local parsing" -- fallback is always
on, which is what makes the switch transparent.

This module deliberately depends on nothing but ``httpx`` and
``xagent.config``. In particular it must not import ``deepdoc.py``: the
image-saving function is injected by the caller instead, which keeps the import
graph acyclic and lets the client be tested without loading any parser.

Server API contract
-------------------

Xinference ships DeepDoc as an *image/OCR model*, not as a whole-document parse
API (xorbitsai/inference#5230). There is no ``/v1/document/parse``. The real,
measured endpoint is:

.. code-block:: text

    POST {base_url}/v1/images/ocr
    Authorization: Bearer <jwt or api key>    # omitted when unauthenticated

    Request (multipart/form-data):
      model   str     required    the launched model UID, e.g. "DeepDoc"
      image   binary  required    the document; PDFs are rasterized server-side
      kwargs  str     optional    a JSON *string*; supports task, pages, dpi,
                                  return_dict, threshold

    Response 200 application/json:
    {"pages": [{"page": 1, "result": {"task": ..., <task payload>}}]}

The per-task payloads this client uses:

- ``task=ocr`` with ``return_dict=true`` yields ``lines``, each
  ``{"box": [[x, y] * 4], "text": ..., "score": ...}``. ``box`` is a
  quadrilateral, not an axis-aligned rectangle.
- ``task=layout`` yields ``layouts``, each
  ``{"type": ..., "bbox": [x0, y0, x1, y1], "score": ...}``. ``type`` comes
  from DeepDoc's label set, lower-cased: ``text``, ``title``, ``figure``,
  ``figure caption``, ``table``, ``table caption``, ``header``, ``footer``,
  ``reference``, ``equation``.

Neither task returns text grouped into blocks, so this client issues both and
joins them spatially. That is sound because the two tasks were measured to
share one coordinate space (both are reported in the rasterized page's pixel
space at the same DPI).

Authentication is a JWT obtained from ``POST {base_url}/token`` with
``{"username", "password"}``. A deployment that issues a static API key instead
can configure that key directly and it is used as the bearer verbatim.

Capability gap versus local parsing
-----------------------------------

The OCR endpoint returns neither table HTML, nor image crops, nor DeepDoc's
XGBoost cross-line paragraph merging, so remote elements are coarser than local
ones: a table's text is its OCR lines rather than reconstructed HTML, ``image``
is always ``None``, and blocks are never merged across pages. See
``docs/deepdoc-remote-parsing.md``.
"""

from __future__ import annotations

import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from ...config import (
    get_deepdoc_xinference_api_key,
    get_deepdoc_xinference_model_uid,
    get_deepdoc_xinference_password,
    get_deepdoc_xinference_timeout_seconds,
    get_deepdoc_xinference_url,
    get_deepdoc_xinference_username,
)

logger = logging.getLogger(__name__)

OCR_ENDPOINT = "/v1/images/ocr"
TOKEN_ENDPOINT = "/token"

# Connecting to and writing headers against a reachable host is fast; only the
# inference itself is slow, so the connect/pool budgets stay short regardless of
# how long the configured read timeout is.
_CONNECT_TIMEOUT_SECONDS = 10.0
_POOL_TIMEOUT_SECONDS = 10.0
# The token exchange is a trivial round trip, so it must not inherit the
# document-sized read timeout: a hung auth endpoint should fail fast into the
# local fallback rather than stall the parse for half an hour.
_TOKEN_TIMEOUT_SECONDS = 30.0

_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}
_DEFAULT_MIME_TYPE = "application/octet-stream"

# DeepDoc's layout label set, lower-cased as the server reports it, mapped onto
# the three element types the downstream translator understands. Anything not
# listed -- including "text", "title", "header", "footer", "reference" and
# "equation" -- becomes a text element, which is the conservative choice: text
# is the only type whose payload is never dropped downstream.
_LAYOUT_TYPE_TO_ELEMENT_TYPE = {
    "table": "table",
    "figure": "figure",
    "figure caption": "figure",
}

# DeepDoc's layout blocks overlap: a caption is routinely reported at the exact
# coordinates of the text or title block covering the same lines, and both would
# otherwise claim those lines and duplicate the text. Blocks are therefore
# ranked and each line is claimed at most once, by the highest-ranked block
# containing it.
#
# Captions rank *below* everything, which was chosen from measured output rather
# than from taste: on the sample document the layout model emitted "table
# caption" over blocks the same model had also classified "title" and "text",
# and those latter labels were the correct ones. Letting a caption lose keeps
# the better label, and costs nothing when the caption is genuine -- the text is
# still emitted, just under a plainer type.
_LAYOUT_TYPE_PRIORITY = {
    "table": 3,
    "figure": 3,
    "figure caption": 1,
    "table caption": 1,
}
_DEFAULT_LAYOUT_PRIORITY = 2

ProgressCallback = Callable[[float, str], None]
SaveImage = Callable[[bytes], str]


class DeepDocRemoteError(Exception):
    """Remote parsing failed and the caller should fall back to local parsing.

    Every failure mode -- unreachable host, timeout, 4xx/5xx, unparsable or
    malformed body -- is reported as this single type so callers need only one
    ``except`` clause.
    """


def is_remote_configured() -> bool:
    """Return whether remote DeepDoc parsing is configured.

    A malformed URL makes the config getter raise. Degrading to local parsing
    with a warning is the right response: a typo in one environment variable
    must not break every document parse.
    """
    try:
        return get_deepdoc_xinference_url() is not None
    except ValueError as exc:
        logger.warning(
            "Ignoring remote DeepDoc configuration and parsing locally: %s", exc
        )
        return False


def _get_access_token(client: httpx.Client, base_url: str) -> Optional[str]:
    """Return the bearer token to authenticate with, or None when unauthenticated.

    A configured key is used verbatim: Xinference accepts both a JWT and a
    static API key in the ``Authorization`` header, and the client cannot tell
    them apart, so it does not try. Only when no key is set and a username and
    password are does it exchange them at ``/token``.

    Args:
        client: The HTTP client to exchange credentials on.
        base_url: Xinference base URL.

    Raises:
        httpx.HTTPError: On any transport failure or non-2xx status.
        ValueError: If the token response is not a JSON object carrying a
            non-empty string ``access_token``.
    """
    api_key = get_deepdoc_xinference_api_key()
    if api_key:
        return api_key

    username = get_deepdoc_xinference_username()
    password = get_deepdoc_xinference_password()
    if not username or not password:
        return None

    response = client.post(
        f"{base_url}{TOKEN_ENDPOINT}",
        json={"username": username, "password": password},
        timeout=_TOKEN_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(
            f"Remote DeepDoc token response is {type(payload).__name__}, "
            "expected a JSON object"
        )
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("Remote DeepDoc token response carries no 'access_token'")
    return token


def _mime_type_for(ext: str) -> str:
    """Return the MIME type to advertise for ``ext``."""
    return _MIME_TYPES.get(ext.lower(), _DEFAULT_MIME_TYPE)


def _post_ocr_task(
    client: httpx.Client,
    file_path: str | BytesIO,
    *,
    base_url: str,
    ext: str,
    task: str,
    headers: dict[str, str],
) -> dict[str, Any]:
    """Run one OCR-endpoint task over the document and return the decoded body.

    The file handle is opened per call rather than shared, because a multipart
    upload consumes it and the caller runs two tasks over the same document.

    Args:
        client: The HTTP client to send on.
        file_path: Path to the original file, or its bytes in memory.
        base_url: Xinference base URL.
        ext: Lower-cased file extension, used for the MIME type and for the
            synthetic filename of an in-memory document.
        task: The DeepDoc task to run, ``"ocr"`` or ``"layout"``.
        headers: Auth headers, empty when unauthenticated.

    Raises:
        httpx.HTTPError: On any transport failure or non-2xx status.
        ValueError: If the body is not JSON, or is JSON but not an object.
        OSError: If a file path cannot be read.
    """
    kwargs: dict[str, Any] = {"task": task}
    if task == "ocr":
        # Without this the server returns bare tuples, which carry the same
        # information but positionally; the dict form is what was measured.
        kwargs["return_dict"] = True
    # kwargs is a JSON *string* form field, not a nested multipart structure.
    data = {"model": get_deepdoc_xinference_model_uid(), "kwargs": json.dumps(kwargs)}
    mime_type = _mime_type_for(ext)

    def _send(filename: str, fh: Any) -> dict[str, Any]:
        response = client.post(
            f"{base_url}{OCR_ENDPOINT}",
            headers=headers,
            data=data,
            files={"image": (filename, fh, mime_type)},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                f"Remote DeepDoc {task} response body is {type(payload).__name__}, "
                "expected a JSON object"
            )
        return payload

    if isinstance(file_path, BytesIO):
        # The buffer is owned by the caller, who may still read from it after a
        # fallback, so put the cursor back where it was found.
        original_position = file_path.tell()
        try:
            file_path.seek(0)
            return _send(f"document{ext}", file_path)
        finally:
            file_path.seek(original_position)

    with open(file_path, "rb") as fh:
        return _send(Path(file_path).name, fh)


def _iter_pages(payload: dict[str, Any], task: str) -> list[tuple[int, dict[str, Any]]]:
    """Return ``(page_number, result)`` pairs from one task's response.

    Args:
        payload: Decoded response body.
        task: The task the payload came from, for error messages.

    Raises:
        ValueError: If the payload does not carry a list of page objects.
    """
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError(
            f"Remote DeepDoc {task} response is missing a 'pages' list "
            f"(got {type(pages).__name__})"
        )

    parsed: list[tuple[int, dict[str, Any]]] = []
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(
                f"Remote DeepDoc {task} page {index} is {type(page).__name__}, "
                "expected an object"
            )
        result = page.get("result")
        if not isinstance(result, dict):
            raise ValueError(
                f"Remote DeepDoc {task} page {index} is missing a 'result' object "
                f"(got {type(result).__name__})"
            )
        # Fall back to the 1-based ordinal so a server that omits the page
        # number still produces monotonically increasing pages.
        page_number = page.get("page", index + 1)
        if not isinstance(page_number, int) or isinstance(page_number, bool):
            raise ValueError(
                f"Remote DeepDoc {task} page {index} has a non-integer page number "
                f"({page_number!r})"
            )
        parsed.append((page_number, result))
    return parsed


def _quad_to_rect(box: Any, page_number: int) -> tuple[float, float, float, float]:
    """Return the axis-aligned ``(x0, top, x1, bottom)`` bounds of an OCR quad.

    OCR boxes are four corner points because recognized text can be skewed. The
    downstream metadata is rectangular, so the quad is reduced to its bounding
    box, which is what local DeepDoc reports too.

    Raises:
        ValueError: If the box is not a sequence of at least three numeric
            ``(x, y)`` points.
    """
    if not isinstance(box, (list, tuple)) or len(box) < 3:
        raise ValueError(
            f"Remote DeepDoc ocr page {page_number} has a malformed line box ({box!r})"
        )
    xs: list[float] = []
    ys: list[float] = []
    for point in box:
        if (
            not isinstance(point, (list, tuple))
            or len(point) < 2
            or isinstance(point[0], bool)
            or isinstance(point[1], bool)
            or not isinstance(point[0], (int, float))
            or not isinstance(point[1], (int, float))
        ):
            raise ValueError(
                f"Remote DeepDoc ocr page {page_number} has a malformed line box "
                f"point ({point!r})"
            )
        xs.append(float(point[0]))
        ys.append(float(point[1]))
    return min(xs), min(ys), max(xs), max(ys)


def _collect_ocr_lines(
    ocr_pages: list[tuple[int, dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """Return per-page OCR lines reduced to rectangles, in reading order.

    Raises:
        ValueError: If a page's ``lines`` are missing or malformed.
    """
    by_page: dict[int, list[dict[str, Any]]] = {}
    for page_number, result in ocr_pages:
        lines = result.get("lines")
        if not isinstance(lines, list):
            raise ValueError(
                f"Remote DeepDoc ocr page {page_number} is missing a 'lines' list "
                f"(got {type(lines).__name__})"
            )
        page_lines: list[dict[str, Any]] = []
        for line in lines:
            if not isinstance(line, dict):
                raise ValueError(
                    f"Remote DeepDoc ocr page {page_number} has a "
                    f"{type(line).__name__} line, expected an object"
                )
            text = line.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"Remote DeepDoc ocr page {page_number} has a line without "
                    "string 'text'"
                )
            x0, top, x1, bottom = _quad_to_rect(line.get("box"), page_number)
            if not text.strip():
                # A blank recognition carries no content but would still occupy
                # a slot in a joined block, so it is dropped rather than joined.
                continue
            page_lines.append(
                {
                    "text": text,
                    "x0": x0,
                    "x1": x1,
                    "top": top,
                    "bottom": bottom,
                }
            )
        # A page repeated across responses would otherwise lose lines.
        by_page.setdefault(page_number, []).extend(page_lines)

    # Sorted after accumulation, not per batch: a page split across two response
    # entries must end up in one reading order, not two concatenated ones.
    for page_lines in by_page.values():
        page_lines.sort(key=lambda item: (item["top"], item["x0"]))
    return by_page


def _collect_layout_blocks(
    layout_pages: list[tuple[int, dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """Return per-page layout blocks, ordered so the most specific claims first.

    Raises:
        ValueError: If a page's ``layouts`` are missing or malformed.
    """
    by_page: dict[int, list[dict[str, Any]]] = {}
    for page_number, result in layout_pages:
        layouts = result.get("layouts")
        if not isinstance(layouts, list):
            raise ValueError(
                f"Remote DeepDoc layout page {page_number} is missing a 'layouts' "
                f"list (got {type(layouts).__name__})"
            )
        page_blocks: list[dict[str, Any]] = []
        for block in layouts:
            if not isinstance(block, dict):
                raise ValueError(
                    f"Remote DeepDoc layout page {page_number} has a "
                    f"{type(block).__name__} block, expected an object"
                )
            bbox = block.get("bbox")
            if (
                not isinstance(bbox, (list, tuple))
                or len(bbox) < 4
                or any(isinstance(value, bool) for value in bbox[:4])
                or not all(isinstance(value, (int, float)) for value in bbox[:4])
            ):
                raise ValueError(
                    f"Remote DeepDoc layout page {page_number} has a malformed "
                    f"bbox ({bbox!r})"
                )
            layout_type = block.get("type")
            if not isinstance(layout_type, str):
                layout_type = "text"
            x0, top, x1, bottom = (float(value) for value in bbox[:4])
            page_blocks.append(
                {
                    "layout_type": layout_type.strip().lower(),
                    # Normalize inverted bounds so containment tests are honest.
                    "x0": min(x0, x1),
                    "x1": max(x0, x1),
                    "top": min(top, bottom),
                    "bottom": max(top, bottom),
                }
            )
        by_page.setdefault(page_number, []).extend(page_blocks)

    # Specific types claim their lines before the plain text block that overlaps
    # them; ties keep reading order so the join is deterministic. Sorted after
    # accumulation so a page split across two response entries is ranked as one.
    for page_blocks in by_page.values():
        page_blocks.sort(
            key=lambda item: (
                -_LAYOUT_TYPE_PRIORITY.get(
                    item["layout_type"], _DEFAULT_LAYOUT_PRIORITY
                ),
                item["top"],
                item["x0"],
            )
        )
    return by_page


def _build_element(
    *,
    element_type: str,
    layout_type: str,
    text: str,
    page_number: int,
    x0: float,
    x1: float,
    top: float,
    bottom: float,
) -> dict[str, Any]:
    """Return one element in the shape ``_translate_remote_elements`` consumes."""
    return {
        "type": element_type,
        "text": text,
        # The OCR endpoint returns no image bytes at all, for any element type.
        "image": None,
        "metadata": {
            "page_number": page_number,
            "x0": x0,
            "x1": x1,
            "top": top,
            "bottom": bottom,
            "layout_type": layout_type,
            "col_id": 0,
            # _build_element_metadata inserts col_id at index 1 when enriching.
            "positions": [[page_number, x0, x1, top, bottom]],
        },
    }


def _join_ocr_and_layout(
    ocr_pages: list[tuple[int, dict[str, Any]]],
    layout_pages: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Group OCR lines into layout blocks and return elements in reading order.

    Both tasks report coordinates in the same rasterized page space, so a line
    belongs to the block whose bbox contains the line's center point. Each line
    is claimed at most once, and a line no block claims becomes its own text
    element: losing recognized text would be strictly worse than emitting it
    unstructured.

    Args:
        ocr_pages: ``(page_number, result)`` pairs from the ``ocr`` task.
        layout_pages: ``(page_number, result)`` pairs from the ``layout`` task.

    Returns:
        Elements sorted by ``(page, top, x0)``.

    Raises:
        ValueError: If either task's page payloads are malformed.
    """
    lines_by_page = _collect_ocr_lines(ocr_pages)
    blocks_by_page = _collect_layout_blocks(layout_pages)

    elements: list[dict[str, Any]] = []
    for page_number in sorted(set(lines_by_page) | set(blocks_by_page)):
        page_lines = lines_by_page.get(page_number, [])
        claimed = [False] * len(page_lines)

        for block in blocks_by_page.get(page_number, []):
            members: list[dict[str, Any]] = []
            for index, line in enumerate(page_lines):
                if claimed[index]:
                    continue
                center_x = (line["x0"] + line["x1"]) / 2.0
                center_y = (line["top"] + line["bottom"]) / 2.0
                if (
                    block["x0"] <= center_x <= block["x1"]
                    and block["top"] <= center_y <= block["bottom"]
                ):
                    claimed[index] = True
                    members.append(line)
            if not members:
                # A block with no text of its own -- a bare figure, or a caption
                # already claimed by a more specific block -- carries nothing
                # downstream, so it is not emitted.
                continue
            # page_lines is already in reading order, so members inherits it.
            elements.append(
                _build_element(
                    element_type=_LAYOUT_TYPE_TO_ELEMENT_TYPE.get(
                        block["layout_type"], "text"
                    ),
                    layout_type=block["layout_type"],
                    text="\n".join(member["text"] for member in members),
                    page_number=page_number,
                    x0=min(member["x0"] for member in members),
                    x1=max(member["x1"] for member in members),
                    top=min(member["top"] for member in members),
                    bottom=max(member["bottom"] for member in members),
                )
            )

        for index, line in enumerate(page_lines):
            if claimed[index]:
                continue
            elements.append(
                _build_element(
                    element_type="text",
                    layout_type="text",
                    text=line["text"],
                    page_number=page_number,
                    x0=line["x0"],
                    x1=line["x1"],
                    top=line["top"],
                    bottom=line["bottom"],
                )
            )

    elements.sort(
        key=lambda element: (
            element["metadata"]["page_number"],
            element["metadata"]["top"],
            element["metadata"]["x0"],
        )
    )
    return elements


def parse_document_remote(
    file_path: str | BytesIO,
    *,
    ext: str,
    save_image: SaveImage,
    callback: Optional[ProgressCallback] = None,
    zoomin: int = 3,
    _transport: Optional[httpx.BaseTransport] = None,
) -> list[dict[str, Any]]:
    """Parse a document on the remote DeepDoc OCR endpoint.

    Two requests are sent over one connection: ``task=ocr`` for the text and
    ``task=layout`` for the block structure. Their results are joined spatially.

    Args:
        file_path: Path to the original file, or its bytes in memory.
        ext: File extension of the document, for example ``".pdf"``.
        save_image: Accepted for signature compatibility with the local path and
            never called -- the OCR endpoint returns no image bytes.
        callback: Optional progress sink taking ``(fraction, message)``. The
            ``"message (1.23s)"`` shape of the completion notice matches what
            local DeepDoc emits, so the existing progress adapter strips the
            timing suffix and dedupes statuses without any change.
        zoomin: Accepted for signature compatibility. The server rasterizes the
            document itself and takes a ``dpi`` rather than a scale factor, so
            this value is not forwarded.
        _transport: Test-only transport override.

    Returns:
        The parsed elements, each with ``image`` set to ``None``.

    Raises:
        DeepDocRemoteError: On any failure. Callers fall back to local parsing.
    """
    del save_image, zoomin  # Signature compatibility only; see above.

    started = time.monotonic()
    if callback is not None:
        callback(0.05, "Uploading document to remote DeepDoc server")

    try:
        base_url = get_deepdoc_xinference_url()
        if base_url is None:
            raise ValueError("Remote DeepDoc parsing is not configured")

        timeout_seconds = float(get_deepdoc_xinference_timeout_seconds())
        timeout = httpx.Timeout(
            connect=_CONNECT_TIMEOUT_SECONDS,
            read=timeout_seconds,
            write=timeout_seconds,
            pool=_POOL_TIMEOUT_SECONDS,
        )
        with httpx.Client(timeout=timeout, transport=_transport) as client:
            token = _get_access_token(client, base_url)
            headers = {"Authorization": f"Bearer {token}"} if token else {}

            ocr_payload = _post_ocr_task(
                client,
                file_path,
                base_url=base_url,
                ext=ext,
                task="ocr",
                headers=headers,
            )
            if callback is not None:
                callback(0.5, "Recognizing document layout on remote DeepDoc server")
            layout_payload = _post_ocr_task(
                client,
                file_path,
                base_url=base_url,
                ext=ext,
                task="layout",
                headers=headers,
            )

        elements = _join_ocr_and_layout(
            _iter_pages(ocr_payload, "ocr"),
            _iter_pages(layout_payload, "layout"),
        )
    except httpx.HTTPError as exc:
        raise DeepDocRemoteError(f"Remote DeepDoc request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DeepDocRemoteError(
            f"Remote DeepDoc returned a non-JSON response: {exc}"
        ) from exc
    except ValueError as exc:
        raise DeepDocRemoteError(
            f"Remote DeepDoc returned an unusable response: {exc}"
        ) from exc
    except OSError as exc:
        raise DeepDocRemoteError(f"Remote DeepDoc parse failed: {exc}") from exc
    except Exception as exc:
        # Anything unforeseen must still end in the local fallback rather than
        # propagating out of the parser.
        raise DeepDocRemoteError(f"Remote DeepDoc parse failed: {exc}") from exc

    elapsed = time.monotonic() - started
    if callback is not None:
        callback(1.0, f"Remote DeepDoc parse finished ({elapsed:.2f}s)")
    logger.info(
        "Remote DeepDoc parsed %s into %d elements in %.2fs",
        ext,
        len(elements),
        elapsed,
    )
    return elements
