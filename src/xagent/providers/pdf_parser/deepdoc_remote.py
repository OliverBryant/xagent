"""HTTP client for offloading DeepDoc document parsing to a remote Xinference server.

Set ``XAGENT_DEEPDOC_XINFERENCE_URL`` to route every format ``DeepDocParser``
supports to a GPU host instead of running local ONNX inference. Any failure
raises :class:`DeepDocRemoteError`, which the caller treats as "fall back to
local parsing" -- fallback is always on, which is what makes the switch
transparent.

This module deliberately depends on nothing but ``httpx`` and
``xagent.config``. In particular it must not import ``deepdoc.py``: the
image-saving function is injected by the caller instead, which keeps the import
graph acyclic and lets the client be tested without loading any parser.

Proposed server API contract (v1)
---------------------------------

Xinference has no DeepDoc serving capability today; the endpoint below is a
proposal tracked alongside this client, mirrored from
``docs/deepdoc-remote-parsing.md`` section 6.

.. code-block:: text

    POST {base_url}/v1/document/parse
    Authorization: Bearer <api_key>          # omitted when the client has no key

    Request (multipart/form-data):
      file         binary  required            original file; filename preserved
                                               (server dispatches on the extension)
      zoomin       int     default 3           PDF only, forwarded to parse_into_bboxes
      image_scope  str     default table_figure    table_figure | all | none

    Response 200 application/json:
    {
      "filename": "report.pdf",
      "file_type": ".pdf",
      "elapsed_ms": 45210,
      "elements": [
        {
          "type": "text",              // "text" | "table" | "figure"
          "text": "...",               // HTML for tables, matching local behavior
          "image_base64": null,        // PNG base64 for table/figure, null otherwise
          "metadata": { ... }          // format specific, see the doc
        }
      ]
    }

    Errors: 400 invalid/unsupported file, 401 auth failure, 413 file too large,
            500 inference failure. Body is always {"detail": "..."}.

Per-format server behavior and the ``metadata`` keys each format must supply
are tabulated in the requirements doc. The intent is semantic parity with
xagent's local parsing so remote and local results stay interchangeable when
fallback kicks in: ``.pdf`` comes from ``parse_into_bboxes(zoomin)`` and
carries ``page_number``, ``x0``, ``x1``, ``top``, ``bottom``, ``layout_type``,
``col_id`` and ``positions``; ``.xlsx`` emits one element per row with
``sheet_name``, ``row_number`` and ``row_type``; the remaining formats follow
their local counterparts.

Only ``table`` and ``figure`` elements carry an image, because that is all the
downstream translation consumes -- stripping text-element crops cuts the
payload substantially.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from ...config import (
    get_deepdoc_xinference_api_key,
    get_deepdoc_xinference_timeout_seconds,
    get_deepdoc_xinference_url,
)

logger = logging.getLogger(__name__)

PARSE_ENDPOINT = "/v1/document/parse"

# Connecting to and writing headers against a reachable host is fast; only the
# parse itself is slow, so the connect/pool budgets stay short regardless of
# how long the configured read timeout is.
_CONNECT_TIMEOUT_SECONDS = 10.0
_POOL_TIMEOUT_SECONDS = 10.0

_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
    ".html": "text/html",
}
_DEFAULT_MIME_TYPE = "application/octet-stream"

ProgressCallback = Callable[[float, str], None]
SaveImage = Callable[[bytes], str]


class DeepDocRemoteError(Exception):
    """Remote parsing failed and the caller should fall back to local parsing.

    Every failure mode -- unreachable host, timeout, 4xx/5xx, unparsable or
    malformed body, undecodable image, failed image write -- is reported as
    this single type so callers need only one ``except`` clause.
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


def _build_headers() -> dict[str, str]:
    """Return the auth headers, which are empty when no API key is configured."""
    api_key = get_deepdoc_xinference_api_key()
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def _mime_type_for(ext: str) -> str:
    """Return the MIME type to advertise for ``ext``."""
    return _MIME_TYPES.get(ext.lower(), _DEFAULT_MIME_TYPE)


def _post_document(
    file_path: str | BytesIO,
    *,
    ext: str,
    zoomin: int,
    transport: Optional[httpx.BaseTransport],
) -> dict[str, Any]:
    """Upload the document and return the decoded JSON response body.

    Args:
        file_path: Path to the original file, or its bytes in memory.
        ext: Lower-cased file extension, used for the MIME type and for the
            synthetic filename of an in-memory document.
        zoomin: PDF render scale forwarded to the server.
        transport: Optional transport override. This is the seam tests use to
            install an ``httpx.MockTransport`` instead of reaching the network.

    Raises:
        httpx.HTTPError: On any transport failure or non-2xx status.
        ValueError: If the body is not JSON, or is JSON but not an object.
        OSError: If a file path cannot be read.
    """
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
    data = {"zoomin": str(zoomin)}
    mime_type = _mime_type_for(ext)

    def _send(filename: str, fh: Any) -> dict[str, Any]:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(
                f"{base_url}{PARSE_ENDPOINT}",
                headers=_build_headers(),
                data=data,
                files={"file": (filename, fh, mime_type)},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                f"Remote DeepDoc response body is {type(payload).__name__}, "
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


def _normalize_elements(
    payload: dict[str, Any], save_image: SaveImage
) -> list[dict[str, Any]]:
    """Validate the response elements and materialize their images on disk.

    Each element's ``image_base64`` is replaced by an ``image`` key holding the
    path string of the saved file, or ``None`` when the element carries no
    image. A path string is exactly what the caller's existing image handling
    already accepts, so no downstream change is needed.

    Args:
        payload: Decoded response body.
        save_image: Writes decoded image bytes and returns the resulting path.

    Raises:
        ValueError: If the payload does not carry a list of elements that each
            look like a parsed element, or if an element's image decodes to no
            bytes at all.
        binascii.Error: If an element's image is not valid base64.
    """
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise ValueError(
            "Remote DeepDoc response is missing an 'elements' list "
            f"(got {type(elements).__name__})"
        )

    normalized: list[dict[str, Any]] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            raise ValueError(
                f"Remote DeepDoc element {index} is {type(element).__name__}, "
                "expected an object"
            )
        missing = [key for key in ("type", "text") if key not in element]
        if missing:
            raise ValueError(
                f"Remote DeepDoc element {index} is missing "
                f"{', '.join(repr(key) for key in missing)}"
            )

        normalized_element = dict(element)
        image_base64 = normalized_element.pop("image_base64", None)
        if image_base64:
            # validate=True rejects non-alphabet characters instead of dropping
            # them. Without it a corrupt-but-right-length payload decodes to
            # b"" and gets written out as a zero-byte PNG, so the caller
            # receives a broken image path rather than falling back to local
            # parsing.
            image_bytes = base64.b64decode(image_base64, validate=True)
            if not image_bytes:
                raise ValueError(
                    f"Remote DeepDoc element {index} carries an empty image"
                )
            normalized_element["image"] = save_image(image_bytes)
        else:
            normalized_element["image"] = None
        normalized.append(normalized_element)

    return normalized


def parse_document_remote(
    file_path: str | BytesIO,
    *,
    ext: str,
    save_image: SaveImage,
    callback: Optional[ProgressCallback] = None,
    zoomin: int = 3,
    _transport: Optional[httpx.BaseTransport] = None,
) -> list[dict[str, Any]]:
    """Parse a document on the remote DeepDoc server.

    Args:
        file_path: Path to the original file, or its bytes in memory.
        ext: File extension of the document, for example ``".pdf"``.
        save_image: Writes decoded image bytes and returns the resulting path.
            Injected by the caller so this module stays independent of the
            parser that owns the artifact directory layout.
        callback: Optional progress sink taking ``(fraction, message)``. The
            ``"message (1.23s)"`` shape of the completion notice matches what
            local DeepDoc emits, so the existing progress adapter strips the
            timing suffix and dedupes statuses without any change.
        zoomin: PDF render scale forwarded to the server.
        _transport: Test-only transport override.

    Returns:
        The parsed elements, each with an ``image`` path string or ``None``.

    Raises:
        DeepDocRemoteError: On any failure. Callers fall back to local parsing.
    """
    started = time.monotonic()
    if callback is not None:
        callback(0.05, "Uploading document to remote DeepDoc server")

    try:
        payload = _post_document(
            file_path, ext=ext, zoomin=zoomin, transport=_transport
        )
        elements = _normalize_elements(payload, save_image)
    except httpx.HTTPError as exc:
        raise DeepDocRemoteError(f"Remote DeepDoc request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DeepDocRemoteError(
            f"Remote DeepDoc returned a non-JSON response: {exc}"
        ) from exc
    except binascii.Error as exc:
        raise DeepDocRemoteError(
            f"Remote DeepDoc returned an undecodable image: {exc}"
        ) from exc
    except ValueError as exc:
        raise DeepDocRemoteError(
            f"Remote DeepDoc returned an unusable response: {exc}"
        ) from exc
    except OSError as exc:
        # Covers both an unreadable source file and a failed image write.
        raise DeepDocRemoteError(f"Remote DeepDoc parse failed: {exc}") from exc
    except Exception as exc:
        # save_image is caller-supplied, so it may fail in ways this module
        # cannot enumerate. Fallback must still be the outcome.
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
