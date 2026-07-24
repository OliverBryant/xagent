"""
Pdf parser providers module.

This module provides various pdf parser backends
"""

import logging
from typing import Type

from xagent.providers.pdf_parser.base import (
    DocumentParser,
    FigureParsing,
    FullTextResult,
    LocalParsing,
    ParsedFigures,
    ParsedTextSegment,
    ParseResult,
    RemoteParsing,
    SegmentedTextResult,
    TextParsing,
)

from . import _libomp
from .basic import PdfPlumberParser, PyMuPdfParser, PyPdfParser, UnstructuredParser

logger = logging.getLogger(__name__)


def _load_deepdoc_parser() -> Type | None:
    """Import the DeepDoc parser, degrading to None on any failure.

    On macOS, deepdoc pulls in xgboost, whose native library
    (``libxgboost.dylib``) cannot be loaded without the OpenMP runtime
    (``libomp.dylib``). That surfaces as a non-``ImportError`` exception
    (``xgboost.core.XGBoostError`` / ``OSError``) which would otherwise crash
    startup. When we recognise that specific failure we try to install the
    OpenMP runtime automatically (conda preferred, Homebrew fallback).

    We do NOT retry the import in-process: xgboost loads (and caches as failed)
    its native extension at first import, so a freshly installed libomp can only
    take effect in a new process. On a successful install we therefore log that
    a restart is needed. Either way this process degrades DeepDoc to unavailable
    and lets the other parsers handle documents rather than crashing startup.
    """
    try:
        from .deepdoc import DeepDocParser

        return DeepDocParser
    except ImportError:
        # deepdoc (an optional extra) is not installed.
        return None
    except Exception as exc:  # noqa: BLE001
        # deepdoc is installed but failed to import.
        if _libomp.is_macos() and _libomp.looks_like_missing_libomp(exc):
            logger.warning(
                "DeepDoc parser failed to import because the OpenMP runtime "
                "(libomp) required by xgboost is missing: %s. Attempting to "
                "install it automatically...",
                exc,
            )
            if _libomp.try_install_libomp():
                logger.warning(
                    "OpenMP runtime (libomp) installed successfully. The DeepDoc "
                    "parser is disabled for this run because xgboost was already "
                    "loaded; RESTART Xagent to enable it. Other document parsers "
                    "remain available."
                )
            else:
                logger.warning(
                    "Could not install the OpenMP runtime automatically. DeepDoc "
                    "parser is unavailable; falling back to other document "
                    "parsers. To enable it, run `%s` and restart.",
                    _libomp.manual_fix_hint(),
                )
            return None

        logger.warning(
            "DeepDoc parser is unavailable because it failed to import: %s. "
            "Falling back to other document parsers.",
            exc,
        )
        return None


DeepDocParser: Type | None = _load_deepdoc_parser()

__all__ = [
    "ParseResult",
    "FigureParsing",
    "DeepDocParser",  # Will be None if deepdoc is not installed
    "PyPdfParser",
    "PdfPlumberParser",
    "UnstructuredParser",
    "PyMuPdfParser",
    "DocumentParser",
    "TextParsing",
    "FullTextResult",
    "SegmentedTextResult",
    "LocalParsing",
    "RemoteParsing",
    "ParsedTextSegment",
    "ParsedFigures",
]
