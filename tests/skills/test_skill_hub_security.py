"""Security tests for skill_hub ZIP extraction and file normalisation guards."""

from __future__ import annotations

import io
import zipfile
from unittest import mock

import pytest
from fastapi import HTTPException

from xagent.web.api.skill_hub import (
    _check_registry_security_gate,
    _normalize_skill_files,
    _safe_zip_extract,
    _safe_zip_to_files,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_zip(members: dict[str, bytes]) -> bytes:
    """Build an in-memory ZIP from a {filename: content} dict."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


SKILL_MD = b"# Test Skill\n\n## Description\nA test skill.\n"


# ── _normalize_skill_files ────────────────────────────────────────────────────


class TestNormalizeSkillFiles:
    def test_happy_path(self):
        result = _normalize_skill_files({"SKILL.md": SKILL_MD})
        assert result == {"SKILL.md": SKILL_MD}

    def test_missing_skill_md_raises(self):
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"other.txt": b"data"})
        assert exc.value.status_code == 400
        assert "SKILL.md" in exc.value.detail

    def test_path_traversal_raises(self):
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, "../escape.py": b"x"})
        assert exc.value.status_code == 400
        assert "traversal" in exc.value.detail.lower()

    def test_dotfile_raises(self):
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, ".env": b"SECRET=1"})
        assert exc.value.status_code == 400
        assert "hidden file" in exc.value.detail
        assert ".env" in exc.value.detail

    def test_absolute_path_stripped(self):
        result = _normalize_skill_files({"/SKILL.md": SKILL_MD})
        assert "SKILL.md" in result

    def test_windows_separator_normalised(self):
        result = _normalize_skill_files({"SKILL.md": SKILL_MD, "sub\\file.md": b"hi"})
        assert "sub/file.md" in result

    def test_size_cap_raises(self):
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        big = b"x" * (_MAX_DOWNLOAD_BYTES + 1)
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, "big.bin": big})
        assert exc.value.status_code == 413


# ── _safe_zip_to_files ────────────────────────────────────────────────────────


class TestSafeZipToFiles:
    def test_happy_path_flat(self):
        data = _make_zip({"SKILL.md": SKILL_MD, "template.md": b"# Template"})
        result = _safe_zip_to_files(data)
        assert "SKILL.md" in result
        assert "template.md" in result

    def test_happy_path_nested(self):
        """ZIP with a top-level directory wrapper."""
        data = _make_zip({"my-skill/SKILL.md": SKILL_MD, "my-skill/extra.md": b"hi"})
        result = _safe_zip_to_files(data)
        assert "SKILL.md" in result
        assert "extra.md" in result

    def test_bad_zip_raises(self):
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(b"not a zip")
        assert exc.value.status_code == 502

    def test_missing_skill_md_raises(self):
        data = _make_zip({"README.md": b"hello"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(data)
        assert exc.value.status_code == 400
        assert "SKILL.md" in exc.value.detail

    def test_path_traversal_in_zip_raises(self):
        data = _make_zip({"SKILL.md": SKILL_MD, "../escape.py": b"evil"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(data)
        assert exc.value.status_code == 400

    def test_dotfile_in_zip_rejected_by_normalize(self):
        data = _make_zip({"SKILL.md": SKILL_MD, ".env": b"SECRET=1"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(data)
        assert exc.value.status_code == 400

    def test_oversized_member_raises(self):
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        big = b"x" * (_MAX_DOWNLOAD_BYTES + 1)
        data = _make_zip({"SKILL.md": SKILL_MD, "large.bin": big})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(data)
        assert exc.value.status_code == 413


# ── _safe_zip_extract ─────────────────────────────────────────────────────────


def _tamper_declared_size(zip_bytes: bytes, declared: int) -> bytes:
    """Rewrite the uncompressed-size field of the first member in both
    the local header and the central directory."""
    import struct

    data = bytearray(zip_bytes)
    local = data.find(b"PK\x03\x04")
    data[local + 22 : local + 26] = struct.pack("<I", declared)
    central = data.find(b"PK\x01\x02")
    data[central + 24 : central + 28] = struct.pack("<I", declared)
    return bytes(data)


def _deflate_zip_with_corrupt_member() -> bytes:
    """A DEFLATE member whose compressed stream has been overwritten."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("SKILL.md", b"B" * 20000)
    data = bytearray(buf.getvalue())
    local = data.find(b"PK\x03\x04")
    header_len = 30 + data[local + 26] + data[local + 28]
    for offset in range(header_len, header_len + 8):
        data[local + offset] = 0xAB
    return bytes(data)


class TestSafeZipExtract:
    def test_returns_nested_root_name(self):
        data = _make_zip({"my-skill/SKILL.md": SKILL_MD, "my-skill/ref.md": b"hi"})
        files, root = _safe_zip_extract(data)
        assert root == "my-skill"
        assert set(files) == {"SKILL.md", "ref.md"}

    def test_returns_empty_root_for_flat_zip(self):
        data = _make_zip({"SKILL.md": SKILL_MD})
        files, root = _safe_zip_extract(data)
        assert root == ""
        assert "SKILL.md" in files

    def test_bad_zip_status_is_configurable(self):
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(b"not a zip", bad_zip_status=400)
        assert exc.value.status_code == 400

    def test_error_wording_is_source_neutral(self):
        # The extractor serves registry installs today and an upload route
        # next, so its messages must not name a specific registry.
        for build in (
            lambda: _safe_zip_extract(b"not a zip"),
            lambda: _safe_zip_extract(_make_zip({"README.md": b"no skill"})),
        ):
            with pytest.raises(HTTPException) as exc:
                build()
            assert "clawhub" not in exc.value.detail.lower()

    def test_decompression_bomb_is_refused_without_inflating_it(self):
        """A bomb must be rejected without its expansion ever being held.

        Reading the whole remaining budget in one call holds the result and
        zipfile's own growing buffer at the same time, so an ~800 KiB archive
        peaked at twice the 50 MiB budget. The declared size stops it before
        any inflation, and members are read in slices so the overshoot on a
        lying header is one chunk rather than the whole budget.
        """
        import tracemalloc

        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", SKILL_MD)
            for i in range(8):
                zf.writestr(f"bomb{i}.bin", b"\0" * (_MAX_DOWNLOAD_BYTES * 2))
        data = buf.getvalue()
        assert len(data) < 4 * 1024 * 1024  # tiny on the wire, huge inflated

        tracemalloc.start()
        try:
            with pytest.raises(HTTPException) as exc:
                _safe_zip_extract(data)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert exc.value.status_code == 413
        # Generous bound: the point is "not a multiple of the budget".
        assert peak < _MAX_DOWNLOAD_BYTES / 2, f"peak {peak} bytes"

    def test_over_declared_member_is_refused_on_its_header(self):
        """An over-declared member is refused before it is inflated.

        The declared size is the cheap first gate. Trusting it costs an
        archive that under-states a small file nothing real -- writers emit
        truthful headers -- while inflating first to find out costs the
        whole budget per bomb, which is the shape an attacker can actually
        produce. The real byte count is still checked as it is read.
        """
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        data = _make_zip({"SKILL.md": SKILL_MD})
        lying = _tamper_declared_size(data, _MAX_DOWNLOAD_BYTES + 1)
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(lying)
        assert exc.value.status_code == 413

    def test_honest_members_within_budget_are_read_whole(self):
        # The chunked read must reassemble a member larger than one chunk
        # byte-for-byte, not truncate it at the chunk boundary.
        from xagent.web.api.skill_hub import _ARCHIVE_CHUNK_BYTES

        body = bytes(range(256)) * ((_ARCHIVE_CHUNK_BYTES * 2) // 256 + 7)
        data = _make_zip({"SKILL.md": SKILL_MD, "big.bin": body})
        files, _root = _safe_zip_extract(data)
        assert files["big.bin"] == body


# ── archive root selection ───────────────────────────────────────────────────


class TestArchiveRootSelection:
    """The root is the shallowest SKILL.md, not the alphabetically first.

    A skill folder that ships its own ``Examples/SKILL.md`` sorts before the
    real root marker, so a plain sort imported the *example* as the skill —
    named "Examples", with the true root's files silently dropped and a 200
    returned. Ordinary shape, not an adversarial one.
    """

    def test_subfolder_does_not_beat_the_true_root(self):
        data = _make_zip(
            {
                "SKILL.md": SKILL_MD,
                "reference.md": b"reference material",
                "Examples/SKILL.md": b"---\ndescription: an example\n---\n# Ex\n",
            }
        )
        files, root = _safe_zip_extract(data)
        assert root == ""
        # The real skill wins and nothing is discarded.
        assert files["SKILL.md"] == SKILL_MD
        assert "reference.md" in files
        assert "Examples/SKILL.md" in files

    def test_wrapper_directory_still_resolves(self):
        data = _make_zip(
            {
                "pdf-tools/SKILL.md": SKILL_MD,
                "pdf-tools/ref.md": b"r",
                "pdf-tools/examples/SKILL.md": b"---\ndescription: e\n---\n# E\n",
            }
        )
        files, root = _safe_zip_extract(data)
        assert root == "pdf-tools"
        assert sorted(files) == ["SKILL.md", "examples/SKILL.md", "ref.md"]

    def test_two_roots_at_the_same_depth_are_refused(self):
        data = _make_zip({"a/SKILL.md": SKILL_MD, "b/SKILL.md": SKILL_MD})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400
        assert "multiple skills" in exc.value.detail

    def test_multiple_roots_reported_without_reflecting_the_whole_archive(self):
        """The rejection names a bounded sample, not every root it found.

        The names come from the archive, so an unbounded list would let a
        crafted upload reflect arbitrary attacker text back through the error.
        """
        long_name = "z" * 500
        data = _make_zip(
            {f"root{i}/SKILL.md": SKILL_MD for i in range(9)}
            | {f"{long_name}/SKILL.md": SKILL_MD}
        )
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400
        assert len(exc.value.detail) < 400
        assert long_name not in exc.value.detail
        assert "5 more" in exc.value.detail

    def test_cruft_is_not_a_root_candidate(self):
        """Cruft must be excluded before candidates are compared.

        At the *same* depth as the real skill it would otherwise look like a
        second skill and trigger a spurious "multiple skills" rejection — which
        is what a Finder-zipped folder actually produces. A deeper __MACOSX
        would be beaten by depth alone, so that shape proves nothing here.
        """
        data = _make_zip(
            {
                "pdf-tools/SKILL.md": SKILL_MD,
                "__MACOSX/SKILL.md": b"junk",
            }
        )
        files, root = _safe_zip_extract(data)
        assert root == "pdf-tools"
        assert files["SKILL.md"] == SKILL_MD

    def test_cruft_alone_is_not_a_skill(self):
        data = _make_zip({"__MACOSX/SKILL.md": b"junk"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400
        assert "no SKILL.md" in exc.value.detail


# ── unreadable archives ──────────────────────────────────────────────────────


class TestUnreadableArchives:
    """Anything zipfile raises on an untrusted archive must be a 4xx/5xx.

    Guarded at one boundary rather than by naming exception types, because the
    per-type list kept missing cases: a tampered end-of-central-directory
    offset raises ValueError from the constructor, well before any member read.
    """

    def test_tampered_eocd_offset(self):
        import struct

        raw = bytearray(_make_zip({"s/SKILL.md": SKILL_MD}))
        eocd = raw.rfind(b"PK\x05\x06")
        struct.pack_into("<I", raw, eocd + 16, 0xFFFFFFF0)
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(bytes(raw), bad_zip_status=400)
        assert exc.value.status_code == 400

    def test_truncated_central_directory(self):
        raw = _make_zip({"s/SKILL.md": SKILL_MD})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(raw[: len(raw) // 2], bad_zip_status=400)
        assert exc.value.status_code == 400

    def test_corrupt_deflate_member(self):
        # zipfile only detects the damage while reading the member, not at
        # open(): zlib.error subclasses neither BadZipFile nor RuntimeError,
        # so it used to escape the extractor as a 500.
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(_deflate_zip_with_corrupt_member(), bad_zip_status=400)
        assert exc.value.status_code == 400

    def test_registry_source_keeps_its_status(self):
        # The boundary guard must not flatten the caller-specific status.
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(b"not a zip at all")
        assert exc.value.status_code == 502

    def test_corrupt_member_keeps_the_registry_status(self):
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(_deflate_zip_with_corrupt_member())
        assert exc.value.status_code == 502

    def test_size_budget_still_reported_as_413(self):
        # Our own HTTPExceptions must pass through the guard unchanged, or
        # every rejection raised inside the try block collapses onto one code.
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", SKILL_MD)
            zf.writestr("big.bin", b"\0" * (_MAX_DOWNLOAD_BYTES + 1024))
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(buf.getvalue(), bad_zip_status=400)
        assert exc.value.status_code == 413

    def test_traversal_still_reported_as_400(self):
        # The other in-try rejection: 400 here and 413 above must stay
        # distinct even when the caller's bad-ZIP status is 502.
        data = _make_zip({"SKILL.md": SKILL_MD, "../escape.py": b"x"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400


# ── macOS archive cruft ──────────────────────────────────────────────────────


class TestArchiveCruft:
    """Zipping a folder in Finder sweeps in .DS_Store and resource forks.

    A skill folder that has been opened in Finder carries them, so refusing
    the whole archive over one would fail an entirely ordinary bundle —
    while real hidden files stay refused.
    """

    @pytest.mark.parametrize(
        "cruft",
        [
            "pdf-tools/.DS_Store",
            "pdf-tools/refs/.DS_Store",
            "__MACOSX/._pdf-tools",
            "pdf-tools/._reference.md",
            "pdf-tools/Thumbs.db",
            "pdf-tools/desktop.ini",
        ],
    )
    def test_cruft_is_dropped_not_rejected(self, cruft):
        data = _make_zip({"pdf-tools/SKILL.md": SKILL_MD, cruft: b"\x00\x01"})
        files, root = _safe_zip_extract(data)
        assert root == "pdf-tools"
        assert sorted(files) == ["SKILL.md"]

    @pytest.mark.parametrize("hidden", [".env", "sub/.env", "a/b/.secret"])
    def test_real_hidden_files_still_rejected(self, hidden):
        # The old check only looked at the first character of the whole path,
        # so ".env" was refused but "sub/.env" was not.
        data = _make_zip({"SKILL.md": SKILL_MD, hidden: b"SECRET=1"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400
        assert "hidden file" in exc.value.detail


# ── _check_registry_security_gate ────────────────────────────────────────────


def _make_registry(display_name: str = "TestHub"):
    """Minimal registry stub with a ClawHub-compatible extract_scan_status."""
    from types import SimpleNamespace

    def extract_scan_status(raw_item):
        latest = raw_item.get("latestVersion") or {}
        security = latest.get("security") or {}
        return security.get("status") if isinstance(security, dict) else None

    return SimpleNamespace(
        display_name=display_name, extract_scan_status=extract_scan_status
    )


def _detail(*, scan_status=None, moderation_state=None):
    """Build a fake registry detail payload."""
    d = {}
    if scan_status is not None:
        d["latestVersion"] = {"security": {"status": scan_status}}
    if moderation_state is not None:
        d["moderation"] = {"moderationState": moderation_state}
    return d


class TestCheckRegistrySecurityGate:
    def test_malicious_scan_status_refused(self):
        with pytest.raises(HTTPException) as exc:
            _check_registry_security_gate(
                _make_registry(), _detail(scan_status="malicious")
            )
        assert exc.value.status_code == 403
        assert "malicious" in exc.value.detail.lower()

    def test_quarantined_refused(self):
        with pytest.raises(HTTPException) as exc:
            _check_registry_security_gate(
                _make_registry(), _detail(moderation_state="quarantined")
            )
        assert exc.value.status_code == 403
        assert "quarantined" in exc.value.detail.lower()

    def test_revoked_refused(self):
        with pytest.raises(HTTPException) as exc:
            _check_registry_security_gate(
                _make_registry(), _detail(moderation_state="revoked")
            )
        assert exc.value.status_code == 403
        assert "revoked" in exc.value.detail.lower()

    def test_clean_scan_status_allowed(self):
        # Must not raise.
        _check_registry_security_gate(_make_registry(), _detail(scan_status="clean"))

    def test_suspicious_scan_status_allowed(self):
        # "suspicious" is a warning, not a hard block.
        _check_registry_security_gate(
            _make_registry(), _detail(scan_status="suspicious")
        )

    def test_no_security_data_allowed(self):
        # Missing keys → None scan status → gate passes.
        _check_registry_security_gate(_make_registry(), {})

    def test_both_signals_malicious_wins(self):
        with pytest.raises(HTTPException) as exc:
            _check_registry_security_gate(
                _make_registry(),
                _detail(scan_status="malicious", moderation_state="quarantined"),
            )
        assert exc.value.status_code == 403
        assert "malicious" in exc.value.detail.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "expected_method"),
    [
        ("create", "create_skill"),
        ("update", "update_skill_file"),
        ("delete", "delete_skill"),
        ("registry_install", "create_skill"),
    ],
)
async def test_team_write_routes_adopt_the_central_provider_invoker(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    expected_method: str,
) -> None:
    from dataclasses import fields
    from types import SimpleNamespace

    from xagent.skills.library import SkillScopeContext
    from xagent.web.api import skill_hub

    captured: list[tuple[str, object, dict]] = []
    scope = SkillScopeContext(user_id=7, metadata={"source_id": 11})

    class _Manager:
        async def get_skill(self, name: str):
            return {"name": name, "scope": "team", "path": ""}

    async def _central_invoker(provider, method, context, **kwargs):
        captured.append((method, context, kwargs))

    registry = SimpleNamespace(
        id="testhub",
        display_name="TestHub",
        get_skill=lambda slug: {},
        extract_scan_status=lambda detail: None,
        download_skill=lambda slug, version: (200, _make_zip({"SKILL.md": SKILL_MD})),
    )
    monkeypatch.setattr(skill_hub, "invoke_skill_write_provider", _central_invoker)
    monkeypatch.setattr(
        "xagent.skills.library.get_skill_write_provider", lambda: object()
    )

    async def _scoped_manager(*args):
        return _Manager()

    monkeypatch.setattr(skill_hub, "_get_scoped_manager", _scoped_manager)
    monkeypatch.setattr(skill_hub, "get_registry", lambda source: registry)

    request = SimpleNamespace()
    user = SimpleNamespace(id=7)
    if route == "create":
        await skill_hub.create_skill(
            skill_hub.CreateSkillRequest(
                name="writer", skill_md="# writer", scope="team"
            ),
            request,
            scope,
            object(),
            user,
        )
    elif route == "update":
        await skill_hub.edit_installed(
            "writer",
            skill_hub.EditSkillRequest(skill_md="# updated"),
            request,
            scope,
            object(),
            user,
        )
    elif route == "delete":
        await skill_hub.delete_installed("writer", request, scope, object(), user)
    else:
        await skill_hub.install_skill(
            "testhub",
            skill_hub.InstallSkillRequest(slug="writer", scope="team"),
            request,
            scope,
            object(),
            user,
        )

    assert len(captured) == 1
    method, context, kwargs = captured[0]
    assert method == expected_method
    assert {field.name for field in fields(context)} == {"user_id", "metadata"}
    assert context.user_id == scope.user_id
    assert context.metadata == scope.metadata
    assert kwargs["scope"] == "team"


# ── construction-time entry bound ────────────────────────────────────────────


def _entry_zip(
    count: int, *, comment: bytes = b"", prefix: bytes = b"", name_len: int = 0
) -> bytes:
    """An archive of ``count`` empty members, optionally commented/prefixed.

    ``prefix`` stands in for a self-extracting stub: bytes before the local
    headers that shift every offset in the archive.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(count):
            name = f"skill/f{i:08d}"
            zf.writestr(name.ljust(name_len, "x"), b"")
        zf.comment = comment
    return prefix + buf.getvalue()


class TestArchiveEntryBound:
    """An over-sized central directory is refused during construction.

    ``ZipFile.__init__`` builds a ``ZipInfo`` per directory entry before any
    of our code runs, so a cap read off ``infolist()`` bounds only what comes
    after. These tests assert the mechanism rather than the status code: a
    pre-construction and a post-construction refusal both answer 400, so a
    status assertion cannot tell them apart.
    """

    def test_entry_cap_boundary(self):
        """Exactly at the cap is accepted; one over is refused.

        The boundary is the test that matters. Counting per read rather than
        across reads under-counted a 2,001-entry directory as 1,993 because
        ``BufferedReader`` splits it into blocks — a bug a 20,000-entry case
        hides completely, since 20,000 trips any cap regardless.
        """
        from xagent.web.api.skill_hub import _MAX_ARCHIVE_ENTRIES

        at_cap = _entry_zip(_MAX_ARCHIVE_ENTRIES)
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(at_cap, bad_zip_status=400)
        # Accepted by the bound — it fails later, on having no SKILL.md.
        assert "no SKILL.md" in exc.value.detail

        over_cap = _entry_zip(_MAX_ARCHIVE_ENTRIES + 1)
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(over_cap, bad_zip_status=400)
        assert exc.value.status_code == 400
        assert "more than" in exc.value.detail

    def test_source_guard_holds_the_boundary_on_its_own(self):
        """The construction-time bound, with the ``infolist()`` gate removed.

        Both gates answer the same 400 with the same wording, so a test that
        only calls ``_safe_zip_extract`` cannot see which one fired: weakening
        the source guard leaves every such test green because the second gate
        covers for it. Drive the source directly, past ``ZipFile``, and assert
        the count it actually produced.

        The exact boundary is the assertion that matters. A per-read tally
        (rather than one accumulated across reads) saw 1,993 headers of a
        2,001-entry directory, because ``BufferedReader`` splits the directory
        into fixed-size blocks — invisible to any test that only tries a
        clearly-hostile count.
        """
        from xagent.web.api.skill_hub import (
            _MAX_ARCHIVE_ENTRIES,
            _ArchiveTooManyEntries,
            _BoundedZipSource,
        )

        source = _BoundedZipSource(_entry_zip(_MAX_ARCHIVE_ENTRIES))
        zf = zipfile.ZipFile(io.BufferedReader(source))
        assert len(zf.namelist()) == _MAX_ARCHIVE_ENTRIES
        # Every header was counted — not a subset that happens to be under.
        assert source._entries == _MAX_ARCHIVE_ENTRIES

        over = _BoundedZipSource(_entry_zip(_MAX_ARCHIVE_ENTRIES + 1))
        with pytest.raises(_ArchiveTooManyEntries):
            zipfile.ZipFile(io.BufferedReader(over))
        # Refused on the first entry past the cap, not somewhere well beyond.
        assert over._entries == _MAX_ARCHIVE_ENTRIES + 1

    def test_over_cap_is_a_content_rejection_not_an_unreadable_one(self):
        """400 regardless of ``bad_zip_status``.

        An archive over the cap parses fine; it just carries more than we
        accept, which is a statement about its contents like the traversal and
        size rejections. ``bad_zip_status`` says who supplied an *unreadable*
        archive, and answering with it here would make a registry-sourced
        upload a 502 the client cannot act on.
        """
        from xagent.web.api.skill_hub import _MAX_ARCHIVE_ENTRIES

        over_cap = _entry_zip(_MAX_ARCHIVE_ENTRIES + 1)
        for bad_zip_status in (400, 502):
            with pytest.raises(HTTPException) as exc:
                _safe_zip_extract(over_cap, bad_zip_status=bad_zip_status)
            assert exc.value.status_code == 400
            assert "more than" in exc.value.detail

    def test_refusal_happens_before_the_directory_is_materialised(self):
        """The refusal must beat ``ZipFile``, not follow it.

        Spy on ``ZipInfo`` construction: an archive of 100,000 padding
        directories costs ~3s and ~56 MiB to materialise, all of it spent
        before a post-construction cap could look. Far fewer than the whole
        directory may be built here — the bound trips partway through it.
        """
        from xagent.web.api import skill_hub
        from xagent.web.api.skill_hub import _MAX_ARCHIVE_ENTRIES

        built = 0
        real_zipinfo = zipfile.ZipInfo

        class CountingZipInfo(real_zipinfo):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                nonlocal built
                built += 1
                super().__init__(*args, **kwargs)

        hostile = _entry_zip(50_000)
        with mock.patch.object(zipfile, "ZipInfo", CountingZipInfo):
            with pytest.raises(HTTPException) as exc:
                skill_hub._safe_zip_extract(hostile, bad_zip_status=400)
        assert exc.value.status_code == 400
        # A materialise-then-check guard would build all 50,000.
        assert built <= _MAX_ARCHIVE_ENTRIES + 1, f"built {built} ZipInfo objects"

    def test_hostile_archive_is_refused_cheaply(self):
        """The whole point is the cost, so bound the cost.

        Unguarded, this archive takes seconds and tens of MiB in the
        constructor; the ingress size limit does not stop it, because 100,000
        empty entries compress to well under the budget.
        """
        import tracemalloc

        hostile = _entry_zip(100_000)
        assert len(hostile) < 16 * 1024 * 1024  # passes the ingress size gate

        tracemalloc.start()
        try:
            with pytest.raises(HTTPException) as exc:
                _safe_zip_extract(hostile, bad_zip_status=400)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert exc.value.status_code == 400
        # Unguarded this peaks around 56 MiB; the archive itself is ~10 MiB.
        assert peak < len(hostile) + 8 * 1024 * 1024, f"peak {peak} bytes"

    @pytest.mark.parametrize(
        "label,build",
        [
            ("plain", lambda n: _entry_zip(n)),
            ("one-byte comment", lambda n: _entry_zip(n, comment=b"x")),
            ("64 KiB comment", lambda n: _entry_zip(n, comment=b"y" * 65535)),
            (
                "comment holding header signatures",
                lambda n: _entry_zip(n, comment=b"PK\x01\x02" * 2001 + b"z" * 100),
            ),
            ("self-extracting prefix", lambda n: _entry_zip(n, prefix=b"MZ" * 3000)),
            ("400-character names", lambda n: _entry_zip(n, name_len=400)),
        ],
    )
    def test_bound_holds_across_archive_shapes(self, label, build):
        """Every shape must land on the same boundary.

        These are the shapes that broke earlier attempts: a comment moves the
        end record and makes CPython re-read the whole file; a comment may
        legally contain a central-directory signature; a self-extracting stub
        shifts every offset. A guard that mis-locates the directory fails
        *open* on one side of this pair and *closed* on the other, so both
        sides are asserted.
        """
        from xagent.web.api.skill_hub import _MAX_ARCHIVE_ENTRIES

        with pytest.raises(HTTPException) as at_cap:
            _safe_zip_extract(build(_MAX_ARCHIVE_ENTRIES), bad_zip_status=400)
        assert "no SKILL.md" in at_cap.value.detail, f"{label} rejected at the cap"

        with pytest.raises(HTTPException) as over:
            _safe_zip_extract(build(_MAX_ARCHIVE_ENTRIES + 1), bad_zip_status=400)
        assert "more than" in over.value.detail, f"{label} not bounded"

    def test_zip64_directory_is_bounded(self):
        """Zip64 must not be a way around the bound.

        Over 65,535 entries the end record is Zip64 and the real values live
        in a separate record CPython locates by its own rules. An attempt that
        hand-mirrored that location was 76 bytes off and counted nothing.
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", allowZip64=True) as zf:
            for i in range(70_000):
                zf.writestr(f"f{i:08d}", b"")
        zip64 = buf.getvalue()

        for label, data in (("plain", zip64), ("with stub", b"MZ" * 3000 + zip64)):
            with pytest.raises(HTTPException) as exc:
                _safe_zip_extract(data, bad_zip_status=400)
            assert exc.value.status_code == 400, label
            assert "more than" in exc.value.detail, label

    def test_member_content_cannot_forge_the_count(self):
        """A member full of header signatures is still one entry.

        The count follows the chain CPython seeks to, so bytes that merely
        look like headers — inside a member, a comment or a stub — are not on
        it. A signature-scanning attempt counted a valid one-entry archive as
        2,002 and refused it.
        """
        forged_header = b"PK\x01\x02" + b"\x00" * 42
        data = _make_zip(
            {"SKILL.md": SKILL_MD, "decoy.bin": forged_header * 5000},
        )
        files, root = _safe_zip_extract(data, bad_zip_status=400)
        assert root == ""
        assert files["SKILL.md"] == SKILL_MD
        assert files["decoy.bin"] == forged_header * 5000

    def test_a_read_landing_on_forged_bytes_does_not_re_anchor(self):
        """A member read that *begins* on a header signature must not re-anchor.

        The anchor latches once, on the directory ``ZipFile`` seeks to, and is
        dropped for good when the chain ends. Re-latching on any later read
        that happens to start on a signature is the failure that sank an
        earlier attempt: it double-counts, and refuses valid archives.

        The shape is reachable, not theoretical. ``BufferedReader`` reads in
        fixed-size blocks, so a member holding a run of forged headers only
        needs its content shifted until a block boundary lands on one — a
        padding member a few dozen bytes long does it. Under a re-anchoring
        guard this three-entry archive counts past 2,000 and is refused.
        """
        forged_header = b"PK\x01\x02" + b"\x00" * 42  # declares 0/0/0 lengths

        # Search the alignment rather than hard-coding it: the offset depends
        # on the block size and on how the writer lays the archive out, and a
        # stale constant would silently stop exercising the shape.
        for pad in range(200):
            data = _make_zip(
                {
                    "SKILL.md": SKILL_MD,
                    "p": b"P" * pad,
                    "decoy.bin": forged_header * 20_000,
                }
            )
            files, root = _safe_zip_extract(data, bad_zip_status=400)
            assert root == ""
            assert files["SKILL.md"] == SKILL_MD
            assert files["decoy.bin"] == forged_header * 20_000

    def test_seek_matches_bytesio_semantics(self):
        """The bounded source must behave as the ``BytesIO`` it replaces.

        ``zipfile`` seeks a fixed distance back from the end to find the end
        record, so an archive shorter than that record seeks past the start.
        ``BytesIO`` clamps that to 0; an unclamped negative position indexes
        ``bytes`` from the *end*, so reads return the wrong bytes — and this
        object is what tells ``zipfile`` where the central directory is.
        """
        from xagent.web.api.skill_hub import _BoundedZipSource

        payload = b"0123456789"
        source = _BoundedZipSource(payload)
        reference = io.BytesIO(payload)

        for offset, whence in ((-22, io.SEEK_END), (-3, io.SEEK_END), (5, io.SEEK_SET)):
            assert source.seek(offset, whence) == reference.seek(offset, whence)
            assert source.read(4) == reference.read(4)

        # Seeking before the start from the current position clamps too.
        source.seek(0, io.SEEK_SET)
        reference.seek(0, io.SEEK_SET)
        assert source.seek(-4, io.SEEK_CUR) == reference.seek(-4, io.SEEK_CUR)

        # An explicitly negative absolute seek is an error, as it is there.
        with pytest.raises(ValueError):
            source.seek(-1, io.SEEK_SET)

    def test_short_archive_is_rejected_not_misread(self):
        """An archive shorter than the end record is refused, not misparsed."""
        for payload in (b"PK\x05\x06", b"PK", b"", b"\x00" * 8):
            with pytest.raises(HTTPException) as exc:
                _safe_zip_extract(payload, bad_zip_status=400)
            assert exc.value.status_code == 400

    def test_ordinary_archives_are_untouched(self):
        """The bounded source must be transparent on archives we accept."""
        data = _make_zip({"my-skill/SKILL.md": SKILL_MD, "my-skill/ref.md": b"hi"})
        files, root = _safe_zip_extract(data)
        assert root == "my-skill"
        assert files == {"SKILL.md": SKILL_MD, "ref.md": b"hi"}

    def test_post_construction_cap_stands_on_its_own(self):
        """Defence in depth: the ``infolist()`` cap without the source guard.

        The two gates are independent. If a future CPython stops reading the
        directory in one pass from its own offset, the source guard could
        under-count; this one bounds the per-entry work regardless, since it
        reads the count the parser produced.
        """
        from xagent.web.api import skill_hub

        class _Passthrough(skill_hub._BoundedZipSource):
            def _count_headers(self, limit: int) -> None:
                return None

        with mock.patch.object(skill_hub, "_BoundedZipSource", _Passthrough):
            with pytest.raises(HTTPException) as exc:
                skill_hub._safe_zip_extract(
                    _entry_zip(skill_hub._MAX_ARCHIVE_ENTRIES + 1), bad_zip_status=400
                )
        assert exc.value.status_code == 400
        assert "more than" in exc.value.detail
