"""Security tests for skill_hub ZIP extraction and file normalisation guards."""

from __future__ import annotations

import io
import pathlib
import zipfile

import pytest
from fastapi import HTTPException

from xagent.web.api.skill_hub import (
    _NAME_RE,
    _check_registry_security_gate,
    _derive_upload_skill_name,
    _normalize_skill_files,
    _safe_zip_extract,
    _slugify_skill_name,
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
        assert "dot" in exc.value.detail.lower()

    def test_absolute_path_stripped(self):
        result = _normalize_skill_files({"/SKILL.md": SKILL_MD})
        assert "SKILL.md" in result

    def test_windows_separator_normalised(self):
        result = _normalize_skill_files({"SKILL.md": SKILL_MD, "sub\\file.md": b"hi"})
        assert "sub/file.md" in result

    def test_control_characters_in_path_rejected(self):
        # A NUL in a stored name is the classic truncation trick against any
        # later consumer that treats it as a C string or a real path.
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, "a\x00.py": b"x"})
        assert exc.value.status_code == 400
        assert "control" in exc.value.detail.lower()

        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, "a\nb.py": b"x"})
        assert exc.value.status_code == 400

    def test_literal_encoded_dots_are_not_traversal(self):
        # These are stored as literal names; they are only dangerous if a
        # consumer decodes or normalizes them, and none does. Pinning the
        # current behaviour so a future normalizer change is a visible break.
        out = _normalize_skill_files({"SKILL.md": SKILL_MD, "%2e%2e/x.py": b"x"})
        assert "%2e%2e/x.py" in out

    def test_overlong_path_rejected_before_the_db(self):
        # UserSkillFile.path is VARCHAR(500); PostgreSQL rejects a longer value
        # at INSERT time and that surfaced as a 500, while SQLite accepted it.
        from xagent.web.api.skill_hub import _MAX_SKILL_FILE_PATH_CHARS

        long_path = "d/" + "x" * _MAX_SKILL_FILE_PATH_CHARS + ".md"
        assert len(long_path) > _MAX_SKILL_FILE_PATH_CHARS
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, long_path: b"y"})
        assert exc.value.status_code == 400
        assert "longer than" in exc.value.detail

    def test_skill_md_limit_counts_characters_not_bytes(self):
        # CreateSkillRequest accepts max_length=200_000 *characters*, so a byte
        # check refused a CJK document the authoring routes accept — the same
        # content rejected only for arriving as an archive.
        from xagent.web.api.skill_hub import _MAX_SKILL_MD_CHARS

        at_limit = ("\u4e2d" * _MAX_SKILL_MD_CHARS).encode("utf-8")
        assert len(at_limit) > _MAX_SKILL_MD_CHARS  # 3 bytes per character
        assert _normalize_skill_files({"SKILL.md": at_limit})["SKILL.md"]

        over = ("\u4e2d" * (_MAX_SKILL_MD_CHARS + 1)).encode("utf-8")
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": over})
        assert exc.value.status_code == 413

    def test_empty_skill_md_rejected(self):
        # create/edit enforce min_length=1; an archive must not be able to
        # register a contentless skill that parses fine and teaches nothing.
        for body in (b"", b"   \n\n  "):
            with pytest.raises(HTTPException) as exc:
                _normalize_skill_files({"SKILL.md": body})
            assert exc.value.status_code == 400
            assert "empty" in exc.value.detail.lower()

    def test_skill_md_size_cap(self):
        # SKILL.md is injected into the LLM system context in full, so it is
        # capped at the same length the create/edit request models enforce,
        # well below the bundle-wide byte budget.
        from xagent.web.api.skill_hub import _MAX_SKILL_MD_CHARS

        oversized = b"#" * (_MAX_SKILL_MD_CHARS + 1)
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": oversized})
        assert exc.value.status_code == 413
        assert "SKILL.md" in exc.value.detail

        # Just under the cap is accepted.
        ok = b"#" * (_MAX_SKILL_MD_CHARS - 1)
        assert _normalize_skill_files({"SKILL.md": ok})["SKILL.md"] == ok

    def test_size_cap_raises(self):
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        big = b"x" * (_MAX_DOWNLOAD_BYTES + 1)
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, "big.bin": big})
        assert exc.value.status_code == 413


# ── _safe_zip_extract (registry path) ────────────────────────────────────────────────────────


class TestSafeZipToFiles:
    def test_happy_path_flat(self):
        data = _make_zip({"SKILL.md": SKILL_MD, "template.md": b"# Template"})
        result = _safe_zip_extract(data)[0]
        assert "SKILL.md" in result
        assert "template.md" in result

    def test_happy_path_nested(self):
        """ZIP with a top-level directory wrapper."""
        data = _make_zip({"my-skill/SKILL.md": SKILL_MD, "my-skill/extra.md": b"hi"})
        result = _safe_zip_extract(data)[0]
        assert "SKILL.md" in result
        assert "extra.md" in result

    def test_bad_zip_raises(self):
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(b"not a zip")[0]
        assert exc.value.status_code == 502

    def test_missing_skill_md_raises(self):
        # Registry-supplied archive: a bad artifact from upstream is a 502,
        # while the same archive uploaded by a user reports 400.
        data = _make_zip({"README.md": b"hello"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)[0]
        assert exc.value.status_code == 502
        assert "SKILL.md" in exc.value.detail
        with pytest.raises(HTTPException) as upload_exc:
            _safe_zip_extract(data, bad_zip_status=400)
        assert upload_exc.value.status_code == 400

    def test_path_traversal_in_zip_raises(self):
        data = _make_zip({"SKILL.md": SKILL_MD, "../escape.py": b"evil"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)[0]
        assert exc.value.status_code == 502
        with pytest.raises(HTTPException) as upload_exc:
            _safe_zip_extract(data, bad_zip_status=400)
        assert upload_exc.value.status_code == 400

    def test_dotfile_in_zip_rejected_by_normalize(self):
        # Registry default is 502 for a bad artifact; an upload reports 400.
        data = _make_zip({"SKILL.md": SKILL_MD, ".env": b"SECRET=1"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)[0]
        assert exc.value.status_code == 502
        with pytest.raises(HTTPException) as upload_exc:
            _safe_zip_extract(data, bad_zip_status=400)
        assert upload_exc.value.status_code == 400

    def test_oversized_member_raises(self):
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        big = b"x" * (_MAX_DOWNLOAD_BYTES + 1)
        data = _make_zip({"SKILL.md": SKILL_MD, "large.bin": big})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)[0]
        assert exc.value.status_code == 413


# ── _safe_zip_extract (upload path) ───────────────────────────────────────────


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
        # The extractor serves both registry installs and user uploads,
        # so its messages must not name a specific registry.
        for build in (
            lambda: _safe_zip_extract(b"not a zip"),
            lambda: _safe_zip_extract(_make_zip({"README.md": b"no skill"})),
        ):
            with pytest.raises(HTTPException) as exc:
                build()
            assert "clawhub" not in exc.value.detail.lower()

    def test_size_budget_uses_actual_bytes_not_declared(self):
        # A header that lies *large* (declares over-budget) must not get
        # a small honest payload rejected: the budget is enforced on the
        # bytes actually decompressed.
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        data = _make_zip({"SKILL.md": SKILL_MD})
        lying = _tamper_declared_size(data, _MAX_DOWNLOAD_BYTES + 1)
        files, _root = _safe_zip_extract(lying)
        assert "SKILL.md" in files

    def test_corrupted_member_maps_to_http_error(self):
        # zipfile only detects a CRC mismatch while reading a member, not
        # at open() — that failure must surface as an HTTP error, not a 500.
        data = bytearray(_make_zip({"SKILL.md": b"A" * 4096}))
        local = data.find(b"PK\x03\x04")
        header_len = 30 + data[local + 26] + data[local + 28]
        data[local + header_len + 10] ^= 0xFF  # flip a byte of member data
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(bytes(data), bad_zip_status=400)
        assert exc.value.status_code == 400
        assert "corrupted, encrypted, or uses an unsupported" in exc.value.detail

    def test_encrypted_member_maps_to_http_error(self):
        # zipfile raises RuntimeError when a member needs a password.
        import shutil
        import subprocess
        import tempfile

        if shutil.which("zip") is None:  # pragma: no cover - depends on the host
            pytest.skip("zip(1) unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "SKILL.md"
            src.write_bytes(SKILL_MD)
            out = pathlib.Path(tmp) / "enc.zip"
            proc = subprocess.run(
                ["zip", "-q", "-P", "hunter2", "-j", str(out), str(src)],
                capture_output=True,
            )
            if proc.returncode != 0:  # pragma: no cover - depends on the host
                pytest.skip(f"zip(1) failed: {proc.stderr[:200]!r}")
            data = out.read_bytes()

        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data, bad_zip_status=400)
        assert exc.value.status_code == 400
        assert "corrupted, encrypted, or uses an unsupported" in exc.value.detail

    def test_unsupported_compression_method_maps_to_http_error(self):
        # An unsupported method raises NotImplementedError, which subclasses
        # RuntimeError, so the (BadZipFile, RuntimeError) handler covers it.
        data = bytearray(_make_zip({"SKILL.md": SKILL_MD}))
        local = data.find(b"PK\x03\x04")
        data[local + 8 : local + 10] = (99).to_bytes(2, "little")
        central = data.find(b"PK\x01\x02")
        data[central + 10 : central + 12] = (99).to_bytes(2, "little")
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(bytes(data), bad_zip_status=400)
        assert exc.value.status_code == 400
        assert "corrupted, encrypted, or uses an unsupported" in exc.value.detail

    def test_multi_root_zip_is_rejected(self):
        # Two sibling skills: picking one would silently discard the other.
        data = _make_zip(
            {
                "a/SKILL.md": SKILL_MD,
                "a/x.md": b"ax",
                "b/SKILL.md": SKILL_MD,
                "b/y.md": b"by",
            }
        )
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 502
        assert "multiple skills" in exc.value.detail.lower()

    def test_independent_skill_outside_the_root_is_rejected(self):
        # Depth does not make a second skill safe to drop: selecting "z" and
        # discarding "a/b/c" is the same data loss as two same-depth roots.
        data = _make_zip(
            {
                "z/SKILL.md": SKILL_MD,
                "z/keep.md": b"k",
                "a/b/c/SKILL.md": SKILL_MD,
            }
        )
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 502
        assert "multiple skills" in exc.value.detail.lower()

    def test_example_skill_inside_the_root_is_kept(self):
        # A SKILL.md nested within the chosen root is a bundled example, not a
        # competing skill: it must survive as an ordinary member.
        data = _make_zip({"s/SKILL.md": SKILL_MD, "s/examples/demo/SKILL.md": SKILL_MD})
        files, root = _safe_zip_extract(data)
        assert root == "s"
        assert sorted(files) == ["SKILL.md", "examples/demo/SKILL.md"]

    def test_directory_entries_count_toward_the_entry_cap(self):
        # Directories carry no bytes and were skipped before the cap, so an
        # archive of nothing but directory entries slipped past it entirely.
        from xagent.web.api.skill_hub import _MAX_SKILL_FILES

        members = {f"d{i}/": b"" for i in range(_MAX_SKILL_FILES + 10)}
        members["SKILL.md"] = SKILL_MD
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(_make_zip(members))
        assert exc.value.status_code == 413
        assert "entries" in exc.value.detail

    def test_corrupted_deflate_stream_maps_to_http_error(self):
        # zlib.error is neither BadZipFile nor RuntimeError, so a damaged
        # DEFLATE stream used to escape as a 500. ZIP_STORED corruption trips
        # the CRC check instead, which is why the earlier test missed this.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", b"B" * 20000)
        data = bytearray(buf.getvalue())
        local = data.find(b"PK\x03\x04")
        header_len = 30 + data[local + 26] + data[local + 28]
        for offset in range(header_len, header_len + 8):
            data[local + offset] = 0xAB
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(bytes(data), bad_zip_status=400)
        assert exc.value.status_code == 400
        assert "corrupted, encrypted, or uses an unsupported" in exc.value.detail

    def test_registry_status_applies_to_normalization_errors(self):
        # A registry-supplied archive must report 502 for content problems too,
        # not only for the branches raised directly by the extractor.
        for members in (
            {"SKILL.md": SKILL_MD, ".env": b"S=1"},
            {"SKILL.md": b""},
        ):
            with pytest.raises(HTTPException) as exc:
                _safe_zip_extract(_make_zip(members))
            assert exc.value.status_code == 502, exc.value.detail

    def test_multi_root_message_is_bounded(self):
        # The message names archive-supplied directories, so a crafted archive
        # must not turn it into kilobytes of reflected content.
        members = {f"{'r' * 200}{i}/SKILL.md": SKILL_MD for i in range(50)}
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(_make_zip(members))
        assert exc.value.status_code == 502
        assert len(exc.value.detail) < 600
        assert "and 45 more" in exc.value.detail

    def test_shallowest_root_wins_over_alphabetical_order(self):
        # "a/b/c/SKILL.md" sorts after "z/SKILL.md", so a lexicographic pick
        # would choose the deep one. Only non-SKILL.md members sit outside the
        # root here, so the shallowest selection is observable without
        # tripping the multiple-skills guard.
        data = _make_zip(
            {"z/SKILL.md": SKILL_MD, "z/keep.md": b"k", "a/b/c/other.md": b"o"}
        )
        files, root = _safe_zip_extract(data)
        assert root == "z"
        assert sorted(files) == ["SKILL.md", "keep.md"]

    def test_member_count_cap_enforced(self):
        from xagent.web.api.skill_hub import _MAX_SKILL_FILES

        members = {f"f{i}.txt": b"" for i in range(_MAX_SKILL_FILES + 5)}
        members["SKILL.md"] = SKILL_MD
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(_make_zip(members))
        assert exc.value.status_code == 413

    def test_member_cap_stops_extraction_before_reading_everything(self, monkeypatch):
        """The cap must fire inside the extraction loop, not only in the
        post-hoc ``_normalize_skill_files`` check.

        Decompressing every member first and rejecting afterwards still does
        all the work an entry-count cap exists to avoid, so assert the loop
        stops early rather than just that a 413 comes out somewhere.
        """
        from xagent.web.api import skill_hub

        members = {f"f{i}.txt": b"" for i in range(skill_hub._MAX_SKILL_FILES + 50)}
        members["SKILL.md"] = SKILL_MD

        calls = {"n": 0}
        real_normalize = skill_hub._normalize_skill_files

        def counting_normalize(files):
            calls["n"] += 1
            return real_normalize(files)

        monkeypatch.setattr(skill_hub, "_normalize_skill_files", counting_normalize)

        with pytest.raises(HTTPException) as exc:
            skill_hub._safe_zip_extract(_make_zip(members))
        assert exc.value.status_code == 413
        # Rejected during extraction, so normalization was never reached.
        assert calls["n"] == 0

    def test_member_cap_counts_duplicate_entries(self):
        # A ZIP may repeat a filename; those collapse onto one dict key, so the
        # cap counts central-directory entries rather than resulting files.
        from xagent.web.api.skill_hub import _MAX_SKILL_FILES

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("SKILL.md", SKILL_MD)
            for _ in range(_MAX_SKILL_FILES + 5):
                zf.writestr("same.txt", b"x")
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(buf.getvalue())
        assert exc.value.status_code == 413
        assert "entries" in exc.value.detail

    def test_max_bytes_tightens_the_decompressed_budget(self):
        # Lowering the configured upload limit must also limit how far an
        # archive may expand, not just its size on the wire.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", SKILL_MD)
            zf.writestr("filler.bin", b"\0" * 8192)
        payload = buf.getvalue()

        # Permissive budget: extracts fine.
        files, _root = _safe_zip_extract(payload, max_bytes=1024 * 1024)
        assert "filler.bin" in files

        # Tight budget: refused, even though the wire size is tiny.
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(payload, max_bytes=2048)
        assert exc.value.status_code == 413

    def test_max_bytes_cannot_raise_the_absolute_ceiling(self):
        """A caller passing a huge ``max_bytes`` must not get a larger budget.

        ``_normalize_skill_files`` would also reject this archive, so asserting
        only "a 413 comes out" would pass with the clamp removed. Assert the
        clamped budget directly, and that the refusal happens during extraction
        rather than falling through to the post-hoc check.
        """
        from xagent.web.api import skill_hub

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", SKILL_MD)
            zf.writestr("bomb.bin", b"\0" * (skill_hub._MAX_DOWNLOAD_BYTES + 1024))

        calls = {"n": 0}
        real_normalize = skill_hub._normalize_skill_files

        def counting_normalize(files):
            calls["n"] += 1
            return real_normalize(files)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(skill_hub, "_normalize_skill_files", counting_normalize)
            with pytest.raises(HTTPException) as exc:
                skill_hub._safe_zip_extract(
                    buf.getvalue(), max_bytes=skill_hub._MAX_DOWNLOAD_BYTES * 10
                )
        assert exc.value.status_code == 413
        assert "ZIP exceeds size budget" in exc.value.detail
        assert calls["n"] == 0

    def test_compression_bomb_rejected_by_actual_byte_budget(self):
        # Highly compressible payload: small on the wire, huge decompressed.
        # This is the direction the bounded read is meant to stop.
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", SKILL_MD)
            zf.writestr("bomb.bin", b"\0" * (_MAX_DOWNLOAD_BYTES + 1024))
        payload = buf.getvalue()
        assert len(payload) < _MAX_DOWNLOAD_BYTES  # small on the wire
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(payload)
        assert exc.value.status_code == 413

    def test_dot_slash_sibling_is_accepted(self):
        data = _make_zip({"SKILL.md": SKILL_MD, "./extra.md": b"x"})
        files, _root = _safe_zip_extract(data)
        assert sorted(files) == ["SKILL.md", "extra.md"]

    def test_windows_drive_letter_path_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, "C:\\evil.py": b"x"})
        assert exc.value.status_code == 400
        assert "colon" in exc.value.detail.lower()


# ── UTF-8 pre-check stays in step with the parser ─────────────────────────────


def test_utf8_precheck_covers_every_file_the_parser_decodes():
    """The upload route pre-checks a fixed list of filenames.

    That list mirrors what ``SkillParser.parse_bundle`` decodes. If the parser
    grows a third decoded file the pre-check silently goes stale and that file
    reaches the DB before failing to parse, so fail here instead.
    """
    import inspect
    import re

    from xagent.skills.parser import SkillParser
    from xagent.web.api import skill_hub

    parser_src = inspect.getsource(SkillParser.parse_bundle)
    decoded = set(re.findall(r'files(?:\.get)?[\[(]"([^"]+)"', parser_src))

    route_src = inspect.getsource(skill_hub.upload_skill)
    checked = set(
        re.findall(r"for path in \(([^)]*)\)", route_src)[0].replace('"', "").split(",")
    )
    checked = {name.strip() for name in checked if name.strip()}

    assert decoded <= checked, (
        f"parse_bundle decodes {sorted(decoded - checked)}, which the upload "
        "route does not pre-check for UTF-8"
    )


# ── upload name derivation ────────────────────────────────────────────────────


SKILL_MD_NAMED = b"""---
name: pdf-tools
description: Handle PDFs.
---

# PDF Tools
"""


class TestDeriveUploadSkillName:
    def test_frontmatter_name_beats_zip_root(self):
        # The folder name is an artifact of how the archive happened to be
        # zipped; frontmatter is the author's explicit declaration.
        name = _derive_upload_skill_name(
            "archive.zip", "skill-v2-1", {"SKILL.md": SKILL_MD_NAMED}
        )
        assert name == "pdf-tools"

    def test_invalid_override_is_refused_not_rewritten(self):
        # Found by end-to-end testing, which no unit test covered: the server
        # silently slugified a bad override ("bad name!" -> "bad-name"), so an
        # API caller got a skill they did not ask for and the client-side check
        # was bypassable.
        for bad in ("bad name!", "../evil", "my/skill", "x" * 65):
            with pytest.raises(HTTPException) as exc:
                _derive_upload_skill_name(
                    "a.zip", "root", {"SKILL.md": SKILL_MD}, override=bad
                )
            assert exc.value.status_code == 400, bad
            assert "must match" in exc.value.detail

    def test_override_with_surrounding_whitespace_is_accepted(self):
        # Trimmed first, so a usable name is not refused for its whitespace.
        name = _derive_upload_skill_name(
            "a.zip", "root", {"SKILL.md": SKILL_MD}, override="  my-skill  "
        )
        assert name == "my-skill"

    def test_explicit_override_beats_everything(self):
        name = _derive_upload_skill_name(
            "archive.zip",
            "zip-root",
            {"SKILL.md": SKILL_MD_NAMED},
            override="user-choice",
        )
        assert name == "user-choice"

    def test_zip_root_used_when_no_frontmatter_name(self):
        name = _derive_upload_skill_name(
            "archive.zip", "my-skill", {"SKILL.md": SKILL_MD}
        )
        assert name == "my-skill"

    def test_bare_markdown_without_name_is_rejected_not_stemmed(self):
        # Dropping a file literally called SKILL.md must not yield "SKILL".
        with pytest.raises(HTTPException) as exc:
            _derive_upload_skill_name("SKILL.md", "", {"SKILL.md": SKILL_MD})
        assert exc.value.status_code == 400
        assert "name" in exc.value.detail.lower()

    def test_numeric_frontmatter_name_is_accepted(self):
        # YAML types a bare "name: 12345" as an int; it is still a valid skill
        # name once slugified, so it must not dead-end the upload.
        md = b"---\nname: 12345\ndescription: d\n---\n# S\n"
        assert _derive_upload_skill_name("a.zip", "", {"SKILL.md": md}) == "12345"

    def test_boolean_frontmatter_name_is_ignored(self):
        # bool subclasses int, and naming a skill "True" is nonsense: fall
        # through to the next candidate instead.
        md = b"---\nname: true\ndescription: d\n---\n# S\n"
        assert (
            _derive_upload_skill_name("a.zip", "real-root", {"SKILL.md": md})
            == "real-root"
        )

    def test_non_scalar_frontmatter_name_is_ignored(self):
        md = b"---\nname:\n  - a\n  - b\ndescription: d\n---\n# S\n"
        assert (
            _derive_upload_skill_name("a.zip", "real-root", {"SKILL.md": md})
            == "real-root"
        )

    def test_cjk_name_falls_back_instead_of_dead_ending(self):
        name = _derive_upload_skill_name("a.zip", "中文技能", {"SKILL.md": SKILL_MD})
        assert name.startswith("skill-")
        assert _NAME_RE.match(name)

    def test_generated_fallback_differs_for_different_bundles(self):
        # Hashing only SKILL.md gave two different skills the same name, and
        # the second upload then hit the duplicate-name 409 unrecoverably.
        base = {"SKILL.md": SKILL_MD}
        with_extra = {"SKILL.md": SKILL_MD, "reference.md": b"more"}
        first = _derive_upload_skill_name("a.zip", "\u4e2d\u6587", base)
        second = _derive_upload_skill_name("a.zip", "\u4e2d\u6587", with_extra)
        assert first != second
        assert _NAME_RE.match(first) and _NAME_RE.match(second)

    def test_cjk_fallback_is_deterministic(self):
        first = _derive_upload_skill_name("a.zip", "中文技能", {"SKILL.md": SKILL_MD})
        second = _derive_upload_skill_name("b.zip", "别的名字", {"SKILL.md": SKILL_MD})
        # Derived from content, so the same bundle yields a stable name.
        assert first == second

    def test_slugify(self):
        assert _slugify_skill_name("  Café menu skill!  ") == "Caf-menu-skill"
        assert _slugify_skill_name("ok_name-1") == "ok_name-1"
        assert _slugify_skill_name("///") == ""

    def test_slugify_truncation_leaves_no_trailing_separator(self):
        raw = "a" * 63 + "-" + "b" * 20
        assert _slugify_skill_name(raw) == "a" * 63


# ── POST /upload route ────────────────────────────────────────────────────────


def _make_upload(filename: str, data: bytes):
    from fastapi import UploadFile

    return UploadFile(file=io.BytesIO(data), filename=filename)


def _upload_args():
    from types import SimpleNamespace

    from xagent.skills.library import SkillScopeContext

    return (
        SimpleNamespace(),
        SkillScopeContext(user_id=7, metadata={}),
        object(),
        SimpleNamespace(id=7),
    )


async def _call_upload(upload, scope_value="personal", name=None):
    """Invoke the route with keyword args.

    Positional calls silently rebind when the signature grows a parameter,
    which is exactly what happened when ``name`` was added.
    """
    from xagent.web.api import skill_hub

    request, scope, db, user = _upload_args()
    return await skill_hub.upload_skill(
        request,
        file=upload,
        scope=scope_value,
        name=name,
        context=scope,
        db=db,
        _user=user,
    )


def _install_recording_write(monkeypatch) -> dict:
    """Record what the route writes, and answer reads only for that name.

    The signature mirrors ``_write_personal_skill`` exactly — no ``**kwargs``
    — so a renamed or added parameter fails here instead of being swallowed.
    """
    from xagent.web.api import skill_hub

    written: dict = {}

    def _fake_write(
        *,
        db,
        user,
        name,
        files,
        origin="custom",
        clawhub_slug=None,
        clawhub_version=None,
    ):
        written.update(
            name=name,
            files=files,
            origin=origin,
            clawhub_slug=clawhub_slug,
            clawhub_version=clawhub_version,
        )

    class _Manager:
        async def get_skill(self, name):
            # Answer only for what was actually written, so the route's
            # identity check is exercised rather than short-circuited.
            if written.get("name") != name:
                return None
            return {"name": name, "scope": "personal", "path": ""}

    async def _scoped(*args):
        return _Manager()

    monkeypatch.setattr(skill_hub, "_write_personal_skill", _fake_write)
    monkeypatch.setattr(skill_hub, "_get_scoped_manager", _scoped)
    return written


class TestUploadRoute:
    @pytest.mark.asyncio
    async def test_zip_with_traversal_rejected(self):
        data = _make_zip({"skill/SKILL.md": SKILL_MD, "skill/../../escape.py": b"evil"})
        with pytest.raises(HTTPException) as exc:
            await _call_upload(_make_upload("skill.zip", data))
        assert exc.value.status_code == 400
        assert "unsafe" in exc.value.detail.lower()

    @pytest.mark.asyncio
    async def test_unsupported_extension_rejected(self):
        with pytest.raises(HTTPException) as exc:
            await _call_upload(_make_upload("skill.tar.gz", b"x"))
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_oversized_upload_rejected(self, monkeypatch):
        from xagent.web.api import skill_hub

        monkeypatch.setattr(skill_hub, "get_max_upload_size_bytes", lambda: 64)
        with pytest.raises(HTTPException) as exc:
            await _call_upload(_make_upload("skill.zip", b"x" * 65))
        assert exc.value.status_code == 413

    @pytest.mark.asyncio
    async def test_zip_happy_path_persists_with_upload_origin(self, monkeypatch):
        written = _install_recording_write(monkeypatch)

        data = _make_zip({"pdf-tools/SKILL.md": SKILL_MD, "pdf-tools/ref.md": b"r"})
        summary = await _call_upload(_make_upload("archive.zip", data))
        assert summary.name == "pdf-tools"
        assert written["name"] == "pdf-tools"
        assert written["origin"] == "upload"
        assert set(written["files"]) == {"SKILL.md", "ref.md"}

    @pytest.mark.asyncio
    async def test_bare_markdown_uses_frontmatter_name(self, monkeypatch):
        written = _install_recording_write(monkeypatch)

        summary = await _call_upload(_make_upload("whatever.md", SKILL_MD_NAMED))
        assert summary.name == "pdf-tools"
        assert written["files"] == {"SKILL.md": SKILL_MD_NAMED}

    @pytest.mark.asyncio
    async def test_non_utf8_skill_md_in_zip_rejected_before_write(self, monkeypatch):
        # The DB write commits before the re-parse check, so a non-UTF-8
        # SKILL.md would leave an unloadable row behind and make the retry
        # collide with the duplicate-name 409. Nothing may be persisted.
        from xagent.web.api import skill_hub

        writes: list = []
        monkeypatch.setattr(
            skill_hub,
            "_write_personal_skill",
            lambda **kwargs: writes.append(kwargs),
        )

        data = _make_zip({"skill/SKILL.md": b"\xff\xfe\x00not utf8"})
        with pytest.raises(HTTPException) as exc:
            await _call_upload(_make_upload("skill.zip", data))
        assert exc.value.status_code == 400
        assert "UTF-8" in exc.value.detail
        assert writes == []

    @pytest.mark.asyncio
    async def test_non_utf8_markdown_rejected(self):
        with pytest.raises(HTTPException) as exc:
            await _call_upload(_make_upload("skill.md", b"\xff\xfe\x00bad"))
        assert exc.value.status_code == 400


# ── _persist_and_reparse failure reporting ───────────────────────────────────


class TestPersistAndReparseFailureMessage:
    """The failure message must describe what actually happened.

    The personal path deletes the row it just wrote, so the name is reusable.
    The team path cannot: a write provider owns its own storage. Claiming a
    rollback there would tell the user to retry a name that is still taken.
    """

    @staticmethod
    async def _run(scope: str, monkeypatch):
        from types import SimpleNamespace

        from xagent.skills.library import SkillScopeContext
        from xagent.web.api import skill_hub

        class _Manager:
            async def get_skill(self, name):
                return None  # force the failure branch

        async def _scoped(*args):
            return _Manager()

        monkeypatch.setattr(skill_hub, "_get_scoped_manager", _scoped)
        monkeypatch.setattr(skill_hub, "_write_personal_skill", lambda **kwargs: None)
        deleted: list = []
        monkeypatch.setattr(
            skill_hub,
            "_delete_personal_skill",
            lambda **kwargs: deleted.append(kwargs["name"]),
        )

        async def _invoker(provider, method, context, **kwargs):
            return None

        monkeypatch.setattr(skill_hub, "invoke_skill_write_provider", _invoker)
        monkeypatch.setattr(
            "xagent.skills.library.get_skill_write_provider", lambda: object()
        )

        with pytest.raises(HTTPException) as exc:
            await skill_hub._persist_and_reparse(
                request=SimpleNamespace(),
                context=SkillScopeContext(user_id=7, metadata={}),
                db=object(),
                user=SimpleNamespace(id=7),
                name="thing",
                files={"SKILL.md": SKILL_MD},
                scope=scope,
                origin="upload",
            )
        return exc.value, deleted

    @pytest.mark.asyncio
    async def test_personal_failure_says_rolled_back(self, monkeypatch):
        err, deleted = await self._run("personal", monkeypatch)
        assert err.status_code == 400
        assert "rolled back" in err.detail
        assert "left in place" not in err.detail
        assert deleted == ["thing"]

    @pytest.mark.asyncio
    async def test_rollback_failure_does_not_mask_the_real_error(self, monkeypatch):
        # If cleanup itself fails the user must still get the original 400,
        # told the copy was left in place — not a second, unrelated error.
        from types import SimpleNamespace

        from xagent.skills.library import SkillScopeContext
        from xagent.web.api import skill_hub

        class _Manager:
            async def get_skill(self, name):
                return None

        async def _scoped(*args):
            return _Manager()

        def _boom(**kwargs):
            raise RuntimeError("delete exploded")

        monkeypatch.setattr(skill_hub, "_get_scoped_manager", _scoped)
        monkeypatch.setattr(skill_hub, "_write_personal_skill", lambda **k: None)
        monkeypatch.setattr(skill_hub, "_delete_personal_skill", _boom)

        with pytest.raises(HTTPException) as exc:
            await skill_hub._persist_and_reparse(
                request=SimpleNamespace(),
                context=SkillScopeContext(user_id=7, metadata={}),
                db=object(),
                user=SimpleNamespace(id=7),
                name="thing",
                files={"SKILL.md": SKILL_MD},
                scope="personal",
                origin="upload",
            )
        assert exc.value.status_code == 400
        assert "left in place" in exc.value.detail
        assert "rolled back" not in exc.value.detail

    @pytest.mark.asyncio
    async def test_team_failure_does_not_claim_a_rollback(self, monkeypatch):
        # No provider-side delete exists, so the message must not promise one.
        err, deleted = await self._run("team", monkeypatch)
        assert err.status_code == 400
        assert "left in place" in err.detail
        assert "rolled back" not in err.detail
        assert deleted == []


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
