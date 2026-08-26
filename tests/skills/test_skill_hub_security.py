"""Security tests for skill_hub ZIP extraction and file normalisation guards."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi import HTTPException

from xagent.web.api.skill_hub import (
    _NAME_RE,
    _check_registry_security_gate,
    _derive_upload_skill_name,
    _normalize_skill_files,
    _safe_zip_extract,
    _safe_zip_to_files,
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


# ── upload name derivation ────────────────────────────────────────────────────


SKILL_MD_NAMED = b"""---
name: pdf-tools
description: Handle PDFs.
---

# PDF Tools
"""


class TestDeriveUploadSkillName:
    def test_zip_root_wins(self):
        name = _derive_upload_skill_name("archive.zip", "my-skill", SKILL_MD_NAMED)
        assert name == "my-skill"

    def test_frontmatter_name_beats_filename(self):
        name = _derive_upload_skill_name("archive.zip", "", SKILL_MD_NAMED)
        assert name == "pdf-tools"

    def test_filename_stem_is_last_resort(self):
        name = _derive_upload_skill_name("My Skill.zip", "", SKILL_MD)
        assert name == "My-Skill"

    def test_no_candidate_raises(self):
        with pytest.raises(HTTPException) as exc:
            _derive_upload_skill_name("---.zip", "", SKILL_MD)
        assert exc.value.status_code == 400

    def test_slugify(self):
        assert _slugify_skill_name("  Café menu skill!  ") == "Caf-menu-skill"
        assert _slugify_skill_name("ok_name-1") == "ok_name-1"
        assert _slugify_skill_name("///") == ""


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


class TestUploadRoute:
    @pytest.mark.asyncio
    async def test_zip_with_traversal_rejected(self):
        from xagent.web.api import skill_hub

        data = _make_zip({"skill/SKILL.md": SKILL_MD, "skill/../../escape.py": b"evil"})
        request, scope, db, user = _upload_args()
        with pytest.raises(HTTPException) as exc:
            await skill_hub.upload_skill(
                request,
                file=_make_upload("skill.zip", data),
                scope="personal",
                name=None,
                context=scope,
                db=db,
                _user=user,
            )
        assert exc.value.status_code == 400
        assert "unsafe" in exc.value.detail.lower()

    @pytest.mark.asyncio
    async def test_unsupported_extension_rejected(self):
        from xagent.web.api import skill_hub

        request, scope, db, user = _upload_args()
        with pytest.raises(HTTPException) as exc:
            await skill_hub.upload_skill(
                request,
                file=_make_upload("skill.tar.gz", b"x"),
                scope="personal",
                name=None,
                context=scope,
                db=db,
                _user=user,
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_oversized_upload_rejected(self, monkeypatch):
        from xagent.web.api import skill_hub

        monkeypatch.setattr(skill_hub, "_MAX_DOWNLOAD_BYTES", 64)
        request, scope, db, user = _upload_args()
        with pytest.raises(HTTPException) as exc:
            await skill_hub.upload_skill(
                request,
                file=_make_upload("skill.zip", b"x" * 65),
                scope="personal",
                name=None,
                context=scope,
                db=db,
                _user=user,
            )
        assert exc.value.status_code == 413

    @pytest.mark.asyncio
    async def test_zip_happy_path_persists_with_upload_origin(self, monkeypatch):
        from xagent.web.api import skill_hub

        written: dict = {}

        def _fake_write(*, db, user, name, files, origin="custom", **kwargs):
            written.update(name=name, files=files, origin=origin)

        class _Manager:
            async def get_skill(self, name):
                return {"name": name, "scope": "personal", "path": ""}

        async def _scoped(*args):
            return _Manager()

        monkeypatch.setattr(skill_hub, "_write_personal_skill", _fake_write)
        monkeypatch.setattr(skill_hub, "_get_scoped_manager", _scoped)

        data = _make_zip({"pdf-tools/SKILL.md": SKILL_MD, "pdf-tools/ref.md": b"r"})
        request, scope, db, user = _upload_args()
        summary = await skill_hub.upload_skill(
            request,
            file=_make_upload("archive.zip", data),
            scope="personal",
            name=None,
            context=scope,
            db=db,
            _user=user,
        )
        assert summary.name == "pdf-tools"
        assert written["name"] == "pdf-tools"
        assert written["origin"] == "upload"
        assert set(written["files"]) == {"SKILL.md", "ref.md"}

    @pytest.mark.asyncio
    async def test_bare_markdown_uses_frontmatter_name(self, monkeypatch):
        from xagent.web.api import skill_hub

        written: dict = {}

        def _fake_write(*, db, user, name, files, origin="custom", **kwargs):
            written.update(name=name, files=files, origin=origin)

        class _Manager:
            async def get_skill(self, name):
                return {"name": name, "scope": "personal", "path": ""}

        async def _scoped(*args):
            return _Manager()

        monkeypatch.setattr(skill_hub, "_write_personal_skill", _fake_write)
        monkeypatch.setattr(skill_hub, "_get_scoped_manager", _scoped)

        request, scope, db, user = _upload_args()
        summary = await skill_hub.upload_skill(
            request,
            file=_make_upload("whatever.md", SKILL_MD_NAMED),
            scope="personal",
            name=None,
            context=scope,
            db=db,
            _user=user,
        )
        assert summary.name == "pdf-tools"
        assert written["files"] == {"SKILL.md": SKILL_MD_NAMED}

    @pytest.mark.asyncio
    async def test_non_utf8_markdown_rejected(self):
        from xagent.web.api import skill_hub

        request, scope, db, user = _upload_args()
        with pytest.raises(HTTPException) as exc:
            await skill_hub.upload_skill(
                request,
                file=_make_upload("skill.md", b"\xff\xfe\x00bad"),
                scope="personal",
                name=None,
                context=scope,
                db=db,
                _user=user,
            )
        assert exc.value.status_code == 400


# ── no ghost rows ────────────────────────────────────────────────────────────


class TestNoGhostRows:
    """A bundle that cannot load must never leave a row behind.

    ``_write_personal_skill`` commits before the re-parse, and
    ``SkillManager.reload`` logs-and-skips a record it cannot decode. Without
    the pre-check and the rollback, the row survives somewhere no API verb can
    reach it: absent from the parsed cache GET enumerates, 404 from DELETE and
    PUT on the same lookup, and 409 on a retry. The name is squatted for good.
    """

    @staticmethod
    async def _upload(
        monkeypatch, members, *, breaks_parse=False, shadowed_by=None, override=None
    ):
        """Drive the route with a manager that can model any cache outcome.

        ``shadowed_by`` returns a record from another scope under the uploaded
        name — what really happens when a personal record fails to parse and a
        same-named builtin stays resident in the cache. The previous stub could
        only answer ``None`` or the skill's own dict, so it could not express
        that case at all, which is why it went unnoticed.
        """
        from types import SimpleNamespace

        from xagent.skills.library import SkillScopeContext
        from xagent.web.api import skill_hub

        written: list[str] = []
        monkeypatch.setattr(
            skill_hub,
            "_write_personal_skill",
            lambda **kw: written.append(kw["name"]),
        )
        monkeypatch.setattr(
            skill_hub,
            "_delete_personal_skill",
            lambda **kw: written.remove(kw["name"]),
        )

        class _Manager:
            async def get_skill(self, name):
                if shadowed_by is not None:
                    # A record from another scope survives under this name.
                    return {
                        "name": name,
                        "scope": None,
                        "source": shadowed_by,
                        "path": f"/{shadowed_by}/{name}",
                        "description": "someone else's skill",
                    }
                # Model reload()'s log-and-skip: an unloadable record is simply
                # absent, which is exactly what produced the ghost row.
                return None if breaks_parse else {"name": name, "scope": "personal"}

        async def _scoped(*args):
            return _Manager()

        monkeypatch.setattr(skill_hub, "_get_scoped_manager", _scoped)

        with pytest.raises(HTTPException) as exc:
            await skill_hub.upload_skill(
                SimpleNamespace(),
                file=_make_upload("bundle.zip", _make_zip(members)),
                scope="personal",
                name=override,
                context=SkillScopeContext(user_id=7, metadata={}),
                db=object(),
                _user=SimpleNamespace(id=7),
            )
        return exc.value, written

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_file", ["SKILL.md", "template.md"])
    async def test_non_utf8_is_refused_before_any_write(self, monkeypatch, bad_file):
        # A SKILL.md saved as Latin-1 by a Windows editor is enough to hit this.
        members = {"s/SKILL.md": SKILL_MD, f"s/{bad_file}": b"\xe9\xff\xfe latin-1"}
        err, written = await self._upload(monkeypatch, members)
        assert err.status_code == 400
        assert bad_file in err.detail
        assert written == [], "nothing may be written when the bundle cannot parse"

    @pytest.mark.asyncio
    async def test_unloadable_bundle_is_rolled_back(self, monkeypatch):
        # Anything the pre-check cannot predict must still free the name.
        err, written = await self._upload(
            monkeypatch, {"s/SKILL.md": SKILL_MD}, breaks_parse=True
        )
        assert err.status_code == 400
        assert "rolled back" in err.detail
        assert written == [], "the committed row must be removed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("override", [None, "my-name"])
    async def test_parser_failure_refused_on_both_naming_paths(
        self, monkeypatch, override
    ):
        """Validation must not depend on which naming branch runs.

        Deriving the name parses frontmatter itself, so a check placed after it
        never sees a bundle that breaks the parser — and an explicit name
        short-circuits that derivation entirely. Both branches must refuse.

        The trigger is injected rather than crafted: a payload deep enough to
        exhaust the stack is a property of the runner's recursion limit, not of
        this code, so a fixed depth passes locally and parses fine on a runner
        with more headroom. What matters is that *any* parser failure is
        refused before the write, whatever provokes it.
        """
        from xagent.skills.parser import SkillParser

        # Explode inside _extract_frontmatter: that is what deep YAML actually
        # blows up, and — crucially — what the *naming* step calls. Injecting
        # at parse_bundle instead would still pass with the check placed after
        # naming, so it would not pin the ordering this test exists for.
        real_extract = SkillParser._extract_frontmatter

        def exploding_extract(content):
            if "BOOM" in content:
                raise RecursionError("simulated deep-YAML stack exhaustion")
            return real_extract(content)

        monkeypatch.setattr(
            SkillParser, "_extract_frontmatter", staticmethod(exploding_extract)
        )

        err, written = await self._upload(
            monkeypatch, {"s/SKILL.md": SKILL_MD + b"\nBOOM\n"}, override=override
        )
        assert err.status_code == 400
        assert written == [], "an unparsable bundle must never be written"

    @pytest.mark.asyncio
    async def test_name_override_does_not_skip_the_parse_check(self, monkeypatch):
        """An explicit name must not opt out of the pre-write parse.

        Naming used to be the only thing that parsed frontmatter, so passing
        ``name=`` skipped it: deeply nested YAML (valid UTF-8, so the old check
        passed it) then reached the database.
        """
        # Non-UTF-8 rather than deep YAML: it fails the parser identically but
        # deterministically, without depending on the runner's stack depth.
        err, written = await self._upload(
            monkeypatch, {"s/SKILL.md": b"\xff\xfe not utf-8"}, override="my-name"
        )
        assert err.status_code == 400
        assert written == [], "an unparsable bundle must never be written"

    @pytest.mark.asyncio
    async def test_shadowing_record_is_not_reported_as_success(self, monkeypatch):
        """A same-named builtin must not satisfy the post-write check.

        ``reload`` keys the cache by name with filesystem records first, so a
        builtin survives when the personal record fails to parse. A bare
        ``is None`` test returned 200 carrying that unrelated skill's content
        while the real row stayed orphaned.
        """
        err, written = await self._upload(
            monkeypatch,
            {"s/SKILL.md": SKILL_MD},
            shadowed_by="builtin",
            override="agent-builder",
        )
        assert err.status_code == 400
        assert "rolled back" in err.detail
        assert written == [], "the orphaned row must be removed"


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
        files, root = _safe_zip_extract(data, bad_zip_status=400)
        assert root == ""
        # The real skill wins and nothing is discarded.
        assert "SKILL.md" in files
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
        files, root = _safe_zip_extract(data, bad_zip_status=400)
        assert root == "pdf-tools"
        assert sorted(files) == ["SKILL.md", "examples/SKILL.md", "ref.md"]

    def test_two_roots_at_the_same_depth_are_refused(self):
        data = _make_zip({"a/SKILL.md": SKILL_MD, "b/SKILL.md": SKILL_MD})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data, bad_zip_status=400)
        assert exc.value.status_code == 400
        assert "multiple skills" in exc.value.detail

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
        files, root = _safe_zip_extract(data, bad_zip_status=400)
        assert root == "pdf-tools"
        assert files["SKILL.md"] == SKILL_MD

    def test_cruft_alone_is_not_a_skill(self):
        data = _make_zip({"__MACOSX/SKILL.md": b"junk"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data, bad_zip_status=400)
        assert exc.value.status_code == 400
        assert "no SKILL.md" in exc.value.detail


# ── name override matches the documented rule ────────────────────────────────


class TestOverrideMatchesNameRule:
    """The override check must accept what _NAME_RE and POST /create accept.

    It round-tripped through the slugifier, which strips leading and trailing
    "-"/"_", so "_foo" and "my_skill_" were rejected citing the very pattern
    they match — and the frontend and /create both accept them.
    """

    @pytest.mark.parametrize(
        "name", ["_foo", "foo_", "my-skill-", "__init__", "my_skill_", "ok-name"]
    )
    def test_names_matching_the_pattern_are_accepted(self, name):
        assert _NAME_RE.match(name), "test premise: this matches the documented rule"
        assert (
            _derive_upload_skill_name("a.zip", "root", SKILL_MD, override=name) == name
        )

    @pytest.mark.parametrize("name", ["bad name!", "../evil", "my/skill", "x" * 65])
    def test_names_breaking_the_pattern_are_refused(self, name):
        with pytest.raises(HTTPException) as exc:
            _derive_upload_skill_name("a.zip", "root", SKILL_MD, override=name)
        assert exc.value.status_code == 400


# ── unreadable archives ──────────────────────────────────────────────────────


class TestUnreadableArchives:
    """Anything zipfile raises on an untrusted archive must be a 4xx.

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

    def test_registry_source_keeps_its_status(self):
        # The boundary guard must not flatten the caller-specific status.
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(b"not a zip at all")
        assert exc.value.status_code == 502

    def test_size_budget_still_reported_as_413(self):
        # Our own HTTPExceptions must pass through the guard unchanged.
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", SKILL_MD)
            zf.writestr("big.bin", b"\0" * (_MAX_DOWNLOAD_BYTES + 1024))
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(buf.getvalue(), bad_zip_status=400)
        assert exc.value.status_code == 413

    def test_traversal_still_reported_as_400(self):
        data = _make_zip({"SKILL.md": SKILL_MD, "../escape.py": b"x"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400


# ── macOS archive cruft ──────────────────────────────────────────────────────


class TestArchiveCruft:
    """Zipping a folder in Finder sweeps in .DS_Store and resource forks.

    The demo path is "zip an Anthropic skill folder and drop it in", so those
    must not fail the upload — while real hidden files stay refused.
    """

    @pytest.mark.parametrize(
        "cruft",
        [
            "pdf-tools/.DS_Store",
            "pdf-tools/refs/.DS_Store",
            "__MACOSX/._pdf-tools",
            "pdf-tools/._reference.md",
            "pdf-tools/Thumbs.db",
        ],
    )
    def test_cruft_is_dropped_not_rejected(self, cruft):
        data = _make_zip({"pdf-tools/SKILL.md": SKILL_MD, cruft: b"\x00\x01"})
        files, root = _safe_zip_extract(data, bad_zip_status=400)
        assert root == "pdf-tools"
        assert sorted(files) == ["SKILL.md"]

    @pytest.mark.parametrize("hidden", [".env", "sub/.env", "a/b/.secret"])
    def test_real_hidden_files_still_rejected(self, hidden):
        # The old check only looked at the first character of the whole path,
        # so ".env" was refused but "sub/.env" was not.
        data = _make_zip({"SKILL.md": SKILL_MD, hidden: b"SECRET=1"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data, bad_zip_status=400)
        assert exc.value.status_code == 400
        assert "hidden file" in exc.value.detail


# ── kept-from-review regressions ─────────────────────────────────────────────


class TestUploadNameOverride:
    """An explicit name is honoured or refused, never silently rewritten.

    End-to-end testing found the server slugifying an override, so
    ``name=bad name!`` returned 200 with a skill called ``bad-name``.
    """

    def test_valid_override_wins(self):
        assert (
            _derive_upload_skill_name("a.zip", "zip-root", SKILL_MD, override="mine")
            == "mine"
        )

    def test_override_is_trimmed(self):
        assert (
            _derive_upload_skill_name("a.zip", "root", SKILL_MD, override="  mine  ")
            == "mine"
        )

    @pytest.mark.parametrize("bad", ["bad name!", "../evil", "my/skill", "x" * 65])
    def test_invalid_override_refused(self, bad):
        with pytest.raises(HTTPException) as exc:
            _derive_upload_skill_name("a.zip", "root", SKILL_MD, override=bad)
        assert exc.value.status_code == 400

    def test_no_override_falls_back_to_zip_root(self):
        assert _derive_upload_skill_name("a.zip", "root", SKILL_MD) == "root"


def test_unreadable_member_is_not_a_500():
    """A corrupt DEFLATE stream raises zlib.error, which subclasses neither
    BadZipFile nor RuntimeError and used to escape as a 500."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("SKILL.md", b"B" * 20000)
    data = bytearray(buf.getvalue())
    local = data.find(b"PK\x03\x04")
    header_len = 30 + data[local + 26] + data[local + 28]
    for offset in range(header_len, header_len + 8):
        data[local + offset] = 0xAB
    with pytest.raises(HTTPException) as exc:
        _safe_zip_to_files(bytes(data))
    assert exc.value.status_code == 502


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
