"""HTTP-level tests for ``POST /api/skill-hub/upload``.

The unit tests in ``test_skill_hub_security.py`` call ``upload_skill`` as a
bare coroutine, which skips FastAPI routing, dependency injection and
multipart parsing, and they stub the SkillManager so the re-parse always
succeeds. That combination cannot observe two of the failure modes this
endpoint has to get right:

* a bundle that commits and then fails to parse must not leave a row behind,
* a name that collides with an existing builtin must not report success with
  the builtin's metadata.

These tests drive the real router against a real sqlite-backed session so
both paths are exercised end to end.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.skills.library import SkillRecord
from xagent.web.api.skill_hub import router as skill_hub_router
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.database import Base, get_db
from xagent.web.models.skill import UserSkill
from xagent.web.models.user import User

SKILL_MD = b"""---
description: A test skill.
---

# Test Skill
"""


def _make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture()
def client_env(monkeypatch: pytest.MonkeyPatch):
    """Router + sqlite session + a stub personal-DB-backed SkillManager.

    ``builtin_records`` lets a test seed filesystem-layer records (the
    builtin skills) that shadow the personal DB in the composite provider,
    which is what makes the false-success path reachable.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False}
    )
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    session = Session()
    user = User(username="tester", password_hash="x")
    session.add(user)
    session.commit()
    user_id = int(user.id)
    session.close()

    builtin_records: list[SkillRecord] = []

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    def override_user():
        db = Session()
        try:
            return db.query(User).filter(User.id == user_id).first()
        finally:
            db.close()

    app = FastAPI()
    app.include_router(skill_hub_router)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_user

    # Build the manager the same way the route does, minus app.state: the
    # filesystem layer is whatever the test seeds, and the personal-DB layer
    # reads the sqlite session above. Nothing about get_skill is stubbed.
    from xagent.skills.library import (
        CompositeSkillLibraryProvider,
        StaticRecordsProvider,
    )
    from xagent.skills.manager import SkillManager
    from xagent.web.api import skill_hub as skill_hub_module

    async def fake_scoped_manager(request: Any, context: Any, db: Any):
        from xagent.skills.personal_db import XagentPersonalDbSkillProvider

        provider = CompositeSkillLibraryProvider(
            [
                StaticRecordsProvider(list(builtin_records)),
                XagentPersonalDbSkillProvider(),
            ]
        )
        mgr = SkillManager(provider=provider, context=context)
        await mgr.reload()
        return mgr

    monkeypatch.setattr(skill_hub_module, "_get_scoped_manager", fake_scoped_manager)

    # XagentPersonalDbSkillProvider calls get_optional_session_local() to find
    # its own session factory, independently of the injected request session.
    # Patch the accessor rather than the module global it reads, so the
    # provider loads back what the route just committed.
    import xagent.web.models.database as database_module

    monkeypatch.setattr(database_module, "get_optional_session_local", lambda: Session)

    with TestClient(app) as client:
        yield {
            "app": app,
            "client": client,
            "Session": Session,
            "user_id": user_id,
            "builtin_records": builtin_records,
        }

    engine.dispose()
    os.unlink(path)


def _skill_rows(env) -> list[str]:
    db = env["Session"]()
    try:
        return [
            row.name
            for row in db.query(UserSkill).filter(UserSkill.user_id == env["user_id"])
        ]
    finally:
        db.close()


class TestUploadHttp:
    def test_zip_upload_round_trip(self, client_env):
        data = _make_zip({"pdf-tools/SKILL.md": SKILL_MD, "pdf-tools/ref.md": b"r"})
        res = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("archive.zip", data, "application/zip")},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["name"] == "pdf-tools"
        assert body["source"] == "user"
        assert _skill_rows(client_env) == ["pdf-tools"]

    def test_name_override_wins(self, client_env):
        data = _make_zip({"pdf-tools/SKILL.md": SKILL_MD})
        res = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("archive.zip", data, "application/zip")},
            data={"name": "my-choice"},
        )
        assert res.status_code == 200, res.text
        assert res.json()["name"] == "my-choice"
        assert _skill_rows(client_env) == ["my-choice"]

    def test_duplicate_name_conflicts(self, client_env):
        data = _make_zip({"dup/SKILL.md": SKILL_MD})
        first = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("a.zip", data, "application/zip")},
        )
        assert first.status_code == 200, first.text
        second = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("a.zip", data, "application/zip")},
        )
        assert second.status_code == 409
        assert _skill_rows(client_env) == ["dup"]

    def test_concurrent_duplicate_name_yields_409_not_500(
        self, client_env, monkeypatch
    ):
        """The duplicate check is not atomic with the INSERT.

        Simulate the lost race by making the pre-check miss. The unique
        constraint then fires on commit, and SQLAlchemy leaves the session
        unusable — every later query raises PendingRollbackError, and get_db
        only closes it. Without an explicit rollback the request 500s and any
        follow-up work on that session fails too.
        """
        from xagent.web.api import skill_hub
        from xagent.web.models.skill import UserSkill

        skill_hub_app = client_env["app"]
        data = _make_zip({"race/SKILL.md": SKILL_MD})
        first = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("a.zip", data, "application/zip")},
        )
        assert first.status_code == 200, first.text

        real_write = skill_hub._write_personal_skill

        def blind_write(**kwargs):
            # Blind only the *first* UserSkill lookup — the pre-check — so it
            # misses exactly as it would in a real race. The handler's own
            # confirmation query afterwards must still see the committed row,
            # which is what lets it distinguish a duplicate from any other
            # constraint violation.
            db = kwargs["db"]
            original_query = db.query
            blinded = {"done": False}

            def patched_query(model, *a, **kw):
                if model is UserSkill and not blinded["done"]:
                    blinded["done"] = True

                    class _Blind:
                        def filter(self, *_a, **_kw):
                            return self

                        def first(self):
                            return None

                    return _Blind()
                return original_query(model, *a, **kw)

            db.query = patched_query
            try:
                return real_write(**kwargs)
            finally:
                db.query = original_query

        monkeypatch.setattr(skill_hub, "_write_personal_skill", blind_write)

        # Probe the session *while the request is still open*: closing a
        # session clears the pending-rollback state, so checking it after the
        # response has been returned would pass either way.
        probe: dict = {}
        original_override = skill_hub_app.dependency_overrides[get_db]

        def tracking_get_db():
            gen = original_override()
            db = next(gen)
            try:
                yield db
            finally:
                # Still inside the request scope, before db.close() runs.
                try:
                    db.query(UserSkill).count()
                    probe["usable"] = True
                except Exception as exc:  # PendingRollbackError without the fix
                    probe["usable"] = False
                    probe["error"] = type(exc).__name__
                for _ in gen:
                    pass

        skill_hub_app.dependency_overrides[get_db] = tracking_get_db
        try:
            second = client_env["client"].post(
                "/api/skill-hub/upload",
                files={"file": ("a.zip", data, "application/zip")},
            )
        finally:
            skill_hub_app.dependency_overrides[get_db] = original_override

        assert second.status_code == 409, second.text
        assert probe.get("usable") is True, (
            f"the failing request left its session unusable: {probe.get('error')}"
        )

        # Exactly one row, and the API still works afterwards.
        assert _skill_rows(client_env) == ["race"]
        listed = client_env["client"].get("/api/skill-hub/installed")
        assert listed.status_code == 200, listed.text

    def test_missing_file_field_returns_422(self, client_env):
        res = client_env["client"].post("/api/skill-hub/upload", data={})
        assert res.status_code == 422
        assert isinstance(res.json()["detail"], list)

    def test_invalid_scope_rejected(self, client_env):
        data = _make_zip({"s/SKILL.md": SKILL_MD})
        res = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("a.zip", data, "application/zip")},
            data={"scope": "everyone"},
        )
        assert res.status_code == 400
        assert _skill_rows(client_env) == []

    def test_team_scope_upload_reports_no_writer(self, client_env):
        # No team write provider is registered in this repo, so the only
        # honest assertion is that the route reports that rather than
        # half-writing or 500ing. Pins the contract for whenever one exists.
        data = _make_zip({"t/SKILL.md": SKILL_MD})
        res = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("t.zip", data, "application/zip")},
            data={"scope": "team"},
        )
        assert res.status_code == 400, res.text
        assert "writer" in res.text.lower()
        # A refused team write must not leave a personal row behind.
        assert _skill_rows(client_env) == []

    def test_zero_byte_upload_rejected(self, client_env):
        res = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("empty.zip", b"", "application/zip")},
        )
        assert res.status_code == 400
        assert _skill_rows(client_env) == []

    def test_directory_only_zip_rejected(self, client_env):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(zipfile.ZipInfo("only-a-dir/"), b"")
        res = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("d.zip", buf.getvalue(), "application/zip")},
        )
        assert res.status_code == 400
        assert _skill_rows(client_env) == []

    def test_failure_message_states_the_name_was_freed(self, client_env, monkeypatch):
        # The personal path does roll back, so the message must say the name is
        # reusable — and must not claim a rollback on a path that has none.
        from xagent.skills.parser import SkillParser

        real_parse = SkillParser.parse_bundle

        def flaky_parse(name: str, files: dict, path: str = "", **kwargs):
            if name == "msg-check":
                raise ValueError("synthetic parse failure")
            return real_parse(name=name, files=files, path=path, **kwargs)

        monkeypatch.setattr(SkillParser, "parse_bundle", staticmethod(flaky_parse))

        data = _make_zip({"msg-check/SKILL.md": SKILL_MD})
        res = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("m.zip", data, "application/zip")},
        )
        assert res.status_code == 400
        detail = res.json()["detail"]
        assert "rolled back" in detail
        assert "left in place" not in detail
        assert _skill_rows(client_env) == []

    def test_configured_limit_bounds_decompressed_size(self, client_env, monkeypatch):
        # Lowering XAGENT_MAX_UPLOAD_SIZE must also stop an archive that is
        # small on the wire but expands past the limit — otherwise the route
        # only caps the upload and leaves expansion on the 50 MiB default.
        from xagent.web.api import skill_hub

        monkeypatch.setattr(skill_hub, "get_max_upload_size_bytes", lambda: 4096)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("big/SKILL.md", SKILL_MD)
            zf.writestr("big/filler.bin", b"\0" * 65536)
        payload = buf.getvalue()
        assert len(payload) < 4096  # passes the wire-size check

        res = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("big.zip", payload, "application/zip")},
        )
        assert res.status_code == 413, res.text
        assert _skill_rows(client_env) == []

    def test_non_utf8_template_is_refused_before_writing(self, client_env):
        # SKILL.md is fine, template.md is not. The parser decodes both, so
        # this is caught up front and never reaches the database.
        data = _make_zip(
            {"broken/SKILL.md": SKILL_MD, "broken/template.md": b"\xff\xfe\x00"}
        )
        res = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("broken.zip", data, "application/zip")},
        )
        assert res.status_code == 400
        assert "template.md" in res.text
        assert _skill_rows(client_env) == []

        # The name must still be usable afterwards.
        good = _make_zip({"broken/SKILL.md": SKILL_MD})
        retry = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("broken.zip", good, "application/zip")},
        )
        assert retry.status_code == 200, retry.text
        assert retry.json()["name"] == "broken"

    def test_parse_failure_after_write_frees_the_name_for_retry(
        self, client_env, monkeypatch
    ):
        """The rollback path itself, reached past the up-front guards.

        The UTF-8 pre-check stops the bundles we can predict, but any parser
        failure after the commit must still release the name — otherwise the
        row is unreachable (GET lists the parsed cache, DELETE 404s on the
        same lookup) and the retry hits the duplicate-name 409 forever.
        """
        from xagent.skills.parser import SkillParser

        real_parse = SkillParser.parse_bundle
        fail = {"on": True}

        def flaky_parse(name: str, files: dict, path: str = "", **kwargs):
            if name == "later-ok" and fail["on"]:
                raise ValueError("synthetic post-commit parse failure")
            return real_parse(name=name, files=files, path=path, **kwargs)

        monkeypatch.setattr(SkillParser, "parse_bundle", staticmethod(flaky_parse))

        data = _make_zip({"later-ok/SKILL.md": SKILL_MD})
        first = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("a.zip", data, "application/zip")},
        )
        assert first.status_code == 400, first.text
        assert _skill_rows(client_env) == []

        # With the parser healthy again the same name must be free: a 409 here
        # would mean the failed write was never rolled back.
        fail["on"] = False
        second = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("a.zip", data, "application/zip")},
        )
        assert second.status_code == 200, second.text
        assert second.json()["name"] == "later-ok"
        assert _skill_rows(client_env) == ["later-ok"]

    def test_unparseable_bundle_is_rolled_back(self, client_env, monkeypatch):
        # Force the re-parse to fail for the record we just wrote, with no
        # same-named builtin in play: the route must roll the row back rather
        # than leave it unreachable.
        from xagent.skills.parser import SkillParser

        real_parse = SkillParser.parse_bundle

        def flaky_parse(name: str, files: dict, path: str = "", **kwargs):
            if name == "doomed":
                raise ValueError("synthetic parse failure")
            return real_parse(name=name, files=files, path=path, **kwargs)

        monkeypatch.setattr(SkillParser, "parse_bundle", staticmethod(flaky_parse))

        data = _make_zip({"doomed/SKILL.md": SKILL_MD})
        res = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("doomed.zip", data, "application/zip")},
        )
        assert res.status_code == 400
        assert _skill_rows(client_env) == []

    def test_builtin_name_collision_is_not_reported_as_success(
        self, client_env, monkeypatch
    ):
        # A builtin named "agent-builder" is cached first; the personal record
        # of the same name then fails to parse. A bare `is None` check would
        # hand back the builtin and answer 200.
        client_env["builtin_records"].append(
            SkillRecord(
                name="agent-builder",
                files={"SKILL.md": b"---\ndescription: builtin one.\n---\n# Builtin\n"},
                source="builtin",
                scope=None,
                path="/builtin/agent-builder",
            )
        )

        from xagent.skills.parser import SkillParser

        real_parse = SkillParser.parse_bundle

        def flaky_parse(name: str, files: dict, path: str = "", **kwargs):
            if name == "agent-builder" and b"builtin one" not in files.get(
                "SKILL.md", b""
            ):
                raise ValueError("synthetic parse failure for the personal record")
            return real_parse(name=name, files=files, path=path, **kwargs)

        monkeypatch.setattr(SkillParser, "parse_bundle", staticmethod(flaky_parse))

        data = _make_zip({"agent-builder/SKILL.md": SKILL_MD})
        res = client_env["client"].post(
            "/api/skill-hub/upload",
            files={"file": ("ab.zip", data, "application/zip")},
        )
        assert res.status_code == 400, res.text
        assert "builtin one" not in res.text
        assert _skill_rows(client_env) == []
