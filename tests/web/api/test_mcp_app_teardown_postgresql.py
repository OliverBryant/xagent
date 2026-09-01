from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import (
    MCPOAuthClient,
    MCPOAuthFlowState,
    MCPOAuthGrant,
)
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth

pytestmark = pytest.mark.postgresql

POSTGRES_TABLES = [
    User.__table__,
    PublicMCPApp.__table__,
    MCPServer.__table__,
    UserMCPServer.__table__,
    UserOAuth.__table__,
    MCPOAuthClient.__table__,
    MCPOAuthGrant.__table__,
    MCPOAuthFlowState.__table__,
]


@pytest.fixture
def postgresql_engine():
    with disposable_database_factory("xagent_mcp_teardown") as make:
        yield make("owner_race")


def _seed(factory) -> tuple[int, int, int]:
    with factory() as db:
        user = User(username="workspace-account", password_hash="hash")
        expected = PublicMCPApp(
            app_id="legacy-mail",
            name="Legacy Mail",
            transport="oauth",
            provider_name="legacy-provider",
            oauth_scopes=[],
            launch_config={},
            is_visible_in_connector=True,
        )
        other = PublicMCPApp(
            app_id="other-mail",
            name="Other Mail",
            transport="oauth",
            provider_name="other-provider",
            oauth_scopes=[],
            launch_config={},
            is_visible_in_connector=True,
        )
        server = MCPServer(
            name="Legacy Mail",
            managed="external",
            transport="oauth",
            auth=None,
        )
        db.add_all([user, expected, other, server])
        db.flush()
        db.add_all(
            [
                UserMCPServer(
                    user_id=user.id,
                    mcpserver_id=server.id,
                    is_owner=True,
                    is_active=True,
                ),
                UserOAuth(
                    user_id=user.id,
                    provider="legacy-provider",
                    access_token=encrypt_value("legacy-token"),
                ),
                UserOAuth(
                    user_id=user.id,
                    provider="other-provider",
                    access_token=encrypt_value("other-token"),
                ),
            ]
        )
        db.commit()
        return int(user.id), int(expected.id), int(server.id)


@pytest.mark.parametrize("mutation", ["delete", "rename-reassign", "provider-drift"])
def test_owner_mutation_between_preflight_and_teardown_fails_closed(
    postgresql_engine, mutation: str
) -> None:
    Base.metadata.create_all(postgresql_engine, tables=POSTGRES_TABLES)
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    user_id, expected_pk, server_id = _seed(factory)
    barrier = threading.Barrier(2)

    def preflight_then_teardown() -> int:
        with factory() as teardown_db:
            current_user = teardown_db.get(User, user_id)
            assert current_user is not None
            # This is the SaaS preflight boundary: exact app and immutable row
            # identity are read before a second Session mutates the catalog.
            expected = teardown_db.get(PublicMCPApp, expected_pk)
            assert expected is not None and expected.app_id == "legacy-mail"
            barrier.wait(timeout=10)
            barrier.wait(timeout=10)
            try:
                asyncio.run(
                    mcp_api.teardown_mcp_app_server(
                        server_id,
                        app_id="legacy-mail",
                        expected_catalog_app_id=expected_pk,
                        expected_provider_name="legacy-provider",
                        current_user=current_user,
                        db=teardown_db,
                    )
                )
            except HTTPException as exc:
                return exc.status_code
            raise AssertionError("teardown unexpectedly accepted a changed owner")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(preflight_then_teardown)
        barrier.wait(timeout=10)
        with factory() as mutation_db:
            expected = mutation_db.get(PublicMCPApp, expected_pk)
            assert expected is not None
            if mutation == "delete":
                mutation_db.delete(expected)
            elif mutation == "provider-drift":
                expected.provider_name = "replacement-provider"
            else:
                expected.name = "Moved Legacy Mail"
                other = (
                    mutation_db.query(PublicMCPApp)
                    .filter(PublicMCPApp.app_id == "other-mail")
                    .one()
                )
                other.name = "Legacy Mail"
            mutation_db.commit()
        barrier.wait(timeout=10)
        assert future.result(timeout=10) == 403

    with factory() as verify_db:
        assert verify_db.get(MCPServer, server_id) is not None
        assert (
            verify_db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == server_id)
            .count()
            == 1
        )
        assert {row.provider for row in verify_db.query(UserOAuth).all()} == {
            "legacy-provider",
            "other-provider",
        }


def test_catalog_mutation_waits_while_teardown_holds_identity_locks(
    postgresql_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    Base.metadata.create_all(postgresql_engine, tables=POSTGRES_TABLES)
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    user_id, expected_pk, server_id = _seed(factory)
    identity_checked = threading.Event()
    release_teardown = threading.Event()
    mutation_committed = threading.Event()
    real_gate = mcp_api._locked_catalog_app_for_server

    def gated_identity(*args, **kwargs):
        answer = real_gate(*args, **kwargs)
        identity_checked.set()
        assert release_teardown.wait(timeout=10)
        return answer

    monkeypatch.setattr(mcp_api, "_locked_catalog_app_for_server", gated_identity)

    def teardown() -> None:
        with factory() as teardown_db:
            current_user = teardown_db.get(User, user_id)
            assert current_user is not None
            asyncio.run(
                mcp_api.teardown_mcp_app_server(
                    server_id,
                    app_id="legacy-mail",
                    expected_catalog_app_id=expected_pk,
                    expected_provider_name="legacy-provider",
                    current_user=current_user,
                    db=teardown_db,
                )
            )

    def rename() -> None:
        with factory() as mutation_db:
            app = mutation_db.get(PublicMCPApp, expected_pk)
            assert app is not None
            app.name = "Renamed After Teardown"
            mutation_db.commit()
        mutation_committed.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        teardown_future = executor.submit(teardown)
        assert identity_checked.wait(timeout=10)
        rename_future = executor.submit(rename)
        assert not mutation_committed.wait(timeout=0.25)
        release_teardown.set()
        teardown_future.result(timeout=10)
        rename_future.result(timeout=10)

    assert mutation_committed.is_set()
    with factory() as verify_db:
        assert verify_db.get(MCPServer, server_id) is None
        assert (
            verify_db.query(UserOAuth).filter_by(provider="legacy-provider").count()
            == 0
        )


def test_concurrent_association_insert_serializes_with_parent_teardown(
    postgresql_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    Base.metadata.create_all(postgresql_engine, tables=POSTGRES_TABLES)
    factory = sessionmaker(bind=postgresql_engine, autoflush=False, autocommit=False)
    user_id, expected_pk, server_id = _seed(factory)
    with factory() as setup_db:
        other_user = User(username="concurrent-account", password_hash="hash")
        setup_db.add(other_user)
        setup_db.commit()
        other_user_id = int(other_user.id)

    identity_checked = threading.Event()
    release_teardown = threading.Event()
    association_insert_sent = threading.Event()
    insert_finished = threading.Event()
    real_gate = mcp_api._locked_catalog_app_for_server

    def gated_identity(*args, **kwargs):
        answer = real_gate(*args, **kwargs)
        identity_checked.set()
        assert release_teardown.wait(timeout=10)
        return answer

    monkeypatch.setattr(mcp_api, "_locked_catalog_app_for_server", gated_identity)

    def observe_association_insert(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ) -> None:
        if statement.lstrip().startswith("INSERT INTO user_mcpservers"):
            association_insert_sent.set()

    event.listen(postgresql_engine, "before_cursor_execute", observe_association_insert)

    def teardown() -> None:
        with factory() as teardown_db:
            current_user = teardown_db.get(User, user_id)
            assert current_user is not None
            asyncio.run(
                mcp_api.teardown_mcp_app_server(
                    server_id,
                    app_id="legacy-mail",
                    expected_catalog_app_id=expected_pk,
                    expected_provider_name="legacy-provider",
                    current_user=current_user,
                    db=teardown_db,
                )
            )

    def insert_association() -> str:
        with factory() as insert_db:
            insert_db.add(
                UserMCPServer(
                    user_id=other_user_id,
                    mcpserver_id=server_id,
                    is_owner=False,
                    is_active=True,
                )
            )
            try:
                insert_db.commit()
            except IntegrityError:
                insert_db.rollback()
                return "foreign-key-rejected"
            finally:
                insert_finished.set()
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            teardown_future = executor.submit(teardown)
            assert identity_checked.wait(timeout=10)
            insert_future = executor.submit(insert_association)
            assert association_insert_sent.wait(timeout=10)
            assert not insert_finished.wait(timeout=0.25)
            release_teardown.set()
            teardown_future.result(timeout=10)
            assert insert_future.result(timeout=10) == "foreign-key-rejected"
        finally:
            release_teardown.set()
            event.remove(
                postgresql_engine,
                "before_cursor_execute",
                observe_association_insert,
            )

    with factory() as verify_db:
        assert verify_db.get(MCPServer, server_id) is None
        assert (
            verify_db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == server_id)
            .count()
            == 0
        )
