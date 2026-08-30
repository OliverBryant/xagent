"""How a stored ``MCPServer`` row is resolved back to its catalog app.

``get_app_for_mcp_server`` decides which app's credentials a disconnect
deletes and which app a connector listing reports, so the rule it encodes is
load-bearing: identify the row by something stable, never by a value that is
both mutable and non-unique.

``PublicMCPApp.app_id`` is unique; ``PublicMCPApp.name`` is neither unique nor
immutable (the admin API can rename an app). Resolving by name therefore had
two failure modes, both fixed here and pinned below:

* an id-named row -- the convention the catalog connect helpers write --
  resolved to nothing, so callers silently skipped whatever they do with the
  result (for the disconnect path, deleting the user's OAuth credentials);
* a rename could move the answer to a *different* app between one caller's
  check and its later use.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from xagent.web.mcp_apps import get_app_for_mcp_server
from xagent.web.models.public_mcp import PublicMCPApp


@pytest.fixture()
def catalog_db():
    engine = create_engine("sqlite:///:memory:")
    PublicMCPApp.__table__.create(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def _app(db, app_id, name, **kwargs):
    db.add(
        PublicMCPApp(
            app_id=app_id,
            name=name,
            transport=kwargs.get("transport", "oauth"),
            provider_name=kwargs.get("provider_name", "acme"),
            category="Email",
            oauth_scopes=[],
            is_visible_in_connector=kwargs.get("visible", True),
            launch_config={},
        )
    )
    db.commit()


class _Row:
    """The two attributes the resolver reads off a stored server row."""

    def __init__(self, name, auth=None):
        self.name = name
        self.auth = auth


class TestAnUnstampedRow:
    """Both provisioning conventions write ``MCPServer.name``: the catalog
    connect helpers store the app id, the builtin OAuth flow the display
    name. Neither may be unresolvable."""

    def test_a_display_named_row_resolves(self, catalog_db):
        _app(catalog_db, "acme-mail", "Acme Mail")
        resolved = get_app_for_mcp_server(catalog_db, _Row("Acme Mail"))
        assert resolved is not None and resolved["id"] == "acme-mail"

    def test_an_id_named_row_resolves(self, catalog_db):
        """The regression: this returned ``None``, so a disconnect of such a
        row reported success while leaving the OAuth credentials stored."""
        _app(catalog_db, "acme-mail", "Acme Mail")
        resolved = get_app_for_mcp_server(catalog_db, _Row("acme-mail"))
        assert resolved is not None and resolved["id"] == "acme-mail"

    def test_a_name_matching_nothing_resolves_to_none(self, catalog_db):
        _app(catalog_db, "acme-mail", "Acme Mail")
        assert get_app_for_mcp_server(catalog_db, _Row("no-such-thing")) is None

    def test_an_exact_id_match_outranks_another_apps_display_name(self, catalog_db):
        """``app_id`` is unique and ``name`` is not, so when one app's id
        equals another app's display name the id must win -- otherwise the
        answer would depend on which row the database happened to scan
        first."""
        _app(catalog_db, "acme-mail", "Acme Mail")
        _app(catalog_db, "other-app", "acme-mail", visible=False)
        resolved = get_app_for_mcp_server(catalog_db, _Row("acme-mail"))
        assert resolved is not None and resolved["id"] == "acme-mail"

    def test_the_id_wins_regardless_of_catalog_insertion_order(self, catalog_db):
        """Same claim, seeded the other way round: a scan-order-dependent
        resolver passes one order and fails the other."""
        _app(catalog_db, "other-app", "acme-mail", visible=False)
        _app(catalog_db, "acme-mail", "Acme Mail")
        resolved = get_app_for_mcp_server(catalog_db, _Row("acme-mail"))
        assert resolved is not None and resolved["id"] == "acme-mail"


class TestAStampedRow:
    def test_the_stamp_decides(self, catalog_db):
        _app(catalog_db, "acme-mail", "Acme Mail")
        _app(catalog_db, "other-app", "Other App")
        resolved = get_app_for_mcp_server(
            catalog_db, _Row("Other App", auth={"app_id": "acme-mail"})
        )
        assert resolved is not None and resolved["id"] == "acme-mail"

    @pytest.mark.parametrize(
        "stamp", [pytest.param("", id="empty"), pytest.param(7, id="non-string")]
    )
    def test_a_malformed_stamp_refuses_rather_than_falling_back(
        self, catalog_db, stamp
    ):
        """A present-but-invalid stamp must not fall back to the row's name:
        that fallback is how one connector's teardown selects another
        connector's credentials."""
        _app(catalog_db, "acme-mail", "Acme Mail")
        assert (
            get_app_for_mcp_server(
                catalog_db, _Row("Acme Mail", auth={"app_id": stamp})
            )
            is None
        )

    def test_an_unknown_stamp_resolves_to_none(self, catalog_db):
        _app(catalog_db, "acme-mail", "Acme Mail")
        row = _Row("Acme Mail", auth={"app_id": "no-such-app"})
        assert get_app_for_mcp_server(catalog_db, row) is None

    def test_a_stamped_row_survives_a_rename_of_another_app_onto_its_name(
        self, catalog_db
    ):
        """The mutable-name race, at the resolver level: renaming app B onto
        the stored server name must not move a stamped row's answer to B."""
        _app(catalog_db, "acme-mail", "Acme Mail")
        _app(catalog_db, "other-app", "Other App")
        row = _Row("acme-mail", auth={"app_id": "acme-mail"})

        renamed = (
            catalog_db.query(PublicMCPApp)
            .filter(PublicMCPApp.app_id == "other-app")
            .one()
        )
        renamed.name = "acme-mail"
        catalog_db.commit()

        resolved = get_app_for_mcp_server(catalog_db, row)
        assert resolved is not None and resolved["id"] == "acme-mail"

    def test_auth_without_an_app_id_key_takes_the_name_path(self, catalog_db):
        """Selection is by key presence, not truthiness: a provider-only blob
        carries no stamp and must still resolve by name."""
        _app(catalog_db, "acme-mail", "Acme Mail")
        resolved = get_app_for_mcp_server(
            catalog_db, _Row("Acme Mail", auth={"provider": "acme"})
        )
        assert resolved is not None and resolved["id"] == "acme-mail"
