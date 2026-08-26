"""Skill Hub API — manage user-installed skills (saas closed-source).

The Hub composes four capabilities on top of xagent's existing skill
machinery (``SkillManager`` + ``SkillParser``):

  1. **Local skill management** — list / detail / delete the skills
     currently visible to the SkillManager, tagging each with
     ``source`` (builtin / user / external) so the UI can gate
     destructive operations on user-installed skills only.

  2. **ClawHub registry browse & install** — a thin proxy in front of
     ``https://clawhub.ai/api/v1/*`` (the public, anonymous-readable
     OpenClaw skill registry). v0 install policy: skills flagged
     ``"malicious"`` or in moderation state ``"quarantined"``/``"revoked"``
     are refused server-side; never trust the client to honor a
     "are you sure?" prompt for malware.

  3. **In-UI authoring** — write a new SKILL.md from scratch
     (``POST /create``) or edit an installed one in place
     (``PUT /installed/{name}``). Edits and creates both invalidate
     the same cache the chat runtime reads from.

  4. **File import** — install a skill the user already has on disk
     (``POST /upload``), either a ``.zip`` bundle of a skill folder or a
     bare ``.md`` used as SKILL.md. The name resolves in order: an explicit
     override, the frontmatter ``name``, the archive's root directory, then
     a content-derived fallback for names that slugify to nothing. Bare
     Markdown with none of the first three is refused rather than named
     after its file.

GitHub-URL import was removed in this iteration: we previously
shipped a ``git clone --depth=1`` path, but ClawHub gives us trusted
binaries with provenance and scan results, so we don't need to
re-implement that surface area. If someone really wants an
unscanned-source install path back, ``git`` is still on the box.

All writes (installs, creates, edits, uploads) persist to the database via
``UserSkill`` / ``UserSkillFile`` models.  The ``XagentPersonalDbSkillProvider``
(``skills/personal_db.py``) surfaces them back to the SkillManager; because
scoped managers are built fresh per request (no per-user cache), changes are
visible immediately on the next API call without an explicit ``reload()``.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
import zipfile
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from xagent.skills.library import SkillScopeContext
from xagent.web.api.skill_hub_registry import (
    _MAX_DOWNLOAD_BYTES,
    SkillRegistry,
    all_registries,
    get_registry,
)
from xagent.web.auth_dependencies import get_current_user
from xagent.web.config import format_file_size, get_max_upload_size_bytes
from xagent.web.models.database import get_db
from xagent.web.models.user import User
from xagent.web.services.skill_runtime import (
    get_skill_runtime_scope,
    handoff_skill_runtime_session,
    invoke_skill_write_provider,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skill-hub", tags=["skill-hub"])


# ──────────────────────────────────────────────────────────────────────
# Schemas — local
# ──────────────────────────────────────────────────────────────────────


class SkillSummary(BaseModel):
    """List-view payload for ``GET /installed``."""

    name: str
    description: str = ""
    when_to_use: str = ""
    tags: List[str] = Field(default_factory=list)
    source: str  # "builtin" | "user" | "external"
    scope: Optional[str] = None
    effective: bool = True
    shadowed_by: Optional[str] = None


class SkillDetail(SkillSummary):
    """Detail-view payload for ``GET /installed/{name}``."""

    content: str = ""
    execution_flow: str = ""
    files: List[str] = Field(default_factory=list)
    path: str


_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class CreateSkillRequest(BaseModel):
    """``POST /create`` body. Name is the on-disk directory name; the
    frontmatter ``name`` inside ``skill_md`` is ignored by the parser
    (xagent always uses the dir name as the source of truth)."""

    name: str = Field(..., min_length=1, max_length=64)
    skill_md: str = Field(..., min_length=1, max_length=200_000)
    scope: str = Field("personal", pattern="^(personal|team)$")


class EditSkillRequest(BaseModel):
    """``PUT /installed/{name}`` body. ``name`` is taken from the URL;
    only the SKILL.md content is mutable in v0."""

    skill_md: str = Field(..., min_length=1, max_length=200_000)


# ──────────────────────────────────────────────────────────────────────
# Schemas — registry (ClawHub proxy)
# ──────────────────────────────────────────────────────────────────────


class RegistrySkillSummary(BaseModel):
    """Card-view payload for a ClawHub skill. We forward only the
    fields the UI actually renders so the frontend contract is stable
    even if upstream evolves."""

    slug: str
    displayName: str = ""
    summary: str = ""
    version: Optional[str] = None
    ownerHandle: Optional[str] = None
    installs: Optional[int] = None
    # ClawHub sends this as a unix-ms integer (e.g. 1778485729679),
    # not a string — the frontend formats it. Typed as int.
    updatedAt: Optional[int] = None
    # Trust badge: "clean" / "suspicious" / "malicious" / None
    scanStatus: Optional[str] = None
    # If installed locally already, the local skill name (so UI can
    # show "Installed" instead of an Install button).
    installedAs: Optional[str] = None


class RegistrySkillDetail(BaseModel):
    """Detail payload returned by ``GET /registry/{slug}``."""

    slug: str
    displayName: str = ""
    summary: str = ""
    version: Optional[str] = None
    ownerHandle: Optional[str] = None
    homepage: Optional[str] = None
    readme: Optional[str] = None  # the SKILL.md body if upstream exposes one
    scanStatus: Optional[str] = None
    moderation: Optional[Dict[str, Any]] = None
    installedAs: Optional[str] = None
    registrySource: str = "clawhub"
    # Raw upstream blob for any UI bits we don't have a typed slot for
    # yet (provenance, capability tags, etc.). UI can poke at this for
    # secondary detail panels.
    raw: Dict[str, Any] = Field(default_factory=dict)


class RegistryListResponse(BaseModel):
    items: List[RegistrySkillSummary]
    nextCursor: Optional[str] = None


class InstallSkillRequest(BaseModel):
    slug: str = Field(..., min_length=1, max_length=128)
    version: Optional[str] = None
    scope: str = Field("personal", pattern="^(personal|team)$")


# ──────────────────────────────────────────────────────────────────────
# Helpers — local skill paths
# ──────────────────────────────────────────────────────────────────────


def _user_skills_root() -> Path:
    """The single writable skills directory we install into. Mirrors
    the third root ``skills/utils._get_default_skill_dirs`` configures
    so anything we write here is picked up by the same SkillManager
    every other code path uses."""
    from xagent.core.storage.manager import get_storage_root

    root = get_storage_root() / "skills"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _builtin_skills_root() -> Path:
    from xagent.skills.manager import SkillManager

    return SkillManager.get_builtin_root().resolve()


def _classify_source(skill_path: str) -> str:
    """Tag a skill as ``builtin`` / ``user`` / ``external`` based on
    where on disk it lives."""
    if not skill_path:
        return "external"
    p = Path(skill_path).resolve()
    user = _user_skills_root().resolve()
    builtin = _builtin_skills_root()
    if str(p).startswith(str(builtin) + "/") or p == builtin:
        return "builtin"
    if str(p).startswith(str(user) + "/"):
        return "user"
    return "external"


def _validate_skill_name(name: str) -> None:
    if not name or not _NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Skill name must match [A-Za-z0-9_-]+ (no spaces, slashes, or dots)."
            ),
        )


async def _get_manager(request: Request) -> Any:
    """Return the process-wide SkillManager singleton from app.state.
    This manager holds only filesystem (builtin / external / project) records.
    Per-request scoped views add the personal-DB layer on top; see
    ``_get_scoped_manager``.  Typed as ``Any`` to keep the skills package
    out of this module's import graph."""
    mgr = getattr(request.app.state, "skill_manager", None)
    if mgr is None:
        from xagent.skills.utils import create_skill_manager

        mgr = create_skill_manager()
        request.app.state.skill_manager = mgr
    await mgr.ensure_initialized()
    return mgr


def _write_context(
    scope: SkillScopeContext,
) -> Any:
    from xagent.skills.library import SkillWriteContext

    return SkillWriteContext(
        user_id=scope.user_id,
        metadata=dict(scope.metadata),
    )


async def _get_scoped_manager(
    request: Request,
    context: SkillScopeContext,
    db: Any,
) -> Any:
    """Build a per-request SkillManager (no persistent per-user cache).

    Caching strategy — decouple by volatility:

    *Default path* (no custom provider registered):
      - Filesystem records (builtin / external / project skills) are stable, so
        we reuse the records already loaded by the process-wide
        ``app.state.skill_manager`` via ``StaticRecordsProvider``.
      - Personal-DB records are volatile, so ``XagentPersonalDbSkillProvider``
        opens its own short session and is queried fresh on every request.

    *Custom-provider path* (SaaS / overlay installed via
    ``set_skill_library_provider``): the provider is used as-is with the
    detached scope identity so that team-scoped records can be included.
    Each request still gets its own ``SkillManager`` instance, so there is no
    shared mutable state between concurrent requests.

    In both paths:
    * No stale-delete bug — the DB layer is always re-queried.
    * No unbounded memory — no persistent per-user dict.
    * No concurrency hazard — each request owns its manager instance.
    """
    from xagent.skills.library import (
        CompositeSkillLibraryProvider,
        StaticRecordsProvider,
        get_skill_library_provider,
    )
    from xagent.skills.manager import SkillManager
    from xagent.skills.personal_db import XagentPersonalDbSkillProvider

    handoff_skill_runtime_session(db)

    custom_provider = get_skill_library_provider()
    if custom_provider is not None:
        # Custom (e.g. SaaS) provider — use as-is; it handles all layers.
        mgr = SkillManager(provider=custom_provider, context=context)
    else:
        # Default path: cached FS records + fresh personal-DB per request.
        global_mgr = await _get_manager(request)
        fs_records = [
            info["_record"]
            for info in global_mgr._skills_cache.values()
            if "_record" in info
        ]
        provider = CompositeSkillLibraryProvider(
            [StaticRecordsProvider(fs_records), XagentPersonalDbSkillProvider()]
        )
        mgr = SkillManager(provider=provider, context=context)

    await mgr.reload()
    return mgr


def _skill_to_summary(skill_dict: dict) -> SkillSummary:
    return SkillSummary(
        name=skill_dict["name"],
        description=skill_dict.get("description", ""),
        when_to_use=skill_dict.get("when_to_use", ""),
        tags=skill_dict.get("tags", []),
        source=_summary_source(skill_dict),
        scope=skill_dict.get("scope"),
        effective=bool(skill_dict.get("effective", True)),
        shadowed_by=skill_dict.get("shadowed_by"),
    )


def _skill_to_detail(skill_dict: dict) -> SkillDetail:
    return SkillDetail(
        name=skill_dict["name"],
        description=skill_dict.get("description", ""),
        when_to_use=skill_dict.get("when_to_use", ""),
        tags=skill_dict.get("tags", []),
        source=_summary_source(skill_dict),
        scope=skill_dict.get("scope"),
        effective=bool(skill_dict.get("effective", True)),
        shadowed_by=skill_dict.get("shadowed_by"),
        content=skill_dict.get("content", ""),
        execution_flow=skill_dict.get("execution_flow", ""),
        files=skill_dict.get("files", []),
        path=skill_dict.get("path", ""),
    )


def _summary_source(skill_dict: dict) -> str:
    scope = skill_dict.get("scope")
    if scope == "personal":
        return "user"
    if isinstance(scope, str) and scope:
        return scope
    return skill_dict.get("source") or _classify_source(skill_dict.get("path", ""))


# One skill is a handful of files: a SKILL.md, maybe a template, some
# reference docs. A cap well above any legitimate bundle still stops an
# archive whose entries are individually tiny — the byte budget alone let a
# ~11 MB upload expand into 100k rows, one INSERT each.
#
# Enforced in two places on purpose, and they are not redundant:
#   * ``_safe_zip_extract`` counts central-directory *entries*, including
#     directories, to bound the work before anything is decompressed.
#   * ``_normalize_skill_files`` counts the resulting *files*, so a caller that
#     builds a bundle without going through the extractor is bounded too.
_MAX_SKILL_FILES = 512

# Matches the ``max_length`` the create/edit request models enforce. SKILL.md
# lands in the LLM system context in full, so the archive paths need the same
# ceiling rather than only the bundle-wide byte budget.
_MAX_SKILL_MD_CHARS = 200_000

# Matches UserSkillFile.path's VARCHAR(500).
_MAX_SKILL_FILE_PATH_CHARS = 500


def _normalize_skill_files(
    files: dict[str, bytes], *, bad_status: int = 400
) -> dict[str, bytes]:
    """Validate and key a skill bundle by relative path.

    ``bad_status`` lets a registry-supplied archive report 502 for content
    problems, matching ``_safe_zip_extract``; a user upload keeps 400.
    """
    if len(files) > _MAX_SKILL_FILES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Skill bundle has {len(files)} files, over the "
                f"{_MAX_SKILL_FILES}-file limit."
            ),
        )
    out: dict[str, bytes] = {}
    total = 0
    for raw_path, content in files.items():
        path = str(raw_path).replace("\\", "/").lstrip("/")
        # Strip a leading "./" so a sibling written as "./extra.md" is not
        # mistaken for a dotfile below.
        while path.startswith("./"):
            path = path[2:]
        if not path or ".." in path.split("/"):
            raise HTTPException(
                status_code=bad_status,
                detail="Skill file path contains a path-traversal sequence.",
            )
        if path.startswith("."):
            raise HTTPException(
                status_code=bad_status,
                detail="Skill file path must not start with a dot.",
            )
        # Reject Windows drive letters and any other colon-bearing path.
        # Nothing writes these to disk today, but a stored "C:/evil.py" is a
        # trap for whatever consumes the bundle next.
        if ":" in path:
            raise HTTPException(
                status_code=bad_status,
                detail="Skill file path must not contain a drive letter or colon.",
            )
        # A NUL or control character in a name is never legitimate and is the
        # classic truncation trick against anything that later treats the
        # stored path as a C string or a filesystem path.
        if any(ch == "\x00" or ord(ch) < 32 for ch in path):
            raise HTTPException(
                status_code=bad_status,
                detail="Skill file path must not contain control characters.",
            )
        # UserSkillFile.path is VARCHAR(500); without this a longer relative
        # path reaches the INSERT and PostgreSQL raises, surfacing as a 500
        # (SQLite silently accepts it, which is why tests missed this).
        if len(path) > _MAX_SKILL_FILE_PATH_CHARS:
            raise HTTPException(
                status_code=bad_status,
                detail=(
                    "Skill file path is longer than "
                    f"{_MAX_SKILL_FILE_PATH_CHARS} characters."
                ),
            )
        total += len(content)
        # Absolute ceiling for every caller, including ones that build a bundle
        # without going through _safe_zip_extract. Archive uploads are already
        # held to the (possibly tighter) configured limit during extraction.
        if total > _MAX_DOWNLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail="Skill files exceed size budget."
            )
        out[path] = bytes(content)
    if "SKILL.md" not in out:
        raise HTTPException(status_code=bad_status, detail="Skill has no SKILL.md.")
    if not out["SKILL.md"].strip():
        # The authoring routes enforce min_length=1; without the same floor an
        # archive could register a skill with no content, which parses fine and
        # then sits in the agent's index as an entry that teaches it nothing.
        raise HTTPException(status_code=bad_status, detail="SKILL.md is empty.")
    # The whole SKILL.md is injected into the agent's system context on every
    # LLM call, so cap it at the same size the authoring routes enforce via
    # their request models; the archive paths otherwise had no limit below the
    # multi-megabyte bundle budget.
    #
    # Count characters, not bytes: CreateSkillRequest/EditSkillRequest accept
    # max_length=200_000 *characters*, so a byte check would reject a
    # 200k-character CJK document that the authoring routes happily accept —
    # the same content, refused only because it arrived as an archive.
    # Non-UTF-8 content is rejected upstream by the route, and this decode is
    # tolerant so a bundle that skipped that path cannot 500 here.
    skill_md_chars = len(out["SKILL.md"].decode("utf-8", errors="replace"))
    if skill_md_chars > _MAX_SKILL_MD_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"SKILL.md is {skill_md_chars} characters, over the "
                f"{_MAX_SKILL_MD_CHARS}-character limit. Move reference "
                "material into separate files in the bundle."
            ),
        )
    return out


def _write_personal_skill(
    *,
    db: Any,
    user: User,
    name: str,
    files: dict[str, bytes],
    origin: str = "custom",
    clawhub_slug: str | None = None,
    clawhub_version: str | None = None,
) -> None:
    from xagent.skills.library import guess_media_type
    from xagent.web.models.skill import UserSkill, UserSkillFile

    _validate_skill_name(name)
    user_id = int(user.id)
    normalized = _normalize_skill_files(files)
    existing = (
        db.query(UserSkill)
        .filter(UserSkill.user_id == user_id, UserSkill.name == name)
        .first()
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"A personal skill named {name!r} already exists.",
        )
    skill = UserSkill(
        user_id=user_id,
        name=name,
        origin=origin,
        clawhub_slug=clawhub_slug,
        clawhub_version=clawhub_version,
        created_by_user_id=user_id,
        updated_by_user_id=user_id,
    )
    db.add(skill)
    try:
        # flush sends the INSERT, so the unique constraint fires here rather
        # than at commit.
        db.flush()
        for path, content in sorted(normalized.items()):
            db.add(
                UserSkillFile(
                    skill_id=skill.id,
                    path=path,
                    content=content,
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                    media_type=guess_media_type(path),
                )
            )
        db.commit()
    except IntegrityError as exc:
        # The duplicate-name SELECT above is not atomic with this INSERT, so a
        # concurrent request for the same name lands here. Roll back either
        # way: SQLAlchemy leaves the session in a state where every later query
        # raises PendingRollbackError, and ``get_db`` only closes the session,
        # so without this the rest of the request fails too.
        db.rollback()
        # Other constraints on these tables (the per-skill file-path unique
        # index, the user foreign keys) would also arrive here, and reporting
        # those as "already exists" would send the user chasing the wrong
        # problem. Only claim a duplicate when the row really is one.
        if (
            db.query(UserSkill)
            .filter(UserSkill.user_id == user_id, UserSkill.name == name)
            .first()
            is not None
        ):
            raise HTTPException(
                status_code=409,
                detail=f"A personal skill named {name!r} already exists.",
            ) from exc
        logger.error(
            "Skill Hub: writing personal skill %r violated a constraint other "
            "than the name uniqueness",
            name,
            exc_info=True,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Skill {name!r} could not be stored: {exc.orig.__class__.__name__}.",
        ) from exc


def _update_personal_skill_md(*, db: Any, user: User, name: str, skill_md: str) -> None:
    from xagent.skills.library import guess_media_type
    from xagent.web.models.skill import UserSkill, UserSkillFile

    skill = (
        db.query(UserSkill)
        .filter(UserSkill.user_id == int(user.id), UserSkill.name == name)
        .first()
    )
    if skill is None:
        raise HTTPException(status_code=404, detail="Personal skill not found")
    content = skill_md.encode("utf-8")
    file = next((item for item in skill.files if item.path == "SKILL.md"), None)
    if file is None:
        file = UserSkillFile(skill_id=skill.id, path="SKILL.md")
        db.add(file)
    file.content = content
    file.size_bytes = len(content)
    file.sha256 = hashlib.sha256(content).hexdigest()
    file.media_type = guess_media_type("SKILL.md")
    skill.updated_by_user_id = int(user.id)
    db.commit()


def _delete_personal_skill(*, db: Any, user: User, name: str) -> None:
    from xagent.web.models.skill import UserSkill

    skill = (
        db.query(UserSkill)
        .filter(UserSkill.user_id == int(user.id), UserSkill.name == name)
        .first()
    )
    if skill is None:
        raise HTTPException(status_code=404, detail="Personal skill not found")
    db.delete(skill)
    db.commit()


def _summary_from_registry_item(
    item: dict, installed_names: set[str], registry: SkillRegistry
) -> RegistrySkillSummary:
    """Normalize one item from ``/api/v1/skills`` or ``/api/v1/search``
    into our typed summary.

    Upstream shape (sampled 2026-05 from clawhub.ai/api/v1/skills):
      {
        slug, displayName, summary,
        tags: {latest: "1.0.0", ...},          ← channel dict, NOT a list!
        stats: {installsCurrent, downloads, stars, ...},
        latestVersion: {version, createdAt, ...},
        metadata: {...},
        createdAt, updatedAt                    ← unix ms
      }

    Search results use the same top-level fields plus ``score`` /
    ``ownerHandle`` (list responses don't carry ownerHandle, only
    detail does). ``scanStatus`` is almost always null today —
    install-time gating happens server-side, not here.
    """
    slug = str(item.get("slug") or "")
    stats = item.get("stats") or {}
    return RegistrySkillSummary(
        slug=slug,
        displayName=str(item.get("displayName") or item.get("name") or slug),
        summary=str(item.get("summary") or item.get("description") or ""),
        version=(
            (item.get("latestVersion") or {}).get("version")
            or (item.get("tags") or {}).get("latest")
            or item.get("version")
        ),
        ownerHandle=item.get("ownerHandle") or (item.get("owner") or {}).get("handle"),
        installs=stats.get("installsCurrent") or item.get("installs"),
        updatedAt=item.get("updatedAt"),
        # ``security`` is almost always missing on list responses
        # today (the registry only attaches it after a scan runs).
        # Read both possible locations defensively.
        scanStatus=registry.extract_scan_status(item),
        installedAs=slug if slug in installed_names else None,
    )


def _installed_slugs(mgr: Any) -> set[str]:
    """Names of skills currently in the SkillManager cache. ClawHub
    slugs and local skill dir names line up because we install to
    ``<user_root>/<slug>/``, so a string-equal check is enough."""
    return set(mgr._skills_cache.keys())  # noqa: SLF001 — internal but stable


def _safe_zip_extract(
    zip_bytes: bytes, *, bad_zip_status: int = 502, max_bytes: int | None = None
) -> tuple[dict[str, bytes], str]:
    """Read a skill ZIP into ``(normalized files, root dir name)``.

    ``bad_zip_status`` distinguishes who supplied the archive: 502 for a
    registry proxy response, 400 for a user upload.

    ``max_bytes`` bounds the *decompressed* total, defaulting to the registry
    download budget. The upload route passes the operator-configured upload
    limit so that lowering ``XAGENT_MAX_UPLOAD_SIZE`` also tightens how far an
    archive may expand, instead of only capping its size on the wire.

    The size budget is enforced on the *actual* decompressed byte count,
    not the sizes declared in the ZIP headers — a hostile archive can
    declare small sizes for members that inflate far larger.
    """
    budget = (
        _MAX_DOWNLOAD_BYTES
        if max_bytes is None
        else min(max_bytes, _MAX_DOWNLOAD_BYTES)
    )
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=bad_zip_status, detail="Skill archive is not a valid ZIP."
        ) from exc

    # Bound total entries before any per-member work. Directories carry no
    # bytes and used to be skipped before the cap, so an archive of nothing but
    # directory entries slipped past it entirely: 200k of them in 17 MiB took
    # 6.3s of event-loop time. The central directory is already materialized by
    # ZipFile(), so this is the earliest point the count is known.
    entries = zf.infolist()
    if len(entries) > _MAX_SKILL_FILES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Skill ZIP has {len(entries)} entries, over the "
                f"{_MAX_SKILL_FILES}-entry limit. Trim the archive to the "
                "skill's own files."
            ),
        )

    total = 0
    raw_files: dict[str, bytes] = {}
    for info in entries:
        if info.is_dir():
            continue
        path = info.filename.replace("\\", "/").lstrip("/")
        if not path or ".." in path.split("/"):
            raise HTTPException(
                status_code=bad_zip_status, detail="Skill ZIP contains unsafe paths."
            )
        remaining = budget - total
        try:
            with zf.open(info) as member:
                content = member.read(remaining + 1)
        except (zipfile.BadZipFile, RuntimeError, zlib.error) as exc:
            # BadZipFile: CRC is validated at member EOF, so lying headers or
            # corrupted data raise here rather than at open().
            # RuntimeError: an encrypted member needs a password, and an
            # unsupported compression method raises NotImplementedError,
            # which subclasses RuntimeError.
            # zlib.error: a damaged DEFLATE stream, which subclasses neither
            # of the above and so escaped as a 500 until it was named here.
            raise HTTPException(
                status_code=bad_zip_status,
                detail="Skill archive is corrupted, encrypted, or uses an unsupported compression method.",
            ) from exc
        if len(content) > remaining:
            raise HTTPException(
                status_code=413, detail="Skill ZIP exceeds size budget."
            )
        total += len(content)
        raw_files[path] = content

    skill_md_paths = [
        path for path in raw_files if path.endswith("/SKILL.md") or path == "SKILL.md"
    ]
    if not skill_md_paths:
        raise HTTPException(
            status_code=bad_zip_status,
            detail="Skill archive has no SKILL.md anywhere in it.",
        )
    # Pick the shallowest SKILL.md as the skill root. Sorting the paths
    # lexicographically instead would pick by alphabet, so "a/b/c/SKILL.md"
    # could beat "z/SKILL.md".
    min_depth = min(path.count("/") for path in skill_md_paths)
    shallowest = sorted(path for path in skill_md_paths if path.count("/") == min_depth)
    skill_root = shallowest[0].removesuffix("SKILL.md").rstrip("/")

    # Anything below is dropped when it falls outside the chosen root, so a
    # second SKILL.md *anywhere* outside it means we would silently discard an
    # independent skill and still report success. Depth is irrelevant: both
    # "a/SKILL.md" + "b/SKILL.md" and "z/SKILL.md" + "a/b/c/SKILL.md" lose one.
    # A SKILL.md nested *within* the chosen root (a bundled example) is kept as
    # an ordinary member, so only genuinely separate skills are refused.
    root_prefix = f"{skill_root}/" if skill_root else ""
    outside = [
        path
        for path in skill_md_paths
        if not (root_prefix and path.startswith(root_prefix))
        and path != f"{root_prefix}SKILL.md"
    ]
    if outside:
        # Name a few roots so the user can find them, but do not echo an
        # archive-controlled list of unbounded length into the response and
        # the logs: 50 long directory names made this a 10 KB error string.
        names = [
            path.removesuffix("SKILL.md").rstrip("/") or "."
            for path in sorted([shallowest[0], *outside])
        ]
        shown = [name[:60] for name in names[:5]]
        roots = ", ".join(shown)
        if len(names) > len(shown):
            roots += f", and {len(names) - len(shown)} more"
        raise HTTPException(
            status_code=bad_zip_status,
            detail=(
                f"Skill archive contains multiple skills ({roots}). "
                "Upload one skill per archive."
            ),
        )
    files: dict[str, bytes] = {}
    for path, content in raw_files.items():
        if skill_root:
            prefix = skill_root + "/"
            if not path.startswith(prefix):
                continue
            rel = path[len(prefix) :]
        else:
            rel = path
        if rel:
            files[rel] = content
    return (
        _normalize_skill_files(files, bad_status=bad_zip_status),
        skill_root.rsplit("/", 1)[-1],
    )


def _check_registry_security_gate(registry: Any, detail: dict) -> None:
    """Raise HTTP 403 if the registry flags this skill as unsafe.

    Checks two independent signals:
    * ``scan_status == "malicious"`` — AV/scanner verdict via
      ``registry.extract_scan_status``
    * ``moderation.moderationState in {"quarantined", "revoked"}`` — human
      moderation verdict embedded directly in the detail payload
    """
    scan_status = registry.extract_scan_status(detail)
    moderation = detail.get("moderation") or {}
    moderation_state = (
        moderation.get("moderationState") if isinstance(moderation, dict) else None
    )
    if scan_status == "malicious":
        raise HTTPException(
            status_code=403,
            detail=f"Install refused: this skill is flagged malicious by {registry.display_name} scanners.",
        )
    if moderation_state in ("quarantined", "revoked"):
        raise HTTPException(
            status_code=403,
            detail=f"Install refused: skill is {moderation_state} by {registry.display_name} moderators.",
        )


# ──────────────────────────────────────────────────────────────────────
# Helpers — persist + verify round trip
# ──────────────────────────────────────────────────────────────────────


def _expected_scope_source(scope: str) -> str:
    """The ``source`` a freshly written record must report back.

    ``_summary_source`` maps ``scope="personal"`` to ``"user"``; team writes
    surface as ``"team"``.
    """
    return "user" if scope == "personal" else scope


async def _persist_and_reparse(
    *,
    request: Request,
    context: SkillScopeContext,
    db: Any,
    user: User,
    name: str,
    files: dict[str, bytes],
    scope: str,
    origin: str,
    clawhub_slug: str | None = None,
    clawhub_version: str | None = None,
) -> Any:
    """Write a skill bundle, re-parse it, and hand back the parsed record.

    Returns the SkillManager's parsed dict. Typed ``Any`` to match the rest of
    this module, which keeps the skills package out of its import graph.

    Shared by ``create_skill`` / ``upload_skill`` / ``install_skill``, which
    previously each carried their own copy of this tail with drifted status
    codes and messages.

    Three failure modes this closes:

    * **Orphaned rows.** ``_write_personal_skill`` commits, but the parser
      that runs afterwards decodes every bundled file as UTF-8 with no
      fallback and ``SkillManager.reload`` logs-and-skips the failure. A
      bundle that commits but cannot parse used to leave a row that no API
      verb could reach — ``GET /installed`` enumerates the parsed cache so it
      never listed, ``DELETE`` 404'd on the same lookup, and a retry hit the
      duplicate-name 409. The name was burned until someone touched the DB by
      hand. We now delete the row before returning the error.

    * **False success on a name collision.** ``reload`` builds a plain dict
      keyed by name, filesystem records first, so a *builtin* of the same name
      stays cached when the personal record fails to parse. A bare
      ``skill is None`` check passed and the endpoint returned 200 with the
      builtin's metadata, sending the user to an unrelated skill's page. We
      verify the record we get back is the one we just wrote.

    * **A durable row with no compensation.** The write commits before the
      readback runs, so a transient read failure or a cancellation used to
      leave the row behind with the client seeing only an error. Every exit
      from the readback now compensates.
    """
    _validate_skill_name(name)
    normalized = _normalize_skill_files(files)

    if scope == "team":
        from xagent.skills.library import get_skill_write_provider

        metadata = None
        if clawhub_slug is not None or clawhub_version is not None:
            metadata = {
                f"{origin}_slug": clawhub_slug,
                f"{origin}_version": clawhub_version,
            }
        kwargs: dict[str, Any] = {
            "scope": "team",
            "name": name,
            "files": normalized,
            "origin": origin,
        }
        if metadata is not None:
            kwargs["metadata"] = metadata
        await invoke_skill_write_provider(
            get_skill_write_provider(),
            "create_skill",
            _write_context(context),
            **kwargs,
        )
    else:
        # Deliberately NOT offloaded. ``run_db_io_cancellation_safe`` — the
        # sanctioned way to move DB work off the loop here — requires the
        # operation to own and close its own Session and return detached data.
        # This writes through the request-scoped session, which is not
        # thread-safe and whose ORM objects would be touched again on the loop
        # afterwards, so handing it to a worker thread would trade a latency
        # problem for a correctness one. Hashing dominates the cost and is
        # already bounded by the file-count and byte caps above.
        _write_personal_skill(
            db=db,
            user=user,
            name=name,
            files=normalized,
            origin=origin,
            clawhub_slug=clawhub_slug,
            clawhub_version=clawhub_version,
        )

    def _compensate() -> bool:
        """Undo the write so the name stays available.

        Personal rows are ours to delete; a team provider owns its own
        storage, so there we can only report. Returns whether the row was
        actually removed, because the caller's message must not claim a
        rollback that never ran.
        """
        if scope != "personal":
            return False
        try:
            _delete_personal_skill(db=db, user=user, name=name)
            return True
        except Exception:
            # Cleanup must never replace the real error with its own.
            logger.warning(
                "Skill Hub: could not roll back unparsable skill %r",
                name,
                exc_info=True,
            )
            return False

    # The write is already durable, so every path out of the readback needs
    # compensation — not just the "it did not parse" one. A transient failure
    # or a cancellation here previously left the row committed and the name
    # taken, with the client seeing only an error.
    try:
        mgr = await _get_scoped_manager(request, context, db)
        skill = await mgr.get_skill(name)
    except BaseException:
        # BaseException so asyncio.CancelledError is covered too.
        _compensate()
        raise

    expected_source = _expected_scope_source(scope)
    actual_source = _summary_source(skill) if skill is not None else None
    if skill is None or actual_source != expected_source:
        rolled_back = _compensate()
        if skill is not None:
            logger.error(
                "Skill Hub: %r re-parsed as a %s skill, not the %s skill just "
                "written — refusing to report success",
                name,
                actual_source,
                expected_source,
            )
        remedy = (
            "It was rolled back, so the name is free to reuse."
            if rolled_back
            else "The written copy was left in place; remove it before retrying."
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Skill {name!r} could not be loaded after writing. {remedy} "
                "Every file in the bundle must be UTF-8 text, and the name "
                "must not collide with an existing skill."
            ),
        )
    return skill


# ──────────────────────────────────────────────────────────────────────
# Routes — local skills (list / detail / delete)
# ──────────────────────────────────────────────────────────────────────


@router.get("/installed", response_model=List[SkillSummary])
async def list_installed(
    request: Request,
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
) -> List[SkillSummary]:
    """List every skill the SkillManager can see, tagged with source."""
    mgr = await _get_scoped_manager(request, context, db)
    summaries: list[SkillSummary] = []
    for skill in mgr._skills_cache.values():  # noqa: SLF001
        summaries.append(_skill_to_summary(skill))
    summaries.sort(key=lambda s: (s.source != "user", s.name.lower()))
    logger.info("Skill Hub: listed %d installed skill(s)", len(summaries))
    return summaries


@router.get("/installed/{name}", response_model=SkillDetail)
async def get_installed(
    name: str,
    request: Request,
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
) -> SkillDetail:
    mgr = await _get_scoped_manager(request, context, db)
    skill = await mgr.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return _skill_to_detail(skill)


@router.delete(
    "/installed/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_installed(
    name: str,
    request: Request,
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> Response:
    """Remove a user-installed skill. Builtin / external are refused."""
    mgr = await _get_scoped_manager(request, context, db)
    skill = await mgr.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    source = _summary_source(skill)
    if source == "team":
        from xagent.skills.library import get_skill_write_provider

        await invoke_skill_write_provider(
            get_skill_write_provider(),
            "delete_skill",
            _write_context(context),
            scope="team",
            name=name,
        )
        logger.info("Skill Hub: deleted team skill %r", name)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    if source != "user":
        raise HTTPException(
            status_code=403,
            detail=(
                f"Cannot delete a {source} skill — only user-installed skills "
                "can be removed."
            ),
        )
    _delete_personal_skill(db=db, user=_user, name=name)
    logger.info("Skill Hub: deleted user skill %r", name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ──────────────────────────────────────────────────────────────────────
# Routes — in-UI authoring
# ──────────────────────────────────────────────────────────────────────


@router.post("/create", response_model=SkillSummary)
async def create_skill(
    body: CreateSkillRequest,
    request: Request,
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Write a brand-new skill from in-UI input.

    The user supplies a name (used verbatim as the on-disk directory
    and the skill's external identifier) and the SKILL.md body. We
    refuse on duplicate names — overwrite via the edit endpoint is
    explicit, not implicit.
    """
    skill = await _persist_and_reparse(
        request=request,
        context=context,
        db=db,
        user=_user,
        name=body.name,
        files={"SKILL.md": body.skill_md.encode("utf-8")},
        scope=body.scope,
        origin="custom",
    )
    logger.info(
        "Skill Hub: created user skill %r (%d bytes)", body.name, len(body.skill_md)
    )
    return _skill_to_summary(skill)


@router.put("/installed/{name}", response_model=SkillSummary)
async def edit_installed(
    name: str,
    body: EditSkillRequest,
    request: Request,
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Replace the SKILL.md of an installed user skill.

    Only ``user`` source is editable — builtin / external skills are
    refused so we don't silently fork a shipped skill (and so symlinked
    external roots stay readonly from our side).
    """
    mgr = await _get_scoped_manager(request, context, db)
    skill = await mgr.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    source = _summary_source(skill)
    if source == "team":
        from xagent.skills.library import get_skill_write_provider

        await invoke_skill_write_provider(
            get_skill_write_provider(),
            "update_skill_file",
            _write_context(context),
            scope="team",
            name=name,
            path="SKILL.md",
            content=body.skill_md.encode("utf-8"),
        )
    elif source != "user":
        raise HTTPException(
            status_code=403,
            detail="Only user-installed skills can be edited via the Hub.",
        )
    else:
        _update_personal_skill_md(db=db, user=_user, name=name, skill_md=body.skill_md)
    mgr = await _get_scoped_manager(request, context, db)
    reloaded = await mgr.get_skill(name)
    if reloaded is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Edit written to disk but the parser rejected it. Fix the "
                "SKILL.md and PUT again — the bad version is still on disk."
            ),
        )
    logger.info("Skill Hub: edited user skill %r", name)
    return _skill_to_summary(reloaded)


def _slugify_skill_name(raw: str) -> str:
    """Collapse arbitrary text into the [A-Za-z0-9_-]+ shape
    ``_validate_skill_name`` accepts; empty string if nothing survives.

    Truncation happens before the final strip so a cut mid-name cannot
    reintroduce a trailing separator.
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "-", raw.strip())[:64].strip("-_")


def _generated_skill_name(files: dict[str, bytes]) -> str:
    """Deterministic fallback for a name that slugifies to nothing.

    Names written entirely in a non-Latin script (Chinese, Japanese,
    Cyrillic, …) collapse to an empty slug, which used to be a dead-end
    400 with no way for the user to proceed. Derive a stable name from the
    content instead so the upload succeeds and can be renamed afterwards.

    The digest covers every file, not just SKILL.md: two bundles sharing a
    SKILL.md but differing in their support files are different skills, and
    hashing only SKILL.md gave them the same name — the second upload then
    hit the duplicate-name 409 with no way to rename it.
    """
    digest = hashlib.sha256()
    for path, content in sorted(files.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    return f"skill-{digest.hexdigest()[:12]}"


def _derive_upload_skill_name(
    filename: str,
    zip_root: str,
    files: dict[str, bytes],
    override: str | None = None,
) -> str:
    """Pick a skill name for an uploaded bundle.

    Priority: caller-supplied ``override`` → frontmatter ``name`` → ZIP root
    directory name → a content-derived fallback.

    Frontmatter outranks the ZIP root because the directory name is an
    accidental artifact of how the folder happened to be zipped, while
    frontmatter is the author's explicit declaration. The bare-``.md`` path
    deliberately does *not* fall back to the filename stem: dropping a file
    named ``SKILL.md`` (exactly what the UI suggests) would otherwise produce
    a skill literally named ``SKILL``.
    """
    from xagent.skills.parser import SkillParser

    frontmatter = SkillParser._extract_frontmatter(  # noqa: SLF001
        files["SKILL.md"].decode("utf-8", errors="replace")
    )
    # YAML types a bare ``name: 12345`` as an int and ``name: true`` as a bool.
    # Those are legitimate skill names once slugified, so accept any scalar
    # rather than dead-ending the upload on the frontmatter's YAML type.
    fm_name = frontmatter.get("name")
    # bool is a subclass of int, and "name: true" naming a skill ``True`` is
    # nonsense, so exclude it explicitly.
    fm_text = (
        str(fm_name)
        if isinstance(fm_name, (str, int, float)) and not isinstance(fm_name, bool)
        else ""
    )
    candidates = [
        override or "",
        fm_text,
        zip_root,
    ]
    for raw in candidates:
        slug = _slugify_skill_name(raw)
        if slug:
            return slug
    if override or any(candidates):
        # Something was supplied but slugified away entirely (e.g. an
        # all-CJK name) — fall back rather than dead-ending the upload.
        return _generated_skill_name(files)
    raise HTTPException(
        status_code=400,
        detail=(
            "Could not determine a skill name. Add a 'name:' field to the "
            "SKILL.md frontmatter, or supply a name with the upload."
        ),
    )


@router.post("/upload", response_model=SkillSummary)
async def upload_skill(
    request: Request,
    file: UploadFile = File(...),
    scope: str = Form("personal"),
    name: Optional[str] = Form(None),
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Install a skill from an uploaded file.

    Accepts either a ``.zip`` skill bundle (a Claude-style skill folder
    with SKILL.md at its root, possibly nested one directory deep) or a
    bare ``.md`` file used verbatim as SKILL.md. Same tail as
    ``install_skill``: persist, re-parse, return the summary.
    """
    if scope not in ("personal", "team"):
        raise HTTPException(
            status_code=400, detail="scope must be 'personal' or 'team'."
        )

    max_bytes = get_max_upload_size_bytes()
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {format_file_size(max_bytes)} limit.",
        )

    filename = (file.filename or "").strip()
    lower = filename.lower()
    if lower.endswith(".zip"):
        # Parsing and decompressing a large archive is CPU-bound and blocking:
        # a 17 MiB archive of 200k entries held the loop for over six seconds.
        # The rest of this module already offloads its blocking work the same
        # way, so keep the event loop free here too.
        files, zip_root = await asyncio.to_thread(
            _safe_zip_extract, data, bad_zip_status=400, max_bytes=max_bytes
        )
    elif lower.endswith(".md"):
        files, zip_root = {"SKILL.md": data}, ""
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported upload — provide a .zip skill bundle or a SKILL.md file.",
        )

    # Reject non-UTF-8 content before persisting anything. ``SkillParser``
    # decodes SKILL.md *and* template.md with no fallback, so checking only
    # SKILL.md left a bundle that committed and then failed to parse.
    # ``_persist_and_reparse`` would roll that back, but failing up front gives
    # the user the actual reason and names the offending file.
    for path in ("SKILL.md", "template.md"):
        content = files.get(path)
        if content is None:
            continue
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"{path} must be UTF-8 text."
            ) from exc

    skill_name = _derive_upload_skill_name(filename, zip_root, files, override=name)

    skill = await _persist_and_reparse(
        request=request,
        context=context,
        db=db,
        user=_user,
        name=skill_name,
        files=files,
        scope=scope,
        origin="upload",
    )
    logger.info(
        "Skill Hub: uploaded skill %r from %r (%d bytes, %d file(s))",
        skill_name,
        # The client controls this and it is not length-bounded anywhere,
        # unlike skill_name; truncate so one request cannot bloat a log line.
        filename[:120],
        len(data),
        len(files),
    )
    return _skill_to_summary(skill)


# ──────────────────────────────────────────────────────────────────────
# Routes — registries list + registry proxy + install
# ──────────────────────────────────────────────────────────────────────


@router.get("/registries")
async def list_registries(
    _context: SkillScopeContext = Depends(get_skill_runtime_scope),
) -> List[Dict[str, str]]:
    """Return available skill registries (ClawHub, etc.).
    The frontend uses this to build the source-selector dropdown."""
    return all_registries()


@router.get("/registry/list", response_model=RegistryListResponse)
async def registry_list(
    request: Request,
    sort: str = Query("installsCurrent"),
    limit: int = Query(24, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    source: str = Query("clawhub"),
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
) -> RegistryListResponse:
    """Browse a skill registry's catalog."""
    registry = get_registry(source)
    payload = await asyncio.to_thread(registry.list_skills, sort, limit, cursor)
    items_raw = payload.get("items", []) if isinstance(payload, dict) else []
    mgr = await _get_scoped_manager(request, context, db)
    installed = _installed_slugs(mgr)
    items = [
        _summary_from_registry_item(i, installed, registry)
        for i in items_raw
        if isinstance(i, dict)
    ]
    next_cursor = payload.get("nextCursor") if isinstance(payload, dict) else None
    logger.info(
        "Skill Hub: registry/list source=%s sort=%s limit=%d → %d item(s), more=%s",
        source,
        sort,
        limit,
        len(items),
        "yes" if next_cursor else "no",
    )
    return RegistryListResponse(items=items, nextCursor=next_cursor)


@router.get("/registry/search", response_model=RegistryListResponse)
async def registry_search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(24, ge=1, le=100),
    source: str = Query("clawhub"),
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
) -> RegistryListResponse:
    """Full-text search a skill registry."""
    registry = get_registry(source)
    payload = await asyncio.to_thread(registry.search_skills, q, limit)
    results_raw = (
        payload.get(registry.search_results_field, [])
        if isinstance(payload, dict)
        else []
    )
    mgr = await _get_scoped_manager(request, context, db)
    installed = _installed_slugs(mgr)
    items = [
        _summary_from_registry_item(i, installed, registry)
        for i in results_raw
        if isinstance(i, dict)
    ]
    logger.info(
        "Skill Hub: registry/search source=%s q=%r → %d result(s)",
        source,
        q[:50],
        len(items),
    )
    return RegistryListResponse(items=items, nextCursor=None)


@router.post("/install/{source}", response_model=SkillSummary)
async def install_skill(
    source: str,
    body: InstallSkillRequest,
    request: Request,
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Install a skill from a registry into ``~/.xagent/skills/<slug>/``."""
    _validate_skill_name(body.slug)

    # --- Look up registry ------------------------------------------
    registry = get_registry(source)

    # --- Scan + moderation gate ------------------------------------
    detail = await asyncio.to_thread(registry.get_skill, body.slug)
    if not isinstance(detail, dict):
        raise HTTPException(
            status_code=502,
            detail=f"{registry.display_name} detail had unexpected shape.",
        )
    _check_registry_security_gate(registry, detail)
    scan_status = registry.extract_scan_status(detail)

    # --- Download ZIP ----------------------------------------------
    dl_status, zip_bytes = await asyncio.to_thread(
        registry.download_skill, body.slug, body.version
    )
    if dl_status == 404:
        raise HTTPException(
            status_code=404,
            detail=f"{registry.display_name} skill or version not found.",
        )
    if dl_status >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"{registry.display_name} /download returned HTTP {dl_status}.",
        )
    if len(zip_bytes) > _MAX_DOWNLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Skill archive exceeds {_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MiB limit.",
        )

    # --- Store DB bundle -----------------------------------------
    files, _root = await asyncio.to_thread(
        _safe_zip_extract, zip_bytes, bad_zip_status=502
    )
    skill = await _persist_and_reparse(
        request=request,
        context=context,
        db=db,
        user=_user,
        name=body.slug,
        files=files,
        scope=body.scope,
        origin=registry.id,
        clawhub_slug=body.slug,
        clawhub_version=body.version,
    )
    logger.info(
        "Skill Hub: installed %s skill %r (v%s, scan=%s)",
        registry.id,
        body.slug,
        body.version or "latest",
        scan_status,
    )
    return _skill_to_summary(skill)


@router.get("/registry/{slug}", response_model=RegistrySkillDetail)
async def registry_detail(
    slug: str,
    request: Request,
    source: str = Query("clawhub"),
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
) -> RegistrySkillDetail:
    """Single-skill detail from a registry."""
    registry = get_registry(source)
    payload = await asyncio.to_thread(registry.get_skill, slug)
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected {registry.display_name} response shape.",
        )
    skill = payload.get("skill") or {}
    latest = payload.get("latestVersion") or {}
    moderation = payload.get("moderation")
    metadata = payload.get("metadata") or {}
    mgr = await _get_scoped_manager(request, context, db)
    installed = _installed_slugs(mgr)
    return RegistrySkillDetail(
        slug=slug,
        displayName=str(skill.get("displayName") or skill.get("name") or slug),
        summary=str(skill.get("summary") or metadata.get("description") or ""),
        version=latest.get("version"),
        ownerHandle=(payload.get("owner") or {}).get("handle")
        or skill.get("ownerHandle"),
        homepage=metadata.get("homepage"),
        readme=metadata.get("readme")
        or latest.get("readme")
        or skill.get("description"),
        scanStatus=registry.extract_scan_status(payload),
        moderation=moderation if isinstance(moderation, dict) else None,
        installedAs=slug if slug in installed else None,
        registrySource=source,
        raw=payload,
    )
