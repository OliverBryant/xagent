"""Skill Hub API — manage user-installed skills (saas closed-source).

The Hub composes three capabilities on top of xagent's existing skill
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

GitHub-URL import was removed in this iteration: we previously
shipped a ``git clone --depth=1`` path, but ClawHub gives us trusted
binaries with provenance and scan results, so we don't need to
re-implement that surface area. If someone really wants an
unscanned-source install path back, ``git`` is still on the box.

All writes (installs, creates, edits) persist to the database via
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

from xagent.skills.library import SkillScopeContext
from xagent.web.api.skill_hub_registry import (
    _MAX_DOWNLOAD_BYTES,
    SkillRegistry,
    all_registries,
    get_registry,
)
from xagent.web.auth_dependencies import get_current_user
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


# Archiving a folder on macOS sweeps in Finder and resource-fork droppings.
# The demo path for this feature is "zip an Anthropic skill folder and drop it
# in", so refusing the whole upload over a .DS_Store would fail that flow on
# any Finder-touched folder. These carry nothing a skill needs: drop them.
_IGNORED_ARCHIVE_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


def _is_archive_cruft(path: str) -> bool:
    """True for OS bookkeeping files that are never part of a skill."""
    if path.startswith("__MACOSX/"):
        return True
    segments = path.split("/")
    return any(
        seg in _IGNORED_ARCHIVE_NAMES or seg.startswith("._") for seg in segments
    )


def _normalize_skill_files(files: dict[str, bytes]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    total = 0
    for raw_path, content in files.items():
        path = str(raw_path).replace("\\", "/").lstrip("/")
        if not path or ".." in path.split("/"):
            raise HTTPException(
                status_code=400,
                detail="Skill file path contains a path-traversal sequence.",
            )
        if _is_archive_cruft(path):
            continue
        # Check every segment, not just the first character of the whole path:
        # ".env" at the root was refused while "sub/.env" sailed through.
        dotted = next((seg for seg in path.split("/") if seg.startswith(".")), None)
        if dotted is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Skill bundle contains a hidden file ({dotted}). Remove it "
                    "and upload again."
                ),
            )
        total += len(content)
        if total > _MAX_DOWNLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail="Skill files exceed size budget."
            )
        out[path] = bytes(content)
    if "SKILL.md" not in out:
        raise HTTPException(status_code=400, detail="Skill has no SKILL.md.")
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


def _safe_zip_to_files(zip_bytes: bytes) -> dict[str, bytes]:
    """Read a skill ZIP into a normalized skill file bundle."""
    files, _root = _safe_zip_extract(zip_bytes)
    return files


def _safe_zip_extract(
    zip_bytes: bytes, *, bad_zip_status: int = 502
) -> tuple[dict[str, bytes], str]:
    """Read a skill ZIP into ``(normalized files, root dir name)``.

    ``bad_zip_status`` distinguishes who supplied the archive: 502 for a
    registry proxy response, 400 for a user upload.

    The size budget is enforced on the *actual* decompressed byte count,
    not the sizes declared in the ZIP headers — a hostile archive can
    declare small sizes for members that inflate far larger.
    """
    # One guard around the whole "read an untrusted archive" region, rather than
    # naming exception types per call site. Enumerating them only ever covers
    # what was thought of: a tampered end-of-central-directory offset raises
    # ValueError from the constructor, an encrypted member RuntimeError, a
    # broken DEFLATE stream zlib.error, BZIP2/LZMA corruption OSError or
    # lzma.LZMAError — none of which subclass one another. Whatever zipfile
    # raises next is covered here too. Our own HTTPExceptions re-raise unchanged
    # so their specific status and message survive.
    total = 0
    raw_files: dict[str, bytes] = {}
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        for info in zf.infolist():
            if info.is_dir():
                continue
            path = info.filename.replace("\\", "/").lstrip("/")
            if not path or ".." in path.split("/"):
                raise HTTPException(
                    status_code=400, detail="Skill ZIP contains unsafe paths."
                )
            remaining = _MAX_DOWNLOAD_BYTES - total
            with zf.open(info) as member:
                content = member.read(remaining + 1)
            if len(content) > remaining:
                raise HTTPException(
                    status_code=413, detail="Skill ZIP exceeds size budget."
                )
            total += len(content)
            raw_files[path] = content
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "Skill Hub: unreadable archive (%s)", type(exc).__name__, exc_info=True
        )
        raise HTTPException(
            status_code=bad_zip_status,
            detail="Skill archive is not a readable ZIP.",
        ) from exc

    # Choose the root by depth, not by alphabet. A plain sort asserted the
    # wrong invariant: a skill folder shipping its own "Examples/SKILL.md"
    # sorts before the real root marker, so the example was imported *as* the
    # skill — named "Examples", with the true root's files silently dropped and
    # a 200 returned. Cruft is filtered first so a crafted "__MACOSX/SKILL.md"
    # cannot win either.
    skill_md_paths = [
        path
        for path in raw_files
        if (path.endswith("/SKILL.md") or path == "SKILL.md")
        and not _is_archive_cruft(path)
    ]
    if not skill_md_paths:
        raise HTTPException(
            status_code=400, detail="Skill archive has no SKILL.md anywhere in it."
        )
    min_depth = min(path.count("/") for path in skill_md_paths)
    shallowest = sorted(p for p in skill_md_paths if p.count("/") == min_depth)
    if len(shallowest) > 1:
        # Two skills at the same depth: picking one would discard the other, so
        # make the user say which. Bounded so a crafted archive cannot reflect
        # an unlimited list of its own directory names back through the error.
        roots = ", ".join(
            (path.removesuffix("SKILL.md").rstrip("/") or ".")[:60]
            for path in shallowest[:5]
        )
        if len(shallowest) > 5:
            roots += f", and {len(shallowest) - 5} more"
        raise HTTPException(
            status_code=bad_zip_status,
            detail=(
                f"Skill archive contains multiple skills ({roots}). "
                "Upload one skill per archive."
            ),
        )
    skill_root = shallowest[0].removesuffix("SKILL.md").rstrip("/")
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
    return _normalize_skill_files(files), skill_root.rsplit("/", 1)[-1]


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
    if body.scope != "personal":
        from xagent.skills.library import get_skill_write_provider

        await invoke_skill_write_provider(
            get_skill_write_provider(),
            "create_skill",
            _write_context(context),
            scope=body.scope,
            name=body.name,
            files={"SKILL.md": body.skill_md.encode("utf-8")},
        )
    else:
        _write_personal_skill(
            db=db,
            user=_user,
            name=body.name,
            files={"SKILL.md": body.skill_md.encode("utf-8")},
        )

    mgr = await _get_scoped_manager(request, context, db)
    skill = await mgr.get_skill(body.name)
    if skill is None:
        # Most likely cause: malformed YAML frontmatter that the parser
        # rejected. Leave the file on disk so the user can fix it via
        # PUT, but tell them why nothing showed up.
        raise HTTPException(
            status_code=400,
            detail=(
                "Skill written to disk but failed to re-parse — check the "
                "YAML frontmatter at the top of SKILL.md."
            ),
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
    ``_validate_skill_name`` accepts; empty string if nothing survives."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", raw.strip()).strip("-_")[:64]


def _is_utf8(content: bytes) -> bool:
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _assert_bundle_parses(files: dict[str, bytes]) -> None:
    """Refuse a bundle that ``SkillManager.reload`` would fail to load.

    Runs the same ``SkillParser.parse_bundle`` the reload path uses, so this
    tracks the parser instead of guessing at its failure modes: non-UTF-8
    content, unparsable frontmatter, and anything added later are all covered
    by construction.
    """
    from xagent.skills.parser import SkillParser

    try:
        SkillParser.parse_bundle(name="upload", files=files)
    except HTTPException:
        raise
    except RecursionError as exc:
        # Deeply nested YAML blows the stack inside yaml.safe_load, which
        # ``_extract_frontmatter`` does not catch (it guards yaml.YAMLError).
        raise HTTPException(
            status_code=400,
            detail="SKILL.md frontmatter is nested too deeply to parse.",
        ) from exc
    except UnicodeDecodeError as exc:
        # parse_bundle does not say which file failed, and that is the
        # actionable half of the message, so identify it here.
        culprit = next(
            (path for path, content in sorted(files.items()) if not _is_utf8(content)),
            None,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"{culprit} must be UTF-8 text."
                if culprit
                else "Every file in the bundle must be UTF-8 text."
            ),
        ) from exc
    except Exception as exc:
        logger.warning(
            "Skill Hub: rejecting an upload that failed to parse", exc_info=True
        )
        raise HTTPException(
            status_code=400,
            detail=f"Skill bundle could not be parsed: {type(exc).__name__}.",
        ) from exc


def _derive_upload_skill_name(
    filename: str, zip_root: str, skill_md: bytes, override: str | None = None
) -> str:
    """Pick a skill name for an uploaded bundle.

    Priority: caller-supplied ``override`` → ZIP root directory name (how
    Claude-style skill folders are usually zipped) → frontmatter ``name`` →
    upload filename stem.

    An override is the caller stating intent, so one that would be rewritten
    by slugification is refused instead: quietly turning "bad name!" into
    "bad-name" hands back a skill nobody asked for.
    """
    from xagent.skills.parser import SkillParser

    if override is not None and override.strip():
        candidate = override.strip()
        # Validate against the rule the message quotes, not a slugifier
        # round-trip. The slugifier strips leading and trailing "-"/"_", so it
        # rejected names like "_foo" and "my_skill_" that _NAME_RE accepts and
        # POST /create takes — with an error citing the regex they do match.
        if not _NAME_RE.match(candidate) or len(candidate) > 64:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Skill name must match [A-Za-z0-9_-]+ and be at most 64 "
                    "characters; it is not rewritten for you."
                ),
            )
        return candidate

    frontmatter = SkillParser._extract_frontmatter(  # noqa: SLF001
        skill_md.decode("utf-8", errors="replace")
    )
    fm_name = frontmatter.get("name")
    candidates = [
        zip_root,
        fm_name if isinstance(fm_name, str) else "",
        Path(filename).stem,
    ]
    for raw in candidates:
        slug = _slugify_skill_name(raw)
        if slug:
            return slug
    raise HTTPException(
        status_code=400, detail="Could not derive a skill name from the upload."
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

    data = await file.read(_MAX_DOWNLOAD_BYTES + 1)
    if len(data) > _MAX_DOWNLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MiB limit.",
        )

    filename = (file.filename or "").strip()
    lower = filename.lower()
    if lower.endswith(".zip"):
        files, zip_root = _safe_zip_extract(data, bad_zip_status=400)
    elif lower.endswith(".md"):
        files, zip_root = {"SKILL.md": data}, ""
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported upload — provide a .zip skill bundle or a SKILL.md file.",
        )

    # Validate the bundle before anything reads it — naming included. Deriving
    # the name parses frontmatter itself, so leaving validation until after
    # would let an unparsable bundle raise from the naming step instead.
    # Validation must not depend on which naming branch a request happens to
    # take; that ordering coupling is what made the override path a bypass.
    # ``name`` only labels the parse error, so validating before the name is
    # known loses nothing and keeps the check independent of naming.
    _assert_bundle_parses(files)

    skill_name = _derive_upload_skill_name(
        filename, zip_root, files["SKILL.md"], override=name
    )

    if scope == "team":
        from xagent.skills.library import get_skill_write_provider

        await invoke_skill_write_provider(
            get_skill_write_provider(),
            "create_skill",
            _write_context(context),
            scope="team",
            name=skill_name,
            files=files,
            origin="upload",
        )
    else:
        _write_personal_skill(
            db=db,
            user=_user,
            name=skill_name,
            files=files,
            origin="upload",
        )

    mgr = await _get_scoped_manager(request, context, db)
    skill = await mgr.get_skill(skill_name)
    # Check identity, not just presence. ``reload`` keys the cache by name with
    # filesystem records loaded first, so when a personal record fails to parse
    # a same-named builtin stays resident — and a bare ``is None`` test would
    # return 200 carrying that unrelated skill's content while the real row sat
    # orphaned. What we wrote must come back as what we wrote.
    expected_source = "user" if scope == "personal" else scope
    if skill is not None and _summary_source(skill) != expected_source:
        logger.error(
            "Skill Hub: %r read back as a %s skill, not the %s skill just "
            "written — refusing to report success",
            skill_name,
            _summary_source(skill),
            expected_source,
        )
        skill = None
    if skill is None:
        # The write is already durable. Undo it so the name stays available;
        # the pre-check above catches the predictable causes, but anything it
        # misses must not squat a name the user can never reclaim.
        rolled_back = False
        if scope == "personal":
            try:
                _delete_personal_skill(db=db, user=_user, name=skill_name)
                rolled_back = True
            except Exception:
                logger.warning(
                    "Skill Hub: could not roll back unloadable skill %r",
                    skill_name,
                    exc_info=True,
                )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Skill {skill_name!r} was written but could not be loaded. "
                + (
                    "It was rolled back, so the name is free to reuse."
                    if rolled_back
                    else "Remove the written copy before retrying."
                )
            ),
        )
    logger.info(
        "Skill Hub: uploaded skill %r from %r (%d bytes, %d file(s))",
        skill_name,
        filename,
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
    files = _safe_zip_to_files(zip_bytes)
    if body.scope == "team":
        from xagent.skills.library import get_skill_write_provider

        await invoke_skill_write_provider(
            get_skill_write_provider(),
            "create_skill",
            _write_context(context),
            scope="team",
            name=body.slug,
            files=files,
            origin=registry.id,
            metadata={
                f"{registry.id}_slug": body.slug,
                f"{registry.id}_version": body.version,
            },
        )
    else:
        _write_personal_skill(
            db=db,
            user=_user,
            name=body.slug,
            files=files,
            origin=registry.id,
            clawhub_slug=body.slug,
            clawhub_version=body.version,
        )

    mgr = await _get_scoped_manager(request, context, db)
    skill = await mgr.get_skill(body.slug)
    if skill is None:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{registry.display_name} skill {body.slug!r} installed but failed "
                "to re-parse. Inspect SKILL.md by hand or remove and retry."
            ),
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
