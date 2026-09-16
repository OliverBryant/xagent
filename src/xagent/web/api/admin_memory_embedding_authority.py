from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user, is_admin_user
from ..models.database import get_db
from ..models.user import User
from ..services.global_memory_embedding_authority import (
    AuthorityConfiguration,
    CredentialSource,
    GlobalMemoryEmbeddingAuthorityRecord,
    GlobalMemoryEmbeddingAuthorityService,
)

router = APIRouter(
    prefix="/api/admin/memory/embedding-authority",
    tags=["admin-memory"],
)


def require_admin(user: User = Depends(get_current_user)) -> User:
    if not is_admin_user(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


class AuthorityState(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    configured: bool = True
    provider: str | None = None
    model_name: str | None = None
    endpoint: str | None = None
    dimension: int | None = None
    instruct: str | None = None
    max_retries: int | None = None
    credential_source: CredentialSource | None = None
    global_sharing_consent: bool | None = None
    consented_by_actor_subject: str | None = None
    consented_at: datetime | None = None
    credential_status: str | None = None
    configured_at: datetime | None = Field(default=None, validation_alias="created_at")
    updated_at: datetime | None = None


def _public_state(service: GlobalMemoryEmbeddingAuthorityService) -> AuthorityState:
    row = service.get_row()
    if row is None:
        return AuthorityState(configured=False)
    return AuthorityState(
        configured=True,
        provider=str(row.model_provider),
        model_name=str(row.model_name),
        endpoint=str(row.base_url),
        dimension=int(row.dimension),
        instruct=str(row.instruct) if row.instruct is not None else None,
        max_retries=int(row.max_retries),
        credential_source=CredentialSource(str(row.credential_source)),
        global_sharing_consent=bool(row.global_sharing_consent),
        consented_by_actor_subject=str(row.consented_by_actor_subject),
        consented_at=row.consented_at,
        credential_status=service.credential_status(),
        configured_at=row.created_at,
        updated_at=row.updated_at,
    )


def _written_state(record: GlobalMemoryEmbeddingAuthorityRecord) -> AuthorityState:
    """Render the write's own detached result instead of re-reading the row.

    A delete landing right after the commit must not turn a durable write into
    an error, and the credential was encrypted inside that same transaction.
    """
    return AuthorityState.model_validate(record).model_copy(
        update={"credential_status": "configured"}
    )


@router.get("", response_model=AuthorityState)
def get_authority(
    _admin: User = Depends(require_admin), db: Session = Depends(get_db)
) -> AuthorityState:
    return _public_state(GlobalMemoryEmbeddingAuthorityService(db))


@router.put("", response_model=AuthorityState)
def set_authority(
    request: AuthorityConfiguration,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> AuthorityState:
    actor_subject = str(admin.actor_subject or "")
    if not actor_subject:
        raise HTTPException(409, detail="Admin actor identity is unavailable")
    service = GlobalMemoryEmbeddingAuthorityService(db)
    try:
        record = service.set(request, actor_subject=actor_subject)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(400, detail=str(exc)) from exc
    except RuntimeError:
        db.rollback()
        raise HTTPException(
            503, detail="Global memory embedding authority could not be stored"
        ) from None
    return _written_state(record)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def delete_authority(
    _admin: User = Depends(require_admin), db: Session = Depends(get_db)
) -> Response:
    GlobalMemoryEmbeddingAuthorityService(db).delete()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
