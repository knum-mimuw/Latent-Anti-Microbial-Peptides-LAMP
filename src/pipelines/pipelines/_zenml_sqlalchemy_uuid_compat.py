"""Work around ZenML 0.94 + SQLAlchemy 2 UUID binding.

``Client._get_entity_by_id_or_name_or_prefix`` passes a UUID *string* into
store ``get_*`` methods that forward to SQLAlchemy with a ``UUID``-typed column.
SQLAlchemy 2's bind processor then calls ``.hex`` on the value and fails for
``str``. Coerce valid UUID strings to ``uuid.UUID`` before the store call.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

_applied = False


def apply() -> None:
    """Patch ``Client._get_entity_by_id_or_name_or_prefix`` (idempotent)."""
    global _applied
    if _applied:
        return
    from zenml.client import Client
    from zenml.utils.uuid_utils import is_valid_uuid

    _orig = Client._get_entity_by_id_or_name_or_prefix

    def _patched(
        self: Any,
        get_method: Any,
        list_method: Any,
        name_id_or_prefix: Any,
        allow_name_prefix_match: bool = True,
        project: Any = None,
        hydrate: bool = True,
        **kwargs: Any,
    ) -> Any:
        if isinstance(name_id_or_prefix, str) and is_valid_uuid(name_id_or_prefix):
            name_id_or_prefix = UUID(name_id_or_prefix)
        return _orig(
            self,
            get_method,
            list_method,
            name_id_or_prefix,
            allow_name_prefix_match=allow_name_prefix_match,
            project=project,
            hydrate=hydrate,
            **kwargs,
        )

    Client._get_entity_by_id_or_name_or_prefix = _patched  # type: ignore[method-assign]
    _applied = True
