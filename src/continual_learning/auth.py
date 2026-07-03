from __future__ import annotations

import base64
import hmac
import os
from functools import wraps
from typing import Any, Callable

from flask import Response, current_app, request


ADMIN_USER_ENV = "DEEPFORMA_ADMIN_USER"
ADMIN_PASSWORD_ENV = "DEEPFORMA_ADMIN_PASSWORD"


def is_admin_auth_enabled() -> bool:
    return bool(os.getenv(ADMIN_USER_ENV) and os.getenv(ADMIN_PASSWORD_ENV))


def _unauthorized() -> Response:
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="Deepforma admin"'},
    )


def check_basic_auth() -> bool:
    admin_user = os.getenv(ADMIN_USER_ENV)
    admin_password = os.getenv(ADMIN_PASSWORD_ENV)
    if not admin_user or not admin_password:
        return False
    auth = request.authorization
    if auth and hmac.compare_digest(auth.username or "", admin_user) and hmac.compare_digest(auth.password or "", admin_password):
        return True
    header = request.headers.get("Authorization", "")
    if header.startswith("Basic "):
        try:
            userpass = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
            username, password = userpass.split(":", 1)
            return hmac.compare_digest(username, admin_user) and hmac.compare_digest(password, admin_password)
        except Exception:
            return False
    return False


def require_admin_auth(view: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(view)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not is_admin_auth_enabled():
            return _unauthorized()
        if not check_basic_auth():
            return _unauthorized()
        return view(*args, **kwargs)

    return wrapper
