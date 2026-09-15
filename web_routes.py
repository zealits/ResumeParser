"""
Serves the browser UI from the same port as the API.

Plain static HTML plus a little JS that talks to the existing JSON endpoints -
no templating engine, so there is no new dependency and nothing to render
server-side. Page-specific config (Square's public application id, the plan
list) is fetched by the browser from /api/billing/config and /api/billing/plans.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
STATIC_DIR = WEB_DIR / "static"

router = APIRouter(tags=["Web UI"])


def _page(filename: str) -> FileResponse:
    """Serve one page, refusing anything that escapes the web directory."""
    path = (WEB_DIR / filename).resolve()
    if not str(path).startswith(str(WEB_DIR.resolve())) or not path.is_file():
        logger.error("UI page missing or outside web dir: %s", filename)
        raise HTTPException(status_code=404, detail="Page not found")
    # These pages are deploy-time assets, but they change on every release, so
    # revalidate rather than letting a browser pin a stale shell.
    return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------------
@router.get("/", include_in_schema=False, response_class=HTMLResponse)
async def landing():
    """Marketing landing page with the pricing section."""
    return _page("index.html")


@router.get("/login", include_in_schema=False, response_class=HTMLResponse)
async def login_page():
    """Customer login. Redirects to /dashboard, or the console for an admin."""
    return _page("login.html")


@router.get("/signup", include_in_schema=False, response_class=HTMLResponse)
async def signup_page():
    """Self-serve signup and checkout. Takes ?plan=<key>."""
    return _page("signup.html")


@router.get("/dashboard", include_in_schema=False, response_class=HTMLResponse)
async def dashboard_page():
    """
    Customer dashboard.

    Auth is enforced client-side for the shell and server-side on every data
    call, so an unauthenticated visitor gets the page but no data.
    """
    return _page("dashboard.html")


# ---------------------------------------------------------------------------
# Admin console
# ---------------------------------------------------------------------------
@router.get("/superadmin", include_in_schema=False, response_class=HTMLResponse)
async def superadmin_login():
    """System administrator sign-in."""
    return _page("superadmin.html")


@router.get("/superadmin/console", include_in_schema=False, response_class=HTMLResponse)
async def superadmin_console():
    """
    Admin console shell.

    Every endpoint it calls is behind `get_current_admin`, so a non-admin who
    loads this URL directly sees an empty console and 403s.
    """
    return _page("superadmin-console.html")


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Avoid a 404 in the log for every page view."""
    icon = STATIC_DIR / "favicon.ico"
    if icon.is_file():
        return FileResponse(icon)
    raise HTTPException(status_code=404, detail="No favicon")
