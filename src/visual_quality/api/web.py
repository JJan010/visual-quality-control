"""Pliki interfejsu są obsługiwane przez ten sam serwer co API."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def install_ui(app: FastAPI) -> None:
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/assets", StaticFiles(directory=static_dir), name="ui-assets")

    @app.get("/", include_in_schema=False)
    async def inspection_page():
        return FileResponse(
            static_dir / "index.html",
            headers={"Cache-Control": "no-store"},
        )
