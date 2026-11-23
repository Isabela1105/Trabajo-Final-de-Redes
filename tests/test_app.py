"""Pruebas de la demo web (FastAPI): carga del modelo y predicción.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.server import create_app  # noqa: E402


def _png(size: tuple[int, int] = (64, 64), mode: str = "RGB") -> bytes:
    channels = {"RGB": 3, "RGBA": 4}[mode]
    array = np.full((size[1], size[0], channels), 120, np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(trained_checkpoint: Path) -> TestClient:
    return TestClient(create_app(trained_checkpoint))


def test_health_before_and_after_loading(client: TestClient) -> None:
    assert client.get("/api/health").json()["model_loaded"] is False
    client.post("/api/restore", files={"file": ("x.png", _png(), "image/png")})
    assert client.get("/api/health").json()["model_loaded"] is True


def test_restore_returns_png_with_same_size(client: TestClient) -> None:
    res = client.post("/api/restore", files={"file": ("x.png", _png((90, 60)), "image/png")})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(res.content)).size == (90, 60)


def test_rejects_non_images(client: TestClient) -> None:
    res = client.post("/api/restore", files={"file": ("x.txt", b"hola", "text/plain")})
    assert res.status_code == 415


def test_rejects_empty_file(client: TestClient) -> None:
    res = client.post("/api/restore", files={"file": ("x.png", b"", "image/png")})
    assert res.status_code == 400


def test_without_checkpoint_returns_503() -> None:
    client = TestClient(create_app(None))
    res = client.post("/api/restore", files={"file": ("x.png", _png(), "image/png")})
    assert res.status_code == 503


def test_serves_web_page(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "/api/restore" in client.get("/static/app.js").text
