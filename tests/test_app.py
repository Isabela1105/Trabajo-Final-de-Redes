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


def test_examples_are_listed_and_served(trained_checkpoint: Path, tmp_path: Path) -> None:
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "foto_001_v0.png")
    (tmp_path / "notas.txt").write_text("no es imagen", encoding="utf-8")
    client = TestClient(create_app(trained_checkpoint, examples_dirs=[tmp_path]))
    items = client.get("/api/examples").json()
    assert [i["name"] for i in items] == ["foto_001_v0.png"]
    assert client.get(items[0]["url"]).status_code == 200
    assert client.get("/api/examples/..%2Fnotas.txt").status_code == 404


def test_restore_large_jpeg_and_rgba(client: TestClient) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (4000, 3000), (80, 90, 100)).save(buffer, format="JPEG")
    res = client.post("/api/restore", files={"file": ("big.jpg", buffer.getvalue(), "image/jpeg")})
    assert res.status_code == 200
    assert max(Image.open(io.BytesIO(res.content)).size) <= 4096
    res = client.post("/api/restore", files={"file": ("a.png", _png(mode="RGBA"), "image/png")})
    assert res.status_code == 200


def test_example_original_and_metrics(trained_checkpoint: Path, tmp_path: Path) -> None:
    """Un ejemplo del test con original: se sirve la original y el restore devuelve métricas."""
    import json

    (tmp_path / "input").mkdir()
    (tmp_path / "target").mkdir()
    clean = np.full((64, 64, 3), 140, np.uint8)
    noisy = np.clip(clean + np.random.default_rng(0).integers(-40, 41, clean.shape), 0, 255)
    Image.fromarray(noisy.astype(np.uint8)).save(tmp_path / "input" / "a.png")
    Image.fromarray(clean).save(tmp_path / "target" / "a.png")
    Image.fromarray(clean).save(tmp_path / "input" / "sin_original.png")
    client = TestClient(create_app(trained_checkpoint, examples_dirs=[tmp_path / "input"]))

    items = {i["name"]: i for i in client.get("/api/examples").json()}
    assert items["a.png"]["original_url"] == "/api/examples/a.png/original"
    assert items["sin_original.png"]["original_url"] is None
    assert client.get("/api/examples/a.png/original").status_code == 200
    assert client.get("/api/examples/sin_original.png/original").status_code == 404

    data = (tmp_path / "input" / "a.png").read_bytes()
    res = client.post(
        "/api/restore", files={"file": ("a.png", data, "image/png")}, data={"example": "a.png"}
    )
    assert res.status_code == 200
    metrics = json.loads(res.headers["X-Metricas"])
    assert set(metrics) == {"entrada", "restaurada"}
    assert metrics["entrada"]["psnr"] > 0 and 0 <= metrics["restaurada"]["ssim"] <= 1
    plain = client.post("/api/restore", files={"file": ("a.png", data, "image/png")})
    assert "X-Metricas" not in plain.headers


def test_web_files_are_not_cached(client: TestClient) -> None:
    for path in ("/", "/static/app.js", "/static/styles.css"):
        assert "no-cache" in client.get(path).headers["cache-control"]
    assert "?v=" in client.get("/").text  # los recursos llevan versión
