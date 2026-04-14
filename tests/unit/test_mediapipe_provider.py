from __future__ import annotations

import sys
from types import ModuleType

import pytest

from visionbeat.mediapipe_provider import _temporary_sounddevice_stub


def test_temporary_sounddevice_stub_installs_and_cleans_up(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)

    with _temporary_sounddevice_stub():
        assert "sounddevice" in sys.modules
        stub = sys.modules["sounddevice"]
        with pytest.raises(ImportError, match="MediaPipe audio Tasks cannot be used"):
            stub.InputStream()

    assert "sounddevice" not in sys.modules


def test_temporary_sounddevice_stub_preserves_existing_module(monkeypatch) -> None:
    existing = ModuleType("sounddevice")
    monkeypatch.setitem(sys.modules, "sounddevice", existing)

    with _temporary_sounddevice_stub():
        assert sys.modules["sounddevice"] is existing

    assert sys.modules["sounddevice"] is existing
