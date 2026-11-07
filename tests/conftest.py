"""Shared setup for API-free regression tests."""

import socket
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in ("src", "scripts", "audits"):
    sys.path.insert(0, str(ROOT / directory))


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def connect(*args, **kwargs):
        raise AssertionError("Regression tests must not make network or model calls")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect)
