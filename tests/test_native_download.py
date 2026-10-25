from __future__ import annotations

import hashlib
import runpy
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("lfs", [False, True])
def test_native_download_rejects_same_size_modified_source(tmp_path, monkeypatch, lfs) -> None:
    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace())
    matches = runpy.run_path("tools/native-speech/fetch_assets.py")["source_matches"]
    payload = b"locked official source"
    item = SimpleNamespace(
        size=len(payload),
        blob_id=hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
        lfs=SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()) if lfs else None,
    )
    path = tmp_path / "asset"
    assert not matches(path, item)
    path.write_bytes(payload)
    assert matches(path, item)
    path.write_bytes(b"X" + payload[1:])
    assert not matches(path, item)
