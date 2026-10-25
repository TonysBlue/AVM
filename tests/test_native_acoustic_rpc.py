from __future__ import annotations

import pytest
import torch
from runtime.codec import CodecIdentity
from runtime.native_acoustics import NativeAcousticClient

from tests.support.physical_workers import CodecWorker


def test_native_worker_rejects_identity_mismatch_and_concurrent_streams(tmp_path) -> None:
    worker = CodecWorker(
        tmp_path / "native.sock", CodecIdentity("test", "hash", "rev", codebook_size=32)
    )
    try:
        with pytest.raises(ValueError, match="identity"):
            NativeAcousticClient(worker.path, {"codec_id": "wrong"})
        first = NativeAcousticClient(worker.path, {"codec_id": "test"})
        second = NativeAcousticClient(worker.path, {"codec_id": "test"})
        first.reset()
        with pytest.raises(RuntimeError, match="active stream lease"):
            second.reset()
        second.close()  # An unowned release must not cancel the first stream.
        assert first.decode([1] * 8, final=False).shape == (960,)
        first.close()
        first.close()
        with NativeAcousticClient(worker.path, {"codec_id": "test"}) as replacement:
            replacement.reset()
            assert replacement.decode([2] * 8, final=False).shape == (960,)
            assert replacement.encode(torch.zeros(6400)).shape == (10,)
    finally:
        worker.close()


def test_native_rpc_rejects_eos_and_requires_explicit_stream_reset(tmp_path) -> None:
    worker = CodecWorker(
        tmp_path / "native.sock", CodecIdentity("test", "hash", "rev", codebook_size=32)
    )
    try:
        with NativeAcousticClient(worker.path, {}) as client:
            with pytest.raises(RuntimeError, match="reset before decode"):
                client.decode([1] * 8, final=False)
            client.reset()
            with pytest.raises(ValueError, match="EOS"):
                client.decode([31] * 8, final=False)
            assert client.decode([1] * 8, final=False).numel() == 960
    finally:
        worker.close()
