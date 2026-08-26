import errno
import json
from collections import namedtuple

import pytest
import torch

from sglang.srt.models import qwen4_ple_mmap
from sglang.srt.models.qwen4_ple_mmap import (
    Qwen4PLEMMapStorage,
    validate_ple_mmap_directory,
)


@pytest.fixture
def mmap_directory(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    monkeypatch.setattr(
        qwen4_ple_mmap, "require_gb10_coherent_pageable_memory", lambda: None
    )
    monkeypatch.setattr(
        Qwen4PLEMMapStorage, "_apply_madv_random", lambda self, pointer: None
    )
    return tmp_path


def make_storage(directory, *, shape=(32, 16), dtype=torch.float8_e4m3fn):
    return Qwen4PLEMMapStorage(
        directory=str(directory),
        shape=shape,
        dtype=dtype,
        identity="ple-layer-00-rows-0-32",
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_mmap_preserves_exact_dtype_shape_and_bytes(mmap_directory, dtype):
    storage = make_storage(mmap_directory, dtype=dtype)
    source = torch.arange(storage.nbytes, dtype=torch.uint8)
    storage.tensor.view(torch.uint8).reshape(-1).copy_(source)
    storage.finalize()

    reopened = make_storage(mmap_directory, dtype=dtype)
    assert reopened.reused
    assert not reopened.initializing_path.exists()
    assert reopened.tensor.dtype == dtype
    assert tuple(reopened.tensor.shape) == storage.shape
    assert torch.equal(reopened.tensor.view(torch.uint8).reshape(-1), source)
    reopened.finalize()


def test_ready_mmap_reopens_without_reinitializing(mmap_directory):
    storage = make_storage(mmap_directory)
    source = torch.arange(storage.nbytes, dtype=torch.uint8)
    storage.tensor.view(torch.uint8).reshape(-1).copy_(source)
    storage.finalize()
    ready_before = storage.ready_path.read_bytes()

    reopened = make_storage(mmap_directory)

    assert reopened.reused
    assert not reopened.initializing_path.exists()
    assert reopened.ready_path.read_bytes() == ready_before
    assert torch.equal(reopened.tensor.view(torch.uint8).reshape(-1), source)
    reopened.finalize()
    assert not reopened.initializing_path.exists()
    assert reopened.ready_path.read_bytes() == ready_before


def test_interrupted_initialization_fails_closed(mmap_directory):
    storage = make_storage(mmap_directory)
    assert storage.initializing_path.is_file()

    with pytest.raises(RuntimeError, match="interrupted initialization"):
        make_storage(mmap_directory)


def test_wrong_existing_file_size_fails_closed(mmap_directory):
    storage = make_storage(mmap_directory)
    storage.finalize()
    with storage.path.open("r+b") as backing:
        backing.truncate(storage.nbytes - 1)

    with pytest.raises(ValueError, match="wrong size"):
        make_storage(mmap_directory)


def test_bad_ready_metadata_fails_closed(mmap_directory):
    storage = make_storage(mmap_directory)
    storage.finalize()
    payload = json.loads(storage.ready_path.read_text())
    payload["dtype"] = "bfloat16"
    storage.ready_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not match dtype/shape"):
        make_storage(mmap_directory)


def test_low_disk_fails_before_creating_state(mmap_directory, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        qwen4_ple_mmap.shutil,
        "disk_usage",
        lambda _path: usage(total=1024, used=1024, free=0),
    )

    with pytest.raises(OSError) as error:
        make_storage(mmap_directory)
    assert error.value.errno == errno.ENOSPC
    assert list(mmap_directory.iterdir()) == []


@pytest.mark.parametrize(
    "shape,dtype,error",
    [
        ((0, 16), torch.float8_e4m3fn, ValueError),
        ((16, 16), torch.float16, TypeError),
    ],
)
def test_bad_shape_and_dtype_fail(mmap_directory, shape, dtype, error):
    with pytest.raises(error):
        make_storage(mmap_directory, shape=shape, dtype=dtype)


def test_directory_validation_rejects_unsafe_paths(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        validate_ple_mmap_directory("relative/path")

    world_writable = tmp_path / "world-writable"
    world_writable.mkdir(mode=0o777)
    world_writable.chmod(0o777)
    with pytest.raises(PermissionError, match="group/world writable"):
        validate_ple_mmap_directory(str(world_writable))

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        validate_ple_mmap_directory(str(link))


def test_gb10_guard_rejects_non_linux_and_non_sm121(monkeypatch):
    monkeypatch.setattr(qwen4_ple_mmap.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux"):
        qwen4_ple_mmap.require_gb10_coherent_pageable_memory()

    monkeypatch.setattr(qwen4_ple_mmap.sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    with pytest.raises(RuntimeError, match="GB10 SM121"):
        qwen4_ple_mmap.require_gb10_coherent_pageable_memory()
