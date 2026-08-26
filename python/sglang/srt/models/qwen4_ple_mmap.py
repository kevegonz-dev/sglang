"""Guarded file-backed storage for Qwen4 PLE tables on GB10."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Iterable

import torch

_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
_MIN_FREE_BYTES_AFTER_ALLOCATION = 1024**3
_METADATA_VERSION = 1


def _shape_tuple(shape: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(dimension) for dimension in shape)
    if not result or any(dimension <= 0 for dimension in result):
        raise ValueError(f"PLE mmap shape must be non-empty and positive: {result}")
    return result


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float8_e4m3fn:
        return "float8_e4m3fn"
    raise TypeError(f"PLE mmap supports bfloat16 or fp8 weights, got {dtype}")


def require_gb10_coherent_pageable_memory(device: int = 0) -> None:
    """Fail closed unless a Linux SM121 CUDA device uses host page tables."""

    if sys.platform != "linux":
        raise RuntimeError("PLE mmap requires Linux")
    if not torch.cuda.is_available():
        raise RuntimeError("PLE mmap requires CUDA")
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != (12, 1):
        raise RuntimeError(f"PLE mmap is guarded to GB10 SM121, found {capability}")

    try:
        cudart = ctypes.CDLL("libcudart.so")
        value = ctypes.c_int()
        result = cudart.cudaDeviceGetAttribute(
            ctypes.byref(value),
            ctypes.c_int(_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES),
            ctypes.c_int(device),
        )
    except (AttributeError, OSError) as error:
        raise RuntimeError("could not query CUDA pageable-memory coherence") from error
    if result != 0:
        raise RuntimeError(
            "cudaDeviceGetAttribute(PageableMemoryAccessUsesHostPageTables) "
            f"failed with CUDA error {result}"
        )
    if value.value != 1:
        raise RuntimeError(
            "PLE mmap requires "
            "cudaDevAttrPageableMemoryAccessUsesHostPageTables == 1"
        )


def validate_ple_mmap_directory(path: str) -> Path:
    if not path:
        raise ValueError("--ple-mmap-dir is required for the mmap backend")
    directory = Path(path)
    if not directory.is_absolute():
        raise ValueError("PLE mmap directory must be absolute")
    if directory == Path("/"):
        raise ValueError("PLE mmap directory cannot be the filesystem root")
    try:
        resolved = directory.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("PLE mmap directory must already exist") from error
    if resolved != directory:
        raise ValueError("PLE mmap directory cannot contain symlinks")
    info = directory.stat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("PLE mmap path must be a directory")
    if info.st_uid != os.geteuid():
        raise PermissionError("PLE mmap directory must be owned by the serving user")
    if info.st_mode & 0o022:
        raise PermissionError("PLE mmap directory cannot be group/world writable")
    return directory


def _write_json_exclusive(path: Path, payload: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _validate_regular_owned_file(path: Path, expected_bytes: int) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"PLE mmap backing file is not a regular file: {path}")
    if info.st_uid != os.geteuid():
        raise PermissionError("PLE mmap backing file must be owned by the serving user")
    if info.st_mode & 0o077:
        raise PermissionError("PLE mmap backing file permissions must be 0600")
    if info.st_size != expected_bytes:
        raise ValueError(
            f"PLE mmap backing file has wrong size: {info.st_size} != {expected_bytes}"
        )


class Qwen4PLEMMapStorage:
    """Own one deterministic mmap file and its fail-closed load markers."""

    def __init__(
        self,
        *,
        directory: str,
        shape: Iterable[int],
        dtype: torch.dtype,
        identity: str,
    ) -> None:
        require_gb10_coherent_pageable_memory()
        self.directory = validate_ple_mmap_directory(directory)
        self.shape = _shape_tuple(shape)
        self.dtype = dtype
        dtype_name = _dtype_name(dtype)
        safe_characters = "-_.0123456789abcdefghijklmnopqrstuvwxyz"
        if not identity or any(
            character not in safe_characters for character in identity
        ):
            raise ValueError(f"invalid internal PLE mmap identity: {identity!r}")
        self.numel = 1
        for dimension in self.shape:
            self.numel *= dimension
        self.nbytes = self.numel * torch.empty((), dtype=dtype).element_size()
        dimensions = "x".join(str(value) for value in self.shape)
        self.path = self.directory / f"{identity}-{dtype_name}-{dimensions}.mmap"
        self.initializing_path = Path(f"{self.path}.initializing.json")
        self.ready_path = Path(f"{self.path}.ready.json")
        self.metadata = {
            "version": _METADATA_VERSION,
            "shape": list(self.shape),
            "dtype": dtype_name,
            "nbytes": self.nbytes,
        }

        if self.initializing_path.exists():
            raise RuntimeError(
                "PLE mmap has an interrupted initialization marker; remove the "
                "lane-owned backing file and marker before rebuilding"
            )
        self._validate_disk()
        self._prepare_file()
        _write_json_exclusive(self.initializing_path, self.metadata)
        try:
            raw = torch.from_file(
                str(self.path), shared=True, size=self.nbytes, dtype=torch.uint8
            )
            self.tensor = raw.view(dtype).view(*self.shape)
            self._apply_madv_random(raw.data_ptr())
        except BaseException:
            # The marker deliberately remains: a later start must not mistake
            # a possibly partial table for completed state.
            raise

    def _validate_disk(self) -> None:
        free = shutil.disk_usage(self.directory).free
        required = _MIN_FREE_BYTES_AFTER_ALLOCATION
        if not self.path.exists():
            required += self.nbytes
        if free < required:
            raise OSError(
                errno.ENOSPC,
                "insufficient disk for PLE mmap plus the 1 GiB safety floor",
                str(self.directory),
            )

    def _prepare_file(self) -> None:
        if self.path.exists():
            _validate_regular_owned_file(self.path, self.nbytes)
            if not self.ready_path.is_file():
                raise RuntimeError("existing PLE mmap is missing its ready metadata")
            try:
                ready = json.loads(self.ready_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError("invalid PLE mmap ready metadata") from error
            if ready != self.metadata:
                raise ValueError("PLE mmap ready metadata does not match dtype/shape")
            return
        if self.ready_path.exists():
            raise RuntimeError(
                "PLE mmap ready metadata exists without its backing file"
            )

        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            os.ftruncate(descriptor, self.nbytes)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _validate_regular_owned_file(self.path, self.nbytes)

    def _apply_madv_random(self, pointer: int) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        madvise = getattr(libc, "madvise", None)
        if madvise is None:
            return
        result = madvise(
            ctypes.c_void_p(pointer),
            ctypes.c_size_t(self.nbytes),
            ctypes.c_int(1),  # POSIX_MADV_RANDOM / MADV_RANDOM on Linux
        )
        if result != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(self.path))

    def finalize(self) -> None:
        """Make completed state durable only after every expected shard loaded."""

        _validate_regular_owned_file(self.path, self.nbytes)
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        temporary_ready = Path(f"{self.ready_path}.tmp-{os.getpid()}")
        _write_json_exclusive(temporary_ready, self.metadata)
        os.replace(temporary_ready, self.ready_path)
        self.initializing_path.unlink()
        directory_fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
