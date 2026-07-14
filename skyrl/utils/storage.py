import gzip
import io
import os
import shutil
import tarfile
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Generator, Optional

from cloudpathlib import AnyPath

from skyrl.utils.log import logger


@contextmanager
def pack_and_upload(dest: AnyPath, rank: Optional[int] = None) -> Generator[Path, None, None]:
    """Give the caller a temp directory that gets uploaded as a tar.gz archive on exit.

    Args:
        dest: Destination path for the tar.gz file
        rank: Process rank for multi-rank deduplication. If provided and a probe
              file exists at {dest}.probe, only rank 0 writes.
    """
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        yield tmp_path

        # If probe file exists, filesystem is shared - only rank 0 should write
        if rank is not None and rank != 0 and dest.with_name(dest.name + ".probe").exists():
            logger.info(f"Skipping write to {dest} (shared filesystem, rank {rank})")
            return

        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open("wb") as f:
            # Use compresslevel=0 to prioritize speed, as checkpoint files don't compress well.
            with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=0) as gz_stream:
                with tarfile.open(fileobj=gz_stream, mode="w:") as tar:
                    tar.add(tmp_path, arcname="")


@contextmanager
def write_and_publish_dir(dest: AnyPath, rank: Optional[int] = None) -> Generator[Path, None, None]:
    """Give the caller a temp directory that gets atomically published to ``dest`` (a directory).

    This is the un-tarred analogue of :func:`pack_and_upload`: instead of packing the
    caller's files into a ``dest`` tar.gz, it publishes them as a plain directory at
    ``dest`` via a single ``os.rename``. External inference engines (vLLM) load a LoRA
    adapter from a directory path, so writing the directory directly avoids the
    pack -> read-back -> un-tar round-trip that ``pack_and_upload`` +
    ``download_and_unpack`` would otherwise incur.

    The staging directory is created on ``dest``'s own filesystem so the publish is an
    atomic ``rename`` (not a cross-device copy). This is what lets a concurrent reader
    racing to load a freshly-written adapter see either no ``dest`` or the complete one,
    never a half-written directory.

    Args:
        dest: Destination directory path for the published contents.
        rank: Process rank for multi-rank deduplication. If provided and a probe
              file exists at {dest}.probe, only rank 0 publishes.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(mkdtemp(dir=str(dest.parent), prefix=f".{dest.name}.tmp."))
    try:
        yield tmp

        # If probe file exists, filesystem is shared - only rank 0 should publish.
        if rank is not None and rank != 0 and dest.with_name(dest.name + ".probe").exists():
            logger.info(f"Skipping write to {dest} (shared filesystem, rank {rank})")
            return

        # Idempotent re-save / retry with the same (deterministic) content: if dest
        # already exists, treat it as already-published. The DB checkpoint status is
        # the readiness source of truth, not the file layout.
        if dest.exists():
            return

        # Atomic on the same filesystem; a concurrent reader never sees a partial dir.
        # Tolerate a lost publish race (e.g. the Ray path where no .probe is written
        # and every rank publishes): if another writer created dest first, os.rename
        # raises (ENOTEMPTY/EEXIST) and we simply keep the winner's copy.
        try:
            os.rename(tmp, dest)
        except OSError:
            if not dest.exists():
                raise
    finally:
        # No-op if the staging dir was renamed away; cleans it up otherwise.
        shutil.rmtree(tmp, ignore_errors=True)


@contextmanager
def download_and_unpack(source: AnyPath, scratch_dir: Optional[Path] = None) -> Generator[Path, None, None]:
    """Download and extract a tar.gz archive and give the content to the caller in a temp directory.

    Args:
        source: Source path for the tar.gz file
        scratch_dir: Directory under which to extract. Pass a path on the
              destination's filesystem when the caller publishes the result via
              an atomic rename (otherwise a cross-device move silently degrades
              to a non-atomic copy); defaults to the system temp dir.
    """
    with TemporaryDirectory(dir=str(scratch_dir) if scratch_dir is not None else None) as tmp:
        tmp_path = Path(tmp)

        # Download and extract tar archive (handles both local and cloud storage).
        # "r:*" auto-detects compression, tolerating both gzip archives and any
        # plain tars that may exist on disk under the historical .tar.gz suffix.
        with source.open("rb") as f:
            with tarfile.open(fileobj=f, mode="r:*") as tar:
                tar.extractall(tmp_path, filter="data")

        yield tmp_path


def read_as_archive(source: AnyPath) -> io.BytesIO:
    """Return the contents of ``source`` as a tar.gz archive in a BytesIO buffer.

    If ``source`` is a directory (the layout used for external-inference sampler
    adapters, published via :func:`write_and_publish_dir`), it is tarred on the
    fly so archive-download endpoints keep working. If it is a file, its bytes
    are returned verbatim (it is already an archive).
    """
    if source.is_dir():
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as gz_stream:
            with tarfile.open(fileobj=gz_stream, mode="w:") as tar:
                tar.add(str(source), arcname="")
        buffer.seek(0)
        return buffer
    return download_file(source)


def download_file(source: AnyPath) -> io.BytesIO:
    """Download a file from storage and return it as a BytesIO object.

    Args:
        source: Source path for the file (local or cloud)

    Returns:
        BytesIO object containing the file contents
    """
    buffer = io.BytesIO()
    with source.open("rb") as f:
        buffer.write(f.read())
    buffer.seek(0)
    return buffer
