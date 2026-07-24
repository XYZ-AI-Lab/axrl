# pyright: strict
import logging
import pickle
import shutil
from pathlib import Path
from time import sleep
from typing import Any

import zstandard as zstd

logger = logging.getLogger(__name__)


def _save_zst(data: Any, filepath: Path, compression_level: int = 8) -> None:
    cctx = zstd.ZstdCompressor(level=compression_level)
    tmp_path = filepath.with_suffix(".tmp")
    with tmp_path.open("wb") as f:
        f.write(cctx.compress(pickle.dumps(data)))
    shutil.move(str(tmp_path), str(filepath))


def _load_zst(filepath: Path) -> Any:
    dctx = zstd.ZstdDecompressor()
    with filepath.open("rb") as f:
        decompressed_data = dctx.decompress(f.read())
    return pickle.loads(decompressed_data)


def save_zst(data: object, filepath: Path, compression_level: int = 8, num_retry: int = 10, *, verbose: bool = False) -> None:
    """Save pickle to zstd file."""
    for retry in range(num_retry + 1):
        try:
            _save_zst(data, filepath, compression_level)
            if verbose:
                logger.info(f"Saved {filepath} successfully.")
            return
        except Exception as _:
            sleep(retry)
            logger.exception(f"retry: {retry}, IO exception when saving pkl {filepath}")
    msg = f"failed to save {filepath} after {num_retry} retries."
    logger.error(msg)
    raise OSError(msg)


def load_zst(filepath: Path, num_retry: int = 10, *, verbose: bool = False) -> Any:
    """Load zstd file."""
    for retry in range(num_retry + 1):
        try:
            data = _load_zst(filepath)
            if verbose:
                logger.info(f"Loaded {filepath} successfully.")
            return data
        except Exception as _:
            sleep(retry)
            logger.exception(f"retry: {retry}, IO exception when loading pkl {filepath}")
    msg = f"failed to load {filepath} after {num_retry} retries."
    logger.error(msg)
    raise OSError(msg)
