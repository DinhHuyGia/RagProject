import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def workspace_temp_path() -> Iterator[Path]:
    """Create a unique test directory without using the Windows temp folder."""
    workspace_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(
        prefix=".test_tmp_",
        dir=workspace_root,
    ) as directory:
        yield Path(directory)

