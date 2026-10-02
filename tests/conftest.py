import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def small_run():
    from src.pipeline import run
    return run(n=12_000, seed=7, write=False)
