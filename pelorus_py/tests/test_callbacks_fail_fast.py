import subprocess
import sys
from pathlib import Path

import pytest
import pelorus


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "pelorus_c" / "examples"
PYTHON_EXAMPLES = REPO_ROOT / "pelorus_py" / "examples"


def test_try_recv_and_recv_fail_when_callback_set():
    lio = pelorus.Handle()
    lio.set_callback(lambda d: None)

    with pytest.raises(RuntimeError):
        lio.try_recv()

    with pytest.raises(RuntimeError):
        lio.recv()

    lio.clear_callback()


def test_async_callback_completes_for_synthetic_dataset(tmp_path):
    dataset = tmp_path / "synthetic_dataset.bin"
    subprocess.run(
        [sys.executable, str(EXAMPLES / "generate_synthetic_dataset.py"), str(dataset)],
        check=True,
        timeout=30,
    )

    completed = subprocess.run(
        [sys.executable, str(PYTHON_EXAMPLES / "async_callback.py"), str(dataset)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "Callback invoked!" in completed.stdout
    assert "scan_id=2" in completed.stdout
