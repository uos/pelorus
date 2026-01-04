import pytest

import pelorus

def test_try_recv_and_recv_fail_when_callback_set():
    lio = pelorus.Handle()
    lio.set_callback(lambda d: None)

    with pytest.raises(RuntimeError):
        lio.try_recv()

    with pytest.raises(RuntimeError):
        lio.recv()

    # Cleanup
    lio.clear_callback()
