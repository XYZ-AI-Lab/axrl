import pytest


def test_sglang_boundary_token_returned_only_when_missing() -> None:
    try:
        from axrl.worker.sglang_worker import SGLangWorker
    except ImportError as exc:
        pytest.skip(f"SGLang worker import requires CUDA kernels on this node: {exc}")

    assert SGLangWorker._assistant_boundary_token_to_append([1, 2], 99) == 99
    assert SGLangWorker._assistant_boundary_token_to_append([], 99) == 99
    assert SGLangWorker._assistant_boundary_token_to_append([1, 99], 99) is None
    assert SGLangWorker._assistant_boundary_token_to_append([1, 2], None) is None
