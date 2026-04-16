import torch

from openpi.models_pytorch.memory_baselines.hamlet_memory import HamletMemoryAdapter
from openpi.models_pytorch.memory_baselines.hamlet_memory import MomentTokenPool


def test_moment_token_pool_shape() -> None:
    pool = MomentTokenPool(num_tokens=4, feature_dim=8)
    tokens = pool(batch_size=3)
    assert tokens.shape == (3, 4, 8)


def test_hamlet_history_buffer_clips_to_configured_length() -> None:
    torch.manual_seed(0)
    mem = HamletMemoryAdapter(feature_dim=8, num_moment_tokens=4, history_length=2, num_heads=2, num_layers=1)

    for _ in range(3):
        _ = mem(torch.randn(1, 4, 8), update_memory=True)

    stats = mem.get_memory_stats()
    assert stats["history_count"] == 2
    assert mem.history_buffer is not None
    assert mem.history_buffer.shape == (1, 2, 4, 8)


def test_hamlet_gate_zero_keeps_output_equal_to_current_tokens() -> None:
    torch.manual_seed(0)
    mem = HamletMemoryAdapter(
        feature_dim=8,
        num_moment_tokens=4,
        history_length=2,
        num_heads=2,
        num_layers=1,
        gate_init=0.0,
    )
    x = torch.randn(2, 4, 8)
    out = mem(x, update_memory=False)
    torch.testing.assert_close(out, x)


def test_hamlet_runtime_state_roundtrip() -> None:
    torch.manual_seed(0)
    mem = HamletMemoryAdapter(feature_dim=8, num_moment_tokens=4, history_length=3, num_heads=2, num_layers=1)
    _ = mem(torch.randn(1, 4, 8), update_memory=True)
    _ = mem(torch.randn(1, 4, 8), update_memory=True)
    state = mem.get_runtime_state()

    mem.reset_runtime_state()
    assert mem.get_memory_stats()["history_count"] == 0

    mem.set_runtime_state(state)
    assert mem.get_memory_stats()["history_count"] == 2
