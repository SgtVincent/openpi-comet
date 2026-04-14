import torch

from openpi.models_pytorch.vlm2.dual_memory import DualMemoryModule


def test_dual_memory_runtime_state_roundtrip() -> None:
    torch.manual_seed(0)
    mem = DualMemoryModule(feature_dim=8, working_memory_size=2, episodic_memory_capacity=3, num_heads=2, hidden_dim=16)

    x0 = torch.randn(2, 4, 8)
    _ = mem(x0, update_memory=True)
    stats_after_1 = mem.get_memory_stats()
    assert stats_after_1["working_memory_count"] == 1
    assert stats_after_1["episodic_memory_count"] == 1

    x1 = torch.randn(2, 4, 8)
    _ = mem(x1, update_memory=True)
    stats_after_2 = mem.get_memory_stats()
    assert stats_after_2["working_memory_count"] == 2
    assert stats_after_2["episodic_memory_count"] == 2

    state = mem.get_runtime_state()

    mem.clear_runtime_state()
    cleared = mem.get_memory_stats()
    assert cleared["working_memory_count"] == 0
    assert cleared["episodic_memory_count"] == 0

    mem.set_runtime_state(state)
    restored = mem.get_memory_stats()
    assert restored["working_memory_count"] == 2
    assert restored["episodic_memory_count"] == 2


def test_dual_memory_set_runtime_state_none_clears() -> None:
    mem = DualMemoryModule(feature_dim=8, working_memory_size=2, episodic_memory_capacity=3, num_heads=2, hidden_dim=16)
    _ = mem(torch.randn(1, 2, 8), update_memory=True)
    mem.set_runtime_state(None)
    stats = mem.get_memory_stats()
    assert stats["working_memory_count"] == 0
    assert stats["episodic_memory_count"] == 0

