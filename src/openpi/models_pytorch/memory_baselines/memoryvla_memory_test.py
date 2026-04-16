import torch

from openpi.models_pytorch.memory_baselines.memoryvla_memory import MemoryVLAModule


def test_memoryvla_bank_insert_and_capacity_clip() -> None:
    torch.manual_seed(0)
    mem = MemoryVLAModule(feature_dim=8, bank_capacity=2, similarity_threshold=0.95)
    _ = mem(torch.randn(1, 1, 8), update_memory=True)
    _ = mem(torch.randn(1, 1, 8), update_memory=True)
    _ = mem(torch.randn(1, 1, 8), update_memory=True)
    assert mem.get_memory_stats()["memory_count"] == 2


def test_memoryvla_retrieval_shape() -> None:
    torch.manual_seed(0)
    mem = MemoryVLAModule(feature_dim=8, bank_capacity=4)
    _ = mem(torch.randn(2, 1, 8), update_memory=True)
    query = torch.randn(2, 8)
    retrieved = mem.retrieve(query)
    assert retrieved is not None
    assert retrieved.shape == (2, 8)


def test_memoryvla_gate_range() -> None:
    torch.manual_seed(0)
    mem = MemoryVLAModule(feature_dim=8, bank_capacity=4)
    current = torch.randn(2, 1, 8)
    _ = mem(current, update_memory=True)
    _, gate = mem(current, update_memory=False)
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)


def test_memoryvla_runtime_state_roundtrip() -> None:
    torch.manual_seed(0)
    mem = MemoryVLAModule(feature_dim=8, bank_capacity=4)
    _ = mem(torch.randn(1, 1, 8), update_memory=True)
    _ = mem(torch.randn(1, 1, 8), update_memory=True)
    state = mem.get_runtime_state()
    mem.reset_runtime_state()
    assert mem.get_memory_stats()["memory_count"] == 0
    mem.set_runtime_state(state)
    assert mem.get_memory_stats()["memory_count"] == 2
