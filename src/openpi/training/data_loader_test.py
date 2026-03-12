import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
        framework="pytorch",
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, framework="pytorch")
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
        num_workers=2,
        framework="pytorch",
    )
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_create_data_loader_fake_pytorch_parallel():
    model = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    config = _config.TrainConfig(
        name="pytest_fake_pytorch",
        exp_name="pytest",
        model=model,
        data=_config.FakeDataConfig(),
        batch_size=8,
        num_workers=2,
        num_train_steps=2,
        wandb_enabled=False,
    )

    loader = _data_loader.create_data_loader(config, framework="pytorch", skip_norm_stats=True, num_batches=2)
    batches = list(loader)
    assert len(batches) == 2

    for observation, actions in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(observation.to_dict()))
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_create_data_loader_fake_jax_single_process():
    model = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    config = _config.TrainConfig(
        name="pytest_fake_jax",
        exp_name="pytest",
        model=model,
        data=_config.FakeDataConfig(),
        batch_size=8,
        num_workers=0,
        num_train_steps=2,
        wandb_enabled=False,
    )

    loader = _data_loader.create_data_loader(config, framework="jax", skip_norm_stats=True, num_batches=2)
    batches = list(loader)
    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_prompt_from_lerobot_item_drops_subtask_text_by_default():
    transform = _transforms.PromptFromLeRobotItem()

    data = transform({"task": "make_pizza", "subtask_text": "pick_up_dough", "state": 1})

    assert data["prompt"] == "make_pizza"
    assert "subtask_text" not in data
    assert data["state"] == 1


def test_prompt_from_lerobot_item_keeps_subtask_text_when_enabled():
    transform = _transforms.PromptFromLeRobotItem(include_subtask_text=True)

    data = transform({"task": "make_pizza", "subtask_text": "pick_up_dough"})

    assert data["prompt"] == "make_pizza"
    assert data["subtask_text"] == "pick_up_dough"


def test_subtask_text_routing_matches_model_type():
    baseline = _config.get_config("pi05_b1k-make_pizza_lr2.5e-6_5ep_sft")
    subtask = _config.get_config("pi05_subtask_b1k-make_pizza_lr2.5e-6_5ep_sft")

    assert baseline.model.model_type.name == "PI05"
    assert not _data_loader._include_subtask_text(baseline.model)

    assert subtask.model.model_type.name == "PI05_SUBTASK"
    assert _data_loader._include_subtask_text(subtask.model)
    assert subtask.pytorch_model_name == "subtask"
