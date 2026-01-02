from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [
                    jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)
                ]

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int) -> Dataset:
    """Create a dataset for training."""
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    args = {}

    if data_config.skill_list != ["all"]:
        args["skill_list"] = data_config.skill_list

    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tolerance_s=data_config.tolerance_s,
        tasks=data_config.tasks,
        modalities=data_config.modalities,
        local_only=True,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=True,
        shuffle=True,
        fine_grained_level=data_config.fine_grained_level,
        return_seg_instance=data_config.return_seg_instance,
        train_rgb_type=data_config.train_rgb_type,
        **args,
    )

    # fixed prompt hard coding
    dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotItem()])

    return dataset


def create_multi_behavior_dataset(
    data_configs: list[_config.DataConfig], sample_weights: list[float] | None, action_horizon: int
) -> Dataset:
    from behavior.learning.datas.dataset import MultiBehaviorLeRobotDataset

    datasets = [
        create_behavior_dataset(data_config, action_horizon) for data_config in data_configs
    ]
    return MultiBehaviorLeRobotDataset(datasets, sample_weights=sample_weights)


def transform_dataset(
    dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    seed_shift: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    if isinstance(config.data, list):
        data_configs = [config_.create(config.assets_dirs, config.model) for config_ in config.data]
        dataset = create_multi_behavior_dataset(
            data_configs,
            sample_weights=config.sample_weights,
            action_horizon=config.model.action_horizon,
        )
        data_config = data_configs[0]
    else:
        data_config = config.data.create(config.assets_dirs, config.model)
        dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon)

    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed + seed_shift,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_torch_behavior_data_loader(
    config: _config.TrainConfig,
    action_horizon: int,
    batch_size: int,
    *,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    if isinstance(config.data, list):
        data_configs = [config_.create(config.assets_dirs, config.model) for config_ in config.data]
        dataset = create_multi_behavior_dataset(
            data_configs,
            sample_weights=config.sample_weights,
            action_horizon=config.model.action_horizon,
        )
        data_config = data_configs[0]
    else:
        data_config = config.data.create(config.assets_dirs, config.model)
        dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon)

    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            drop_last=True,
        )
        local_batch_size = batch_size // torch.distributed.get_world_size()
    else:
        local_batch_size = batch_size

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_workers=num_workers,
        seed=seed,
        framework="pytorch",
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        from behavior.learning.datas.dataset import MultiBehaviorLeRobotDataset
        if jax.process_count() > 1:
            logging.info(f"Subsetting dataset for process {jax.process_index()}.")
            if isinstance(dataset._dataset, MultiBehaviorLeRobotDataset):
                for dataset_index in range(len(dataset._dataset.datasets)):
                    indices = list(range(jax.process_index(), len(dataset._dataset.datasets[dataset_index]._dataset.chunks), jax.process_count()))
                    dataset._dataset.datasets[dataset_index]._dataset.chunks = [dataset._dataset.datasets[dataset_index]._dataset.chunks[i] for i in indices]
                total_chunks = sum(len(dataset._dataset.datasets[dataset_index]._dataset.chunks) for dataset_index in range(len(dataset._dataset.datasets)))
                logging.info(
                    f"[P{jax.process_index()}] After subset, Dataset has {total_chunks} chunks."
                )
            else:
                indices = list(range(jax.process_index(), len(dataset._dataset._dataset.chunks), jax.process_count()))
                dataset._dataset._dataset.chunks = [dataset._dataset._dataset.chunks[i] for i in indices]
                logging.info(
                    f"[P{jax.process_index()}] After subset, Dataset has {len(dataset._dataset._dataset.chunks)} chunks."
                )

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        # For multi-process JAX training, each process should have a different seed
        process_seed = seed + jax.process_index()
        generator = torch.Generator()
        generator.manual_seed(process_seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration as e:
                    logging.info(f"Stop Iteration ... {e}")
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(
                        lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch
                    )
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
