import abc
from collections.abc import Sequence
import dataclasses
from enum import Enum
from enum import auto
import logging
import pathlib
from typing import Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.vlm2_vla_config as vlm2_vla_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.b1k_policy as b1k_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None

    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None

    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None

    # Only used for B1K data loader.
    behavior_dataset_root: str = None

    # Action space for DROID dataset.
    action_space: DroidActionSpace | None = None

    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # episodes index to use for training
    episodes_index: list[int] | None = None

    # tasks to use for training
    tasks: list[str] | None = None

    # tasks to use for training
    modalities: list[str] = dataclasses.field(default_factory=lambda: ["rgb"])

    # tolerance decoding
    tolerance_s: float = 1e-4

    # fine-grained level of orchestrators to use for training
    fine_grained_level: int = (0,)  # 0, 1, 2

    # whether to return seg instance
    return_seg_instance: bool = False

    # type of rgb to use for training
    train_rgb_type: str = "regular"  # regular | box | point

    # skill list to use for training
    skill_list: list[str] = dataclasses.field(default_factory=lambda: ["all"])


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    rearrange_action_indices: Sequence[int] | None = None

    model_delta_action_mask: Sequence[int] | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        meta_input_transforms = []
        meta_output_transforms = []

        if self.model_delta_action_mask:
            delta_action_mask = _transforms.make_bool_mask(*self.model_delta_action_mask)
            meta_input_transforms.append(_transforms.DeltaActions(delta_action_mask))
            meta_output_transforms.append(_transforms.AbsoluteActions(delta_action_mask))

        if self.rearrange_action_indices is not None:
            meta_input_transforms.append(_transforms.ArrangeStateActions(indices=self.rearrange_action_indices))
            meta_output_transforms.append(_transforms.RearrangeStateActions(indices=self.rearrange_action_indices))

        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        *meta_input_transforms,
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                    outputs=[
                        *meta_output_transforms[::-1],  # inverse
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, (pi0_config.Pi0Config, vlm2_vla_config.VLM2VLAConfig))
                return _transforms.Group(
                    inputs=[
                        *meta_input_transforms,
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                    outputs=[
                        *meta_output_transforms[::-1],  # inverse
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        *meta_input_transforms,
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        *meta_output_transforms[::-1],  # inverse
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        ),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    # Meta image keys to use for training
    meta_image_keys: list[str] = dataclasses.field(default_factory=list)

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("action",)

    delta_action_mask: Sequence[int] | None = None

    subsample_action_stride: int = 1

    rearrange_action_indices: Sequence[int] | None = None

    model_delta_action_mask: Sequence[int] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/egocentric_camera": "observation.images.rgb.head",
                        "observation/wrist_image_left": "observation.images.rgb.left_wrist",
                        "observation/wrist_image_right": "observation.images.rgb.right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[b1k_policy.B1kInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[b1k_policy.B1kOutputs(action_dim=23)],
        )

        if self.subsample_action_stride > 1:
            data_transforms = data_transforms.push(
                inputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
                outputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
            )

        if self.delta_action_mask is not None:
            delta_action_mask = _transforms.make_bool_mask(*self.delta_action_mask)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(
            rearrange_action_indices=self.rearrange_action_indices,
            model_delta_action_mask=self.model_delta_action_mask,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_quantile_norm=True,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotB1KRGBDDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("action",)

    delta_action_mask: Sequence[int] | None = None

    rearrange_action_indices: Sequence[int] | None = None

    model_delta_action_mask: Sequence[int] | None = None

    subsample_action_stride: int = 1

    depth_as_pcd: bool = False

    pcd_downsample: int = 9

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/egocentric_camera": "observation.images.rgb.head",
                        "observation/wrist_image_left": "observation.images.rgb.left_wrist",
                        "observation/wrist_image_right": "observation.images.rgb.right_wrist",
                        "observation/egocentric_depth": "observation.images.depth.head",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                b1k_policy.B1kInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    meta_image_keys=self.meta_image_keys,
                    depth_as_pcd=self.depth_as_pcd,
                    pcd_downsample=self.pcd_downsample,
                )
            ],
            outputs=[b1k_policy.B1kOutputs(action_dim=23)],
        )

        if self.subsample_action_stride > 1:
            data_transforms = data_transforms.push(
                inputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
                outputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
            )

        if self.delta_action_mask is not None:
            delta_action_mask = _transforms.make_bool_mask(*self.delta_action_mask)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(
            rearrange_action_indices=self.rearrange_action_indices,
            model_delta_action_mask=self.model_delta_action_mask,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_quantile_norm=True,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotB1KRGBSegmentationDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("action",)

    delta_action_mask: Sequence[int] | None = None

    rearrange_action_indices: Sequence[int] | None = None

    model_delta_action_mask: Sequence[int] | None = None

    subsample_action_stride: int = 1

    depth_as_pcd: bool = False

    pcd_downsample: int = 9

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/egocentric_camera": "observation.images.rgb.head",
                        "observation/wrist_image_left": "observation.images.rgb.left_wrist",
                        "observation/wrist_image_right": "observation.images.rgb.right_wrist",
                        "observation/egocentric_seg": "observation.images.seg.head",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                b1k_policy.B1kInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    meta_image_keys=self.meta_image_keys,
                    depth_as_pcd=self.depth_as_pcd,
                    pcd_downsample=self.pcd_downsample,
                )
            ],
            outputs=[b1k_policy.B1kOutputs(action_dim=23)],
        )

        if self.subsample_action_stride > 1:
            data_transforms = data_transforms.push(
                inputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
                outputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
            )

        if self.delta_action_mask is not None:
            delta_action_mask = _transforms.make_bool_mask(*self.delta_action_mask)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(
            rearrange_action_indices=self.rearrange_action_indices,
            model_delta_action_mask=self.model_delta_action_mask,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_quantile_norm=True,
        )
