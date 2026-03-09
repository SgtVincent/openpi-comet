import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# Keep the minimal R1Pro proprio layout locally so B1K policy setup does not
# import the full OmniGibson package during training/config startup.
PROPRIOCEPTION_INDICES = {
    "R1Pro": {
        "arm_left_qpos": np.s_[158:165],
        "gripper_left_qpos": np.s_[193:195],
        "arm_right_qpos": np.s_[197:204],
        "gripper_right_qpos": np.s_[232:234],
        "trunk_qpos": np.s_[236:240],
        "base_qvel": np.s_[253:256],
    }
}

MAX_DEPTH = 10.0

CAMERA_INTRINSICS = {
    "head": np.array([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),  # 720x720
    "left_wrist": np.array(
        [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
    ),  # 480x480
    "right_wrist": np.array(
        [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
    ),  # 480x480
}


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(23),
        "prompt": "do something",
    }


def extract_state_from_proprio(proprio_data):
    """
    We assume perfect correlation for the two gripper fingers.
    """
    # extract joint position
    base_qvel = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]]  #  7
    arm_right_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]]  #  7
    left_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(
        axis=-1, keepdims=True
    )  # 1
    right_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(
        axis=-1, keepdims=True
    )  # 1
    return np.concatenate(
        [
            base_qvel,
            trunk_qpos,
            arm_left_qpos,
            # left_gripper_width,
            arm_right_qpos,
            left_gripper_width,  # NOTE: we rearrange the gripper from 21 to 14 to match the action space
            right_gripper_width,
        ],
        axis=-1,
    )


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_seg_image(image) -> np.ndarray:
    MAX_SEG = 8
    image = np.asarray(image)
    image = image / MAX_SEG * 255
    image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    return image.astype(np.uint8)


def depth_to_pcd(depth_image: np.ndarray, camera_intrinsics: np.ndarray, downsample: int = 6) -> np.ndarray:
    """
    Convert depth image to point cloud.
    """
    depth_image = np.asarray(depth_image)
    h, w = depth_image.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
    y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
    z = depth_image
    pcd_xyz = np.stack([x, y, z], axis=-1)  # (h, w, 3)

    pcd_xyz = pcd_xyz[::downsample, ::downsample].reshape(16, -1, 3)
    return pcd_xyz  # (16, 2025, 3)


@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    meta_image_keys: list[str] = dataclasses.field(default_factory=list)

    depth_as_pcd: bool = False

    pcd_downsample: int = 6

    def __call__(self, data: dict) -> dict:
        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data)
        if "actions" in data:
            action = data["actions"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        meta_images, meta_image_names = [], []

        if self.depth_as_pcd:
            depth_image = data["observation/egocentric_depth"]
            pcd_xyz = depth_to_pcd(depth_image, CAMERA_INTRINSICS["head"], self.pcd_downsample)

        if "observation/egocentric_seg" in self.meta_image_keys:
            seg_image = _parse_seg_image(data["observation/egocentric_seg"])
            meta_images.append(seg_image)
            meta_image_names.append("base_0_seg")

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05 | _model.ModelType.PI05_HYBRID:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        names += tuple(meta_image_names)
        images += tuple(meta_images)
        image_masks += tuple(np.True_ for _ in meta_image_names)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "subtask_text" in data:
            inputs["subtask_text"] = data["subtask_text"]

        if self.depth_as_pcd:
            inputs["pcd_xyz"] = pcd_xyz
        return inputs


@dataclasses.dataclass(frozen=True)
class B1kOutputs(transforms.DataTransformFn):
    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        # Only return the first 23 dims.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
