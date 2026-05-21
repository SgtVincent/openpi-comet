import os

from omnigibson.envs import Environment
from omnigibson.envs import EnvironmentWrapper
from omnigibson.learning.utils.eval_utils import HEAD_RESOLUTION
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.eval_utils import WRIST_RESOLUTION
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
logger = create_module_logger("RGBWrapper")


class RGBWrapper(EnvironmentWrapper):
    """
    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        if os.environ.get("PERSISTENT_EVAL_DISABLE_SENSOR_RECONFIG", "0").lower() in {"1", "true", "yes"}:
            logger.warning("PERSISTENT_EVAL_DISABLE_SENSOR_RECONFIG is set; skip RGBWrapper sensor reconfiguration")
            env.load_observation_space()
            logger.info("Reloaded observation space!")
            return
        # Note that from eval.py we only set rgb modality, here we include more (depth + seg_instance_id)
        # Here, we change the camera resolution and head camera aperture to match the one we used in data collection
        robot = env.robots[0]
        # Update robot sensors:
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            sensor = robot.sensors.get(sensor_name)
            if sensor is None:
                logger.warning(f"Sensor '{sensor_name}' not found on robot; skip sensor reconfiguration")
                continue
            try:
                if camera_id == "head":
                    sensor.horizontal_aperture = 40.0
                    sensor.set_image_resolution(width=HEAD_RESOLUTION[1], height=HEAD_RESOLUTION[0])
                else:
                    sensor.set_image_resolution(width=WRIST_RESOLUTION[1], height=WRIST_RESOLUTION[0])
            except Exception as e:  # noqa: BLE001
                # Some Isaac/Replicator versions can fail while mutating render
                # product settings in headless persistent eval. Keep default
                # sensor settings instead of failing the whole rollout.
                logger.warning(
                    f"Failed to reconfigure sensor '{sensor_name}' (camera_id='{camera_id}'): "
                    f"{type(e).__name__}: {e}. Keep defaults."
                )
                continue
            # # add depth and segmentation
            # robot.sensors[sensor_name].add_modality("depth_linear")
            # robot.sensors[sensor_name].add_modality("seg_semantic")
            # robot.sensors[sensor_name].add_modality("seg_instance_id")
        # reload observation space
        env.load_observation_space()
        logger.info("Reloaded observation space!")
