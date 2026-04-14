import numpy as np
from omnigibson.learning.datas import BehaviorLeRobotDataset
from tqdm import tqdm

from openpi.policies import policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config

print("loading policy")
# TODO: change to the actual checkpoint dir
checkpoint_dir = "outputs/checkpoints/pi05_b1k/openpi/49999"
policy = policy_config.create_trained_policy(config.get_config("pi05_b1k-base"), checkpoint_dir)
openpi_policy = B1KPolicyWrapper(policy, control_mode="receeding_horizon")

ds = BehaviorLeRobotDataset(
    repo_id="rollout/demo",
    root="openpi",
    tasks=["turning_on_radio"],
    modalities=["rgb"],
    local_only=True,
    shuffle=False,
)


def get_action(idx: int):
    data = ds[idx]
    gt_action = data["action"]
    example = {
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": data["observation.images.rgb.head"].permute(1, 2, 0).numpy(),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": data["observation.images.rgb.left_wrist"]
        .permute(1, 2, 0)
        .numpy(),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": data["observation.images.rgb.right_wrist"]
        .permute(1, 2, 0)
        .numpy(),
        "robot_r1::proprio": data["observation.state"],
    }
    return openpi_policy.act(example), gt_action


gt_actions = []
pred_actions = []

print("start testing")
for idx in tqdm(range(len(ds))):
    action, gt_action = get_action(idx)
    print(action.shape, gt_action.shape)
    gt_actions.append(gt_action)
    pred_actions.append(action)

gt_actions = np.stack(gt_actions, axis=0)
pred_actions = np.stack(pred_actions, axis=0)

# plot the actions vs gt actions on dim = 23
import matplotlib.pyplot as plt

dims = 23
fig, axes = plt.subplots(dims, 1, figsize=(3, 4 * dims))
for i in range(dims):
    axes[i].plot(pred_actions[:, i], label="pred action")
    axes[i].plot(gt_actions[:, i], label="gt action")
    axes[i].set_title(f"dim {i}")
    axes[i].set_xlabel("step")
    axes[i].set_ylabel("action")
    axes[i].legend()

plt.savefig("actions.png")
