"""Run the physical SO101 ACT color-sorting policy in the MuJoCo scene.

The policy was trained on LeRobot observations at 30 Hz:
  - observation.state: five calibrated joint angles in degrees and a gripper
    position normalized to 0-100
  - observation.images.front: 480x640 RGB
  - observation.images.overhead: 720x1280 RGB

MuJoCo stores all six joint positions in radians, so this runner performs the
LeRobot/MuJoCo conversion at the simulation boundary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import build_inference_frame


HERE = Path(__file__).resolve().parent
SCENE_PATH = HERE / "scene_workspace.xml"
DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\doris\.cache\huggingface\lerobot\robododo\place-rectangle-colored-box_20260718_195758"
)
DEFAULT_POLICY_CACHE = Path(
    r"C:\Users\doris\.cache\huggingface\hub\models--robododo--sort_blocks_by_color_act_v2"
)

DATASET_REPO_ID = "robododo/place-rectangle-colored-box"
FPS = 30
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Median state at frame zero across the 81 real training episodes. Starting the
# simulation here avoids giving ACT an unfamiliar initial arm configuration.
HOME_STATE = np.array(
    [-4.571, -103.824, 96.308, 78.901, -89.363, 1.270], dtype=np.float64
)


def resolve_snapshot(policy_path: Path) -> Path:
    """Accept either a pretrained_model directory or a Hugging Face cache."""
    policy_path = policy_path.expanduser().resolve()
    if (policy_path / "config.json").is_file():
        return policy_path

    revision_file = policy_path / "refs" / "main"
    if not revision_file.is_file():
        raise FileNotFoundError(
            f"Expected config.json or refs/main under policy path: {policy_path}"
        )
    revision = revision_file.read_text(encoding="utf-8").strip()
    snapshot = policy_path / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Policy snapshot does not exist: {snapshot}")
    return snapshot


class SO101MujocoPolicyEnv:
    """Small adapter exposing the real robot's LeRobot observation interface."""

    def __init__(self, seed: int = 0):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        # High-resolution policy cameras require a larger offscreen framebuffer
        # than MuJoCo's default.
        self.model.vis.global_.offwidth = 1280
        self.model.vis.global_.offheight = 720
        self.front_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.overhead_renderer = mujoco.Renderer(self.model, height=720, width=1280)

        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        )
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES]
        )
        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1]

        self.block_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "rectangle_obj"
        )
        self.block_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "rectangle_geom"
        )
        block_joint = self.model.body_jntadr[self.block_body]
        self.block_qpos_address = self.model.jnt_qposadr[block_joint]
        self.block_dof_address = self.model.jnt_dofadr[block_joint]
        self.tray_bodies = {
            color: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"tray_{color}")
            for color in ("green", "yellow")
        }
        self.block_color = "green"

    def lerobot_to_mujoco(self, state: np.ndarray) -> np.ndarray:
        """Map five degrees-valued joints plus 0-100 gripper to radians."""
        state = np.asarray(state, dtype=np.float64)
        target = np.empty(6, dtype=np.float64)
        target[:5] = np.deg2rad(state[:5])
        gripper_fraction = np.clip(state[5], 0.0, 100.0) / 100.0
        target[5] = self.ctrl_low[5] + gripper_fraction * (
            self.ctrl_high[5] - self.ctrl_low[5]
        )
        return np.clip(target, self.ctrl_low, self.ctrl_high)

    def mujoco_to_lerobot(self, qpos: np.ndarray) -> np.ndarray:
        """Map MuJoCo radians to the physical dataset's calibrated units."""
        qpos = np.asarray(qpos, dtype=np.float64)
        state = np.empty(6, dtype=np.float64)
        state[:5] = np.rad2deg(qpos[:5])
        state[5] = 100.0 * (qpos[5] - self.ctrl_low[5]) / (
            self.ctrl_high[5] - self.ctrl_low[5]
        )
        state[5] = np.clip(state[5], 0.0, 100.0)
        return state

    def reset(
        self,
        initial_state: np.ndarray | None = None,
        block_xy: np.ndarray | None = None,
        block_yaw: float | None = None,
        block_color: str | None = None,
    ) -> dict[str, np.ndarray | float]:
        mujoco.mj_resetData(self.model, self.data)

        start_state = HOME_STATE if initial_state is None else np.asarray(initial_state)
        home_qpos = self.lerobot_to_mujoco(start_state)
        self.data.qpos[self.qpos_addresses] = home_qpos
        self.data.ctrl[self.actuator_ids] = home_qpos

        self.block_color = block_color or str(self.rng.choice(["green", "yellow"]))
        if self.block_color not in ("green", "yellow"):
            raise ValueError("block_color must be 'green' or 'yellow'")
        rgba = (
            np.array([0.42, 0.78, 0.32, 1.0])
            if self.block_color == "green"
            else np.array([0.97, 0.78, 0.05, 1.0])
        )
        self.model.geom_rgba[self.block_geom] = rgba

        if block_xy is None:
            block_xy = np.array([0.21948, -0.00645]) + self.rng.uniform(
                -0.025, 0.025, size=2
            )
        else:
            block_xy = np.asarray(block_xy, dtype=np.float64)
        self.block_yaw = (
            float(self.rng.uniform(-np.pi / 2, np.pi / 2))
            if block_yaw is None
            else float(block_yaw)
        )
        adr = self.block_qpos_address
        self.data.qpos[adr : adr + 3] = [block_xy[0], block_xy[1], 0.00843]
        self.data.qpos[adr + 3 : adr + 7] = [
            np.cos(self.block_yaw / 2),
            0.0,
            0.0,
            np.sin(self.block_yaw / 2),
        ]
        self.data.qvel[self.block_dof_address : self.block_dof_address + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def _render_camera(self, renderer: mujoco.Renderer, camera: str) -> np.ndarray:
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def observation(self) -> dict[str, np.ndarray | float]:
        joint_state = self.mujoco_to_lerobot(
            self.data.qpos[self.qpos_addresses]
        ).astype(np.float32)
        obs: dict[str, np.ndarray | float] = {
            f"{name}.pos": float(value) for name, value in zip(JOINT_NAMES, joint_state)
        }
        # The physical dataset calls its wrist-mounted view "front".
        obs["front"] = self._render_camera(self.front_renderer, "wrist_cam")
        obs["overhead"] = self._render_camera(self.overhead_renderer, "overhead_cam")
        return obs

    def step(self, action_state: np.ndarray) -> dict[str, np.ndarray | float]:
        self.data.ctrl[self.actuator_ids] = self.lerobot_to_mujoco(action_state)

        # The MJCF timestep is 2 ms. Step to the next exact 30 Hz policy tick;
        # this naturally alternates between 16 and 17 physics steps.
        target_time = self.data.time + 1.0 / FPS
        while self.data.time + 1e-12 < target_time:
            mujoco.mj_step(self.model, self.data)
        return self.observation()

    def success(self) -> bool:
        block_xyz = self.data.xpos[self.block_body]
        tray_xyz = self.data.xpos[self.tray_bodies[self.block_color]]
        return bool(np.linalg.norm(block_xyz[:2] - tray_xyz[:2]) < 0.045 and block_xyz[2] < 0.06)

    def close(self) -> None:
        self.front_renderer.close()
        self.overhead_renderer.close()


def video_frame(observation: dict[str, np.ndarray | float]) -> np.ndarray:
    front = np.asarray(observation["front"])
    overhead = cv2.resize(np.asarray(observation["overhead"]), (640, 480), interpolation=cv2.INTER_AREA)
    return np.concatenate([front, overhead], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500, help="Maximum 30 Hz control steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "mujoco_act_rollout.mp4")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--policy-path",
        "--policy-cache",
        dest="policy_path",
        type=Path,
        default=DEFAULT_POLICY_CACHE,
        help="Checkpoint pretrained_model directory or Hugging Face cache directory",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto",
        help="Inference device; auto uses CUDA when available",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Save the initial policy camera views without loading ACT",
    )
    args = parser.parse_args()

    env = SO101MujocoPolicyEnv(seed=args.seed)
    observation = env.reset()

    if args.preview_only:
        output = args.output.with_suffix(".png")
        cv2.imwrite(str(output), cv2.cvtColor(video_frame(observation), cv2.COLOR_RGB2BGR))
        print(f"Saved policy-camera preview to {output}")
        print(f"Block color: {env.block_color}")
        env.close()
        return

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is unavailable.")
    device = torch.device(device_name)
    snapshot = resolve_snapshot(args.policy_path)
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID, root=args.dataset_root)

    config = PreTrainedConfig.from_pretrained(str(snapshot))
    config.device = device_name
    policy = ACTPolicy.from_pretrained(str(snapshot), config=config).to(device)
    policy.eval()
    policy.reset()
    # Recreate the checkpoint's MEAN_STD transforms from the explicitly chosen
    # real dataset. This also makes the processor's device follow --device.
    preprocess, postprocess = make_pre_post_processors(
        policy.config, dataset_stats=metadata.stats
    )

    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 480)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")

    success = False
    try:
        for step in range(args.steps):
            writer.write(cv2.cvtColor(video_frame(observation), cv2.COLOR_RGB2BGR))
            frame = build_inference_frame(
                observation=observation,
                ds_features=metadata.features,
                device=device,
                robot_type=metadata.robot_type,
            )
            processed = preprocess(frame)
            action = postprocess(policy.select_action(processed))
            action_state = action.squeeze(0).detach().cpu().numpy()
            observation = env.step(action_state)

            if env.success():
                success = True
                print(f"Success at step {step} ({step / FPS:.2f}s)")
                break
    finally:
        writer.release()
        env.close()

    print(f"Block color: {env.block_color}")
    print(f"Success: {success}")
    print(f"Saved rollout to {args.output}")


if __name__ == "__main__":
    main()
