"""Generate SO101 color-sorting demonstrations with a scripted MuJoCo expert.

The expert reads exact object poses from simulation, solves damped least-squares
inverse kinematics for the gripper site, and executes approach, grasp, lift,
transfer, release, and retreat phases at 30 Hz. Successful episodes can be
stored with the same LeRobot schema as the physical ACT dataset.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from record_sim_teleop import DEFAULT_OUTPUT_ROOT, source_features
from run_act_policy_in_mujoco import (
    DEFAULT_DATASET_ROOT,
    FPS,
    JOINT_NAMES,
    SO101MujocoPolicyEnv,
    video_frame,
)


HERE = Path(__file__).resolve().parent
DEFAULT_SCRIPTED_ROOT = DEFAULT_OUTPUT_ROOT.with_name("so101-mujoco-scripted")
DEFAULT_REPO_ID = "robododo/so101-mujoco-scripted"
DEFAULT_TASK = "Pick up the colored rectangle and place it in the matching colored tray"

# A collision-free model pose near the center of the workspace. The wrist roll
# is near the physical dataset's median rather than zero.
SAFE_Q = np.array([0.0, -1.0, 1.0, 0.5, -np.pi / 2], dtype=np.float64)
# This calibration maps larger 0-100 values to a wider jaw opening. The real
# dataset spans roughly 1 (closed) to 61 (fully open).
OPEN_GRIPPER = 55.0
CLOSED_GRIPPER = 0.0
GRASP_SITE_Z = 0.025
RELEASE_SITE_Z = 0.035


@dataclass
class MotionFrame:
    q: np.ndarray
    gripper: float
    phase: str


class GraspMonitor:
    """Require opposing contacts and an actual lift before transferring."""

    def __init__(self, env):
        self.env = env
        self.initial_z = float(env.data.xpos[env.block_body, 2])
        self.previous_phase = None

    def check(self, phase):
        if phase == self.previous_phase:
            return
        if phase in ("lift", "transfer", "lower"):
            contacts = set()
            for contact in self.env.data.contact:
                if self.env.block_geom in (contact.geom1, contact.geom2):
                    other = contact.geom2 if contact.geom1 == self.env.block_geom else contact.geom1
                    contacts.add(mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, other))
            if not {"fixed_finger_pad", "moving_finger_pad"}.issubset(contacts):
                raise RuntimeError(f"Grasp failed before {phase}: opposing finger contacts missing")
            if phase in ("transfer", "lower") and self.env.data.xpos[self.env.block_body, 2] < self.initial_z + 0.025:
                raise RuntimeError(f"Grasp failed before {phase}: block did not stay lifted")
        self.previous_phase = phase


class ScriptedExpert:
    def __init__(self, env: SO101MujocoPolicyEnv):
        self.env = env
        self.model = env.model
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center"
        )
        self.arm_joint_ids = env.joint_ids[:5]
        self.qpos_addresses = env.qpos_addresses[:5]
        self.dof_addresses = self.model.jnt_dofadr[self.arm_joint_ids]
        self.lower = self.model.jnt_range[self.arm_joint_ids, 0]
        self.upper = self.model.jnt_range[self.arm_joint_ids, 1]

    @staticmethod
    def target_rotation(yaw: float) -> np.ndarray:
        """Tool X points down; tool Y follows the requested planar angle."""
        tool_x = np.array([0.0, 0.0, -1.0])
        tool_y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        tool_z = np.cross(tool_x, tool_y)
        return np.column_stack((tool_x, tool_y, tool_z))

    def solve_ik(
        self,
        position: np.ndarray,
        yaw: float,
        initial_q: np.ndarray,
        orientation_weight: float = 0.2,
        max_iterations: int = 300,
    ) -> np.ndarray:
        q = np.asarray(initial_q, dtype=np.float64).copy()
        target_rotation = self.target_rotation(yaw)

        for _ in range(max_iterations):
            self.data.qpos[self.qpos_addresses] = q
            mujoco.mj_forward(self.model, self.data)
            current_position = self.data.site_xpos[self.site_id].copy()
            current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
            position_error = np.asarray(position) - current_position
            rotation_error = Rotation.from_matrix(
                target_rotation @ current_rotation.T
            ).as_rotvec()

            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(
                self.model, self.data, jac_pos, jac_rot, self.site_id
            )
            jacobian = np.vstack(
                (
                    jac_pos[:, self.dof_addresses],
                    orientation_weight * jac_rot[:, self.dof_addresses],
                )
            )
            error = np.concatenate(
                (position_error, orientation_weight * rotation_error)
            )
            damping = 1e-4
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            q = np.clip(q + np.clip(delta, -0.08, 0.08), self.lower, self.upper)

            if np.linalg.norm(position_error) < 5e-4 and np.linalg.norm(rotation_error) < 0.03:
                return q

        # Position matters most for grasping. Reject plans that miss by >8 mm.
        self.data.qpos[self.qpos_addresses] = q
        mujoco.mj_forward(self.model, self.data)
        residual = np.linalg.norm(np.asarray(position) - self.data.site_xpos[self.site_id])
        if residual > 0.008:
            raise RuntimeError(f"IK target is unreachable; position residual={residual:.4f} m")
        return q

    @staticmethod
    def smooth_joint_segment(
        start: np.ndarray,
        end: np.ndarray,
        frames: int,
        gripper_start: float,
        gripper_end: float,
        phase: str,
    ) -> list[MotionFrame]:
        result = []
        for alpha in np.linspace(0.0, 1.0, frames, endpoint=True):
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            result.append(
                MotionFrame(
                    q=(1.0 - smooth) * start + smooth * end,
                    gripper=(1.0 - smooth) * gripper_start + smooth * gripper_end,
                    phase=phase,
                )
            )
        return result

    def cartesian_segment(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray,
        yaw: float,
        start_q: np.ndarray,
        frames: int,
        gripper: float,
        phase: str,
    ) -> list[MotionFrame]:
        result = []
        q = start_q.copy()
        for alpha in np.linspace(0.0, 1.0, frames, endpoint=True):
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            position = (1.0 - smooth) * start_position + smooth * end_position
            q = self.solve_ik(position, yaw, q)
            result.append(MotionFrame(q=q.copy(), gripper=gripper, phase=phase))
        return result

    def plan(self) -> list[MotionFrame]:
        block = self.env.data.xpos[self.env.block_body].copy()
        tray = self.env.data.xpos[
            self.env.tray_bodies[self.env.block_color]
        ].copy()

        # The mesh's long axis is local Y. A 180-degree ambiguity is harmless
        # for this symmetric rectangle.
        grasp_yaw = self.env.block_yaw + np.pi / 2
        approach = np.array([block[0], block[1], 0.080])
        # grasp_center lies between the opposing finger surfaces, above their
        # tips. At 25 mm the fingers straddle the 30 mm block above the mat.
        grasp = np.array([block[0], block[1], GRASP_SITE_Z])
        lift = np.array([block[0], block[1], 0.080])
        transfer = np.array([tray[0], tray[1], 0.080])
        release = np.array([tray[0], tray[1], RELEASE_SITE_Z])

        q_approach = self.solve_ik(approach, grasp_yaw, SAFE_Q)
        motion = self.smooth_joint_segment(
            SAFE_Q, q_approach, 45, OPEN_GRIPPER, OPEN_GRIPPER, "approach"
        )
        q = motion[-1].q
        motion += self.cartesian_segment(
            approach, grasp, grasp_yaw, q, 30, OPEN_GRIPPER, "descend"
        )
        q = motion[-1].q
        motion += self.smooth_joint_segment(
            q, q, 18, OPEN_GRIPPER, CLOSED_GRIPPER, "close"
        )
        motion += self.smooth_joint_segment(
            q, q, 10, CLOSED_GRIPPER, CLOSED_GRIPPER, "secure"
        )
        motion += self.cartesian_segment(
            grasp, lift, grasp_yaw, q, 30, CLOSED_GRIPPER, "lift"
        )
        q = motion[-1].q
        motion += self.cartesian_segment(
            lift, transfer, grasp_yaw, q, 50, CLOSED_GRIPPER, "transfer"
        )
        q = motion[-1].q
        motion += self.cartesian_segment(
            transfer, release, grasp_yaw, q, 24, CLOSED_GRIPPER, "lower"
        )
        q = motion[-1].q
        motion += self.smooth_joint_segment(
            q, q, 18, CLOSED_GRIPPER, OPEN_GRIPPER, "release"
        )
        motion += self.cartesian_segment(
            release, transfer, grasp_yaw, q, 25, OPEN_GRIPPER, "retreat"
        )
        return motion


def action_from_motion(frame: MotionFrame) -> np.ndarray:
    return np.concatenate((np.rad2deg(frame.q), [frame.gripper]))


def run_episode(
    env: SO101MujocoPolicyEnv,
    dataset: LeRobotDataset | None,
    writer: cv2.VideoWriter | None,
    task: str,
    grasp_assist: bool = False,
) -> bool:
    if grasp_assist:
        raise ValueError('Grasp assist is disabled: demonstrations require physical contact.')
    start_state = np.concatenate((np.rad2deg(SAFE_Q), [OPEN_GRIPPER]))
    needs_images = dataset is not None or writer is not None
    observation = env.reset(initial_state=start_state, render_cameras=needs_images)
    expert = ScriptedExpert(env)
    motion = expert.plan()
    monitor = GraspMonitor(env)

    for frame in motion:
        monitor.check(frame.phase)
        action_array = action_from_motion(frame)
        action = {
            f"{name}.pos": float(value)
            for name, value in zip(JOINT_NAMES, action_array)
        }
        if dataset is not None:
            observation_frame = build_dataset_frame(
                dataset.features, observation, prefix=OBS_STR
            )
            action_frame = build_dataset_frame(
                dataset.features, action, prefix=ACTION
            )
            dataset.add_frame({**observation_frame, **action_frame, "task": task})

        if writer is not None:
            display = cv2.cvtColor(video_frame(observation), cv2.COLOR_RGB2BGR)
            cv2.putText(
                display,
                f"{frame.phase} | {env.block_color} | yaw {np.degrees(env.block_yaw):.1f} deg",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            writer.write(display)
        observation = env.step(action_array, render_cameras=False)
        if needs_images:
            observation = env.observation(render_cameras=True)

    # Allow the released object to settle before grading the trajectory.
    final_action = action_from_motion(motion[-1])
    for _ in range(30):
        observation = env.step(final_action, render_cameras=needs_images)
        if writer is not None:
            writer.write(cv2.cvtColor(video_frame(observation), cv2.COLOR_RGB2BGR))
    return env.success()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_SCRIPTED_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-grasp-assist",
        action="store_true",
        help="Compatibility flag; contact-only physics is now always used",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Write an MP4 preview instead of creating a dataset",
    )
    args = parser.parse_args()

    dataset = None
    writer = None
    if args.preview is not None:
        writer = cv2.VideoWriter(
            str(args.preview), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 480)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create preview video: {args.preview}")
    else:
        if args.dataset_root.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Dataset exists: {args.dataset_root}. Use --overwrite to replace it."
                )
            shutil.rmtree(args.dataset_root)
        features, robot_type = source_features(args.source_dataset_root)
        encoder = RGBEncoderConfig.from_video_info(
            features["observation.images.front"].get("info")
        )
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=FPS,
            features=features,
            root=args.dataset_root,
            robot_type=robot_type,
            use_videos=True,
            image_writer_threads=8,
            rgb_encoder=encoder,
        )

    env = SO101MujocoPolicyEnv(seed=args.seed)
    successes = 0
    try:
        for episode in range(args.episodes):
            try:
                success = run_episode(
                    env,
                    dataset,
                    writer,
                    args.task,
                    grasp_assist=False,
                )
            except RuntimeError as error:
                success = False
                print(f"Episode {episode}: plan failed: {error}")
            if success:
                successes += 1
                if dataset is not None:
                    dataset.save_episode(parallel_encoding=False)
            elif dataset is not None:
                dataset.clear_episode_buffer()
            print(
                f"Episode {episode}: color={env.block_color}, "
                f"yaw={np.degrees(env.block_yaw):.1f}, success={success}"
            )
    finally:
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()
        if writer is not None:
            writer.release()
        env.close()

    print(f"Successful episodes: {successes}/{args.episodes}")
    if args.preview is not None:
        print(f"Saved preview to {args.preview}")
    else:
        print(f"Saved dataset to {args.dataset_root}")


if __name__ == "__main__":
    main()
