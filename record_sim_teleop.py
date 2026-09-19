"""Record successful SO101 MuJoCo demonstrations with a physical leader arm.

The output is a LeRobot dataset with the same state, action, and camera schema
as the existing physical color-sorting dataset, so the two can be merged and
used to fine-tune the existing ACT checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from run_act_policy_in_mujoco import (
    DEFAULT_DATASET_ROOT,
    FPS,
    JOINT_NAMES,
    SO101MujocoPolicyEnv,
    video_frame,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "lerobot"
    / "robododo"
    / "so101-mujoco-color-sort"
)
DEFAULT_REPO_ID = "robododo/so101-mujoco-color-sort"
DEFAULT_TASK = "Pick up the colored rectangle and place it in the matching colored tray"


def source_features(root: Path) -> tuple[dict, str]:
    """Read the exact policy-facing schema without loading source videos."""
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    wanted = {
        "action",
        "observation.state",
        "observation.images.front",
        "observation.images.overhead",
    }
    return {key: value for key, value in info["features"].items() if key in wanted}, info["robot_type"]


def action_array(action: dict[str, float]) -> np.ndarray:
    return np.asarray([action[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float64)


def ask_episode_result(success: bool) -> str:
    default = "s" if success else "r"
    answer = input(
        f"Episode ended (automatic success={success}). "
        f"[s]ave, [r]etry/discard, or [q]uit [{default}]: "
    ).strip().lower()
    return answer or default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Leader arm serial port, for example COM5")
    parser.add_argument("--leader-id", default="my_awesome_leader_arm")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-seconds", type=float, default=25.0)
    parser.add_argument("--success-hold-frames", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output dataset before recording",
    )
    args = parser.parse_args()

    if args.dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset already exists: {args.dataset_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(args.dataset_root)

    features, robot_type = source_features(args.source_dataset_root)
    rgb_encoder = RGBEncoderConfig.from_video_info(
        features["observation.images.front"].get("info")
    )
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        root=args.dataset_root,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        image_writer_threads=8,
        rgb_encoder=rgb_encoder,
    )
    leader = SO101Leader(
        SO101LeaderConfig(port=args.port, id=args.leader_id, use_degrees=True)
    )
    env = SO101MujocoPolicyEnv(seed=args.seed)

    saved = 0
    try:
        leader.connect()
        print("Leader connected. Its first five joints use degrees; gripper uses 0-100.")
        print("During an episode, press Q in the camera window to stop it early.")

        while saved < args.episodes:
            input(
                f"\nMove the leader to a safe starting pose for episode {saved + 1}, "
                "then press ENTER..."
            )
            first_action = leader.get_action()
            observation = env.reset(initial_state=action_array(first_action))
            stable_success_frames = 0
            start = time.perf_counter()

            while time.perf_counter() - start < args.episode_seconds:
                tick = time.perf_counter()
                action = leader.get_action()
                observation_frame = build_dataset_frame(
                    dataset.features, observation, prefix=OBS_STR
                )
                action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
                dataset.add_frame({**observation_frame, **action_frame, "task": args.task})
                observation = env.step(action_array(action))

                display = cv2.cvtColor(video_frame(observation), cv2.COLOR_RGB2BGR)
                label = f"episode {saved + 1} | block: {env.block_color} | Q: stop"
                cv2.putText(display, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.imshow("SO101 MuJoCo teleoperation", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

                stable_success_frames = stable_success_frames + 1 if env.success() else 0
                if stable_success_frames >= args.success_hold_frames:
                    break
                remaining = 1.0 / FPS - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)

            success = stable_success_frames >= args.success_hold_frames
            result = ask_episode_result(success)
            if result == "s":
                # Sequential encoding is slower than a process pool but avoids
                # Windows spawn/serial-port interactions while the leader is open.
                dataset.save_episode(parallel_encoding=False)
                saved += 1
                print(f"Saved episode {saved}/{args.episodes}")
            else:
                dataset.clear_episode_buffer()
                print("Discarded episode.")
            if result == "q":
                break
    finally:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        dataset.finalize()
        env.close()
        cv2.destroyAllWindows()
        if leader.is_connected:
            leader.disconnect()

    print(f"Finalized {saved} episodes at {args.dataset_root}")


if __name__ == "__main__":
    main()
