#!/usr/bin/env python
"""
Safe, manual-step replay of a single recorded training episode on the physical
SO-101 follower arm. Used to diagnose whether the ACT rollout offset is caused
by a physical workspace/calibration shift (replay also misses) or an ACT
inference issue (replay lands correctly).

Usage:
    python tools/replay_physical_episode.py --episode 0 --speed 0.25
    python tools/replay_physical_episode.py --episode 0 --dry-run
    python tools/replay_physical_episode.py --episode 0 --speed 0.25 --record out.mp4
    python tools/replay_physical_episode.py --episode 0 --speed 0.25 --display-rerun

--display-rerun opens a Rerun viewer with the live overhead + front camera
feeds and the original training video for the same episode, laid out as a
2x2 grid, so you can visually compare where the arm/block ends up now vs.
during the original demonstration. The original clips are decoded directly
from the dataset's mp4s with OpenCV (not torchcodec) to sidestep the
torchcodec native-lib issue.

Safety:
    - Does NOT auto-move the robot to the episode's first pose.
    - Prints the first recorded observation.state and waits for you to
      manually position the arm and press ENTER before sending any action.
    - Sends actions one at a time with a small sleep derived from --speed
      (1.0 = real time at the dataset fps, 0.25 = 4x slower).
    - Ctrl+C at any point stops sending further actions (robot stays where it
      last was commanded; it does not snap anywhere).
"""

import argparse
import threading
import time

import cv2

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_REPO_ID = "robododo/place-rectangle-colored-box"
ROBOT_PORT = "COM5"
ROBOT_ID = "my_awesome_follower_arm"

MOTOR_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# cam_key -> (camera index, capture width, capture height, dataset video feature key)
CAMERAS = {
    "overhead": (2, 1280, 720, "observation.images.overhead"),
    "front": (0, 640, 480, "observation.images.front"),
}

# Camera used for --record (single-stream mp4 output)
RECORD_CAMERA_KEY = "overhead"


def open_original_video_reader(dataset, episode: int, video_key: str):
    """Return (VideoCapture positioned at the start of this episode's clip, start_frame)."""
    video_rel_path = dataset.meta.get_video_file_path(episode, video_key)
    video_path = dataset.root / video_rel_path
    ep_meta = dataset.meta.episodes[episode]
    from_timestamp = ep_meta[f"videos/{video_key}/from_timestamp"]
    start_frame = round(from_timestamp * dataset.meta.fps)

    # Force the FFMPEG backend: the default (often MSMF on Windows) has COM
    # apartment-threading issues that make reads from a background thread
    # return stale/repeated frames instead of advancing.
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"Could not open original training video at {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return cap, start_frame


def build_robot(camera_keys: list[str]):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower

    cameras = {}
    for key in camera_keys:
        index, width, height, _ = CAMERAS[key]
        cameras[key] = OpenCVCameraConfig(index_or_path=index, width=width, height=height, fps=30)
    config = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID, cameras=cameras)
    return SO101Follower(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.25, help="1.0 = real-time playback speed")
    parser.add_argument("--dry-run", action="store_true", help="Don't connect to the robot, just print")
    parser.add_argument(
        "--record", type=str, default=None,
        help="Path to save an .mp4 recording of the overhead camera during replay (e.g. out.mp4)",
    )
    parser.add_argument(
        "--display-rerun", action="store_true",
        help="Show live overhead+front camera feeds side by side with the original training clips in Rerun",
    )
    parser.add_argument(
        "--align-only", action="store_true",
        help="Only stream live vs. original camera feeds for visual alignment (camera extrinsics check). "
             "Never sends any action to the arm. Requires --display-rerun. Ctrl+C to stop.",
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="Add a 50/50 alpha-blended overlay panel (live + original) per camera in Rerun. "
             "Requires --display-rerun.",
    )
    args = parser.parse_args()

    if args.overlay and not args.display_rerun:
        raise SystemExit("--overlay requires --display-rerun")

    if args.align_only and not args.display_rerun:
        raise SystemExit("--align-only requires --display-rerun")

    if args.display_rerun:
        camera_keys = list(CAMERAS.keys())
    elif args.record:
        camera_keys = [RECORD_CAMERA_KEY]
    else:
        camera_keys = []

    print(f"Loading dataset {DATASET_REPO_ID} ...")
    dataset = LeRobotDataset(DATASET_REPO_ID)
    fps = dataset.fps

    # Only pull the non-video columns we need. dataset[idx] / dataset.hf_dataset
    # decode video frames via torchcodec, which can fail to load its native
    # libs (unrelated to this replay) — select_columns avoids that path.
    cols = dataset.select_columns(["episode_index", "observation.state", "action"])
    ep_col = cols["episode_index"]
    indices = [i for i, e in enumerate(ep_col) if int(e) == args.episode]
    if not indices:
        raise SystemExit(f"No frames found for episode {args.episode}")
    print(f"Episode {args.episode}: {len(indices)} frames at {fps} fps")

    first = cols[indices[0]]
    first_state = [float(v) for v in first["observation.state"]]
    print("\nFirst recorded observation.state (radians/normalized, order = "
          f"{MOTOR_ORDER}):")
    print([round(v, 4) for v in first_state])

    if args.dry_run:
        print("\n--dry-run set: not connecting to robot. Exiting.")
        return

    robot = build_robot(camera_keys)
    robot.connect(calibrate=False)

    writer = None
    if args.record:
        _, rec_w, rec_h, _ = CAMERAS[RECORD_CAMERA_KEY]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, fps, (rec_w, rec_h))
        print(f"\nRecording {RECORD_CAMERA_KEY} camera to {args.record}")

    original_caps = {}
    rr = None
    if args.display_rerun:
        import rerun as rr
        import rerun.blueprint as rrb

        rr.init("replay_episode_comparison", spawn=True)
        if args.overlay:
            views = [
                rrb.Spatial2DView(origin="overlay/overhead", name="Overlay overhead (blend)"),
                rrb.Spatial2DView(origin="overlay/front", name="Overlay front (blend)"),
            ]
            grid_columns = 2
        else:
            views = [
                rrb.Spatial2DView(origin="live/overhead", name="Live overhead (now)"),
                rrb.Spatial2DView(origin="original/overhead", name=f"Original overhead (ep {args.episode})"),
                rrb.Spatial2DView(origin="live/front", name="Live front (now)"),
                rrb.Spatial2DView(origin="original/front", name=f"Original front (ep {args.episode})"),
            ]
            grid_columns = 2
        rr.send_blueprint(rrb.Blueprint(rrb.Grid(*views, grid_columns=grid_columns)))
        original_starts = {}
        for cam_key in camera_keys:
            _, _, _, video_feature_key = CAMERAS[cam_key]
            cap, start_frame = open_original_video_reader(dataset, args.episode, video_feature_key)
            original_caps[cam_key] = cap
            original_starts[cam_key] = start_frame
        print("Opened Rerun viewer. Streaming live camera feed for alignment...")

        # Grab the original clip's first frame as a reference for calibration
        # (it should NOT play during preview). Kept as a raw RGB array so it
        # can also be alpha-blended with the live frame for the overlay panel.
        # Logged repeatedly below (not via static=True) so it's purely
        # temporal on the "frame" timeline, same as the replay loop's later
        # logs — avoids any static/temporal precedence ambiguity that could
        # keep the panel stuck once replay starts logging real playback
        # frames to the same entity path.
        original_first_frame_rgb = {}
        for cam_key in camera_keys:
            ok, orig_bgr = original_caps[cam_key].read()
            if ok:
                original_first_frame_rgb[cam_key] = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            original_caps[cam_key].set(cv2.CAP_PROP_POS_FRAMES, original_starts[cam_key])

        # Stream the live feed while the user physically aligns the block,
        # before any action is sent. Uses the same "frame" timeline as the
        # replay loop below (continuing the same counter) so the Rerun viewer
        # keeps advancing seamlessly across the preview -> replay transition
        # instead of looking frozen.
        stop_preview = threading.Event()
        preview_frame_count = [0]

        def _preview_loop():
            pn = 0
            while not stop_preview.is_set():
                rr.set_time("frame", sequence=pn)
                for cam_key in camera_keys:
                    img = robot.cameras[cam_key].read_latest()
                    rr.log(f"live/{cam_key}", rr.Image(img))
                    orig_rgb = original_first_frame_rgb.get(cam_key)
                    if orig_rgb is not None:
                        rr.log(f"original/{cam_key}", rr.Image(orig_rgb))
                        if args.overlay:
                            if orig_rgb.shape == img.shape:
                                overlay = cv2.addWeighted(img, 0.5, orig_rgb, 0.5, 0)
                            else:
                                overlay = cv2.resize(orig_rgb, (img.shape[1], img.shape[0]))
                                overlay = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
                            rr.log(f"overlay/{cam_key}", rr.Image(overlay))
                pn += 1
                preview_frame_count[0] = pn
                time.sleep(1.0 / fps)

        preview_thread = threading.Thread(target=_preview_loop, daemon=True)
        preview_thread.start()

    if args.align_only:
        print(
            "\n--align-only: streaming live vs. original camera feeds only. "
            "No action will be sent to the arm. Press Ctrl+C to stop."
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            stop_preview.set()
            preview_thread.join()
            for cap in original_caps.values():
                cap.release()
            robot.disconnect()
        return

    input(
        "\nWatch the Live overhead/front panels in Rerun and manually align the "
        "block/arm to match the Original panels, then press ENTER to begin "
        "replay. Ctrl+C to abort at any time."
    )

    frame_offset = 0
    if args.display_rerun:
        stop_preview.set()
        preview_thread.join()
        frame_offset = preview_frame_count[0]
        # Rewind the original clips back to the episode start so their
        # playback is in lockstep with the replay loop's frame index n.
        for cam_key in camera_keys:
            original_caps[cam_key].set(cv2.CAP_PROP_POS_FRAMES, original_starts[cam_key])

    try:
        dt = (1.0 / fps) / max(args.speed, 1e-6)
        print(f"\nReplaying {len(indices)} actions at {args.speed}x speed (dt={dt:.3f}s)...")
        for n, idx in enumerate(indices):
            frame = cols[idx]
            action_vec = [float(v) for v in frame["action"]]
            action = {f"{m}.pos": v for m, v in zip(MOTOR_ORDER, action_vec)}
            robot.send_action(action)

            if writer is not None:
                live_img = robot.cameras[RECORD_CAMERA_KEY].read_latest()
                writer.write(cv2.cvtColor(live_img, cv2.COLOR_RGB2BGR))
            if args.display_rerun:
                rr.set_time("frame", sequence=frame_offset + n)
                for cam_key in camera_keys:
                    live_img = robot.cameras[cam_key].read_latest()
                    rr.log(f"live/{cam_key}", rr.Image(live_img))
                    ok, orig_bgr = original_caps[cam_key].read()
                    if ok:
                        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                        rr.log(f"original/{cam_key}", rr.Image(orig_rgb))
                        if args.overlay:
                            if orig_rgb.shape == live_img.shape:
                                overlay = cv2.addWeighted(live_img, 0.5, orig_rgb, 0.5, 0)
                            else:
                                overlay = cv2.resize(orig_rgb, (live_img.shape[1], live_img.shape[0]))
                                overlay = cv2.addWeighted(live_img, 0.5, overlay, 0.5, 0)
                            rr.log(f"overlay/{cam_key}", rr.Image(overlay))

            if n % 10 == 0:
                print(f"  frame {n}/{len(indices)}")
            time.sleep(dt)
        print("Replay finished.")
    except KeyboardInterrupt:
        print("\nInterrupted — stopped sending actions. Robot left at last commanded pose.")
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved recording to {args.record}")
        for cap in original_caps.values():
            cap.release()
        robot.disconnect()


if __name__ == "__main__":
    main()
