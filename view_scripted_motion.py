"""Inspect the scripted motion live, with ordinary contact physics only."""
import argparse
import time
import threading

import mujoco
import mujoco.viewer
import numpy as np

from generate_scripted_dataset import (
    SAFE_Q, OPEN_GRIPPER, ScriptedExpert, GraspMonitor, action_from_motion,
)
from run_act_policy_in_mujoco import SO101MujocoPolicyEnv, FPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    env = SO101MujocoPolicyEnv(seed=args.seed)
    paused = threading.Event()
    paused.set()
    restart = threading.Event()
    cycle_camera = threading.Event()

    def keypress(key):
        if key == 32:
            paused.clear() if paused.is_set() else paused.set()
        elif key in (82, 114):
            restart.set()
        elif key in (67, 99):
            cycle_camera.set()

    def reset():
        env.rng = np.random.default_rng(args.seed)
        env.reset(initial_state=np.r_[np.rad2deg(SAFE_Q), OPEN_GRIPPER],
                  render_cameras=False)
        return ScriptedExpert(env).plan()

    try:
        motion = reset()
        index = 0
        monitor = GraspMonitor(env)
        print('SPACE: play/pause; R: restart; C: free/wrist/overhead camera. Contact physics only.', flush=True)
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=keypress) as viewer:
            camera_index = 0
            cameras = [None, 'wrist_cam', 'overhead_cam']
            viewer.cam.lookat[:] = [0.22, 0, 0.08]
            viewer.cam.distance = 0.65
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -30
            while viewer.is_running():
                tick = time.perf_counter()
                with viewer.lock():
                    if cycle_camera.is_set():
                        camera_index = (camera_index + 1) % len(cameras)
                        name = cameras[camera_index]
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE if name is None else mujoco.mjtCamera.mjCAMERA_FIXED
                        if name is not None:
                            viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                        cycle_camera.clear()
                    if restart.is_set():
                        motion = reset()
                        monitor = GraspMonitor(env)
                        index = 0
                        restart.clear()
                        paused.set()
                    if not paused.is_set():
                        if index < len(motion):
                            frame = motion[index]
                            try:
                                monitor.check(frame.phase)
                            except RuntimeError as error:
                                print(error, flush=True)
                                paused.set()
                                continue
                            if index == 0 or frame.phase != motion[index - 1].phase:
                                print(frame.phase, flush=True)
                            action = action_from_motion(frame)
                        else:
                            action = action_from_motion(motion[-1])
                        env.step(action, render_cameras=False)
                        index += 1
                        if index == len(motion) + FPS:
                            print(f'Block settled inside matching tray: {env.success()}', flush=True)
                            paused.set()
                viewer.sync()
                time.sleep(max(0, 1 / FPS - (time.perf_counter() - tick)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
