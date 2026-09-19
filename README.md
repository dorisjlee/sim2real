# SO101 MuJoCo color sorting

MuJoCo simulation of the physical SO101 color-sorting workspace. The scene
contains the SO101 arm, lightbox, black mat, two colored trays, a movable STL
block, wrist camera, and external camera.

## What is version controlled

- All SO101 MJCF/URDF descriptions and STL meshes needed to load the robot
- The custom workspace MJCF and user-provided object/tray meshes
- The Gymnasium environment, leader-arm demonstration recorder, and ACT
  closed-loop evaluation runner
- An experimental scripted IK generator that automatically attempts randomized
  pick-and-place demonstrations and retains successful episodes only
- Physical episode replay/alignment and training-log replay utilities
- The mesh recentering utility and session notes

The physical LeRobot dataset and ACT weights are intentionally not copied into
this repository. They are large generated artifacts already held in the local
Hugging Face cache. Their paths can be passed to the runner as arguments.

## Environment

This project currently uses the existing `lerobot` conda environment and the
LeRobot checkout at `C:\Users\doris\lerobot`.

```powershell
conda activate lerobot
cd C:\Users\doris\Desktop\robotics\sim2real
```

## Open the workspace

```powershell
python -m mujoco.viewer --mjcf=scene_workspace.xml
```

## Inspect the policy camera inputs

This saves a side-by-side PNG containing the exact simulated wrist and external
views sent to ACT.

```powershell
python run_act_policy_in_mujoco.py --preview-only --seed 0
```

## Run the physical ACT policy in MuJoCo

```powershell
python run_act_policy_in_mujoco.py --steps 500 --seed 0
```

The defaults point at:

- Dataset: `place-rectangle-colored-box_20260718_195758`
- Policy: `robododo/sort_blocks_by_color_act_v2` local cache

Use `--dataset-root` and `--policy-path` if those locations move. The runner
uses CUDA when available and falls back to CPU.

The policy was trained at 30 Hz on five joint positions in calibrated degrees,
a gripper value normalized to 0-100, plus `front` (wrist, 640x480) and
`overhead` (external, 1280x720) RGB images. The adapter converts these values
to and from MuJoCo radians.

## Record simulation data and train

The complete workflow, including leader-arm teleoperation, dataset merging,
ACT fine-tuning, cloud/local CUDA options, and multi-seed evaluation, is in
[`docs/TRAINING_WORKFLOW.md`](docs/TRAINING_WORKFLOW.md).

The recording entry point is:

```powershell
python record_sim_teleop.py --port COM5 --episodes 20
```

Preview one automatically planned IK trajectory:

```powershell
python generate_scripted_dataset.py --episodes 1 --seed 0 --preview scripted_preview.mp4
```

Generate a local LeRobot dataset after the grasp parameters have been
validated across several preview seeds:

```powershell
python generate_scripted_dataset.py --episodes 100 --overwrite
```

The scripted generator is experimental. It currently solves and executes the
complete approach/grasp/lift/transfer/release motion, but contact parameters and
grasp height still need tuning before it should be used to produce training
data. Failed episodes are discarded automatically.

Supporting utilities are under `tools/`:

- `replay_physical_episode.py` safely replays or visually aligns a recorded
  physical episode.
- `replay_training_log.py` replays the preserved ACT training transcript.

## Source and provenance

The SO101 descriptions and robot meshes came from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), under
the Apache License 2.0. `docs/SO_ARM100_README.md` preserves the upstream model
notes. The lightbox was rebuilt as MJCF using dimensions and poses from
NVIDIA's Sim-to-Real SO-101 workshop USD asset. The files under
`assets/objects/` came from the user's physical workspace; the `*.orig.stl`
files preserve the original exports and the other STL files have centered
origins for stable MuJoCo placement.

## Repository boundaries

Large Hugging Face datasets, model checkpoints, generated videos, and training
outputs are excluded from Git. The NVIDIA Isaac Sim workshop remains in its
own upstream repository; this repository contains the native MuJoCo workspace
derived from its documented dimensions and USD asset poses.
