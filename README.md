# SO101 MuJoCo color sorting

MuJoCo simulation of the physical SO101 color-sorting workspace. The scene
contains the SO101 arm, lightbox, black mat, two colored trays, a movable STL
block, wrist camera, and external camera.

## What is version controlled

- All SO101 MJCF/URDF descriptions and STL meshes needed to load the robot
- The custom workspace MJCF and user-provided object/tray meshes
- The Gymnasium environment, automatic IK demonstration generator, leader-arm
  recorder, and ACT closed-loop evaluation runner
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

Inspect the automatic motion live (starts paused):

```powershell
python view_scripted_motion.py --seed 0
```

Space plays/pauses, R restarts, and C cycles free, wrist, and overhead views.
Grasping uses physical contacts only. The expert stops if opposing finger
contacts or the lift check fail; placement requires the whole block to settle
inside the matching tray. Run `python -m unittest test_scripted_physics` to
check contact-only placements and rejection of invalid grasps.

The complete workflow, including leader-arm teleoperation, dataset merging,
ACT fine-tuning, cloud/local CUDA options, and multi-seed evaluation, is in
[`docs/TRAINING_WORKFLOW.md`](docs/TRAINING_WORKFLOW.md).

Generate varied demonstrations automatically:

```powershell
python generate_scripted_dataset.py --episodes 100
```

Preview one automatically planned motion without creating a dataset:

```powershell
python generate_scripted_dataset.py --seed 0 --preview scripted_preview.mp4
```

Manual leader-arm recording remains available for corrections and behaviors
that the scripted expert does not cover:

```powershell
python record_sim_teleop.py --port COM5 --episodes 20
```

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
