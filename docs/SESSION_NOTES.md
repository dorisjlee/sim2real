# SO101 MuJoCo Color-Sort Sim — Session Notes

Context dump so a fresh session can pick this up without re-deriving it.

## Environment

- Conda env: `lerobot` (existing LeRobot install at `C:\Users\doris\lerobot`)
- Installed into it: `mujoco>=3.1` (3.13.0), `mujoco-python-viewer`, `trimesh`, `scipy`
- Activate with: `conda activate lerobot`

## What exists, where

The version-controlled project now lives under
`C:\Users\doris\Desktop\robotics\sim2real\`:

- **`scene.xml`** — plain SO101 arm on a floor, no workspace. Original download.
- **`scene_workspace.xml`** — the real scene: arm + lightbox + mat + 2 trays + 1 pick block + cameras. **This is the one to open/edit.**
- **`so101_new_calib.xml`** / **`so101_new_calib_camera.xml`** — arm MJCF, downloaded from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (`Simulation/SO101/`). The `_camera` variant adds the wrist-camera-mount housing mesh; I added an actual `<camera name="wrist_cam">` element inside its `wrist_camera` body (the stock file only had the housing geometry, no sensor) — **its orientation is an unverified guess**, check it in the viewer.
- **`assets/objects/`** — your pick-and-place STL meshes (`rectangle.stl`, `rectangle_wide.stl`, `tray.stl`), copied in from `C:\Users\doris\Desktop\robotics\3d print\`. Originals backed up as `*.orig.stl`. **They were re-centered** (see `recenter_meshes.py`) because the as-exported STLs carried raw CAD placement offsets — without this, bodies loaded far from their specified `pos` and objects would explode on contact. Recentered origin = `(x_center, y_center, z_min)` of the bounding box, so `body pos.z` = the height of the surface the object rests on.
- **`so101_color_sort_env.py`** — Gymnasium env wrapping the MuJoCo sim (see below).
- **`view_scene.py`** — minimal interactive-viewer launcher script.

Quick launch:
```bash
conda activate lerobot
cd /c/Users/doris/Desktop/robotics/sim2real
python -m mujoco.viewer --mjcf=scene_workspace.xml
```

## Scene layout (scene_workspace.xml)

**Lightbox** — reverse-engineered from NVIDIA's [Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) (`lightbox-simple.usd`), since that repo is Isaac Sim/USD, not MuJoCo — no MJCF existed to reuse, so I extracted exact panel poses/sizes from the USD via `pxr` (OpenUSD Python bindings) and rebuilt as native MJCF boxes:
- Open-front enclosure, ~0.762 × 0.508 × 0.508m, white foam-board Base/Left/Right/Back panels (contype/conaffinity=0, visual only)
- Open top (the earlier translucent diffuser panel was removed to match the physical workspace)
- 3 spotlights along a 0.3m span above the opening, approximating the real LED bar light (~4000K)

**Cameras:**
- `external_cam` — pose copied directly from the USD `camera_mount` (matches doc's Logitech C920: 40cm height, 45° down, fovy≈49 for ~78° horizontal @ 16:9). MuJoCo and USD/OpenGL share the same -Z-forward/+Y-up camera convention, so the quaternion transferred with no remapping needed.
- `wrist_cam` — inside `so101_new_calib_camera.xml`'s `wrist_camera` body. **Orientation unverified**, see above.
- `overhead_cam`, `front_cam` — convenience cams I added, not from any reference asset.

**Mat** — 30cm × 42cm × 6mm (user-corrected from an initial wrong guess of 45.72×30.48cm derived from a since-superseded NVIDIA mat.usda asset). Long edge (42cm) runs side-to-side (Y), short edge (30cm) runs front-to-back (X, depth from arm). Positioned at `pos="0.25198 0 0.00543"`, top surface at `z=0.00843`.

**Trays + block** — placed from a user-provided calibration diagram (`C:\Users\doris\Downloads\2.jpg`, top-down view, cm-labeled gaps). Diagram shapes: green square + yellow square = the two trays (both using `tray.stl`, 108.55mm footprint), small rectangle between them = the pick block (`rectangle.stl`).
  - Gaps used: 4cm (mat-left → green tray), 11cm (green tray → yellow tray, the gap the block centers in), 12.5cm (mat back edge → tray back edge), 6cm (tray front edge → mat front edge).
  - At the corrected 30×42cm mat size, these gaps are now self-consistent to within ~0.5cm of the tray's real 10.855cm footprint (an earlier mat-size guess made a labeled "5cm" gap come out ~9cm — confirms 30×42cm was the fix).
  - Current placements: `tray_green` at `(0.21948, -0.11572)`, `tray_yellow` at `(0.21948, 0.10282)`, `rectangle_obj` at `(0.21948, -0.00645)`, all at `z=0.00843`.
  - `rectangle_wide_obj` was removed per "for now just have one block."

## The Gym env (so101_color_sort_env.py)

`SO101ColorSortEnv(gym.Env)` — wraps `scene_workspace.xml`.

- **Task**: place the pick block in the tray matching its color. **Per-episode randomization** (confirmed requirement): `reset()` randomly recolors `rectangle_obj`'s geom to green or yellow (`model.geom_rgba[...]`, mutated directly at runtime — no need to recompile the model) and jitters its spawn position ±1.5cm around the nominal diagram-derived spot.
- **Action space**: `Box(6,)`, one target angle per actuator (`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`), written to `data.ctrl` — these are **position actuators** (PD-controlled, like the real servos), not direct joint torques.
- **Observation**: dict of `qpos`, `qvel`, `block_color` (0=green,1=yellow). No image obs wired into the space yet — `render(camera=...)` works standalone (tested with `overhead_cam` at 480×480) but isn't in `observation_space`.
- **Reward/termination**: sparse — 1.0 and `terminated=True` when block XY is within 6cm of the correct tray's body position and block Z has settled back down (<5cm), else 0.0.
- Tested: smoke-tested standalone (`python so101_color_sort_env.py`), and reset-color-randomization verified visually across 6 seeds (both colors appear).

## Known gaps / open threads (not yet done)

1. **Wrist camera orientation** is an unverified guess — needs checking against the render, likely needs quat correction.
2. **Scripted IK grasp needs contact tuning** — `generate_scripted_dataset.py`
   now plans approach, grasp, lift, transfer, release, and retreat with damped
   Jacobian IK. It discards failed episodes, but grasp height, jaw orientation,
   and contact parameters still need validation before bulk generation.
3. **Image observations aren't in the Gym env's `observation_space`** — only `qpos`/`qvel`/`block_color`. Add camera renders there if training a vision policy.
4. **Success/reward tuning is a first pass** — 6cm XY tolerance and 5cm Z ceiling were picked without validating against actual tray placement geometry; may need tightening once real pick-and-place trajectories are tried.
5. **Home pose (`HOME_QPOS` in the env)** is an arbitrary guess, not validated against a real "arm tucked out of the workspace" pose.
6. Diagram's dimensions weren't perfectly self-consistent even after the mat-size fix — worth a sanity pass against the physical rig if precision matters beyond what's already implemented.

## Reference material consulted

- Arm/scene source: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- Lightbox/camera reference: [NVIDIA SO-101 sim-to-real workshop docs](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/05-building-workspace.html) + [isaac-sim/Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) (USD assets, extracted via `pxr`/OpenUSD — cloned to a temp dir and deleted after extracting the numbers, not vendored into the repo)
- User's own 3D-printed pick objects: `C:\Users\doris\Desktop\robotics\3d print\{Rectangle,Rectangle-Wide,tray}.stl`
- User's calibration diagram: `C:\Users\doris\Downloads\2.jpg`
