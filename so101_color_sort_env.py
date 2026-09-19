"""Gymnasium wrapper around the SO101 MuJoCo color-sort workspace.

Each reset() randomizes which color (green/yellow) the pick block is, and
where it starts within a small jitter region on the mat. The task is solved
when the block is placed in the tray matching its own color.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

HERE = Path(__file__).parent
SCENE_PATH = HERE / "scene_workspace.xml"

# Tray colors (defined in scene_workspace.xml materials) -- reused here so the
# block can be recolored at runtime to match one of them.
COLOR_GREEN = np.array([0.42, 0.78, 0.32, 1.0])
COLOR_YELLOW = np.array([0.97, 0.78, 0.05, 1.0])

# Nominal block spawn point from the calibration diagram (assets/so101/scene_workspace.xml)
BLOCK_NOMINAL_XY = np.array([0.21948, -0.00645])
BLOCK_Z = 0.00843
SPAWN_JITTER = 0.015  # +/- 1.5cm randomization around the nominal spawn point

# Home pose (radians): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
HOME_QPOS = np.array([0.0, -1.0, 1.0, 0.5, 0.0, 0.4])

SUCCESS_XY_TOL = 0.06  # meters, "in the tray" tolerance
SUCCESS_Z_MAX = 0.05  # block must have settled back down (not mid-air/being held)


class SO101ColorSortEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, camera: str = "overhead_cam",
                 image_size: tuple[int, int] = (128, 128), control_substeps: int = 5):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.camera = camera
        self.control_substeps = control_substeps

        self._block_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rectangle_obj")
        self._block_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "rectangle_geom")
        block_jnt = self.model.body_jntadr[self._block_body]
        self._block_qposadr = self.model.jnt_qposadr[block_jnt]
        self._block_dofadr = self.model.jnt_dofadr[block_jnt]

        self._tray_body = {
            "green": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tray_green"),
            "yellow": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tray_yellow"),
        }

        self.n_actuators = self.model.nu
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.n_actuators,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "qpos": spaces.Box(-np.inf, np.inf, shape=(self.model.nq,), dtype=np.float32),
            "qvel": spaces.Box(-np.inf, np.inf, shape=(self.model.nv,), dtype=np.float32),
            "block_color": spaces.Discrete(2),  # 0 = green, 1 = yellow
        })

        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=image_size[0], width=image_size[1])

        self._rng = np.random.default_rng()
        self.block_color_name = "green"

    def _randomize_block(self):
        self.block_color_name = self._rng.choice(["green", "yellow"])
        rgba = COLOR_GREEN if self.block_color_name == "green" else COLOR_YELLOW
        self.model.geom_rgba[self._block_geom] = rgba

        jitter = self._rng.uniform(-SPAWN_JITTER, SPAWN_JITTER, size=2)
        xy = BLOCK_NOMINAL_XY + jitter
        adr = self._block_qposadr
        self.data.qpos[adr:adr + 3] = [xy[0], xy[1], BLOCK_Z]
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]  # identity quat
        self.data.qvel[self._block_dofadr:self._block_dofadr + 6] = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[: self.n_actuators] = HOME_QPOS
        self.data.ctrl[:] = HOME_QPOS
        self._randomize_block()
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {"block_color": self.block_color_name}

    def step(self, action: np.ndarray):
        self.data.ctrl[:] = np.clip(action, self.action_space.low, self.action_space.high)
        for _ in range(self.control_substeps):
            mujoco.mj_step(self.model, self.data)

        terminated = self._check_success()
        reward = 1.0 if terminated else 0.0
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {"block_color": self.block_color_name}

    def _check_success(self) -> bool:
        block_xyz = self.data.xpos[self._block_body]
        target_body = self._tray_body[self.block_color_name]
        tray_xyz = self.data.xpos[target_body]
        xy_dist = np.linalg.norm(block_xyz[:2] - tray_xyz[:2])
        return bool(xy_dist < SUCCESS_XY_TOL and block_xyz[2] < SUCCESS_Z_MAX)

    def _get_obs(self):
        return {
            "qpos": self.data.qpos.copy().astype(np.float32),
            "qvel": self.data.qvel.copy().astype(np.float32),
            "block_color": 0 if self.block_color_name == "green" else 1,
        }

    def render(self):
        if self._renderer is None:
            raise RuntimeError("Env was constructed without render_mode='rgb_array'")
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()


if __name__ == "__main__":
    env = SO101ColorSortEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    print("block color:", info["block_color"])
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0.05)
    img = env.render()
    print("rendered image shape:", img.shape)
