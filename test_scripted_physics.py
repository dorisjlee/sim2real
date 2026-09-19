"""Regression checks: real finger contacts, lift, and settled placement."""
import unittest
import numpy as np
from generate_scripted_dataset import (
    DEFAULT_TASK, GraspMonitor, SO101MujocoPolicyEnv, run_episode,
)


class ScriptedPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = SO101MujocoPolicyEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_contact_only_placements(self):
        for seed in (0, 1, 3, 8):
            with self.subTest(seed=seed):
                self.env.rng = np.random.default_rng(seed)
                self.assertTrue(run_episode(self.env, None, None, DEFAULT_TASK))

    def test_missing_grasp_blocks_transfer(self):
        self.env.reset(render_cameras=False)
        with self.assertRaisesRegex(RuntimeError, 'opposing finger contacts'):
            GraspMonitor(self.env).check('transfer')

    def test_initial_block_is_not_success(self):
        self.env.reset(render_cameras=False)
        self.assertFalse(self.env.success())

    def test_assist_cannot_generate_demonstrations(self):
        with self.assertRaisesRegex(ValueError, 'disabled'):
            run_episode(self.env, None, None, DEFAULT_TASK, grasp_assist=True)


if __name__ == '__main__':
    unittest.main()
