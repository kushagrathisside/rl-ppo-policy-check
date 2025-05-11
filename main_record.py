import os
import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

class UnitreeH1Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        xml_path = "mujoco_menagerie/unitree_h1/h1.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.render_mode = render_mode

        obs_dim = self.model.nq + self.model.nv
        act_dim = self.model.nu

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

    def step(self, action):
        ctrl = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

        obs = np.concatenate([self.data.qpos, self.data.qvel])
        reward = -np.linalg.norm(ctrl)  # example reward
        done = False
        info = {}
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        info = {}
        return obs, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model)
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            del self.renderer

def make_env():
    env = UnitreeH1Env(render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="videos",
        episode_trigger=lambda e: True,
        name_prefix="unitree_h1",
    )
    env = Monitor(env)
    return env

def train():
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500_000)
    env.close()

if __name__ == "__main__":
    os.makedirs("videos", exist_ok=True)
    print("Using cpu device")
    train()
