import os
import numpy as np
import git
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from mujoco import viewer


# 1. Clone the MuJoCo Menagerie if not already present
def get_h1_model_path():
    repo_dir = "mujoco_menagerie"
    h1_model_path = os.path.join(repo_dir, "unitree_h1", "h1.xml")

    if not os.path.exists(h1_model_path):
        print("Cloning MuJoCo Menagerie...")
        git.Repo.clone_from("https://github.com/google-deepmind/mujoco_menagerie", repo_dir)

    return h1_model_path


# 2. Define the custom Gym environment
class UnitreeH1Env(gym.Env):
    def __init__(self, xml_path, render_mode=False):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer_launched = False  # Only launch viewer once

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = 0.01 * np.random.randn(self.model.nq)
        self.data.qvel[:] = 0.01 * np.random.randn(self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}


    def step(self, action):
        self.data.ctrl[:] = action * 100  # scale torques
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = float(self.data.qvel[0])  # forward velocity
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def render(self):
        if self.render_mode and not self.viewer_launched:
            print("Launching viewer... Close the window to continue.")
            viewer.launch(self.model, self.data)
            self.viewer_launched = True

    def close(self):
        pass  # No cleanup needed for mujoco.viewer.launch()


# 3. Train with PPO
def train():
    xml_path = get_h1_model_path()
    env = UnitreeH1Env(xml_path)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_h1_log/")
    model.learn(total_timesteps=500_000)
    model.save("ppo_h1_unitree")
    env.close()


# 4. Visualize policy using mujoco.viewer.launch()
def visualize(model_path="ppo_h1_unitree", steps=1000):
    xml_path = get_h1_model_path()
    model = PPO.load(model_path)

    model_env = mujoco.MjModel.from_xml_path(xml_path)
    model_data = mujoco.MjData(model_env)

    print("Launching viewer. Close window to stop.")
    viewer.launch(model_env, model_data)  # This is blocking

    # Optional: For continuous visualization (requires non-blocking viewer)
    # for _ in range(steps):
    #     obs = np.concatenate([model_data.qpos, model_data.qvel])
    #     action, _ = model.predict(obs)
    #     model_data.ctrl[:] = action * 100
    #     mujoco.mj_step(model_env, model_data)


if __name__ == "__main__":
    train()
    # To visualize after training, uncomment:
    visualize()
