import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
import time
import os
import shutil
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import highway_env
highway_env.register_highway_envs()

# Function to check if TensorBoard is installed
def is_tensorboard_installed():
    try:
        import tensorboard
        return True
    except ImportError:
        return False

# Custom callback to track performance metrics
class CustomCallback(BaseCallback):
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.collisions = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if any(self.locals['dones']):
            self.episode_rewards.append(sum(self.locals['rewards']))
            self.episode_lengths.append(len(self.locals['rewards']))
            # Add logic to detect collisions from the 'info' dictionary if available
            if self.locals['infos'][0]['crashed'] == True:
                self.collisions += 1
        return True

    def _on_training_end(self):
        self.end_time = time.time()
        self.training_time = self.end_time - self.start_time

    def get_metrics(self):
        average_reward = np.mean(self.episode_rewards)
        average_length = np.mean(self.episode_lengths)
        collision_rate = self.collisions / len(self.episode_rewards)
        return {
            'average_reward': average_reward,
            'average_length': average_length,
            'collision_rate': collision_rate,
            'training_time': self.training_time
        }

# Training A2C using vectorized env for faster speed
if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env(
            "custom-highway-v0",
            n_envs=n_cpu,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={"config": {"render_mode": "rgb_array", "spawn_probability": 0}},
        )
        log_dir = "highway_a2c/A2C"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        tensorboard_log = log_dir if is_tensorboard_installed() else None
        callback = CustomCallback()
        model = A2C(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            n_steps=batch_size // n_cpu,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log=tensorboard_log,
        )
        model.learn(total_timesteps=int(1000), callback=callback)
        model.save("highway_a2c/model")
        metrics = callback.get_metrics()
        print("A2C Metrics:", metrics)
        # Save metrics to a file for comparison later
        np.save("a2c_metrics.npy", metrics)
    model = A2C.load("highway_a2c/model")
    env = gym.make('custom-highway-v0', render_mode="rgb_array")
    env.unwrapped.configure({"spawn_probability": 0})
    for _ in tqdm(range(100), desc="Testing A2C"):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()
