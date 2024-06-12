import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import time
import os
import shutil
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

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
            if 'collision' in self.locals['infos'][0]:
                self.collisions += self.locals['infos'][0]['collision']
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

# Create custom highway env
if __name__ == "__main__":
    train = True
    if train:
        env = gym.make('custom-highway-v0', render_mode="rgb_array")
        env.unwrapped.configure({"spawn_probability": 0})
        log_dir = "highway_dqn/DQN"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        tensorboard_log = log_dir if is_tensorboard_installed() else None
        callback = CustomCallback()
        model = DQN(
            'MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )
        model.learn(int(10000), callback=callback)
        model.save("highway_dqn/model")
        metrics = callback.get_metrics()
        print("DQN Metrics:", metrics)
        # Save metrics to a file for comparison later
        np.save("dqn_metrics.npy", metrics)
    model = DQN.load("highway_dqn/model")
    env = gym.make('custom-highway-v0', render_mode="rgb_array")
    env.unwrapped.configure({"spawn_probability": 0})
    for _ in tqdm(range(100), desc="Testing DQN"):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()
