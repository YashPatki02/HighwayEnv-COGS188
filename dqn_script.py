import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import time
import os
import shutil
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import highway_env
from gymnasium.wrappers import RecordVideo

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
    def __init__(self, video_freq, video_dir, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.video_freq = video_freq
        self.video_dir = video_dir
        os.makedirs(video_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        self.collisions = 0
        self.start_time = time.time()
        self.episode_count = 0

    def _on_step(self) -> bool:
        if any(self.locals['dones']):
            self.episode_count += 1
            self.episode_rewards.append(sum(self.locals['rewards']))
            self.episode_lengths.append(len(self.locals['rewards']))
            # Add logic to detect collisions from the 'info' dictionary if available
            if self.locals['infos'][0]['crashed'] == True:
                self.collisions += 1
            if self.episode_count % self.video_freq == 0:
                self.model.save(os.path.join(self.video_dir, f"dqn_model_episode_{self.episode_count}"))
                self.record_video(f"episode_{self.episode_count}")
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
    def record_video(self, prefix):
        video_env = gym.make('custom-highway-v0', render_mode="rgb_array")
        video_env.unwrapped.configure({"spawn_probability": 0})
        video_env = RecordVideo(video_env, video_folder=self.video_dir, episode_trigger=lambda x: True, name_prefix=prefix)
        model = DQN.load(os.path.join(self.video_dir, f"dqn_model_episode_{self.episode_count}"))
        obs, info = video_env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = video_env.step(action)
            video_env.render()
        video_env.close()

# Create custom highway env
if __name__ == "__main__":
    train = True
    if train:
        vid_freq = 1000
        vid_dir = "dqn_training_videos"
        env = gym.make('custom-highway-v0', render_mode="rgb_array")
        env.unwrapped.configure({"spawn_probability": 0})
        log_dir = "highway_dqn/DQN"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        tensorboard_log = log_dir if is_tensorboard_installed() else None
        callback = CustomCallback(video_freq=vid_freq, video_dir=vid_dir)
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
        model.learn(int(100000), callback=callback)
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
