import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import time
import os
import shutil
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
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
    def __init__(self, threshold=200, video_freq, video_dir, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.video_freq = video_freq
        self.video_dir = video_dir
        os.makedirs(video_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        self.collisions = 0
        self.start_time = time.time()
        self.threshold = threshold
        self.convergence_rate = None
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
                self.model.save(os.path.join(self.video_dir, f"ppo_model_episode_{self.episode_count}"))
                self.record_video(f"episode_{self.episode_count}")
        return True

    def _on_training_end(self):
        self.end_time = time.time()
        self.training_time = self.end_time - self.start_time

        # Calculate convergence rate
        for i in range(len(self.episode_rewards)):
            if np.mean(self.episode_rewards[i:i+100]) >= self.threshold:
                self.convergence_rate = i + 100
                break
        if self.convergence_rate is None:
            self.convergence_rate = "Did not converge"

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
        model = PPO.load(os.path.join(self.video_dir, f"ppo_model_episode_{self.episode_count}"))
        obs, info = video_env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = video_env.step(action)
            video_env.render()
        video_env.close()

# Training PPO using vectorized env for faster speed
if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 6
        batch_size = 64
        vid_freq = 1000
        vid_dir = "ppo_training_videos"
        env = make_vec_env(
            "custom-highway-v0",
            n_envs=n_cpu,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={"config": {"render_mode": "rgb_array", "spawn_probability": 0}},
        )
        log_dir = "highway_ppo/PPO"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        tensorboard_log = log_dir if is_tensorboard_installed() else None
        callback = CustomCallback(video_freq=vid_freq, video_dir=vid_dir)
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log=tensorboard_log,
        )
        model.learn(total_timesteps=int(200000), callback=callback)
        model.save("highway_ppo/model")
        metrics = callback.get_metrics()
        print("PPO Metrics:", metrics)
        # Save metrics to a file for comparison later
        np.save("ppo_metrics.npy", metrics)
    model = PPO.load("highway_ppo/model")
    env = gym.make('custom-highway-v0', render_mode="rgb_array")
    env.unwrapped.configure({"spawn_probability": 0})
    
    collision_count = 0
    total_episodes = 100
    episode_rewards = []

    for _ in tqdm(range(100), desc="Testing PPO"):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if info['crashed']:
                collision_count += 1
            env.render()
        episode_rewards.append(episode_reward)
    env.close()

    print("Collision rate:", collision_count / total_episodes)
    print("Average reward:", np.mean(episode_rewards))

    metrics['eval_collision_rate'] = collision_count / total_episodes
    metrics['eval_average_reward'] = np.mean(episode_rewards)
    metrics['total_reward'] = sum(episode_rewards)
    np.save("ppo_metrics.npy", metrics)
