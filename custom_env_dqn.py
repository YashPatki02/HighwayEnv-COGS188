import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
import os
import shutil


# Create custom highway env
env = gym.make('custom-highway-v0', render_mode="rgb_array")

env.unwrapped.configure({
    "collision_reward": -50,
})

# Clear directory to avoid repeat logs
log_dir = "highway_dqn/DQN"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

# Create DQN model
model = DQN('MlpPolicy', env,
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
              tensorboard_log=log_dir)

# Learn and save trained agent
model.learn(int(100000))
model.save("highway_dqn/model")

# Load and test saved model 5 times
model = DQN.load("highway_dqn/model")
for _ in range(5):
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()