import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3 import A2C
import shutil

import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import logging



# Training A2C using vectorized env for faster speed
if __name__ == "__main__":
  
    train = True

    if train:
        n_cpu = 6  
        batch_size = 64

        # Create vectorized custom highway env
        env = make_vec_env(
            "custom-highway-v0",
            n_envs=n_cpu,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={
                "config": {
                    "render_mode": "rgb_array",
                    "spawn_probability": 0,
                }
            }
        )
        
        # Clear directory to avoid repeat logs
        log_dir = "highway_a2c/A2C"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            
        # Create A2C model
        model = A2C(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]), 
            n_steps=batch_size // n_cpu,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log=log_dir,
        )

        # Learn and save trained agent
        model.learn(total_timesteps=int(200000))
        model.save("highway_a2c/model")

    # Load and test saved model 5 times
    model = A2C.load("highway_a2c/model")
    
    env = gym.make('custom-highway-v0', render_mode="rgb_array")
    env.unwrapped.configure({
        "spawn_probability": 0,
    })
    
    for _ in range(100):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()