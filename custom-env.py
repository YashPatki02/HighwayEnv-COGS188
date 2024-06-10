import gymnasium as gym
from matplotlib import pyplot as plt

env = gym.make('custom-highway-v0', render_mode="rgb_array")

env.unwrapped.configure({
    "collision_reward": -50,
})
env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print("Reward:", reward)
    print("Info:", info)
    print("Action:", action)
    env.render()
env.reset()

plt.imshow(env.render())
plt.show()
