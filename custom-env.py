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
    # print("Shape:", obs.shape)
    print("Reward:", reward)
    # print("Done:", done)
    # print("Truncated:", truncated)
    print("Info:", info)
    print("Action:", action)
    # print("Actions Array:", env.unwrapped.get_available_actions())
    env.render()
env.reset()
# for _ in range(3):
#     action = env.action_type.actions_indexes["IDLE"]
#     obs, reward, done, truncated, info = env.step(action)
    # env.render()

plt.imshow(env.render())
plt.show()
