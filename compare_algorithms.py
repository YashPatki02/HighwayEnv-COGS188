import numpy as np
import pandas as pd

# Load metrics from each algorithm
a2c_metrics = np.load("a2c_metrics.npy", allow_pickle=True).item()
ppo_metrics = np.load("ppo_metrics.npy", allow_pickle=True).item()
dqn_metrics = np.load("dqn_metrics.npy", allow_pickle=True).item()

# Print comparison
print("A2C Metrics:", a2c_metrics)
print("PPO Metrics:", ppo_metrics)
print("DQN Metrics:", dqn_metrics)

# Create a comparison table
data = {
    'Algorithm': ['A2C', 'PPO', 'DQN'],
    'Average Reward': [a2c_metrics['average_reward'], ppo_metrics['average_reward'], dqn_metrics['average_reward']],
    'Average Length': [a2c_metrics['average_length'], ppo_metrics['average_length'], dqn_metrics['average_length']],
    'Collision Rate': [a2c_metrics['collision_rate'], ppo_metrics['collision_rate'], dqn_metrics['collision_rate']],
    'Training Time': [a2c_metrics['training_time'], ppo_metrics['training_time'], dqn_metrics['training_time']],
    'Convergence Rate': [a2c_metrics['convergence_rate'], ppo_metrics['convergence_rate'], dqn_metrics['convergence_rate']],
    'Evaluation Average Reward': [a2c_metrics['eval_average_reward'], ppo_metrics['eval_average_reward'], dqn_metrics['eval_average_reward']],
    'Evaluation Collision Rate': [a2c_metrics['eval_collision_rate'], ppo_metrics['eval_collision_rate'], dqn_metrics['eval_collision_rate']],
}

df = pd.DataFrame(data)
print(df)

# Optionally, save the comparison table to a file
df.to_csv("algorithm_comparison.csv", index=False)
