import gymnasium as gym
import numpy as np
from tqdm import tqdm
from stable_baselines3.common.env_util import make_vec_env

def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable

def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state])
    return action

def epsilon_greedy_policy(Qtable, state, epsilon):
    if np.random.random() > epsilon:
        return greedy_policy(Qtable, state)
    action = np.random.randint(0, len(Qtable[state][:]))
    return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)

            current_value = Qtable[state][action] 
            Qtable[state][action] = current_value + learning_rate * (reward + gamma * Qtable[new_state][greedy_policy(Qtable, new_state)] - current_value)

            if terminated or truncated:
                break

            state = new_state
    return Qtable

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
      if seed:
        state, info = env.reset(seed=seed[episode])
      else:
        state, info = env.reset()
      step = 0
      truncated = False
      terminated = False
      total_rewards_ep = 0

      for step in range(max_steps):
        action = greedy_policy(Q, state)
        new_state, reward, terminated, truncated, info = env.step(action)
        total_rewards_ep += reward

        if terminated or truncated:
          break
        state = new_state
      episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

n_training_episodes = 10000
n_eval_episodes = 100

learning_rate = 0.7

gamma = 0.95
max_steps = 99

min_epsilon = 0.05
max_epsilon = 1
decay_epsilon = 0.0005

eval_seed = []

train_env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery = False,
    render_mode="rgb_array"
)

state_space = train_env.observation_space.n
action_space = train_env.action_space.n

Qtable_frozen_lake = initialize_q_table(state_space, action_space)

Qtable_frozen_lake = train(n_training_episodes, min_epsilon, max_epsilon, decay_epsilon, train_env, max_steps, Qtable_frozen_lake)

mean_reward, std_reward = evaluate_agent(train_env, max_steps, n_eval_episodes, Qtable_frozen_lake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")