import gymnasium as gym
import numpy as np
from tqdm import tqdm

n_train_episodes = 25000
n_eval_episodes = 100

learning_rate = 0.7
gamma = 0.95
max_steps = 99

env_seed = [
    16,
    54,
    165,
    177,
    191,
    191,
    120,
    80,
    149,
    178,
    48,
    38,
    6,
    125,
    174,
    73,
    50,
    172,
    100,
    148,
    146,
    6,
    25,
    40,
    68,
    148,
    49,
    167,
    9,
    97,
    164,
    176,
    61,
    7,
    54,
    55,
    161,
    131,
    184,
    51,
    170,
    12,
    120,
    113,
    95,
    126,
    51,
    98,
    36,
    135,
    54,
    82,
    45,
    95,
    89,
    59,
    95,
    124,
    9,
    113,
    58,
    85,
    51,
    134,
    121,
    169,
    105,
    21,
    30,
    11,
    50,
    65,
    12,
    43,
    82,
    145,
    152,
    97,
    106,
    55,
    31,
    85,
    38,
    112,
    102,
    168,
    123,
    97,
    21,
    83,
    158,
    26,
    80,
    63,
    5,
    81,
    32,
    11,
    28,
    148,
]

max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005


def initialize_q_table(state_space, action_space):
    qtable = np.zeros((state_space, action_space))
    return qtable

def greedy_policy(qtable, state):
    action = np.argmax(qtable[state][:])
    return action

def epsilon_greedy_policy(qtable, state, eps):
    prob = np.random.random()
    if prob > eps:
        return greedy_policy(qtable, state)
    action = np.random.randint(0, len(qtable[0]))
    return action

def train(env, qtable, n_train_episodes, max_steps, min_epsilon, max_epsilon, decay_rate, gamma, learning_rate):

    for episode in tqdm(range(n_train_episodes)):
        epsilon = max_epsilon - (episode * decay_rate)
        if epsilon < min_epsilon:
            epsilon = min_epsilon

        state, info = env.reset()
        truncated = False
        terminated = False

        for step in range(max_steps):

            action = epsilon_greedy_policy(qtable, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)

            current_value = qtable[state][action]
            qtable[state][action] = current_value + learning_rate * (reward + gamma * qtable[new_state][greedy_policy(qtable, new_state)] - current_value)

            if terminated or truncated:
                break

            state = new_state

    return qtable

def evaluation(env, n_eval_episodes, env_seed, max_steps, Qtable):
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        state, info = env.reset(seed=env_seed[episode])

        step = 0
        terminated = False
        truncated = False
        total_ep_rewards = 0

        for step in range(max_steps):
            action = greedy_policy(Qtable, state)
            new_state, reward, terminated, truncated, info = env.step(action)

            total_ep_rewards += reward

            if truncated or terminated:
                break

            state = new_state
        episode_rewards.append(total_ep_rewards)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

import time
import gymnasium as gym
import numpy as np

def evaluation2(Qtable, n_episodes=3, max_steps=99, delay=0.3):
    env = gym.make("Taxi-v4", render_mode="human")

    for _ in range(n_episodes):
        state, _ = env.reset()

        for _ in range(max_steps):
            action = np.argmax(Qtable[state])
            state, _, terminated, truncated, _ = env.step(action)
            time.sleep(delay)

            if terminated or truncated:
                time.sleep(0.5)
                break

    env.close()

env = gym.make("Taxi-v4", render_mode="rgb_array")

state_space = env.observation_space.n
action_space = env.action_space.n

qtable = initialize_q_table(state_space, action_space)

qtable_taxi = train(env, qtable, n_train_episodes, max_steps, min_epsilon, max_epsilon, decay_rate, gamma, learning_rate)

mean_reward, std_reward = evaluation(env, n_eval_episodes, env_seed, max_steps, qtable_taxi)

print (f"Mean reward: {mean_reward:2f} +/- {std_reward}")

env.close()

evaluation2(qtable_taxi)