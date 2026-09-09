import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

train_env = make_vec_env("LunarLander-v3", n_envs=16)

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)

model.learn(total_timesteps=1_000_000)

model_name = "lunar_lander_test"
model.save(model_name)

train_env.close()

eval_env = Monitor(gym.make("LunarLander-v3"))

mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

eval_env.close()