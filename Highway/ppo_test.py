import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from dqn_train import make_configure_env, env_kwargs
import torch.nn as nn
import os

# Inicjalizacja środowiska
env = make_configure_env(**env_kwargs)
n_actions = env.action_space.n
obs_space = env.reset()[0].shape

# Inicjalizacja TensorBoard Logger
log_path = os.path.join('logs', 'ppo')

# Konfiguracja modelu PPO
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

# Trening modelu
total_timesteps = 100000  # Zmień na żądaną liczbę kroków treningowych
model.learn(total_timesteps=total_timesteps)

# Zapisanie modelu
model.save("ppo_highway")
