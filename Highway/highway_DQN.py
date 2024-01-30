import gymnasium as gym
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo

env_kwargs = {
    "id": "highway-fast-v0",
    "config": {
        "lanes_count": 3,
        "vehicles_count": 15,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
        "duration": 40,
        "policy_frequency": 2,
        "initial_spacing": 2,
        "collision_reward": -2,
        "right_lane_reward": 0.2,  
        "high_speed_reward": 0.6,  
        "lane_change_reward": 0,    
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "offroad_terminal": False
    },
}

env_kwargs2 = {
    "id": "highway-fast-v0",
    "config": {
        "lanes_count": 3,
        "vehicles_count": 15,
        "observation": {
            "type": "Kinematics",
        },
    "action": {
        "type": "DiscreteMetaAction",
    },
      "lanes_count": 4,
      "vehicles_count": 50,
      "duration": 40,  # [s]
      "initial_spacing": 2,
      "collision_reward": -1,  # The reward received when colliding with a vehicle.
      "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
      "simulation_frequency": 15,  # [Hz]
      "policy_frequency": 1,  # [Hz]
      "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
      "screen_width": 600,  # [px]
      "screen_height": 150,  # [px]
      "centering_position": [0.3, 0.5],
      "scaling": 5.5,
      "show_trajectories": False,
      "render_agent": True,
      "offscreen_rendering": False
    },
}


def test_model_with_recording(env, model, num_episodes=10):
    env = RecordVideo(env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Ustawienie wyższego FPS dla renderingu

    #test_model_with_recording(env, policy_net)
    IleZderzen = 0
    nagrody = []
    for _ in range(num_episodes):
        nagroda = 0
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            proba = _
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done == True:
                IleZderzen+=1
            nagroda = nagroda+reward
            env.render()
        nagrody.append(nagroda)
    print("Nagroda " + str(proba) + " : " + str(nagroda))
    print(nagrody)
    print(" = " + str(sum(nagrody) / len(nagrody)) )
    print(IleZderzen)

def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"], render_mode="rgb_array")
    env.configure(kwargs["config"])
    env.reset()
    return env

if __name__ == "__main__":
    train = False
    if train:
      #env = make_configure_env(**env_kwargs2)
      env = gym.make("highway-fast-v0", render_mode ="rgb_array")
      model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=180,
                    batch_size=128,
                    gamma=0.9,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log="highway_dqn/")
      model.learn(int(3e4))
      model.save("highway_dqn/model4")

# Load and test saved model
model = DQN.load("highway_dqn/model4")
if train == False:
  #env = make_configure_env(**env_kwargs2)
  env = gym.make("highway-fast-v0", render_mode ="rgb_array")
  #test_model_with_recording(env, model)
  IleZderzen = 0
  nagrody = []
  for _ in range(10):
      nagroda = 0
      obs, info = env.reset()
      done = truncated = False
      while not (done or truncated):
          proba = _
          action, _ = model.predict(obs)
          obs, reward, done, truncated, info = env.step(action)
          #print(truncated)
          #print(reward)
          if done == True:
              IleZderzen+=1
          nagroda = nagroda+reward
          env.render()
      nagrody.append(nagroda)
      print("Nagroda " + str(proba) + " : " + str(nagroda))
      print(nagrody)
      print(" = " + str(sum(nagrody) / len(nagrody)) )
      print(IleZderzen)