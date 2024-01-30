import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env  # noqa: F401

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = False
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=100,
            learning_rate=5e-4,
            gamma=0.9,
            verbose=2,
            tensorboard_log="highway_ppo/",
        )
        # Train the agent
        model.learn(total_timesteps=int(2e4))
        # Save the agent
        model.save("highway_ppo/model4")

    model = PPO.load("highway_ppo/model3")
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    IleZderzen = 0
    nagrody = []
    for _ in range(100):
        nagroda = 0
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            proba = _
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            #print(done)
            if done == True:
                IleZderzen+=1
            nagroda = nagroda+reward
            #env.render()
        nagrody.append(nagroda)
        print("Nagroda " + str(proba) + " : " + str(nagroda))
        print(nagrody)
        print(" = " + str(sum(nagrody) / len(nagrody)) )
        print(IleZderzen)