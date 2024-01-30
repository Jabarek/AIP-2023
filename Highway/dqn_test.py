import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
from dqn_train import DQN, select_action, make_configure_env, env_kwargs
from itertools import count

# Testowanie nauczonego modelu
def test_model2(env, policy_net, num_episodes=100):
    IleZderzen = 0
    nagrody = []
    for i in range(num_episodes):
        nagroda = 0
        state = torch.tensor([env.reset()[0]], dtype=torch.float)
        for t in count():
            action = select_action(state, policy_net, 0, n_actions)  # Epsilon = 0 dla czystej polityki
            next_state, reward, done,truncated, info = env.step(action.item())
            if done == True :
                IleZderzen+=1
            nagroda = nagroda+reward
            if not done and not truncated:
                next_state = torch.tensor([next_state], dtype=torch.float)
            else:
                next_state = None
                print(f"Episode {i + 1} finished after {t + 1} timesteps")
                break
            state = next_state
        nagrody.append(nagroda)
        print("Nagroda " + str(i) + " : " + str(nagroda))
        print(nagrody)
        print(" = " + str(sum(nagrody) / len(nagrody)) )
        print("Ile razy kraksa: " + str(IleZderzen))

# Inicjalizacja środowiska
#env = gym.make("highway-fast-v0", render_mode='rgb_array')
env = make_configure_env(**env_kwargs)
n_actions = env.action_space.n
obs_space = env.reset()[0].shape

# Wczytanie i testowanie modelu
model_path = 'D:\\ISA_w4n\\IIsem\\AIP\\Highway\\trained_dqn_model_1.pth'
loaded_model = DQN(obs_space, n_actions)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()

print("Visualizing the loaded model")
#test_model2(env, loaded_model)
#test_model_with_recording(env, loaded_model)
print("Complete")
env.close()
