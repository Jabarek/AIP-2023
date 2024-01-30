import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
#plt.ion()

# Definicja modelu sieci neuronowej
class DQN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_space[0] * obs_space[1], 128),  # Dostosowanie rozmiaru wejścia
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Spłaszczanie tensora do 1D
        return self.fc(x)

# Doświadczenie Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Funkcje pomocnicze
def select_action(state, policy_net, epsilon, n_actions):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = tuple(zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch[2])), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch[2] if s is not None])

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[3])

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Obliczanie wartości dla stanów, które nie są końcowe
    next_state_values = torch.zeros(batch_size)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Obliczanie oczekiwane wartości Q
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Obliczanie straty i optymalizacja modelu
    loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_model(env, policy_net, target_net, optimizer, memory, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
    episode_durations = []
    for i_episode in range(num_episodes):
        env_state = env.reset()
        state = torch.tensor([env_state[0]], dtype=torch.float)
        for t in count():
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * i_episode / epsilon_decay)
            action = select_action(state, policy_net, epsilon, env.action_space.n)
            next_state, reward, done, info = env.step(action.item())[:4]
            reward = torch.tensor([reward], dtype=torch.float)

            if not done:
                next_state = torch.tensor([next_state], dtype=torch.float)
            else:
                next_state = None

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma)
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

        if i_episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Zapisanie modelu po treningu
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

# Testowanie nauczonego modelu
def test_model(env, policy_net, num_episodes=5):
    for i in range(num_episodes):
        state = torch.tensor([env.reset()[0]], dtype=torch.float)
        for t in count():
            action = select_action(state, policy_net, 0, n_actions)  # Epsilon = 0 dla czystej polityki
            next_state, reward, done, info = env.step(action.item())[:4]
            if not done:
                next_state = torch.tensor([next_state], dtype=torch.float)
            else:
                next_state = None
                print(f"Episode {i + 1} finished after {t + 1} timesteps")
                break
            state = next_state
   
def test_model_with_recording(env, policy_net, num_episodes=10, max_steps_per_episode=100):
    env = RecordVideo(env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Ustawienie wyższego FPS dla renderingu

    for i in range(num_episodes):
        state = torch.tensor([env.reset()[0]], dtype=torch.float)
        for t in range(max_steps_per_episode):
            action = select_action(state, policy_net, 0, n_actions)  # Epsilon = 0 dla czystej polityki
            next_state, reward, done, _ = env.step(action.item())[:4]
            env.render()  # Renderowanie do zapisania wideo
            if done:
                break  # Zakończ epizod jeśli 'done' jest True
            state = torch.tensor([next_state], dtype=torch.float)
        print(f"Episode {i + 1} finished after {t + 1} timesteps")

# Testowanie nauczonego modelu
def test_model2(env, policy_net, num_episodes=15):
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

def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"], render_mode="rgb_array")
    env.configure(kwargs["config"])
    env.reset()
    return env

env_kwargs = {
    "id": "highway-fast-v0",
    "config": {
        "lanes_count": 3,
        "vehicles_count": 50,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
        "duration": 40,
        "policy_frequency": 1,
        "initial_spacing": 2,
        "collision_reward": -2,
        "right_lane_reward": 0.1,  
        "high_speed_reward": 0.6,  
        "lane_change_reward": 0.2,    
        "reward_speed_range": [20, 30],
        'scaling': 5.5,
        'screen_height': 150,
        'screen_width': 600,
        'vehicles_density': 1,
        'simulation_frequency': 15,
        'show_trajectories': False,
        "normalize_reward": False,
        "offroad_terminal": False
    },
}

#####################################
# Inicjalizacja środowiska i modeli #
#####################################

if __name__ == "__main__":
    env = make_configure_env(**env_kwargs)
    obs_space = env.reset()[0].shape
    n_actions = env.action_space.n

    policy_net = DQN(obs_space, n_actions)
    target_net = DQN(obs_space, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(10000)

    # Parametry treningu
    num_episodes = 300
    batch_size = 32
    gamma = 0.8
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 200

    # Wywołanie funkcji treningowej
    train_model(env, policy_net, target_net, optimizer, memory, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay)

    # Test nauczonego modelu
    print("\nTesting trained model:")
    #test_model(env, policy_net)
    test_model_with_recording(env, policy_net)

    # Zapisz model po zakończeniu treningu
    #save_model(policy_net, "trained_dqn_model.pth")

    print('Complete')
    env.close()
