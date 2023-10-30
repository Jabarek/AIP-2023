import random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import pickle


#episodes = Ile epizodów(int), is_traning = Czy to trening (True/false), render = Czy wyswietlać (true/False)
#allMoves = Wszystkie mozliwe ruchy czy wyciac niemozliwe (True/False)
def run(episodes, is_training=True, render=False, recording=False):

    if recording:
        env = gym.make('Taxi-v3', render_mode='rgb_array' if render else None)
        env.metadata['render_fps'] = 6
        env = gym.wrappers.RecordVideo(env,"przebieg")
    else:
        env = gym.make('Taxi-v3', render_mode='human' if render else None)
        env.metadata['render_fps'] = 6

    q = np.zeros([env.observation_space.n,env.action_space.n])
    learning_rate_a = 0.1
    discount_factor_g = 0.6
    epsilon = 0.1

    if is_training != True:
        f = open('taxiq.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    for i in range(episodes):
        state,info = env.reset()  # states: 0 to 500,
        terminated = False      # True when dropping off the passanger
        truncated = False       # True when actions > 200
        if is_training:
            rewardWhole = 0

        while(not terminated and not truncated):

            if is_training and random.uniform(0,1) < epsilon:
                    action = env.action_space.sample()  # actions: 0=down,1=up,2=right,3=left,4=pickup,5=drop off
            elif is_training:
                    action = np.argmax(q[state])  # actions: 0=down,1=up,2=right,3=left,4=pickup,5=drop off
            else:
                    action = np.argmax(q[state])  # actions: 0=down,1=up,2=right,3=left,4=pickup,5=drop off
            new_state,reward,terminated,truncated, info = env.step(action)
            if is_training:
                rewardWhole = rewardWhole + reward
            old_q = q[state,action]
            next_max = np.max(q[new_state])

            new_q = (1-learning_rate_a) * old_q + learning_rate_a*(reward+discount_factor_g*next_max)
            q[state,action] = new_q
            state = new_state
        if is_training:
            print("Nagroda w " + str(i+1) + " epizodzie: " + str(rewardWhole))
        env.close()
    if(is_training):
        f = open("taxiq.pkl","wb")
        pickle.dump(q, f)
        f.close()
#

if __name__ == '__main__':

    run(5000, is_training=True, render=False)
    run(1, is_training=False, render=True)
    #run(1, is_training=False, render=True,recording=True)