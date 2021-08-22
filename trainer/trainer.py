from agents import DDPG_Agent as Agent
import unityagents
from collections import deque
import pickle
import numpy as np


class AgentTrainer():
    ''' Skeleton adapted from Udacity exercise sample code.'''
    def __init__(self,
                 env: unityagents.environment.UnityEnvironment,
                 max_t: int = 1000,
                 max_n_episodes: int = 1000,
                 target_score: float =  30
                 ):
        self.env = env
        self.max_t = max_t
        self.max_n_episodes = max_n_episodes
        self.brain_name = env.brain_names[0]
        self.target_score = target_score

    def env_step(self, action):

        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        # see if episode has finished
        done = env_info.local_done[0]

        return next_state, reward, done

    def train_agent(self,
                    agent: Agent,
                    hyperparams):
        """Deep Q-Learning.

        Params
        ======
            agent:Agent
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores

        solved = False

        for i_episode in range(1, self.max_n_episodes+1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            agent.reset()
            # get the current state
            state = env_info.vector_observations[0]

            score = 0
            for t in range(self.max_t):
                action = agent.act(state)
                next_state, reward, done = self.env_step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                scores_window_mean = np.mean(scores_window)
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                    i_episode, scores_window_mean))

            if np.mean(scores_window) >= self.target_score and not solved:
                solved = True
                print('Env solved in {:d} episodes! Avg Score: {:.2f}'.format(
                    i_episode-100, np.mean(scores_window)))
                break

        self.save_scores( hyperparams, f'''./{hyperparams['description']}''',scores)
        return agent

    def save_scores(self, hyperparams, filename,scores):
        obj = {'scores': scores,
               'hyperparams': hyperparams}

        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
