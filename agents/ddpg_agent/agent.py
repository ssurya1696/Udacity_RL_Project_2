import numpy as np
import random
from collections import namedtuple, deque

import copy

from .models import Actor,Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

BUFFER_SIZE = int(1e6)  # replay buffer size
DEFAULT_BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 4       # how often to update the network
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

# Ou noise hyper params.

STARTING_THETA_DEFAULT =0.20
END_THETA_DEFAULT = 0.15
FACTOR_THETA_DEFAULT =0.99

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def annealing_generator(start: float,
                        end: float,
                        factor: float):
    decreasing = start > end

    eps = start
    while True:
        yield eps
        f = max if decreasing else min
        eps = f(end, factor*eps)


class Agent():
    '''
    DDPG Agent for solving the navegation system project.

    Skeleton adapted from Udacity exercise sample code. 

    '''
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 hyperparams,
                 seed=13):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.do_gradient_clipping_critic = hyperparams['do_gradient_clipping_critic']
        # Noise process
        self.noise = OUNoise(action_size, 
                             seed,
                             starting_theta= hyperparams.get('starting_theta',STARTING_THETA_DEFAULT),
                             end_theta= hyperparams.get('end_theta',END_THETA_DEFAULT),
                             factor_theta= hyperparams.get('factor_theta',FACTOR_THETA_DEFAULT),
                             noise_func=hyperparams['noise_generation_function'],)
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed,do_batch_norm=hyperparams['do_batch_norm']).to(device)
        self.actor_target = Actor(state_size, action_size, seed,do_batch_norm=hyperparams['do_batch_norm']).to(device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters())
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed,do_batch_norm=hyperparams['do_batch_norm']).to(device)
        self.critic_target = Critic(state_size, action_size, seed,do_batch_norm=hyperparams['do_batch_norm']).to(device)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC)

        # Replay memory
        self.batch_size = hyperparams.get('batch_size') or DEFAULT_BATCH_SIZE

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.batch_size, seed)
    
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.critic_criterion = nn.MSELoss(reduce=False)

        self.gamma = GAMMA

    def step(self,
             state: torch.Tensor,
             action: int,
             reward: float,
             next_state: torch.Tensor,
             done: bool):
        '''
        Function to be called after every interaction between the agent
        and the environment.
        
        Updates the memory and learns.
        '''
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            self.learn()

    def act(self,
            state: np.array,
            training: bool = True) -> torch.Tensor:
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            training (bool): whether the agent is training or not.
        """
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval() 
        with torch.no_grad():
            output_actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if training:
            output_actions += self.noise.sample()
        
        return np.clip(output_actions,-1,1)
        
    def reset(self):
        self.noise.reset()

    def learn(self):

        self.actor_local.train() 

        # 1) Sample experience tuples.

        experiences = self.memory.sample()
        states, mem_actions, rewards, next_states, dones = experiences

        # 2) Optimize the critic.

        # 2.1 Use the actor target network for estimating the actions 
        # and calculate their value using the critic local network.

        critic_output = self.critic_local(states,mem_actions)

        # 2.2 Use the critic target network for using the estimated value.

        actions_next = self.actor_target(next_states)
        critic_next_action_estimated_values = self.critic_target(next_states,actions_next)

        critic_estimated_values=rewards + (1-dones)*self.gamma*critic_next_action_estimated_values

        critic_loss = F.mse_loss(critic_output, critic_estimated_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.do_gradient_clipping_critic:
            torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # 3) Optimize the actor.

        

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        if self.t_step == 0:
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)
            

    def soft_update(self,
                    local_model: nn.Module,
                    target_model: nn.Module,
                    tau: float):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        This is an alterative to the original formulation of the DQN 
        paper, in which the target agent is updated with the local 
        model every X steps.
        
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, starting_theta, end_theta, factor_theta, noise_func,sigma=0.2,mu=0.):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = starting_theta
        self.theta_gen = annealing_generator(starting_theta, end_theta, factor_theta)
        self.noise_func = noise_func
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.theta = next(self.theta_gen)
        print('theta:',self.theta)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([self.noise_func() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

