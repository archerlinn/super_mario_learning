import cv2
import gym
import numpy as np
from collections import deque
from gym import spaces
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
import gym_super_mario_bros
from tqdm import tqdm




class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame
    """
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._skip = skip
        self._buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._buffer.append(obs)
            total_reward += reward
            if done:
                break
        # take the max of the last two frames
        max_frame = np.max(np.stack(self._buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._buffer.clear()
        obs = self.env.reset()
        self._buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    Resize to 84x84 grayscale
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        # frame shape is (240, 256, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = np.expand_dims(frame, axis=-1)
        return frame


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image shape to [C, H, W]
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_shape[-1], obs_shape[0], obs_shape[1]),
            dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    """
    Stacks consecutive frames
    """
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), dtype=self.dtype)
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        self.n_steps = n_steps

    def reset(self):
        self.buffer = np.zeros_like(self.buffer, dtype=self.dtype)
        obs = self.env.reset()
        self.update_buffer(obs)
        return self.buffer

    def observation(self, observation):
        self.update_buffer(observation)
        return self.buffer

    def update_buffer(self, obs):
        channels = self.observation_space.shape[0] // self.n_steps
        self.buffer[:-channels] = self.buffer[channels:]
        self.buffer[-channels:] = obs


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Normalize pixel values from [0, 255] to [0, 1]
    """
    def observation(self, obs):
        return np.array(obs, dtype=np.float32) / 255.0
    
def make_env(env, skip=4, actions=RIGHT_ONLY):
    env = MaxAndSkipEnv(env, skip=skip)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = ScaledFloatFrame(env)
    env = BufferWrapper(env, 4)  # stack 4 frames
    env = JoypadSpace(env, actions)
    return env

class DQNSolver(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Compute size of the output of conv layers
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # forward pass
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)
    
class DQNAgent:
    def __init__(
        self, 
        state_shape,     # e.g. (4, 84, 84)
        n_actions,       # number of valid actions
        gamma=0.99,
        lr=1e-4,
        batch_size=32,
        replay_size=10000,
        min_replay_size=1000,
        eps_start=1.0,
        eps_end=0.02,
        eps_decay=1000000,
        target_update_interval=1000,
        device="cuda"
    ):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.min_replay_size = min_replay_size
        
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        self.n_actions = n_actions

        # Current (local) and target networks
        self.local_net = DQNSolver(state_shape, n_actions).to(device)
        self.target_net = DQNSolver(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.local_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.local_net.parameters(), lr=lr)

        # Replay memory
        self.memory = deque(maxlen=replay_size)
        self.steps_done = 0
        self.target_update_interval = target_update_interval

    def act(self, state):
        """
        Epsilon-greedy action selection.
        """
        # Update epsilon
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1.0 * self.steps_done / self.eps_decay)
        
        self.steps_done += 1
        
        if random.random() < eps_threshold:
            # random action
            return random.randrange(self.n_actions)
        else:
            # exploit: pick best action according to local_net
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.local_net(state_t)
                return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Store the transition in replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        """
        Sample a batch and train the local network, using the DDQN approach.
        """
        if len(self.memory) < self.min_replay_size:
            return

        # Sample a batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values
        q_values = self.local_net(states_t).gather(1, actions_t)

        # Double DQN:
        # Use local net to choose best action in next_state
        next_actions = self.local_net(next_states_t).argmax(dim=1, keepdim=True)

        # Use target net to evaluate Q-values of that next action
        target_q_values_next = self.target_net(next_states_t).gather(1, next_actions)

        # The TD target
        target = rewards_t + (1 - dones_t) * self.gamma * target_q_values_next

        # Loss
        loss = F.smooth_l1_loss(q_values, target.detach())

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_target_net(self):
        """
        Copy local_net weights into target_net.
        """
        self.target_net.load_state_dict(self.local_net.state_dict())

def train_mario(num_episodes=10000, skip=4):
    """
    Train a DDQN agent to play Super Mario Bros-1-1 using a restricted action space.
    """
    # Create the environment
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = make_env(env, skip=skip, actions=RIGHT_ONLY)
    
    # Extract shapes and sizes
    state_shape = env.observation_space.shape  # e.g. (4, 84, 84)
    n_actions = env.action_space.n            # e.g. 5 for RIGHT_ONLY
    
    import wandb
    wandb.init(project="super_mario_bros", name="manual-grayscale-v0")

    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        gamma=0.90,
        lr=0.00025,
        batch_size=32,
        replay_size=30000,
        min_replay_size=1000,  
        eps_start=0.02,
        eps_end=0.02,
        eps_decay=1000000,  # can tune
        target_update_interval=1000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()        
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            # Clipping the reward can sometimes help
            # reward = np.clip(reward, -1, 1)
            
            agent.remember(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            total_reward += reward
            
            if done:
                break

        # Update target network
        if episode % (agent.target_update_interval // 10) == 0:
            agent.sync_target_net()

        all_rewards.append(total_reward)

        if (episode+1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(f"Episode: {episode+1}, Avg Reward (last 10): {avg_reward:.2f}")
    
    wandb.log({
            "Episode": episode,
            "Reward": total_reward,
            "Epsilon": agent.eps_end
        })

    env.close()
    wandb.finish()

    return agent, all_rewards

if __name__ == "__main__":
    agent, rewards = train_mario(num_episodes=10000)
