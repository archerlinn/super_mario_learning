import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.tensorboard import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ====================
#  NEW-API COMPAT WRAPPER
# ====================
class OldEnvCompatibility(gym.Wrapper):
    """
    Converts old-style (obs, reward, done, info) into new-style
    (obs, reward, terminated, truncated, info).
    Also makes reset return (obs, info).
    """
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)  # old-style reset -> single value
        info = {}
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)  # old style, 4-tuple
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info


# ========================== #
#    WRAPPERS FOR PREPROCESS #
# ========================== #

class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame (frames are maxed over the two most recent).
    """
    def __init__(self, env=None, skip=4):
        super().__init__(env)
        self._skip = skip
        obs_shape = env.observation_space.shape
        self._obs_buffer = np.zeros((2,) + obs_shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if term or trunc:
                terminated = term
                truncated = trunc
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    """
    Convert to grayscale, resize to 84x84 (single 2D array).
    """
    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs  # shape => (84,84)


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Convert [0,255] uint8 to [0,1] float32
    """
    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Expand dims from (84,84) -> (1,84,84) for a single grayscale frame.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(1, 84, 84),
            dtype=np.float32
        )

    def observation(self, obs):
        # (84,84)->(1,84,84)
        return np.expand_dims(obs, axis=0)


class BufferWrapper(gym.ObservationWrapper):
    """
    Stacks the last n frames. Each frame is (1,84,84),
    so final shape after 4 frames is (4,84,84).
    """
    def __init__(self, env, n_steps=4, dtype=np.float32):
        super().__init__(env)
        self.n_steps = n_steps

        # We store frames in a buffer of shape=(4, 84, 84),
        # flattening the single channel dimension
        self.buffer = np.zeros((n_steps, 84, 84), dtype=dtype)

        low  = np.zeros((n_steps, 84, 84), dtype=dtype)
        high = np.ones((n_steps, 84, 84), dtype=dtype)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)  # obs => shape (1,84,84)
        obs = obs[0]  # => shape (84,84)
        self.buffer[...] = 0
        self.buffer[-1] = obs
        return self.buffer, info

    def observation(self, obs):
        # obs => (1,84,84)
        obs = obs[0]  # => (84,84)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs
        return self.buffer


def make_env():
    """
    Create a Super Mario environment using the old 4-tuple step API,
    forcibly unwrapping TimeLimit, then convert it to new 5-tuple style
    with OldEnvCompatibility, and apply custom wrappers.
    """
    # 1) Create env
    env = gym.make("SuperMarioBros-1-1-v3", apply_api_compatibility=False)
    
    # 2) Force-unwrap TimeLimit if it's there
    from gym.wrappers import TimeLimit
    while isinstance(env, TimeLimit):
        env = env.env

    # 3) Remove default time limit
    if env.spec is not None:
        env.spec.max_episode_steps = None

    # 4) Convert old 4-tuple to new 5-tuple
    env = OldEnvCompatibility(env)

    # 5) Limit action space
    env = JoypadSpace(env, RIGHT_ONLY)

    # 6) Apply wrappers
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame84(env)
    env = ScaledFloatFrame(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return env

# ================
#    DQN Network
# ================
import torch
class DQNSolver(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        # shape => (4,84,84) if 4 frames stacked
        test_input = torch.zeros(1, *shape)  # => shape (1,4,84,84)
        o = self.conv(test_input)           # => shape (1,64,7,7) for example
        return int(np.prod(o.size()))

    def forward(self, x):
        # x => (batch_size,4,84,84)
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)


# ===================== #
#    DQN AGENT CLASS    #
# ===================== #
class DQNAgent:
    def __init__(
        self,
        state_space,
        action_space,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.90,
        lr=0.00025,
        exploration_max=1.0,
        exploration_min=0.02,
        exploration_decay=0.99,
        copy_every=1000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.state_space = state_space      # (4,84,84)
        self.action_space = action_space

        self.max_memory_size = max_memory_size
        self.memory_sample_size = batch_size
        self.gamma = gamma

        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.device = device

        # Networks
        self.local_net = DQNSolver(self.state_space, self.action_space).to(self.device)
        self.target_net = DQNSolver(self.state_space, self.action_space).to(self.device)
        self.copy_model()  # start identical

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.step_counter = 0
        self.copy_every = copy_every

        # Replay memory
        # => shape: (30000,4,84,84)
        self.STATE_MEM = torch.zeros((self.max_memory_size,) + self.state_space, dtype=torch.float32)
        self.ACTION_MEM = torch.zeros((self.max_memory_size, 1), dtype=torch.float32)
        self.REWARD_MEM = torch.zeros((self.max_memory_size, 1), dtype=torch.float32)
        self.STATE2_MEM = torch.zeros((self.max_memory_size,) + self.state_space, dtype=torch.float32)
        self.DONE_MEM = torch.zeros((self.max_memory_size, 1), dtype=torch.float32)

        self.ending_position = 0
        self.num_in_queue = 0

    def copy_model(self):
        self.target_net.load_state_dict(self.local_net.state_dict())

    def act(self, state):
        """
        Epsilon-greedy action selection.
        state => (4,84,84)
        """
        if random.random() < self.exploration_rate:
            action = random.randrange(self.action_space)
        else:
            with torch.no_grad():
                # Add batch dimension => (1,4,84,84)
                state_batched = state.unsqueeze(0).to(self.device)
                q_values = self.local_net(state_batched)
                action = torch.argmax(q_values, dim=1).item()

        self.step_counter += 1
        # Decay epsilon
        self.exploration_rate = max(
            self.exploration_min, 
            self.exploration_rate * self.exploration_decay
        )
        return action

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in replay. 
        state => (4,84,84)
        next_state => (4,84,84)
        """
        idx = self.ending_position
        self.STATE_MEM[idx]  = state.cpu().float()
        self.ACTION_MEM[idx] = torch.tensor([action], dtype=torch.float32)
        self.REWARD_MEM[idx] = torch.tensor([reward], dtype=torch.float32)
        self.STATE2_MEM[idx] = next_state.cpu().float()
        self.DONE_MEM[idx]   = torch.tensor([done], dtype=torch.float32)

        self.ending_position = (idx + 1) % self.max_memory_size
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def recall(self):
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        states = self.STATE_MEM[idx].to(self.device)
        actions = self.ACTION_MEM[idx].to(self.device)
        rewards = self.REWARD_MEM[idx].to(self.device)
        next_states = self.STATE2_MEM[idx].to(self.device)
        dones = self.DONE_MEM[idx].to(self.device)
        return states, actions, rewards, next_states, dones

    def experience_replay(self):
        # periodically sync
        if self.step_counter % self.copy_every == 0:
            self.copy_model()

        if self.num_in_queue < self.memory_sample_size:
            return

        states, actions, rewards, next_states, dones = self.recall()

        with torch.no_grad():
            q_next = self.target_net(next_states)
            q_next_max = q_next.max(dim=1, keepdim=True)[0]
            target = rewards + (1 - dones) * self.gamma * q_next_max

        q_vals = self.local_net(states)
        q_a = q_vals.gather(1, actions.long())

        loss = self.loss_fn(q_a, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# =====================
#   TRAINING FUNCTION
# =====================
def run_training(num_episodes=500, render=True, save_model_path="mario_dqn.pth"):
    writer = SummaryWriter(log_dir="logs")  # "logs/" is the default folder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    env = make_env()
    obs_shape = env.observation_space.shape  # e.g. (4,84,84)
    n_actions = env.action_space.n

    agent = DQNAgent(
        state_space=obs_shape,
        action_space=n_actions,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.90,
        lr=0.00025,
        exploration_max=1.0,
        exploration_min=0.02,
        exploration_decay=0.99,
        copy_every=1000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    rewards_history = []

    for ep in tqdm(range(num_episodes), desc="Training"):
        # new-style reset => (obs, info)
        obs, info = env.reset()  # shape => (4,84,84)
        state_t = torch.tensor(obs, dtype=torch.float32)  # => (4,84,84)
        total_reward = 0.0
        done = False

        while not done:
            if render:
                env.render()

            action = agent.act(state_t)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # next_obs => (4,84,84)
            next_state_t = torch.tensor(next_obs, dtype=torch.float32)

            agent.remember(state_t, action, reward, next_state_t, float(done))
            agent.experience_replay()

            state_t = next_state_t

        rewards_history.append(total_reward)
        print(f"Episode {ep + 1} | Reward: {total_reward:.2f} "
              f"| Eps: {agent.exploration_rate:.3f}")

    env.close()

    # Save the model
    torch.save(agent.local_net.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

    # Plot training curve
    plt.figure(figsize=(8,5))
    plt.plot(rewards_history, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Super Mario DQN Training')
    plt.legend()
    plt.show()

    # Print final stats
    if len(rewards_history) >= 50:
        avg_50 = np.mean(rewards_history[-50:])
    else:
        avg_50 = np.mean(rewards_history)
    print(f"Average reward over last 50 episodes: {avg_50:.2f}")


if __name__ == "__main__":
    run_training(
        num_episodes=500,
        render=True,         # Set True to attempt a live window
        save_model_path="mario_dqn.pth"
    )