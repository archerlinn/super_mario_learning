import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import torch
import torch.nn as nn
import torch.optim as optim

import wandb  # <-- Weights & Biases

# -------------------------------
#     Check Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# -------------------------------
#   OLD-STYLE TO NEW-STYLE API
# -------------------------------
class OldEnvCompatibility(gym.Wrapper):
    """
    Converts old-style (obs, reward, done, info) into new-style
    (obs, reward, terminated, truncated, info).
    Also makes reset return (obs, info).
    """
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)   # old (obs)
        info = {}
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)  # old (obs, reward, done, info)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info


# -------------------------------
#  MAKE FULL-COLOR ENV
# -------------------------------
def make_env_full_color():
    """
    Creates Super Mario with no grayscale/resize wrappers
    so env.render() is colorful. We only do minimal old->new step conversion
    and limit actions to RIGHT_ONLY.
    """
    env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=False)

    # Remove default TimeLimit if present
    from gym.wrappers import TimeLimit
    while isinstance(env, TimeLimit):
        env = env.env

    if env.spec is not None:
        env.spec.max_episode_steps = None

    # Convert old 4-tuple -> new 5-tuple
    env = OldEnvCompatibility(env)
    env = JoypadSpace(env, RIGHT_ONLY)

    return env


# -------------------------------
#  MANUAL PREPROCESS FUNCTIONS
# -------------------------------
def preprocess_frame(obs):
    """
    Convert a color (H,W,3) frame into grayscale (1,84,84).
    """
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # => (H,W)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    # Add channel dimension => (1,84,84)
    return np.expand_dims(gray, axis=0)


def stack_frames(stacked_frames, new_frame):
    """
    Takes existing (4,84,84) buffer, appends new_frame (1,84,84) at end,
    discarding the oldest. Returns updated (4,84,84).
    """
    stacked_frames[:-1] = stacked_frames[1:]
    stacked_frames[-1] = new_frame
    return stacked_frames


# -------------------------------
#   DQN NETWORK
# -------------------------------
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
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # x => (batch, 4, 84, 84)
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)


# -------------------------------
#   DQN AGENT
# -------------------------------
class DQNAgent:
    def __init__(
        self,
        state_shape=(4,84,84),
        n_actions=6,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.99,
        lr=0.00025,
        exploration_max=1.0,
        exploration_min=0.02,
        exploration_decay=0.995,
        copy_every=1000,
        device=device
    ):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.max_memory_size = max_memory_size
        self.memory_sample_size = batch_size
        self.gamma = gamma

        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.device = device

        # Networks
        self.local_net = DQNSolver(self.state_shape, self.n_actions).to(self.device)
        self.target_net = DQNSolver(self.state_shape, self.n_actions).to(self.device)
        self.copy_model()

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.step_counter = 0
        self.copy_every = copy_every

        # Replay buffers
        self.STATE_MEM = torch.zeros((self.max_memory_size,) + self.state_shape)
        self.ACTION_MEM = torch.zeros((self.max_memory_size, 1))
        self.REWARD_MEM = torch.zeros((self.max_memory_size, 1))
        self.STATE2_MEM = torch.zeros((self.max_memory_size,) + self.state_shape)
        self.DONE_MEM = torch.zeros((self.max_memory_size, 1))

        self.ending_position = 0
        self.num_in_queue = 0

    def copy_model(self):
        self.target_net.load_state_dict(self.local_net.state_dict())

    def act(self, state_4ch):
        """Epsilon-greedy action selection."""
        if random.random() < self.exploration_rate:
            action = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(
                    state_4ch, dtype=torch.float32, device=self.device
                ).unsqueeze(0)  # => (1,4,84,84)
                q_values = self.local_net(state_tensor)
                action = q_values.argmax(dim=1).item()
        return action
    
    def end_of_episode(self):
        self.exploration_rate = max(
            self.exploration_min,
            self.exploration_rate * self.exploration_decay
        )

    def remember(self, s, a, r, s2, done):
        idx = self.ending_position
        self.STATE_MEM[idx]  = torch.tensor(s, dtype=torch.float32)
        self.ACTION_MEM[idx] = torch.tensor([a], dtype=torch.float32)
        self.REWARD_MEM[idx] = torch.tensor([r], dtype=torch.float32)
        self.STATE2_MEM[idx] = torch.tensor(s2, dtype=torch.float32)
        self.DONE_MEM[idx]   = torch.tensor([done], dtype=torch.float32)

        self.ending_position = (idx + 1) % self.max_memory_size
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def recall(self):
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATES  = self.STATE_MEM[idx].to(self.device)
        ACTIONS = self.ACTION_MEM[idx].to(self.device)
        REWARDS = self.REWARD_MEM[idx].to(self.device)
        STATES2 = self.STATE2_MEM[idx].to(self.device)
        DONE    = self.DONE_MEM[idx].to(self.device)
        return STATES, ACTIONS, REWARDS, STATES2, DONE

    def experience_replay(self):
        if self.num_in_queue < self.memory_sample_size:
            return

        # Periodically copy local -> target
        if self.step_counter % self.copy_every == 0:
            self.copy_model()

        S, A, R, S2, D = self.recall()

        with torch.no_grad():
            q_next = self.target_net(S2)
            q_next_max = q_next.max(dim=1, keepdim=True)[0]
            # target = r + gamma * maxQ(s') * (1 - done)
            target = R + (1 - D) * self.gamma * q_next_max

        q_vals = self.local_net(S)
        current = q_vals.gather(1, A.long())

        loss = self.loss_fn(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# -------------------------------
#   TRAIN
# -------------------------------
def run_training(
    num_episodes=500,
    render=True,
    load_model=None,    # Path to existing model or None
    save_model="mario_dqn.pth"
):
    import wandb
    wandb.init(project="super_mario_bros", name="manual-grayscale-v0")

    env = make_env_full_color()
    agent = DQNAgent(
        state_shape=(4,84,84),
        n_actions=env.action_space.n,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.99,                # Higher gamma to emphasize long-term reward (finishing)
        lr=0.00025,
        exploration_max=1.0,
        exploration_min=0.02,
        exploration_decay=0.995,   # Slow decay to ensure enough exploration
        copy_every=1000,
        device=device
    )

    # Optionally load a preexisting model
    if load_model is not None:
        import os
        if os.path.exists(load_model):
            print(f"Loading model from {load_model} ...")
            agent.local_net.load_state_dict(torch.load(load_model, map_location=device))
            agent.copy_model()  # sync target net
            print("Model loaded successfully. Training will resume from this checkpoint.")
        else:
            print(f"No model found at {load_model}; starting from scratch.")

    # Hyperparameters for reward shaping
    FLAG_REWARD = 5000      # Large reward for reaching the flag
    DISTANCE_SCALE = 0.1    # Scales reward for moving forward
    STEP_PENALTY = -0.01    # Small negative reward each step, encourages speed
    STUCK_PENALTY = -1      # Penalty if stuck for too many steps
    MAX_STUCK_STEPS = 60    # How many steps allowed without x_pos change
    DEATH_PENALTY = -20     # Penalty if Mario dies (done but no flag)

    rewards_history = []

    for ep in tqdm(range(num_episodes), desc="Training"):
        obs, info = env.reset()
        frame_1ch = preprocess_frame(obs)
        stacked_state = np.zeros((4,84,84), dtype=np.float32)
        stacked_state = stack_frames(stacked_state, frame_1ch)

        total_reward = 0.0
        done = False

        # Track x-position for distance reward & stuck penalty
        prev_x_pos = info.get("x_pos", 0)
        stuck_counter = 0

        while not done:
            if render:
                env.render()

            action = agent.act(stacked_state)
            next_obs, env_reward, terminated, truncated, info = env.step(action)

            # Start shaping with the environment's default reward (often 0 or small)
            shaped_reward = float(env_reward)

            # 1) Small negative penalty each step to encourage speed.
            shaped_reward += STEP_PENALTY

            # 2) Reward forward distance
            current_x_pos = info.get("x_pos", prev_x_pos)
            distance_gain = current_x_pos - prev_x_pos
            if distance_gain > 0:
                shaped_reward += DISTANCE_SCALE * distance_gain

            # 3) Stuck penalty if Mario doesn't move for too long
            if current_x_pos == prev_x_pos:
                stuck_counter += 1
                if stuck_counter >= MAX_STUCK_STEPS:
                    shaped_reward += STUCK_PENALTY
                    stuck_counter = 0
            else:
                stuck_counter = 0

            # 4) Huge reward if Mario reaches the flag
            flag_reached = info.get("flag_get", False)
            if flag_reached:
                print("Mario reached the flag! Awarding huge reward!")
                shaped_reward += FLAG_REWARD

            # Check if episode ended
            done = terminated or truncated

            # 5) If the episode ends and Mario did NOT reach the flag => penalty for dying
            if done and not flag_reached:
                shaped_reward += DEATH_PENALTY

            # Accumulate for logging
            total_reward += shaped_reward

            # Preprocess next observation
            next_frame_1ch = preprocess_frame(next_obs)
            next_stacked_state = stack_frames(stacked_state.copy(), next_frame_1ch)

            # IMPORTANT: Store the *shaped* reward in replay memory
            agent.remember(stacked_state, action, shaped_reward, next_stacked_state, float(done))
            agent.experience_replay()

            stacked_state = next_stacked_state
            prev_x_pos = current_x_pos
        
        agent.end_of_episode()

        # Save reward to history
        rewards_history.append(total_reward)

        # Save model periodically
        if (ep + 1) % 10 == 0:
            torch.save(agent.local_net.state_dict(), save_model)
            print(f"[Episode {ep+1}] Model saved to {save_model}")

        # Log metrics
        wandb.log({
            "Episode": ep,
            "Reward": total_reward,
            "Epsilon": agent.exploration_rate
        })

        print(f"Episode {ep+1} | Reward: {total_reward:.2f} "
              f"| Eps: {agent.exploration_rate:.3f}")

    env.close()
    wandb.finish()

    # Plot results
    plt.figure(figsize=(8,5))
    plt.plot(rewards_history, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Super Mario DQN Training (Manual Grayscale) - v0")
    plt.legend()
    plt.show()

    # Print final average
    if len(rewards_history) >= 50:
        avg_50 = np.mean(rewards_history[-50:])
    else:
        avg_50 = np.mean(rewards_history)
    print(f"Average reward over last 50 episodes: {avg_50:.2f}")


if __name__ == "__main__":
    run_training(num_episodes=5000, render=True, load_model="mario_dqn.pth")
