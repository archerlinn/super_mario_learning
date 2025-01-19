import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import wandb  # <-- Weights & Biases
from tqdm import tqdm

# ==============================
# Device
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==============================
# Old -> New API Wrapper
# ==============================
class OldEnvCompatibility(gym.Wrapper):
    """
    Converts old-style (obs, reward, done, info) into new-style
    (obs, reward, terminated, truncated, info).
    Also makes reset return (obs, info).
    """
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)  # old: returns just obs
        info = {}
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)  # old style
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

# ==============================
# Single Env Creation
# ==============================
def make_single_env():
    """
    Returns a callable that creates one Super Mario environment (with old->new wrapper).
    For vectorized usage.
    """
    def _init():
        env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=False)
        # Remove default TimeLimit if present
        from gym.wrappers import TimeLimit
        while isinstance(env, TimeLimit):
            env = env.env

        if env.spec is not None:
            env.spec.max_episode_steps = None

        env = OldEnvCompatibility(env)
        env = JoypadSpace(env, RIGHT_ONLY)
        return env
    return _init

# ==============================
# Vectorized Env (4 copies)
# ==============================
def make_vec_env(num_envs=4):
    """
    Creates a SyncVectorEnv of Super Mario with 4 parallel copies.
    """
    from gym.vector import SyncVectorEnv
    env_fns = [make_single_env() for _ in range(num_envs)]
    vec_env = SyncVectorEnv(env_fns)
    return vec_env

# ==============================
# Preprocessing
# ==============================
def preprocess_frame(obs):
    """
    Convert a color (H,W,3) frame into grayscale (1,84,84).
    """
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    return np.expand_dims(gray, axis=0)  # shape (1,84,84)

def stack_frames(stacked, new):
    """
    stacked: (4,84,84)
    new: (1,84,84)
    returns updated stacked
    """
    stacked[:-1] = stacked[1:]
    stacked[-1] = new
    return stacked

# ==============================
# PPO Policy
# ==============================
class PPOPolicy(nn.Module):
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
            nn.ReLU()
        )
        # Actor head => logits
        self.actor_head = nn.Linear(512, n_actions)
        # Critic head => scalar value
        self.critic_head = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        x: (batch,4,84,84)
        returns (logits, value)
        """
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_action_value(self, obs):
        """
        obs: (batch,4,84,84)
        returns action, log_prob, value
         each shape: (batch,)
        """
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values.squeeze(-1)

    def evaluate_actions(self, obs, actions):
        """
        For PPO update: (obs, actions) => (log_probs, entropy, values).
        """
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, entropy, values.squeeze(-1)

# ==============================
# PPO Agent
# ==============================
class PPOAgent:
    def __init__(
        self,
        input_shape=(4,84,84),
        n_actions=6,
        lr=2.5e-4,
        gamma=0.99,
        lam=0.95,
        clip_range=0.1,
        n_steps=128,
        n_epochs=4,
        batch_size=128
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.policy = PPOPolicy(input_shape, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Buffers for rollouts
        self.obs_buffer = []
        self.actions_buffer = []
        self.logp_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []
        self.dones_buffer = []

    def remember(self, obs, action, logp, reward, value, done):
        """
        obs: shape (num_envs, 4, 84, 84)
        action, logp, reward, value, done: shape (num_envs,)
        """
        self.obs_buffer.append(obs)
        self.actions_buffer.append(action)
        self.logp_buffer.append(logp)
        self.rewards_buffer.append(reward)
        self.values_buffer.append(value)
        self.dones_buffer.append(done)

    def finish_trajectory(self, last_value):
        """
        Compute GAE-lambda advantages for each environment in the vector.
          last_value: shape (num_envs,)
        """
        # Convert to torch
        rewards = torch.tensor(self.rewards_buffer, dtype=torch.float32, device=device)  # (T, num_envs)
        values = torch.tensor(self.values_buffer, dtype=torch.float32, device=device)    # (T, num_envs)
        dones = torch.tensor(self.dones_buffer, dtype=torch.float32, device=device)      # (T, num_envs)

        T, N = rewards.shape  # T steps, N envs
        advantages = torch.zeros_like(rewards, device=device)  # (T, N)
        gae = torch.zeros(N, device=device)                    # (N,)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_value  # shape (N,)
            else:
                next_values = values[t+1] * (1 - dones[t+1])
            delta = rewards[t] + self.gamma * next_values - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = values + advantages
        return advantages, returns

    def learn(self, advantages, returns):
        """
        Flatten (T, N) => (T*N,).
        Then run PPO updates in mini-batches.
        """
        obs_arr = torch.tensor(np.array(self.obs_buffer), dtype=torch.float32, device=device)  # shape (T, N, 4,84,84)
        T, N, C, H, W = obs_arr.shape
        obs_arr = obs_arr.view(T*N, C, H, W)

        actions_arr = torch.tensor(np.array(self.actions_buffer), dtype=torch.long, device=device)  # (T, N)
        actions_arr = actions_arr.view(T*N)

        old_logp_arr = torch.tensor(np.array(self.logp_buffer), dtype=torch.float32, device=device) # (T, N)
        old_logp_arr = old_logp_arr.view(T*N)

        advantages_arr = advantages.view(T*N)
        returns_arr = returns.view(T*N)

        total_steps = T*N
        for _ in range(self.n_epochs):
            idxs = np.arange(total_steps)
            np.random.shuffle(idxs)

            start = 0
            while start < total_steps:
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                batch_obs = obs_arr[batch_idx]
                batch_actions = actions_arr[batch_idx]
                batch_old_logp = old_logp_arr[batch_idx]
                batch_adv = advantages_arr[batch_idx]
                batch_ret = returns_arr[batch_idx]

                new_logp, entropy, values = self.policy.evaluate_actions(batch_obs, batch_actions)
                ratio = torch.exp(new_logp - batch_old_logp)
                unclipped = ratio * batch_adv
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = F.mse_loss(values, batch_ret)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                start = end

    def reset_buffers(self):
        self.obs_buffer = []
        self.actions_buffer = []
        self.logp_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []
        self.dones_buffer = []

    def get_action_value(self, obs_tensor):
        """
        obs_tensor: shape (num_envs,4,84,84)
        Return (actions, log_probs, values) => numpy arrays
        """
        with torch.no_grad():
            actions, log_probs, values = self.policy.get_action_value(obs_tensor)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

# ==============================
#  PPO Training with 4 Env
# ==============================
def run_ppo_vec_training(
    total_timesteps=200_000,
    num_envs=4,
    rollout_steps=128,
    render=False
):
    """
    We'll collect rollouts from multiple parallel envs.
    """
    wandb.init(project="super_mario_bros", name="ppo-4env-grayscale")

    # Create 4 parallel Mario envs
    env = make_vec_env(num_envs=num_envs)
    n_actions = env.single_action_space.n

    # Create PPO agent
    agent = PPOAgent(
        input_shape=(4,84,84),
        n_actions=n_actions,
        lr=2.5e-4,
        gamma=0.99,
        lam=0.95,
        clip_range=0.1,
        n_steps=rollout_steps,
        n_epochs=4,
        batch_size=128
    )

    # Reward shaping constants
    FLAG_REWARD = 5000
    DISTANCE_SCALE = 0.1
    STEP_PENALTY = -0.01
    STUCK_PENALTY = -2
    MAX_STUCK_STEPS = 60
    DEATH_PENALTY = -20

    # Track stats for each env
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_counts = np.zeros(num_envs, dtype=np.int32)

    # For stuck detection
    stuck_counters = np.zeros(num_envs, dtype=np.int32)
    prev_x_pos = np.zeros(num_envs, dtype=np.float32)

    # Reset the vector env
    reset_return = env.reset()
    if isinstance(reset_return, tuple) and len(reset_return) == 2:
        obs_array, info_array = reset_return
    else:
        obs_array = reset_return
        info_array = [{} for _ in range(num_envs)]  # Ensure compatibility

    prev_x_pos = np.zeros(num_envs, dtype=np.float32)

    # Prepare stacked states (num_envs,4,84,84)
    stacked_states = np.zeros((num_envs, 4, 84, 84), dtype=np.float32)
    for i in range(num_envs):
        frame_gray = preprocess_frame(obs_array[i])
        stacked_states[i] = stack_frames(stacked_states[i], frame_gray)
        # Attempt to read x_pos if available
        if isinstance(info_array[i], dict):
            prev_x_pos[i] = info_array[i].get("x_pos", 0.0)
        else:
            prev_x_pos[i] = 0.0

    global_steps = 0
    pbar = tqdm(total=total_timesteps, desc="Training PPO (4-env)")

    while global_steps < total_timesteps:
        agent.reset_buffers()

        # Collect rollout_steps from the 4 envs
        for step_i in range(agent.n_steps):
            if render:
                env.render()  # Might show multiple windows or a combined view

            # (num_envs,4,84,84) => torch
            obs_tensor = torch.tensor(stacked_states, dtype=torch.float32, device=device)
            actions, log_probs, values = agent.get_action_value(obs_tensor)

            # Step the env
            step_return = env.step(actions)

            # Because of different Gym versions, handle 4 or 5 elements
            if len(step_return) == 5:
                next_obs_array, rewards_array, done_array, truncated_array, info_array = step_return
                # Combine done & truncated
                done_array = np.logical_or(done_array, truncated_array)
            elif len(step_return) == 4:
                next_obs_array, rewards_array, done_array, info_array = step_return
            else:
                # fallback if older version returns (obs, reward, done)
                next_obs_array, rewards_array, done_array = step_return
                info_array = [{} for _ in range(num_envs)]

            # Convert to arrays
            rewards_array = np.array(rewards_array, dtype=np.float32)
            done_array = np.array(done_array, dtype=bool)

            # Reward shaping
            shaped_rewards = np.zeros(num_envs, dtype=np.float32)
            for i in range(num_envs):
                shaped = rewards_array[i]
                # Step penalty
                shaped += STEP_PENALTY

                # Ensure info_array is treated correctly based on returned format
                if isinstance(info_array, list):
                    current_info = info_array[i] if i < len(info_array) and isinstance(info_array[i], dict) else {}
                else:
                    current_info = info_array if isinstance(info_array, dict) else {}

                # Extract x_pos safely
                current_x = current_info.get("x_pos", prev_x_pos[i])

                distance_gain = current_x - prev_x_pos[i]
                if distance_gain > 0:
                    shaped += DISTANCE_SCALE * distance_gain

                # Stuck
                if current_x == prev_x_pos[i]:
                    stuck_counters[i] += 1
                    if stuck_counters[i] >= MAX_STUCK_STEPS:
                        shaped += STUCK_PENALTY
                        stuck_counters[i] = 0
                else:
                    stuck_counters[i] = 0

                # Flag
                flag_reached = False
                if isinstance(info_array[i], dict):
                    flag_reached = info_array[i].get("flag_get", False)
                if flag_reached:
                    shaped += FLAG_REWARD

                # Death penalty if done but no flag
                if done_array[i] and not flag_reached:
                    shaped += DEATH_PENALTY

                shaped_rewards[i] = shaped
                episode_rewards[i] += shaped
                prev_x_pos[i] = current_x

            # Store transitions in PPO buffer
            agent.remember(
                obs=stacked_states.copy(),    # shape (num_envs,4,84,84)
                action=actions,              # shape (num_envs,)
                logp=log_probs,
                reward=shaped_rewards,
                value=values,
                done=done_array.astype(np.float32)
            )

            # Update stacked states for next step
            next_stacked = stacked_states.copy()
            for i in range(num_envs):
                # Preprocess new obs
                frame_gray = preprocess_frame(next_obs_array[i])
                next_stacked[i] = stack_frames(next_stacked[i], frame_gray)

                # If done => environment might have auto-reset internally
                if done_array[i]:
                    # Log episode
                    episode_counts[i] += 1
                    wandb.log({
                        "episode_reward": episode_rewards[i],
                        "env_id": i,
                        "episode_count": episode_counts[i],
                        "global_steps": global_steps
                    })
                    episode_rewards[i] = 0.0
                    stuck_counters[i] = 0

                    # Attempt to reset x_pos
                    if isinstance(info_array[i], dict):
                        prev_x_pos[i] = info_array[i].get("x_pos", 0.0)
                    else:
                        prev_x_pos[i] = 0.0

            stacked_states = next_stacked
            global_steps += num_envs
            pbar.update(num_envs)

            if global_steps >= total_timesteps:
                break

        # After collecting rollout_steps => compute advantage
        with torch.no_grad():
            obs_tensor = torch.tensor(stacked_states, dtype=torch.float32, device=device)
            # same get_action_value => returns (a, logp, v)
            # we only need v
            _, _, last_values = agent.policy.get_action_value(obs_tensor)

        # For envs that are done, last_value = 0
        for i in range(num_envs):
            # If done => 0
            if done_array[i]:
                last_values[i] = 0.0

        advantages, returns = agent.finish_trajectory(last_values)
        agent.learn(advantages, returns)

    env.close()
    wandb.finish()
    pbar.close()

    print("Training complete!")
    # You could track average episode rewards in a separate list if desired.
    # This snippet logs each env's reward to wandb at the end of each episode.


# ==============================
#  Main Entry
# ==============================
if __name__ == "__main__":
    run_ppo_vec_training(
        total_timesteps=200_000,
        num_envs=4,
        rollout_steps=128,
        render=False
    )
