import torch
import torch.nn as nn
import numpy as np

# Actor Network - same as PPO, takes local observation
class Actor(nn.Module):
    def __init__(self, obs_dim=115, action_dim=19):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, obs):
        return self.network(obs)

    def get_action(self, obs):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(
            logits=logits
        )
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()


# Centralized Critic - takes GLOBAL state of all players
class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim=230):
        super(CentralizedCritic, self).__init__()
        # 230 = 115 obs per team x 2 teams
        self.network = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, global_state):
        return self.network(global_state)


# Global state builder
# Combines both teams observations into one vector
def build_global_state(obs):
    """
    Takes GRF raw observation dict and builds
    a 230-dim global state vector for the Critic.
    Concatenates left_team + right_team positions,
    velocities, ball position and possession.
    """
    left_team  = obs['left_team'].flatten()       # 10 players x 2 = 20
    right_team = obs['right_team'].flatten()      # 10 players x 2 = 20
    left_vel   = obs['left_team_direction'].flatten()   # 20
    right_vel  = obs['right_team_direction'].flatten()  # 20
    ball       = obs['ball']                      # 3 (x,y,z)
    ball_dir   = obs['ball_direction']            # 3
    possession = np.array([obs['ball_owned_team'],
                           obs['ball_owned_player']]) # 2

    global_state = np.concatenate([
        left_team, right_team,
        left_vel, right_vel,
        ball, ball_dir,
        possession
    ])

    # Pad or trim to exactly 230 dims
    if len(global_state) < 230:
        global_state = np.pad(
            global_state, (0, 230 - len(global_state))
        )
    else:
        global_state = global_state[:230]

    return global_state.astype(np.float32)