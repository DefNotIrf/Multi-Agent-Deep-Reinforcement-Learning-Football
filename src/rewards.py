import gym
import numpy as np


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def compute_xg(ball_x, ball_y):
    """
    Compute Expected Goals (xG) based on ball position.
    Higher xG = better shooting position.
    Range: 0.0 (impossible) to 1.0 (certain goal)
    """
    goal_pos = np.array([1.0, 0.0])
    ball_pos = np.array([ball_x, ball_y])
    distance = np.linalg.norm(ball_pos - goal_pos)
    angle    = np.arctan2(0.05, max(distance, 0.01))
    xg       = (angle * 2.0) / (distance + 1.0)
    return float(np.clip(xg, 0.0, 1.0))


def compute_packing_rate(ball_x, last_ball_x, obs):
    """
    Counts opponents bypassed by ball progression.
    Higher packing = more defenders beaten = better pass.
    """
    opponents = obs['right_team']
    packed = 0
    for opp in opponents:
        opp_x = opp[0]
        if last_ball_x < opp_x < ball_x:
            packed += 1
    return packed / max(len(opponents), 1)


# -----------------------------------------------------------------------------
# SPECIALIST REWARD WRAPPER
# -----------------------------------------------------------------------------

class SpecialistRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, tactic="normal"):
        super().__init__(env)
        self.tactic          = tactic
        self.last_ball_owner = -1
        self.last_ball_x     = 0.0
        self.last_active     = -1
        print(f"--- Loaded Specialist Reward "
              f"(Thesis Strict Mode): {tactic.upper()} ---")

    def reward(self, reward):
        obs = self.env.unwrapped.observation()
        if obs is None:
            return reward

        ball_pos        = obs[0]['ball']
        ball_x          = ball_pos[0]
        ball_y          = ball_pos[1]
        ball_owned_team = obs[0]['ball_owned_team']

        ball_recovered  = (self.last_ball_owner != 0
                           and ball_owned_team == 0)
        ball_progression = ball_x - self.last_ball_x

        # Detect actual pass — active player changed
        current_active = obs[0]['active']
        actual_pass    = (ball_owned_team == 0
                          and self.last_active != current_active
                          and self.last_active != -1)

        # Compute xG for current ball position
        xg = compute_xg(ball_x, ball_y)

        # ── TACTIC 1: NORMAL ────────────────────────
        if self.tactic in ("normal", "normal2", "normal3"):

            # Ball progression toward goal
            if ball_progression > 0 and ball_owned_team == 0:
                reward += 0.9 * ball_progression

            # Zone crossing rewards
            if ball_owned_team == 0:
                if ball_x > 0.0 and self.last_ball_x <= 0.0:
                    reward += 0.5   # crossed halfway
                if ball_x > 0.5 and self.last_ball_x <= 0.5:
                    reward += 0.8   # entered final third
                if ball_x > 0.8 and self.last_ball_x <= 0.8:
                    reward += 1.0   # entered penalty box

            # Shot on target proxy
            if ball_x > 0.85 and abs(ball_y) < 0.2:
                reward += 0.5

            # xG position reward
            if ball_owned_team == 0 and xg > 0.05:
                reward += xg * 0.5

            # Actual pass detected
            if actual_pass:
                reward += 0.5
                if ball_progression > 0.02:
                    reward += 0.4  # progressive pass bonus

            # Possession retention drip
            if ball_owned_team == 0:
                reward += 0.01
                if ball_x > 0.0:
                    reward += 0.02  # in opponent half bonus

            # Stagnation penalty
            if (ball_owned_team == 0 and
                    abs(ball_progression) < 0.005):
                reward -= 0.03

            # Ball recovery reward
            if ball_recovered:
                reward += 0.2
            if ball_recovered and ball_x > 0.0:
                reward += 0.2

        # ── TACTIC 2: ALL-OUT ATTACK ─────────────────
        elif self.tactic in ("attack", "attack2", "attack3"):

            # Ball recovery in opponent half
            if ball_recovered and ball_x > 0:
                reward += 0.3

            # Zone crossing rewards stronger than normal
            if ball_owned_team == 0:
                if ball_x > 0.0 and self.last_ball_x <= 0.0:
                    reward += 1.0   # crossed halfway
                if ball_x > 0.5 and self.last_ball_x <= 0.5:
                    reward += 1.5   # entered final third
                if ball_x > 0.8 and self.last_ball_x <= 0.8:
                    reward += 2.0   # entered penalty box

            # Forward progression reward
            if ball_owned_team == 0 and ball_progression > 0:
                reward += 0.5 * ball_progression

            # xG position reward — main shooting incentive
            # Rewards being in dangerous positions
            if ball_owned_team == 0 and xg > 0.05:
                reward += xg * 1.5

            # Direct shot from good position bonus
            if ball_owned_team == 0 and xg > 0.10:
                reward += xg * 1.0  # extra bonus for great positions

            # Shot zone drip reward
            if ball_x > 0.7 and ball_owned_team == 0:
                reward += 0.05

            # Backward pass penalty
            if ball_owned_team == 0 and ball_progression < -0.05:
                reward -= 0.5

            # Stagnation penalty — stronger for attack
            if (ball_owned_team == 0
                    and abs(ball_progression) < 0.005):
                reward -= 0.05

            # Packing rate — reward bypassing defenders
            if ball_owned_team == 0 and ball_progression > 0.02:
                pack = compute_packing_rate(
                    ball_x, self.last_ball_x, obs[0]
                )
                if pack > 0:
                    reward += pack * 0.3

        # ── TACTIC 3: PARK THE BUS ──────────────────
        elif self.tactic in ("defense", "defense2", "defense3"):

            # Interception inside own penalty box
            if ball_recovered and ball_x < -0.7:
                reward += 0.5

            # Zone penalty — leave defensive third
            active_player_idx = obs[0]['active']
            active_player_pos = obs[0]['left_team'][
                active_player_idx
            ]
            if active_player_pos[0] > 0.0:
                reward -= 2.0   # attacking third = severe
            elif active_player_pos[0] > -0.3:
                reward -= 1.0   # middle third = strong

            # Clearance reward — get ball forward quickly
            if ball_owned_team == 0 and ball_x < -0.5:
                if ball_progression > 0.05:
                    reward += 0.3

            # Penalty for carrying ball back toward own goal
            if ball_owned_team == 0 and ball_x < -0.7:
                if ball_progression < -0.02:
                    reward -= 0.2

            # Reward being between ball and own goal
            ball_x = obs[0]['ball'][0]
            ball_y = obs[0]['ball'][1]
            active_x = active_player_pos[0]
            if (ball_owned_team == 1 and  # opponent has ball
                    active_x < ball_x and  # agent between ball and goal
                    active_x < -0.3):      # in defensive position
                reward += 0.05

            reward += 0.005  # survival reward
 
        # ── UPDATE HISTORY ──────────────────────────
        self.last_ball_owner = ball_owned_team
        self.last_ball_x     = ball_x
        self.last_active     = current_active

        return reward