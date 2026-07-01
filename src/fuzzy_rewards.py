import gym
import numpy as np

# -----------------------------------------------------------------------------
# SECTION 1: FUZZY MEMBERSHIP FUNCTIONS
# -----------------------------------------------------------------------------

def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a + 1e-9)
    else:
        return (c - x) / (c - b + 1e-9)


# --- Opponent Density Membership Functions ---
def opponent_density_low(density):
    return trimf(density, -0.1, 0.0, 0.4)

def opponent_density_medium(density):
    return trimf(density, 0.2, 0.5, 0.8)

def opponent_density_high(density):
    return trimf(density, 0.6, 1.0, 1.1)


# --- Distance to Goal Membership Functions ---
def distance_to_goal_near(dist):
    return trimf(dist, -0.1, 0.0, 0.4)

def distance_to_goal_moderate(dist):
    return trimf(dist, 0.2, 0.5, 0.8)

def distance_to_goal_far(dist):
    return trimf(dist, 0.6, 1.0, 1.1)


# --- Passing Lane Openness Membership Functions ---
def lane_closed(openness):
    return trimf(openness, -0.1, 0.0, 0.4)

def lane_partially_open(openness):
    return trimf(openness, 0.2, 0.5, 0.8)

def lane_open(openness):
    return trimf(openness, 0.6, 1.0, 1.1)


# -----------------------------------------------------------------------------
# SECTION 2: CONTEXT VARIABLE COMPUTATION
# -----------------------------------------------------------------------------

def compute_opponent_density(obs, radius=0.3):
    ball_pos = np.array(obs['ball'][:2])
    opponents = obs['right_team']
    count = 0
    for opp in opponents:
        dist = np.linalg.norm(np.array(opp[:2]) - ball_pos)
        if dist < radius:
            count += 1
    return min(count / 5.0, 1.0)


def compute_distance_to_goal(obs):
    ball_pos = np.array(obs['ball'][:2])
    opponent_goal = np.array([1.0, 0.0])
    dist = np.linalg.norm(ball_pos - opponent_goal)
    return min(dist / 2.0, 1.0)


def compute_passing_lane_openness(obs, check_distance=0.12):
    ball_pos = np.array(obs['ball'][:2])
    teammates = obs['left_team']
    opponents = obs['right_team']
    active_idx = obs['active']

    lane_scores = []

    for i, teammate in enumerate(teammates):
        if i == active_idx:
            continue

        teammate_pos = np.array(teammate[:2])
        seg = teammate_pos - ball_pos
        seg_len = np.linalg.norm(seg) + 1e-9

        min_clearance = float('inf')
        for opp in opponents:
            opp_pos = np.array(opp[:2])
            t = np.dot(opp_pos - ball_pos, seg) / (seg_len ** 2)
            t = np.clip(t, 0, 1)
            closest = ball_pos + t * seg
            clearance = np.linalg.norm(opp_pos - closest)
            if clearance < min_clearance:
                min_clearance = clearance

        score = np.clip(min_clearance / (check_distance * 2), 0.0, 1.0)
        lane_scores.append(score)

    if not lane_scores:
        return 1.0

    return float(np.mean(lane_scores))


# -----------------------------------------------------------------------------
# SECTION 3: FUZZY INFERENCE ENGINE
# -----------------------------------------------------------------------------

def compute_fuzzy_scaling(density, dist_to_goal, lane_openness):
    d_low  = opponent_density_low(density)
    d_med  = opponent_density_medium(density)
    d_high = opponent_density_high(density)

    g_far  = distance_to_goal_far(dist_to_goal)
    g_mod  = distance_to_goal_moderate(dist_to_goal)
    g_near = distance_to_goal_near(dist_to_goal)

    l_open    = lane_open(lane_openness)
    l_partial = lane_partially_open(lane_openness)
    l_closed  = lane_closed(lane_openness)

    rules = [
        (min(d_low,  g_far,  l_open),    0.5),   # R1
        (min(d_med,  g_far,  l_open),    1.0),   # R2
        (min(d_high, g_far,  l_open),    1.5),   # R3
        (min(d_med,  g_mod,  l_partial), 1.5),   # R4
        (min(d_high, g_mod,  l_partial), 1.8),   # R5
        (min(d_high, g_near, l_partial), 2.0),   # R6
        (min(d_low,  g_near, l_open),    1.0),   # R7
        (min(d_high, g_near, l_closed),  0.2),   # R8
    ]

    total_strength = sum(strength for strength, _ in rules)
    if total_strength < 1e-9:
        return 1.0

    weighted_sum = sum(strength * output for strength, output in rules)
    return weighted_sum / total_strength


# -----------------------------------------------------------------------------
# SECTION 4: FUZZY REWARD WRAPPER
# -----------------------------------------------------------------------------

class FuzzyRewardWrapper(gym.Wrapper):

    def __init__(self, env, tactic="normal", log_fuzzy=True):
        super().__init__(env)
        self.tactic       = tactic
        self.log_fuzzy    = log_fuzzy
        self._step_count  = 0
        self.last_ball_x  = 0.0  # for stagnation detection
        print(f"--- FYP2 Fuzzy Reward Wrapper ACTIVE: {tactic.upper()} ---")
        print(f"    Fuzzy inputs: Opponent Density | Distance to Goal | Lane Openness")
        print(f"    Scaling range: [0.2 (penalised), 2.0 (maximum reward)]")

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._step_count += 1

        raw_obs = self.env.unwrapped.observation()
        if raw_obs is None:
            return obs, reward, done, info

        agent_obs = raw_obs[0]

        # --- Fuzzy input variables ---
        density      = compute_opponent_density(agent_obs)
        dist_to_goal = compute_distance_to_goal(agent_obs)
        lane_open_v  = compute_passing_lane_openness(agent_obs)

        # --- Fuzzy scaling factor ---
        scale = compute_fuzzy_scaling(density, dist_to_goal, lane_open_v)

        # --- Separate sparse and shaped reward ---
        # Sparse: goal rewards (±1.0) and checkpoints (±0.1)
        # Clamp shaped to 0 minimum to prevent penalty amplification
        sparse_component = reward if abs(reward) >= 0.09 else 0.0
        shaped_component = reward - sparse_component
        shaped_component = max(shaped_component, 0.0)

        ball_x          = agent_obs['ball'][0]
        ball_owned_team = agent_obs['ball_owned_team']

        # --- Tactic drip rewards + stagnation penalties ---
        if self.tactic in ("normal", "normal2", "normal3"):
            # Possession drip
            if ball_owned_team == 0:
                shaped_component += 0.02
            # Stagnation penalty — punish holding ball in same position
            if (ball_owned_team == 0 and
                    abs(ball_x - self.last_ball_x) < 0.005):
                shaped_component -= 0.03

        elif self.tactic in ("attack", "attack2", "attack3"):
            if ball_x > 0.0:
                shaped_component += 0.05  # stronger forward drip
            if ball_owned_team == 0 and ball_x > 0.0:
                shaped_component += 0.05  # stronger possession drip
            if ball_owned_team == 0 and ball_x > 0.5:
                shaped_component += 0.05  # extra for final third
            if (ball_owned_team == 0 and
                    abs(ball_x - self.last_ball_x) < 0.005):
                shaped_component -= 0.05

        elif self.tactic in ("defense", "defense2", "defense3"):
            active_idx = agent_obs['active']
            active_pos = agent_obs['left_team'][active_idx]
            active_x   = active_pos[0]

            # Base drip always
            shaped_component += 0.02

            # Ball in own half bonus
            if ball_x < 0.0:
                shaped_component += 0.03

            # Position-based reward
            if active_x < -0.5:
                shaped_component += 0.05   # deep = best
            elif active_x < -0.3:
                shaped_component += 0.03   # defensive third = good
            elif active_x < 0.0:
                shaped_component += 0.01   # own half = ok
            else:
                shaped_component -= 0.05   # opp half = bad

        # --- Apply fuzzy scaling ---
        fuzzy_scaled_reward = sparse_component + (shaped_component * scale)

        # --- Tactic-specific fuzzy context bonuses ---
        if self.tactic in ("attack", "attack2", "attack3"):
            # Strong forward drive bonus — scaled by context
            if ball_owned_team == 0 and ball_x > 0.0:
                fuzzy_scaled_reward += 0.2 * scale

        elif self.tactic in ("normal", "normal2", "normal3"):
            # Passing incentive — reward open lanes strongly
            # Lower threshold (0.2) and higher bonus (0.5)
            # directly combats ball hoarding
            if ball_owned_team == 0 and lane_open_v > 0.2:
                fuzzy_scaled_reward += 0.5 * lane_open_v

        elif self.tactic in ("defense", "defense2", "defense3"):
            active_idx = agent_obs['active']
            active_pos = agent_obs['left_team'][active_idx]
            active_x   = active_pos[0]

            # Compact shape under pressure
            if ball_x < -0.2 and density > 0.4:
                fuzzy_scaled_reward += 0.05 * density

            # Shoot penalty from own half
            if action == 12 and ball_x < 0.0:
                fuzzy_scaled_reward -= 1.0

            # Shape holding when opp has ball
            if (ball_owned_team != 0 and
                    abs(ball_x - self.last_ball_x) < 0.005):
                fuzzy_scaled_reward += 0.01

            # Position-based penalties
            if active_x > 0.0:
                fuzzy_scaled_reward -= 1.0    # attacking third severe
            elif active_x > -0.3:
                fuzzy_scaled_reward -= 0.3    # middle third moderate

            # Position-based rewards
            if active_x < -0.5:
                fuzzy_scaled_reward += 0.05   # deep = reward

            # Block shooting lanes
            if ball_owned_team == 1:
                if (active_x < ball_x and
                        active_x < -0.3):
                    fuzzy_scaled_reward += 0.05

            # NO stagnation penalty for defense
            

        # --- Update last ball position for stagnation detection ---
        self.last_ball_x = ball_x

        # --- Logging every 500 steps ---
        if self.log_fuzzy and self._step_count % 500 == 0:
            print(f"\n[Fuzzy @ step {self._step_count}] "
                  f"Tactic: {self.tactic.upper()}")
            print(f"  Opponent Density  : {density:.3f}  "
                  f"(Low={opponent_density_low(density):.2f}, "
                  f"Med={opponent_density_medium(density):.2f}, "
                  f"High={opponent_density_high(density):.2f})")
            print(f"  Distance to Goal  : {dist_to_goal:.3f}  "
                  f"(Near={distance_to_goal_near(dist_to_goal):.2f}, "
                  f"Mod={distance_to_goal_moderate(dist_to_goal):.2f}, "
                  f"Far={distance_to_goal_far(dist_to_goal):.2f})")
            print(f"  Lane Openness     : {lane_open_v:.3f}  "
                  f"(Closed={lane_closed(lane_open_v):.2f}, "
                  f"Partial={lane_partially_open(lane_open_v):.2f}, "
                  f"Open={lane_open(lane_open_v):.2f})")
            print(f"  Fuzzy Scale       : {scale:.4f}")
            print(f"  Shaped Reward     : {shaped_component:.4f}  ->  "
                  f"Fuzzy Scaled: {fuzzy_scaled_reward:.4f}")

        # --- Store fuzzy context in info for TensorBoard ---
        info['fuzzy_scale']        = scale
        info['fuzzy_density']      = density
        info['fuzzy_dist_to_goal'] = dist_to_goal
        info['fuzzy_lane_open']    = lane_open_v

        return obs, fuzzy_scaled_reward, done, info

    def reset(self):
        self._step_count = 0
        self.last_ball_x = 0.0
        return self.env.reset()