import gfootball.env as football_env
from rewards import SpecialistRewardWrapper
from fuzzy_rewards import FuzzyRewardWrapper
from mappo_network import Actor, build_global_state
import torch
import numpy as np
import os
import pickle

# ── CONFIG ──────────────────────────────────────────
TACTIC        = "normal"
NUM_EPISODES  = 50
MODEL_VERSION = "v1"      # "ppo_static", "ppo_fuzzy",
                           # "v1", "v2", "v3", "v4"
# ────────────────────────────────────────────────────

DEVICE = torch.device("cpu")

# ── LOAD MODEL ──────────────────────────────────────
if MODEL_VERSION == "ppo_static":
    from stable_baselines3 import PPO
    ppo_path = f"/workspace/models/specialist_{TACTIC}.zip"
    if not os.path.exists(ppo_path):
        raise FileNotFoundError(f"Not found: {ppo_path}")
    ppo_model = PPO.load(ppo_path)

    class PPOActor:
        def __init__(self, model):
            self.model = model
        def get_action(self, obs_tensor):
            obs_np = obs_tensor.squeeze(0).numpy()
            action, _ = self.model.predict(
                obs_np, deterministic=True)
            return (torch.tensor(action),
                    torch.tensor(0.0),
                    torch.tensor(0.0))
        def eval(self):
            pass

    actor = PPOActor(ppo_model)
    print(f"Loaded PPO Static: specialist_{TACTIC}.zip")

elif MODEL_VERSION == "ppo_fuzzy":
    from stable_baselines3 import PPO
    ppo_path = f"/workspace/models/specialist_{TACTIC}2_fuzzy.zip"
    if not os.path.exists(ppo_path):
        raise FileNotFoundError(f"Not found: {ppo_path}")
    ppo_model = PPO.load(ppo_path)

    class PPOActor:
        def __init__(self, model):
            self.model = model
        def get_action(self, obs_tensor):
            obs_np = obs_tensor.squeeze(0).numpy()
            action, _ = self.model.predict(
                obs_np, deterministic=True)
            return (torch.tensor(action),
                    torch.tensor(0.0),
                    torch.tensor(0.0))
        def eval(self):
            pass

    actor = PPOActor(ppo_model)
    print(f"Loaded PPO Fuzzy: specialist_{TACTIC}2_fuzzy.zip")

elif MODEL_VERSION == "v1":
    model_path = f"/workspace/models/mappo_{TACTIC}3_fuzzy_actor.pth"
    actor = Actor(obs_dim=115, action_dim=19).to(DEVICE)
    actor.load_state_dict(torch.load(
        model_path, map_location=DEVICE))
    actor.eval()
    print(f"Loaded MAPPO v1: {model_path}")

elif MODEL_VERSION == "v2":
    model_path = f"/workspace/models/mappo_{TACTIC}3_v2_fuzzy_actor.pth"
    actor = Actor(obs_dim=115, action_dim=19).to(DEVICE)
    actor.load_state_dict(torch.load(
        model_path, map_location=DEVICE))
    actor.eval()
    print(f"Loaded MAPPO v2: {model_path}")

elif MODEL_VERSION == "v3":
    model_path = f"/workspace/models/mappo_{TACTIC}3_v3_fuzzy_actor.pth"
    actor = Actor(obs_dim=115, action_dim=19).to(DEVICE)
    actor.load_state_dict(torch.load(
        model_path, map_location=DEVICE))
    actor.eval()
    print(f"Loaded MAPPO v3: {model_path}")

elif MODEL_VERSION == "v4":
    model_path = f"/workspace/models/mappo_{TACTIC}3_v4_fuzzy_actor.pth"
    actor = Actor(obs_dim=115, action_dim=19).to(DEVICE)
    actor.load_state_dict(torch.load(
        model_path, map_location=DEVICE))
    actor.eval()
    print(f"Loaded MAPPO v4: {model_path}")

else:
    raise ValueError(f"Unknown MODEL_VERSION: {MODEL_VERSION}")

# ── ENVIRONMENT ─────────────────────────────────────
env = football_env.create_environment(
    env_name='5_vs_5',
    stacked=False,
    representation='simple115v2',
    rewards='scoring',
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False
)
env = SpecialistRewardWrapper(env, tactic=TACTIC)
env = FuzzyRewardWrapper(env, tactic=TACTIC, log_fuzzy=False)

# ── METRICS STORAGE ─────────────────────────────────
results          = []
ball_positions   = []
shoot_positions  = []
player_positions = []
pass_events      = []

print(f"\n── EVALUATING: {TACTIC.upper()} "
      f"({NUM_EPISODES} matches) ──")
print(f"Model version : {MODEL_VERSION}")
print(f"─────────────────────────────────────────\n")

# ── EVALUATION LOOP ─────────────────────────────────
for ep in range(NUM_EPISODES):
    obs              = env.reset()
    done             = False
    step             = 0
    goals_scored     = 0
    goals_conceded   = 0
    last_ball_player = -1
    ep_ball_pos      = []
    ep_shoot_pos     = []
    ep_player_pos    = []
    ep_passes        = []

    while not done:
        raw_obs = env.unwrapped.observation()
        if raw_obs is not None:
            agent_obs  = raw_obs[0]
            ball_x     = agent_obs['ball'][0]
            ball_y     = agent_obs['ball'][1]
            ball_owned = agent_obs['ball_owned_team']
            active_idx = agent_obs['active']
            left_team  = agent_obs['left_team']
            right_team = agent_obs['right_team']

            # Record ball position
            ep_ball_pos.append((ball_x, ball_y))

            # Record player positions
            ep_player_pos.append({
                'left'      : [(p[0], p[1])
                               for p in left_team],
                'right'     : [(p[0], p[1])
                               for p in right_team],
                'ball'      : (ball_x, ball_y),
                'possession': ball_owned
            })

            # Detect pass via ball_owned_player change
            if ball_owned == 0:
                ball_player = agent_obs.get(
                    'ball_owned_player', -1)
                if (last_ball_player != -1 and
                        ball_player != -1 and
                        ball_player != last_ball_player):
                    ep_passes.append(
                        (last_ball_player, ball_player))
                last_ball_player = ball_player
            else:
                last_ball_player = -1

        # Actor selects action
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = actor.get_action(obs_tensor)
        action_np = action.item()

        # Track shooting positions
        if action_np == 12 and raw_obs is not None:
            ep_shoot_pos.append((
                raw_obs[0]['ball'][0],
                raw_obs[0]['ball'][1]
            ))

        obs, reward, done, info = env.step(action_np)

        # Track goals
        score_r = info.get('score_reward', 0)
        if score_r > 0.5:
            goals_scored += 1
        elif score_r < -0.5:
            goals_conceded += 1

        step += 1

    # Episode result
    result = {
        'episode'       : ep + 1,
        'goals_scored'  : goals_scored,
        'goals_conceded': goals_conceded,
        'result'        : 'W' if goals_scored > goals_conceded
                          else 'D' if goals_scored == goals_conceded
                          else 'L',
        'steps'         : step
    }
    results.append(result)
    ball_positions.extend(ep_ball_pos)
    shoot_positions.extend(ep_shoot_pos)
    player_positions.extend(ep_player_pos)
    pass_events.extend(ep_passes)

    print(f"Match {ep+1:>3} | "
          f"Score {goals_scored}-{goals_conceded} | "
          f"{result['result']} | "
          f"Steps {step}")

# ── SUMMARY ─────────────────────────────────────────
total    = len(results)
wins     = sum(1 for r in results if r['result'] == 'W')
draws    = sum(1 for r in results if r['result'] == 'D')
losses   = sum(1 for r in results if r['result'] == 'L')
scored   = sum(r['goals_scored']   for r in results)
conceded = sum(r['goals_conceded'] for r in results)

print(f"\n── EVALUATION SUMMARY: {TACTIC.upper()} "
      f"({MODEL_VERSION.upper()}) ──")
print(f"Total Matches  : {total}")
print(f"Wins           : {wins}  ({wins/total*100:.1f}%)")
print(f"Draws          : {draws} ({draws/total*100:.1f}%)")
print(f"Losses         : {losses} ({losses/total*100:.1f}%)")
print(f"Goals Scored   : {scored} "
      f"(avg {scored/total:.2f}/match)")
print(f"Goals Conceded : {conceded} "
      f"(avg {conceded/total:.2f}/match)")
print(f"Clean Sheets   : "
      f"{sum(1 for r in results if r['goals_conceded']==0)}")
print(f"Failed To Score: "
      f"{sum(1 for r in results if r['goals_scored']==0)}")
print(f"Pass Events    : {len(pass_events)}")
print(f"Shoot Events   : {len(shoot_positions)}")

# ── SAVE DATA ────────────────────────────────────────
save_dir = f"/workspace/analysis/{TACTIC}_{MODEL_VERSION}"
os.makedirs(save_dir, exist_ok=True)

with open(f"{save_dir}/results.pkl", 'wb') as f:
    pickle.dump(results, f)
with open(f"{save_dir}/ball_positions.pkl", 'wb') as f:
    pickle.dump(ball_positions, f)
with open(f"{save_dir}/shoot_positions.pkl", 'wb') as f:
    pickle.dump(shoot_positions, f)
with open(f"{save_dir}/player_positions.pkl", 'wb') as f:
    pickle.dump(player_positions, f)
with open(f"{save_dir}/pass_events.pkl", 'wb') as f:
    pickle.dump(pass_events, f)

print(f"\nData saved to: {save_dir}/")
print(f"Ready for analysis.")