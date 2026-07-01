import gfootball.env as football_env
from rewards import SpecialistRewardWrapper
from fuzzy_rewards import FuzzyRewardWrapper
from mappo_network import Actor, build_global_state
import torch
import time
import numpy as np

CURRENT_TACTIC = "attack3"  # change to attack3 or defense3

env = football_env.create_environment(
    env_name='5_vs_5',
    stacked=False,
    representation='simple115v2',
    rewards='scoring,checkpoints',
    render=True
)
env = SpecialistRewardWrapper(env, tactic=CURRENT_TACTIC)
env = FuzzyRewardWrapper(env, tactic=CURRENT_TACTIC,
                         log_fuzzy=False)

DEVICE = torch.device("cpu")
actor = Actor(obs_dim=115, action_dim=19).to(DEVICE)
actor.load_state_dict(torch.load(
    f"/workspace/models/mappo_attack3_fuzzy_actor.pth",
    map_location=DEVICE
))
actor.eval()

print(f"--- VIEWING: {CURRENT_TACTIC.upper()} ---")
print("Watch the game window!")

obs = env.reset()
goals_scored   = 0
goals_conceded = 0

for step in range(3000):
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        action, _, _ = actor.get_action(obs_tensor)
    
    obs, reward, done, info = env.step(action.item())
    time.sleep(0.01)
    
    if done:
        print(f"Match ended | Score: {goals_scored}-{goals_conceded}")
        obs = env.reset()

env.close()