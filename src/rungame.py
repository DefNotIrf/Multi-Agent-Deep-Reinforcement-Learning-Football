import gfootball.env as football_env
from stable_baselines3 import PPO
from rewards import SpecialistRewardWrapper         
from fuzzy_rewards import FuzzyRewardWrapper        
import time
import os

CURRENT_TACTIC = "defense"

model_path = f"models/specialist_{CURRENT_TACTIC}_fuzzy"
fallback_path = f"models/specialist_{CURRENT_TACTIC}" 
env = football_env.create_environment(
    env_name='5_vs_5',
    stacked=False,
    representation='simple115v2',
    rewards='scoring,checkpoints',
    render=True
)

env = SpecialistRewardWrapper(env, tactic=CURRENT_TACTIC)
env = FuzzyRewardWrapper(env, tactic=CURRENT_TACTIC, log_fuzzy=True)

if os.path.exists(model_path + ".zip"):
    print(f"Loading FYP2 fuzzy model: {model_path}")
    model = PPO.load(model_path, env=env)
elif os.path.exists(fallback_path + ".zip"):
    print(f"FYP2 model not found. Loading FYP1 baseline: {fallback_path}")
    print(f"(Fuzzy wrapper is still ACTIVE - you will see fuzzy logs in terminal)")
    model = PPO.load(fallback_path, env=env)
else:
    raise FileNotFoundError(f"No model found at {model_path} or {fallback_path}")

obs = env.reset()
print(f"\n--- PREVIEWING {CURRENT_TACTIC.upper()} STRATEGY (FYP2 Fuzzy Active) ---")
print(f"Watch the terminal for live fuzzy context values every 500 steps.")
print(f"Press Win+Shift+S to capture screenshots for your report.\n")

for _ in range(3000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    time.sleep(0.01)

    if done:
        obs = env.reset()

env.close()
