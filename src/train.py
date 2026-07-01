import gfootball.env as football_env
from stable_baselines3 import PPO
from rewards import SpecialistRewardWrapper         
from fuzzy_rewards import FuzzyRewardWrapper        
import os

# "normal", "attack", or "defense"
CURRENT_TACTIC = "defense2"
TOTAL_TIMESTEPS = 100000  # Use 2M+ for final results

os.makedirs("/gfootball/models", exist_ok=True)
os.makedirs("/gfootball/logs", exist_ok=True)

env = football_env.create_environment(
    env_name='5_vs_5',
    stacked=False,
    representation='simple115v2',
    rewards='scoring,checkpoints',
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False
)

# --- FYP2 STACKED WRAPPERS ---
# Layer 1: FYP1 static dense rewards (SpecialistRewardWrapper)
env = SpecialistRewardWrapper(env, tactic=CURRENT_TACTIC)
# Layer 2: FYP2 fuzzy dynamic scaling on top (FuzzyRewardWrapper)
env = FuzzyRewardWrapper(env, tactic=CURRENT_TACTIC, log_fuzzy=True)

print(f"--- STARTING FYP2 TRAINING: {CURRENT_TACTIC.upper()} AGENT (Fuzzy Reward Active) ---")
print(f"Logs will be saved to /gfootball/logs/ for Tensorboard")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="/gfootball/logs/",
    learning_rate=0.0005,
    batch_size=64
)

model.learn(total_timesteps=TOTAL_TIMESTEPS)

save_path = f"/gfootball/models/specialist_{CURRENT_TACTIC}_fuzzy.zip"
model.save(save_path)
print(f"Training Complete. Model saved to {save_path}")
