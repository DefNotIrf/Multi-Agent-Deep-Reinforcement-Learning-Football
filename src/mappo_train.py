import time
import gfootball.env as football_env
from stable_baselines3 import PPO
from rewards import SpecialistRewardWrapper
from fuzzy_rewards import FuzzyRewardWrapper
from mappo_network import Actor, CentralizedCritic, build_global_state
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────
CURRENT_TACTIC   = "defense3"
TOTAL_TIMESTEPS  = 3_000_000
LEARNING_RATE    = 0.0005
N_STEPS          = 3000
BATCH_SIZE       = 300
N_EPOCHS         = 10
CLIP_RANGE       = 0.2
ENTROPY_COEF     = 0.01
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
LOG_INTERVAL     = 1
# ────────────────────────────────────────────────────

os.makedirs("/workspace/models", exist_ok=True)
os.makedirs("/workspace/logs", exist_ok=True)
writer = SummaryWriter(
    log_dir=f"/workspace/logs/mappo_{CURRENT_TACTIC}_v4"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")
print(f"Training on: {DEVICE}")

# ── ENVIRONMENT ─────────────────────────────────────
env = football_env.create_environment(
    env_name='5_vs_5',
    stacked=False,
    representation='simple115v2',
    rewards='scoring',              # scoring only, no checkpoints
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False
)
env = SpecialistRewardWrapper(env, tactic=CURRENT_TACTIC)
env = FuzzyRewardWrapper(env, tactic=CURRENT_TACTIC,
                         log_fuzzy=True)

# ── NETWORKS ────────────────────────────────────────
actor  = Actor(obs_dim=115, action_dim=19).to(DEVICE)
critic = CentralizedCritic(global_state_dim=230).to(DEVICE)

# ── WARM START ──────────────────────────────────────
ppo_path = "/workspace/models/specialist_defense2_fuzzy.zip"
if os.path.exists(ppo_path):
    ppo_model = PPO.load(ppo_path)
    try:
        mappo_state = actor.state_dict()
        ppo_state   = ppo_model.policy.state_dict()
        layer_map   = {
            'network.0.weight': 'mlp_extractor.policy_net.0.weight',
            'network.0.bias'  : 'mlp_extractor.policy_net.0.bias',
            'network.2.weight': 'mlp_extractor.policy_net.2.weight',
            'network.2.bias'  : 'mlp_extractor.policy_net.2.bias',
        }
        loaded = 0
        for mappo_key, ppo_key in layer_map.items():
            if (mappo_key in mappo_state and
                    ppo_key in ppo_state and
                    mappo_state[mappo_key].shape ==
                    ppo_state[ppo_key].shape):
                mappo_state[mappo_key].copy_(
                    ppo_state[ppo_key])
                loaded += 1
        actor.load_state_dict(mappo_state)
        print(f"Warm start: loaded {loaded} layers from PPO")
    except Exception as e:
        print(f"Warm start failed: {e}. Training from scratch.")
else:
    print("No PPO model found. Training from scratch.")

# ── OPTIMIZERS ──────────────────────────────────────
actor_optimizer  = torch.optim.Adam(
    actor.parameters(),  lr=LEARNING_RATE
)
critic_optimizer = torch.optim.Adam(
    critic.parameters(), lr=LEARNING_RATE
)

# ── TRAINING LOOP ───────────────────────────────────
print(f"\n--- MAPPO + CTDE TRAINING: {CURRENT_TACTIC.upper()} ---")
print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
print(f"Device          : {DEVICE}")
print(f"─────────────────────────────────────────────\n")

obs               = env.reset()
timestep          = 0
episode           = 0
ep_rewards        = []
ep_reward         = 0
scoreline_log     = []
xg_log            = []
shoot_counts      = []
ep_our_goals      = 0
ep_opp_goals      = 0
prev_score_reward = 0

# Storage buffers
obs_buf    = []
act_buf    = []
logp_buf   = []
rew_buf    = []
val_buf    = []
global_buf = []
done_buf   = []

def compute_xg(ball_x, ball_y):
    goal_pos = np.array([1.0, 0.0])
    ball_pos = np.array([ball_x, ball_y])
    distance = np.linalg.norm(ball_pos - goal_pos)
    angle    = np.arctan2(0.05, max(distance, 0.01))
    xg       = (angle * 2.0) / (distance + 1.0)
    return float(np.clip(xg, 0.0, 1.0))

while timestep < TOTAL_TIMESTEPS:

    # Get raw obs for global state
    raw_obs = env.unwrapped.observation()
    if raw_obs is not None:
        global_state = build_global_state(raw_obs[0])
    else:
        global_state = np.zeros(230, dtype=np.float32)

    # Actor selects action
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        action, log_prob, _ = actor.get_action(obs_tensor)
        value = critic(
            torch.FloatTensor(global_state)
            .unsqueeze(0).to(DEVICE)
        )

    action_np = action.item()

    # Track shooting action (action 12)
    if action_np == 12 and raw_obs is not None:
        bx = raw_obs[0]['ball'][0]
        by = raw_obs[0]['ball'][1]
        xg = compute_xg(bx, by)
        xg_log.append(xg)

    next_obs, reward, done, info = env.step(action_np)

    # ── GOAL TRACKING ───────────────────────────────
    # scoring only mode: score_reward = +1 we score
    #                                   -1 they score
    # It is per-frame event not cumulative
    raw_score = info.get('score_reward', 0)
    if raw_score > 0.5:
        ep_our_goals += 1
        print(f"  ⚽ GOAL SCORED! ep={episode+1} "
              f"step={timestep}")
    elif raw_score < -0.5:
        ep_opp_goals += 1
        print(f"  ❌ GOAL CONCEDED! ep={episode+1} "
              f"step={timestep}")

    # Store transition
    obs_buf.append(obs)
    act_buf.append(action_np)
    logp_buf.append(log_prob.item())
    rew_buf.append(reward / 5.0)
    val_buf.append(value.item())
    global_buf.append(global_state)
    done_buf.append(done)

    ep_reward += reward
    obs        = next_obs
    timestep  += 1

    if done:
        episode += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0

        # ── SCORELINE ───────────────────────────────
        goals_scored   = ep_our_goals
        goals_conceded = ep_opp_goals
        ep_our_goals   = 0
        ep_opp_goals   = 0

        obs = env.reset()

        scoreline_log.append({
            'episode'  : episode,
            'scored'   : goals_scored,
            'conceded' : goals_conceded,
            'result'   : 'W' if goals_scored > goals_conceded
                         else 'D' if goals_scored == goals_conceded
                         else 'L'
        })

        # ── SHOOT STATS ─────────────────────────────
        shoot_count = act_buf.count(12)
        shoot_rate  = shoot_count / max(len(act_buf), 1) * 100
        shoot_counts.append(shoot_rate)

        avg_scored   = float(np.mean(
            [s['scored']   for s in scoreline_log[-10:]]
        )) if scoreline_log else 0.0
        avg_conceded = float(np.mean(
            [s['conceded'] for s in scoreline_log[-10:]]
        )) if scoreline_log else 0.0
        win_rate     = float(np.mean(
            [1 if s['result'] == 'W' else 0
             for s in scoreline_log[-10:]]
        )) if scoreline_log else 0.0
        avg_xg       = float(np.mean(
            xg_log[-100:]
        )) if len(xg_log) > 0 else 0.0

        # ── UPDATE NETWORKS ─────────────────────────
        returns    = []
        advantages = []
        gae        = 0

        for t in reversed(range(len(rew_buf))):
            next_val = 0 if t == len(rew_buf) - 1 \
                       else val_buf[t + 1]
            delta_t  = (rew_buf[t]
                        + GAMMA * next_val * (1 - done_buf[t])
                        - val_buf[t])
            gae      = (delta_t
                        + GAMMA * GAE_LAMBDA
                        * (1 - done_buf[t]) * gae)
            advantages.insert(0, gae)
            returns.insert(0, gae + val_buf[t])

        adv_tensor  = torch.FloatTensor(advantages).to(DEVICE)
        adv_tensor  = ((adv_tensor - adv_tensor.mean())
                       / (adv_tensor.std() + 1e-8))
        ret_tensor  = torch.FloatTensor(returns).to(DEVICE)
        obs_tensor  = torch.FloatTensor(
                          np.array(obs_buf)).to(DEVICE)
        act_tensor  = torch.LongTensor(act_buf).to(DEVICE)
        logp_tensor = torch.FloatTensor(logp_buf).to(DEVICE)
        gs_tensor   = torch.FloatTensor(
                          np.array(global_buf)).to(DEVICE)

        for _ in range(N_EPOCHS):
            logits   = actor.network(obs_tensor)
            dist     = torch.distributions.Categorical(
                           logits=logits)
            new_logp = dist.log_prob(act_tensor)
            entropy  = dist.entropy().mean()

            ratio  = torch.exp(new_logp - logp_tensor)
            clip1  = ratio * adv_tensor
            clip2  = (torch.clamp(ratio,
                                  1 - CLIP_RANGE,
                                  1 + CLIP_RANGE)
                      * adv_tensor)
            actor_loss = (-torch.min(clip1, clip2).mean()
                          - ENTROPY_COEF * entropy)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                actor.parameters(), 0.5)
            actor_optimizer.step()

            new_values  = critic(gs_tensor).squeeze()
            critic_loss = nn.MSELoss()(new_values, ret_tensor)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                critic.parameters(), 0.5)
            critic_optimizer.step()

        obs_buf.clear()
        act_buf.clear()
        logp_buf.clear()
        rew_buf.clear()
        val_buf.clear()
        global_buf.clear()
        done_buf.clear()

        # ── LOGGING ─────────────────────────────────
        if episode % LOG_INTERVAL == 0:
            mean_rew = np.mean(ep_rewards[-LOG_INTERVAL:])
            print(f"Episode {episode:>5} | "
                  f"Timestep {timestep:>8,} | "
                  f"Mean Reward {mean_rew:>8.2f} | "
                  f"Actor Loss {actor_loss.item():>8.4f} | "
                  f"Critic Loss {critic_loss.item():>8.4f} | "
                  f"Score {goals_scored}-{goals_conceded} | "
                  f"Avg {avg_scored:.1f}-{avg_conceded:.1f} | "
                  f"WR {win_rate*100:.0f}% | "
                  f"Shoot {shoot_rate:.1f}% | "
                  f"xG {avg_xg:.3f}")

            writer.add_scalar("Reward/mean_reward",
                               mean_rew, episode)
            writer.add_scalar("Loss/actor_loss",
                               actor_loss.item(), episode)
            writer.add_scalar("Loss/critic_loss",
                               critic_loss.item(), episode)
            writer.add_scalar("Goals/scored",
                               avg_scored, episode)
            writer.add_scalar("Goals/conceded",
                               avg_conceded, episode)
            writer.add_scalar("Goals/win_rate",
                               win_rate, episode)
            writer.add_scalar("Actions/shoot_rate",
                               shoot_rate, episode)
            writer.add_scalar("Actions/avg_xg",
                               avg_xg, episode)
            writer.add_scalar("Fuzzy/scale",
                               info.get('fuzzy_scale', 0),
                               episode)
            writer.add_scalar("Fuzzy/density",
                               info.get('fuzzy_density', 0),
                               episode)
            writer.add_scalar("Fuzzy/lane_open",
                               info.get('fuzzy_lane_open', 0),
                               episode)
            writer.add_scalar("Fuzzy/dist_to_goal",
                               info.get('fuzzy_dist_to_goal', 0),
                               episode)

            if episode % 10 == 0:
                plt.figure(figsize=(10, 5))
                plt.plot(ep_rewards)
                plt.title(f"MAPPO {CURRENT_TACTIC} v4"
                          f" - Live Reward (Ep {episode})")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.grid(True)
                plt.savefig(
                    f"/workspace/models/"
                    f"mappo_{CURRENT_TACTIC}_v4_live.png"
                )
                plt.close()

# ── SAVE ────────────────────────────────────────────
save_path = (f"/workspace/models/"
             f"mappo_{CURRENT_TACTIC}_v4_fuzzy")
torch.save(actor.state_dict(),
           save_path + "_actor.pth")
torch.save(critic.state_dict(),
           save_path + "_critic.pth")
writer.close()
print(f"\nTraining Complete.")
print(f"Actor  saved: {save_path}_actor.pth")
print(f"Critic saved: {save_path}_critic.pth")

# ── FINAL PLOTS ─────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(ep_rewards)
plt.title(f"MAPPO {CURRENT_TACTIC} v4 - Final Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig(
    f"/workspace/models/"
    f"mappo_{CURRENT_TACTIC}_v4_reward.png"
)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(shoot_counts)
plt.title(f"MAPPO {CURRENT_TACTIC} v4 - Shoot Rate %")
plt.xlabel("Episode")
plt.ylabel("Shoot Rate (%)")
plt.grid(True)
plt.savefig(
    f"/workspace/models/"
    f"mappo_{CURRENT_TACTIC}_v4_shootrate.png"
)
plt.close()
print("Reward and shoot rate graphs saved.")

# ── FINAL SCORELINE SUMMARY ─────────────────────────
total_w        = sum(1 for s in scoreline_log
                     if s['result'] == 'W')
total_d        = sum(1 for s in scoreline_log
                     if s['result'] == 'D')
total_l        = sum(1 for s in scoreline_log
                     if s['result'] == 'L')
total_scored   = sum(s['scored']   for s in scoreline_log)
total_conceded = sum(s['conceded'] for s in scoreline_log)
total_episodes = len(scoreline_log)

print(f"\n── FINAL SCORELINE SUMMARY ──────────────")
print(f"Total Episodes : {total_episodes}")
print(f"Wins           : {total_w} "
      f"({total_w/max(total_episodes,1)*100:.1f}%)")
print(f"Draws          : {total_d} "
      f"({total_d/max(total_episodes,1)*100:.1f}%)")
print(f"Losses         : {total_l} "
      f"({total_l/max(total_episodes,1)*100:.1f}%)")
print(f"Goals Scored   : {total_scored} "
      f"(avg {total_scored/max(total_episodes,1):.2f}/match)")
print(f"Goals Conceded : {total_conceded} "
      f"(avg {total_conceded/max(total_episodes,1):.2f}/match)")
print(f"Avg xG         : {np.mean(xg_log):.3f}" if xg_log
      else "Avg xG: N/A (no shots taken)")