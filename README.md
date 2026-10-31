# Multi-Agent Deep Reinforcement Learning Football

Tactical football simulation using MAPPO with Centralised Training and Decentralised Execution (CTDE) and context-aware fuzzy reward shaping, in the 5v5 Google Research Football (GRF) environment.

Developed as part of a Final Year Project at the Department of Mechatronics Engineering, Kulliyyah of Engineering, International Islamic University Malaysia (IIUM).

---

## Main Finding

Under the evaluated reward, training, and opponent settings, fully parameter-shared MAPPO produced an **asymmetric form of coordination**:

- **Team level:** tactic-dependent territorial behaviour, strongly reduced shooting spam, and strong defensive stability in several configurations (Normal MAPPO v1 conceded 0.12 goals per match).
- **Within team:** possession stayed concentrated in individual agents (dominant player ≈ 61% in Normal MAPPO v1, 40% in Attack MAPPO v1, 89% in Defense PPO Static), and no passing events were recorded in the MAPPO evaluation logs.

Parameter sharing is a *plausible* architectural explanation for this pattern, but the experiments do not isolate it from reward design, training history, and opponent choice. Heterogeneous, modular, or partially shared actor architectures are suggested as future work.

---

## Three Tactical Profiles

| Tactic | Description |
|---|---|
| Normal | Balanced possession and progression |
| All-Out Attack | High press, territorial dominance, forward runs |
| Park the Bus | Deep defensive low block |

---

## Key Results

Evaluated across 13 trained models and 650 evaluation matches (50 per model) against the built-in GRF opponent.

| Model | GA | D% | CS | GF | Total shots (50 matches) |
|---|---|---|---|---|---|
| Normal PPO Static | 1.08 | 22% | 11 | 0 | 6,589 |
| Normal PPO Fuzzy | 1.64 | 16% | 8 | 0.10 | 568 |
| Normal MAPPO v1 | **0.12** | **88%** | **44** | 0 | 0 |
| Normal MAPPO v2 | 0.82 | 36% | 18 | 0 | 3 |
| Attack PPO Static | 2.04 | 6% | 3 | 0 | 5,111 |
| Attack PPO Fuzzy | 2.08 | 12% | 6 | 0.18 | 16,183 |
| Attack MAPPO v1 | 0.24 | 78% | 39 | 0 | 26 |
| Attack MAPPO v2 | 2.26 | 2% | 1 | 0 | 4,339 |
| Defense PPO Static | 0.94 | 48% | 24 | 0 | 25 |
| Defense PPO Fuzzy | 1.58 | 38% | 19 | 0 | 0 |
| Defense MAPPO v1 | 2.54 | 10% | 5 | 0 | 16,057 |
| Defense MAPPO v2 | 2.44 | 8% | 4 | 0 | 1 |
| Defense MAPPO v3 | 2.34 | 16% | 8 | 0 | 32 |

GA = goals against per match | D% = draw percentage | CS = clean sheets out of 50 | GF = goals for per match | Shots = total shooting actions across all 50 evaluation matches (divide by 50 for per-match values)

**Highlights:**
- Fuzzy reward shaping changed Normal PPO mean episodic reward from −0.85 (static) to 30.1, and explained variance from 0.254 to 0.829. Reward values are not directly comparable across reward definitions, so this is a change in the learning signal, not a proportional gain in football performance.
- Shooting dropped from 6,589 shots (≈132 per match) in Normal PPO Static to none in Normal MAPPO v1, and from 16,183 (Attack PPO Fuzzy) to 26 (Attack MAPPO v1).
- Normal MAPPO v1 conceded 0.12 goals per match with 44 clean sheets, although the Normal profile was not designed as a defensive low block.

---

## Architecture

### Three-Stage Pipeline

**Stage 1: PPO + static dense rewards**
Behavioural baseline, ~100k timesteps per tactic | Normal: ep_rew −0.85, exp_var 0.254

**Stage 2: PPO + fuzzy dynamic rewards**
Fuzzy inference with 8 rules over 3 inputs, scale 0.2–2.0 | Normal: ep_rew 30.1, exp_var 0.829

**Stage 3: MAPPO + CTDE + fuzzy rewards**
Shared actor, 230-D centralised critic | 2–3M timesteps per configuration | iterative reward engineering (v1–v3)

### Actor Network

115 → 256 → 256 → 19 | Local Simple115v2 observation to action logits | Parameters shared across all 5 agents

### Centralised Critic

230 → 256 → 256 → 1 | Global state built from player positions and velocities (both teams), ball position and direction, and possession indicators (108 values), zero-padded to 230 | Used only during training (CTDE)

### Fuzzy Inference Engine

**Inputs:**
- Opponent density (Low / Medium / High)
- Distance to goal (Near / Moderate / Far)
- Passing-lane openness (Closed / Partial / Open)

**Rules:** 8 rules over the 27 possible input combinations (sparse rule base). If no rule is active, the scale defaults to 1.0.

| Rule | Density | Distance | Lane | Scale |
|---|---|---|---|---|
| R1 | Low | Far | Open | 0.5 |
| R2 | Medium | Far | Open | 1.0 |
| R3 | High | Far | Open | 1.5 |
| R4 | Medium | Moderate | Partial | 1.5 |
| R5 | High | Moderate | Partial | 1.8 |
| R6 | High | Near | Partial | 2.0 |
| R7 | Low | Near | Open | 1.0 |
| R8 | High | Near | Closed | 0.2 |

**Method:** Triangular membership functions, minimum operator for rule strength, weighted-average defuzzification

**Output:** Scaling factor 0.2 to 2.0

**Final reward:** `r_final = r_sparse + (r_shaped × fuzzy_scale) + tactic-specific terms`

This is context-aware reward shaping, not potential-based reward shaping, so no policy-invariance guarantee is claimed.

---

## Repository Structure

```
Multi-Agent-Deep-Reinforcement-Learning-Football/
│
├── src/
│   ├── mappo_train.py        MAPPO training loop
│   ├── mappo_network.py      Actor and Critic networks
│   ├── rewards.py            Static dense reward functions
│   ├── fuzzy_rewards.py      Fuzzy inference reward wrapper
│   ├── evaluate.py           50-match silent evaluation
│   ├── maprungame.py         Run MAPPO agent with rendering
│   ├── rungame.py            Run PPO agent with rendering
│   └── train.py              PPO baseline training
│
├── analysis/
│   ├── analyze_spatial.py    Ball heatmaps, xG proxy, time-in-thirds
│   └── analyze_network.py    Possession networks, player positions
│
├── figures/
│   ├── fig1_ball_heatmaps.png
│   ├── fig2_player_heatmaps.png
│   ├── fig3_time_in_thirds.png
│   ├── fig4_shoot_xg_map.png
│   ├── fig5_xg_distribution.png
│   ├── fig6_goals_conceded.png
│   ├── fig7_shoot_events.png
│   ├── fig8_draw_cleansheet.png
│   ├── fig9_avg_ball_x.png
│   ├── fig10_possession_network.png
│   ├── fig11_possession_distribution.png
│   ├── fig12_player_avg_positions.png
│   └── fig13_opp_half_presence.png
│
├── models/
│   ├── mappo_normal3_fuzzy_actor.pth
│   ├── mappo_normal3_fuzzy_critic.pth
│   ├── mappo_normal3_v2_fuzzy_actor.pth
│   ├── mappo_normal3_v2_fuzzy_critic.pth
│   ├── mappo_attack3_fuzzy_actor.pth
│   ├── mappo_attack3_fuzzy_critic.pth
│   ├── mappo_attack3_v2_fuzzy_actor.pth
│   ├── mappo_attack3_v2_fuzzy_critic.pth
│   ├── mappo_defense3_fuzzy_actor.pth
│   ├── mappo_defense3_fuzzy_critic.pth
│   ├── mappo_defense3_v2_fuzzy_actor.pth
│   ├── mappo_defense3_v2_fuzzy_critic.pth
│   ├── mappo_defense3_v3_fuzzy_actor.pth
│   └── mappo_defense3_v3_fuzzy_critic.pth
│
├── logs/
│   ├── mappo_normal3/        Normal tactic v1 logs
│   ├── mappo_normal3_v2/     Normal tactic v2 logs
│   ├── mappo_attack3/        Attack tactic v1 logs
│   ├── mappo_attack3_v2/     Attack tactic v2 logs
│   ├── mappo_defense3/       Defense tactic v1 logs
│   ├── mappo_defense3_v2/    Defense tactic v2 logs
│   └── mappo_defense3_v3/    Defense tactic v3 logs
│
├── .gitignore
└── README.md
```

---

## Requirements

- Ubuntu 20.04 or 22.04
- Docker with NVIDIA GPU support
- NVIDIA GPU with CUDA
- Python 3.10

---

## Installation

### Step 1 — Clone this repository

```bash
git clone https://github.com/DefNotIrf/Multi-Agent-Deep-Reinforcement-Learning-Football.git
cd Multi-Agent-Deep-Reinforcement-Learning-Football
```

### Step 2 — Install Google Research Football

```bash
git clone https://github.com/google-research/football.git
```

Follow the official GRF Docker installation: https://github.com/google-research/football

### Step 3 — Build Docker container

Build using the Dockerfile provided in the official GRF repository:

```bash
cd football
docker build -t gfootball_with_training .
cd ..
```

### Step 4 — Mount workspace and run container

```bash
docker run -it --rm \
  --gpus all \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /path/to/this/repo:/workspace \
  -v /path/to/models:/gfootball/models \
  gfootball_with_training bash
```

Replace `/path/to/this/repo` with your actual cloned repository path.

---

## Training

### Train MAPPO model

Inside Docker:

```bash
python3 /workspace/src/mappo_train.py
```

Change tactic in `mappo_train.py`:

```python
CURRENT_TACTIC = "normal3"   # Normal
CURRENT_TACTIC = "attack3"   # All-Out Attack
CURRENT_TACTIC = "defense3"  # Park the Bus
```

Change training steps (reported models used 2–3M):

```python
TOTAL_TIMESTEPS = 2_000_000
```

### Train PPO baseline

```bash
python3 /workspace/src/train.py
```

The reported PPO baselines used `TOTAL_TIMESTEPS = 100000`.

---

## Evaluation

Run the 50-match silent evaluation:

```bash
python3 /workspace/src/evaluate.py
```

Results are saved as .pkl files in the analysis data folder for each tactic.

---

## Analysis

After evaluation, generate all 13 figures:

```bash
python3 analysis/analyze_spatial.py
python3 analysis/analyze_network.py
```

Figures are saved to `analysis/figures/`.

The xG values are a **positional xG proxy** (based on shot distance and angle), not a calibrated expected-goals model.

---

## Demo

### Watch MAPPO agent play

Edit `src/maprungame.py`:

```python
CURRENT_TACTIC = "normal3"   # or attack3 or defense3
```

Run:

```bash
python3 /workspace/src/maprungame.py
```

### Watch PPO agent play

Edit `src/rungame.py`:

```python
CURRENT_TACTIC = "defense"   # or normal or attack
```

Run:

```bash
python3 /workspace/src/rungame.py
```

---

## TensorBoard

View training curves:

```bash
tensorboard --logdir logs/
```

Open your browser at http://localhost:6006

---

## Note on Model Weights

The `models/` folder contains Actor and Critic weights for the seven MAPPO configurations (Normal v1–v2, Attack v1–v2, Defense v1–v3). The Critic weights are used during training only and are not needed for evaluation or demo playback.

To run a pre-trained model without retraining, use `maprungame.py` or `rungame.py` as described above.

---

## Disclaimer

The pre-trained model weights in this repository were trained for research and educational purposes within the Google Research Football 5v5 environment.

**Known limitations:**

- All models were trained and evaluated against the built-in GRF bot at fixed difficulty. Performance against adaptive opponents or other trained agents is not validated.
- Each configuration comes from a single training run; multiple seeds and statistical testing were not performed.
- PPO baselines were trained for ~100k timesteps and MAPPO models for 2–3M, so stage-to-stage comparisons also reflect different training durations.
- The 13 configurations come from iterative development (v1 → v2 → v3), not a controlled factorial experiment.
- All MAPPO models show strong possession concentration in individual agents, with no recorded passing events. Parameter sharing is a plausible cause, but this was not isolated experimentally.
- Defense MAPPO models show erratic shooting caused by a conflict between stagnation penalties and zone constraints. Defense PPO Static gives the most visually coherent defensive behaviour.
- Normal MAPPO v1 shows strong defensive results (GA 0.12) but near-zero offensive output. Do not expect goal-scoring behaviour.
- Attack MAPPO v1 shows high territorial presence in the opponent half, but offensive teamwork remains limited.

These limitations are documented in the accompanying research paper. They are empirical findings rather than implementation errors, and they motivate future work on heterogeneous or partially shared actor architectures.

---

## Citation

If you use this code or findings in your research, please cite:

```
Muhammad Irfan bin Rosdin, Salman Ibne Eunus, Azhar Mohd Ibrahim.
"Emergent Possession Concentration and Limited Offensive Coordination
in Parameter-Sharing MAPPO for Cooperative Football."
Manuscript in preparation, 2026.

GitHub: https://github.com/DefNotIrf/Multi-Agent-Deep-Reinforcement-Learning-Football
```

---

## Acknowledgement

Department of Mechatronics Engineering, Kulliyyah of Engineering, International Islamic University Malaysia (IIUM), and the Advanced Multi-Agent System Laboratory for providing GPU computational resources.

Supervisor: Assoc. Prof. Dr. Azhar Bin Mohd Ibrahim
