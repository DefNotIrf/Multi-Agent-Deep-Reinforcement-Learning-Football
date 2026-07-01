# Multi-Agent Deep Reinforcement Learning Football

Interpretable tactical football simulation
using MAPPO with Centralised Training
Decentralised Execution (CTDE) and Fuzzy
Dynamic Reward Shaping.

Developed as part of a Final Year Project
at the Department of Mechatronics Engineering,
Kulliyyah of Engineering,
International Islamic University Malaysia (IIUM).

---

## Novel Finding

MAPPO parameter sharing produces emergent
defensive discipline while preventing
offensive teamwork, attributable to averaged
policy convergence creating a Nash Equilibrium
of ball retention that persists despite CTDE
resolving non-stationarity. Heterogeneous
role-specific networks are identified as the
necessary architectural progression for
balanced MADRL tactical football.

This finding is not previously documented
in football simulation literature.

---

## Three Tactical Profiles

| Tactic | Description |
|---|---|
| Normal | Balanced possession and defensive shape |
| All-Out Attack | High press, territorial dominance, forward runs |
| Park the Bus | Deep defensive low block, clearances |

---

## Key Results

Evaluated across 13 trained models and
650 evaluation matches (50 per model).

| Model | GA | D% | CS | GF | Shots |
|---|---|---|---|---|---|
| Normal PPO Static | 1.08 | 22% | 11 | 0 | 6,589 |
| Normal PPO Fuzzy | 1.64 | 16% | 8 | 0.1 | 5 |
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

GA = Goals Against per match
D% = Draw Percentage
CS = Clean Sheets out of 50
GF = Goals For per match

**Key metrics:**
- 54x episode reward improvement (0.559 to 30.1)
  via fuzzy dynamic reward shaping
- 99.8% shoot spam reduction (6,589 to near-zero)
  via MAPPO CTDE
- GA 0.12 emergent defensive discipline without
  any explicit defensive reward signals

---

## Architecture

### Three Stage Pipeline
Stage 1: PPO + Static Dense Rewards
Establish behavioural baseline
ep_rew: 0.559 | exp_var: 0.254
     ↓
Stage 2: PPO + Fuzzy Dynamic Rewards
Mamdani inference engine
27 rules, 3 inputs, scale 0.2-2.0
ep_rew: 30.1 | exp_var: 0.829
     ↓
Stage 3: MAPPO + CTDE + Fuzzy Rewards
230-dim Centralised Critic
GA 0.12 | 99.8% shoot spam reduction

### Actor Network
115 -> 256 -> 256 -> 19
Local observation to action logits
Parameter sharing across 5 agents

### Centralised Critic
230 -> 256 -> 256 -> 1
Global state (both teams) to value estimate
Used only during training (CTDE)

### Fuzzy Inference Engine
Inputs:
Opponent Density    (Low / Medium / High)
Distance to Goal    (Near / Moderate / Far)
Passing Lane        (Closed / Partial / Open)
Rules: 27 (complete rule base)
Output: Scaling factor 0.2 to 2.0
Method: Mamdani + Centroid defuzzification
Final reward:
r_final = r_sparse + (r_shaped x fuzzy_scale)

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
│   ├── analyze_spatial.py    Ball heatmaps, xG, time-in-thirds
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

Follow the official GRF Docker installation:
https://github.com/google-research/football

### Step 3 — Build Docker container

```bash
docker build -t gfootball_with_training .
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

Replace /path/to/this/repo with your
actual cloned repository path.

---

## Training

### Train MAPPO model

Inside Docker:

```bash
python3 /workspace/src/mappo_train.py
```

Change tactic in mappo_train.py:

```python
CURRENT_TACTIC = "normal3"   # Normal
CURRENT_TACTIC = "attack3"   # All-Out Attack
CURRENT_TACTIC = "defense3"  # Park the Bus
```

Change training steps:

```python
TOTAL_TIMESTEPS = 2_000_000
```

### Train PPO baseline

```bash
python3 /workspace/src/train.py
```

---

## Evaluation

Run 50-match silent evaluation:

```bash
python3 /workspace/src/evaluate.py
```

Results saved as .pkl files
in analysis data folders per tactic.

---

## Analysis

After evaluation, generate all 13 figures:

```bash
python3 analysis/analyze_spatial.py
python3 analysis/analyze_network.py
```

Figures saved to analysis/figures/

---

## Demo

### Watch MAPPO agent play

Edit src/maprungame.py:

```python
CURRENT_TACTIC = "normal3"   # or attack3 or defense3
```

Run:

```bash
python3 /workspace/src/maprungame.py
```

### Watch PPO agent play

Edit src/rungame.py:

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

Open browser at http://localhost:6006

---

## Note on Model Weights

Pre-trained model weights (.pth files) for
all 13 trained models are included in the
models/ folder:

- mappo_normal3_fuzzy_actor.pth
- mappo_normal3_v2_fuzzy_actor.pth
- mappo_attack3_fuzzy_actor.pth
- mappo_attack3_v2_fuzzy_actor.pth
- mappo_defense3_fuzzy_actor.pth
- mappo_defense3_v2_fuzzy_actor.pth
- mappo_defense3_v3_fuzzy_actor.pth

Each tactic has a corresponding Critic
weight file (_critic.pth) used during
training only.

To run a pre-trained model directly
without retraining, use maprungame.py
or rungame.py as described above.

## Disclaimer

The pre-trained model weights included in
this repository were trained for research
and educational purposes within the Google
Research Football 5v5 environment.

**Known Limitations:**

- All MAPPO models exhibit ball hoarding
  behaviour due to parameter sharing
  architecture. Agents converge to a Nash
  Equilibrium of possession retention
  rather than active passing or shooting.

- Normal MAPPO v1 produces emergent
  defensive discipline (GA 0.12) but
  near-zero offensive output. Do not
  expect goal scoring behaviour.

- Attack MAPPO v1 shows high territorial
  presence in the opponent half but
  offensive teamwork is limited by the
  same parameter sharing constraint.

- Defense MAPPO models exhibit erratic
  shooting behaviour due to reward
  conflict between stagnation penalties
  and zone constraints. Defense PPO
  Static is recommended for the most
  visually coherent defensive behaviour.

- All models were trained against the
  built-in GRF bot at fixed difficulty.
  Performance against adaptive opponents
  or other trained agents is not validated.

These limitations are documented and
discussed in the accompanying research
report. They represent
empirical findings rather than
implementation errors, and directly
motivate future work on heterogeneous
role-specific networks.
