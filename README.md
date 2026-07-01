# Multi-Agent-Deep-Reinforcement-Learning-Football

Interpretable tactical football simulation
using MAPPO with Centralised Training
Decentralised Execution (CTDE) and Fuzzy
Dynamic Reward Shaping.

## Novel Finding

MAPPO parameter sharing produces emergent
defensive discipline while preventing
offensive teamwork, attributable to
averaged policy convergence creating a
Nash Equilibrium of ball retention.
Heterogeneous role-specific networks are
identified as the necessary architectural
progression.

## Three Tactical Profiles

- Normal: Balanced possession play
- All-Out Attack: High press, territorial dominance
- Park the Bus: Deep defensive low block

## Key Results

| Model | GA | D% | CS | Shots |
|---|---|---|---|---|
| Normal MAPPO v1 | 0.12 | 88% | 44 | 0 |
| Attack MAPPO v1 | 0.24 | 78% | 39 | 26 |
| Defense PPO Static | 0.94 | 48% | 24 | 25 |

- 54x episode reward improvement via fuzzy shaping
- 99.8% shoot spam reduction via MAPPO CTDE
- 650 evaluation matches across 13 trained models

## Architecture

Three stage pipeline:
1. PPO with static dense reward shaping
2. Mamdani fuzzy inference engine (27 rules,
   3 inputs, scaling 0.2 to 2.0)
3. MAPPO with CTDE (230-dim Centralised Critic)

## Environment

- Google Research Football 5v5
- Simple115v2 observation space
- Docker with GPU acceleration

## Repository Structure
src/              Training and evaluation scripts
analysis/         Spatial and network analysis
figures/          13 behavioural analysis figures
logs/             TensorBoard training logs
