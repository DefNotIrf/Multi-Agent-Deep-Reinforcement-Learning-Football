import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

try:
    import networkx as nx
except ImportError:
    os.system("pip install networkx")
    import networkx as nx

# ── CONFIG ──────────────────────────────────────────
ANALYSIS_DIR = "/workspace/analysis"
OUTPUT_DIR   = "/workspace/analysis/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODELS = {
    "Normal\n(MAPPO v1)"   : "normal_v1",
    "Attack\n(MAPPO v1)"   : "attack_v1",
    "Defense\n(PPO Static)": "defense_ppo_static"
}

def load_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def get_closest_player(ball_pos, players):
    bx, by   = ball_pos
    min_dist = float('inf')
    closest  = 0
    for i, (px, py) in enumerate(players):
        dist = ((px-bx)**2 + (py-by)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            closest  = i
    return closest

def draw_pitch(ax, bg='#2d6a2d', lc='white'):
    ax.set_facecolor(bg)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.7, 0.7)
    # Pitch outline
    ax.add_patch(patches.Rectangle(
        (-1,-0.65), 2, 1.3,
        lw=2, edgecolor=lc, facecolor='none',
        zorder=2))
    # Halfway line
    ax.axvline(0, color=lc, lw=1.5, zorder=2)
    # Centre circle
    ax.add_patch(plt.Circle(
        (0,0), 0.15, color=lc,
        fill=False, lw=1.5, zorder=2))
    ax.plot(0, 0, 'o', color=lc, ms=3, zorder=2)
    # Penalty boxes
    for x0,y0,w,h in [
        (-1,-0.22,0.16,0.44),
        (0.84,-0.22,0.16,0.44)]:
        ax.add_patch(patches.Rectangle(
            (x0,y0), w, h,
            lw=1.5, edgecolor=lc,
            facecolor='none', zorder=2))
    # Goals
    for x0,y0,w,h in [
        (-1.04,-0.05,0.04,0.10),
        (1.0,-0.05,0.04,0.10)]:
        ax.add_patch(patches.Rectangle(
            (x0,y0), w, h,
            lw=1.5, edgecolor=lc,
            facecolor='none', zorder=2))
    ax.set_aspect('equal')
    ax.axis('off')

def compute_xg(bx, by):
    goal_pos = np.array([1.0, 0.0])
    ball_pos = np.array([bx, by])
    distance = np.linalg.norm(ball_pos - goal_pos)
    angle    = np.arctan2(0.05, max(distance, 0.01))
    xg       = (angle * 2.0) / (distance + 1.0)
    return float(np.clip(xg, 0.0, 1.0))

# ════════════════════════════════════════════════════
# FIGURE 4 UPDATED — Shoot Position + xG
# (fix: visible pitch lines)
# ════════════════════════════════════════════════════
print("Figure 4: Shoot Position Maps (updated)...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Shooting Position Maps (Color = xG Quality)"
    " — Best Model Per Tactic",
    fontsize=14, fontweight='bold')

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

for ax, (title, key) in zip(axes, BEST_MODELS.items()):
    data = load_pkl(
        f"{ANALYSIS_DIR}/{key}/shoot_positions.pkl")

    # Draw pitch first with zorder=1
    draw_pitch(ax, bg='#2d6a2d', lc='white')

    if data and len(data) > 0:
        xs  = [p[0] for p in data]
        ys  = [p[1] for p in data]
        xgs = [compute_xg(x, y) for x, y in zip(xs, ys)]

        # Plot scatter with higher zorder than pitch
        sc = ax.scatter(xs, ys, c=xgs,
                        cmap='RdYlGn',
                        s=200,
                        alpha=0.95,
                        vmin=0, vmax=0.15,
                        edgecolors='black',
                        linewidths=0.8,
                        zorder=5)
        plt.colorbar(sc, ax=ax, fraction=0.03,
                     label='xG')
        avg_xg   = np.mean(xgs)
        total_xg = sum(xgs)
        ax.set_title(
            f"{title}\n"
            f"{len(data)} shots | "
            f"Avg xG:{avg_xg:.3f} | "
            f"Total xG:{total_xg:.2f}",
            fontsize=9, fontweight='bold', pad=8)
    else:
        ax.set_title(
            f"{title}\nNo shooting events",
            fontsize=10, fontweight='bold', pad=8)
        ax.text(0, 0, 'No shots taken',
                ha='center', va='center',
                color='white', fontsize=11,
                zorder=5)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_shoot_xg_map.png",
            dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved: fig4_shoot_xg_map.png (updated)")

# ════════════════════════════════════════════════════
# FIGURE 10 — Possession Dominance Network
# ════════════════════════════════════════════════════
print("Figure 10: Possession Dominance Networks...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#1a3a1a')
fig.suptitle(
    "Player Possession Dominance — Best Model Per Tactic\n"
    "(Node size = possession %, "
    "Position = average location on pitch)",
    fontsize=12, fontweight='bold',
    color='white')

network_stats = {}

for ax, (title, key) in zip(axes, BEST_MODELS.items()):
    player_data = load_pkl(
        f"{ANALYSIS_DIR}/{key}/player_positions.pkl")

    ax.set_facecolor('#1a3a1a')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.75, 0.75)

    # Draw pitch lines
    ax.add_patch(patches.Rectangle(
        (-1,-0.65), 2, 1.3,
        lw=2, edgecolor='white',
        facecolor='none', zorder=2))
    ax.axvline(0, color='white', lw=1.5,
               alpha=0.7, zorder=2)
    ax.add_patch(plt.Circle(
        (0,0), 0.15, color='white',
        fill=False, lw=1.5,
        alpha=0.7, zorder=2))
    ax.plot(0, 0, 'o', color='white',
            ms=3, alpha=0.7, zorder=2)
    for x0,y0,w,h in [
        (-1,-0.22,0.16,0.44),
        (0.84,-0.22,0.16,0.44)]:
        ax.add_patch(patches.Rectangle(
            (x0,y0), w, h,
            lw=1.5, edgecolor='white',
            facecolor='none',
            alpha=0.7, zorder=2))

    if player_data:
        possession_count = {i: 0 for i in range(5)}
        avg_pos_sum      = {i: [0.0, 0.0] for i in range(5)}
        frame_count      = {i: 0 for i in range(5)}

        for frame in player_data:
            owned    = frame.get('possession', -1)
            ball_pos = frame['ball']
            players  = frame['left']

            for i, (px, py) in enumerate(players):
                avg_pos_sum[i][0] += px
                avg_pos_sum[i][1] += py
                frame_count[i]    += 1

            if owned == 0:
                closest = get_closest_player(
                    ball_pos, players)
                possession_count[closest] += 1

        avg_pos = {}
        for i in range(5):
            if frame_count[i] > 0:
                avg_pos[i] = (
                    avg_pos_sum[i][0] / frame_count[i],
                    avg_pos_sum[i][1] / frame_count[i]
                )
            else:
                avg_pos[i] = (0.0, 0.0)

        total_possession = max(
            sum(possession_count.values()), 1)
        poss_pct = {
            i: possession_count[i] /
               total_possession * 100
            for i in range(5)
        }

        node_colors = [
            '#e74c3c', '#e67e22',
            '#f1c40f', '#2ecc71', '#3498db'
        ]

        for i in range(5):
            px, py = avg_pos[i]
            # Scale node by possession %
            radius = max(poss_pct[i] / 100 * 0.3, 0.03)
            circle = plt.Circle(
                (px, py), radius,
                color=node_colors[i],
                alpha=0.85, zorder=5)
            ax.add_patch(circle)
            ax.annotate(
                f"P{i}\n{poss_pct[i]:.0f}%",
                xy=(px, py),
                ha='center', va='center',
                fontsize=7, fontweight='bold',
                color='white', zorder=6)

        max_poss_player = max(
            possession_count,
            key=possession_count.get)
        network_stats[key] = {
            'possession_count': possession_count,
            'poss_pct'        : poss_pct,
            'max_poss_player' : max_poss_player,
            'max_poss_pct'    : poss_pct[max_poss_player]
        }

        ax.set_title(
            f"{title}\n"
            f"Dominant: P{max_poss_player} "
            f"({poss_pct[max_poss_player]:.0f}%)",
            fontsize=9, fontweight='bold',
            color='white', pad=8)
    else:
        ax.set_title(
            f"{title}\nNo data",
            fontsize=10, fontweight='bold',
            color='white', pad=8)
        network_stats[key] = {
            'possession_count': {i:0 for i in range(5)},
            'poss_pct'        : {i:0 for i in range(5)},
            'max_poss_player' : 0,
            'max_poss_pct'    : 0
        }

    ax.axis('off')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig10_possession_network.png",
            dpi=150, bbox_inches='tight',
            facecolor='#1a3a1a')
plt.close()
print("  Saved: fig10_possession_network.png")

# ════════════════════════════════════════════════════
# FIGURE 11 — Possession Distribution Per Player
# ════════════════════════════════════════════════════
print("Figure 11: Possession Distribution...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Possession Distribution Per Player"
    " — Best Model Per Tactic\n"
    "(Shows role specialisation and ball distribution)",
    fontsize=13, fontweight='bold')

tactic_colors = ['#3498db', '#e74c3c', '#27ae60']

for ax, (title, key), color in zip(
        axes, BEST_MODELS.items(), tactic_colors):
    stats = network_stats.get(key, {})
    poss  = stats.get(
        'poss_pct', {i:0 for i in range(5)})

    players = [f"P{i}" for i in range(5)]
    vals    = [poss.get(i, 0) for i in range(5)]

    bars = ax.bar(players, vals,
                  color=color, alpha=0.85,
                  edgecolor='black')
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.1f}%',
            xy=(bar.get_x()+bar.get_width()/2,
                bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9)

    ax.set_title(title, fontsize=10,
                 fontweight='bold')
    ax.set_xlabel('Player Index', fontsize=10)
    ax.set_ylabel('Possession %', fontsize=10)
    ax.set_ylim(0, 75)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(20, color='red', linestyle='--',
               lw=1, alpha=0.5,
               label='Equal share (20%)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/fig11_possession_distribution.png",
    dpi=150, bbox_inches='tight',
    facecolor='white')
plt.close()
print("  Saved: fig11_possession_distribution.png")

# ════════════════════════════════════════════════════
# FIGURE 12 — Average Player X Position Per Tactic
# ════════════════════════════════════════════════════
print("Figure 12: Average Player Positions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Average Player X Position — Best Model Per Tactic\n"
    "(Shows tactical positioning and role distribution)",
    fontsize=13, fontweight='bold')

for ax, (title, key), color in zip(
        axes, BEST_MODELS.items(), tactic_colors):
    player_data = load_pkl(
        f"{ANALYSIS_DIR}/{key}/player_positions.pkl")

    if player_data:
        pos_sum   = {i: 0.0 for i in range(5)}
        pos_count = {i: 0   for i in range(5)}

        for frame in player_data:
            for i, (px, py) in enumerate(frame['left']):
                pos_sum[i]   += px
                pos_count[i] += 1

        avg_x   = [pos_sum[i] / max(pos_count[i], 1)
                   for i in range(5)]
        players = [f"P{i}" for i in range(5)]

        bar_colors = [color if v >= 0
                      else '#e74c3c'
                      for v in avg_x]
        bars = ax.bar(players, avg_x,
                      color=bar_colors,
                      alpha=0.85,
                      edgecolor='black')
        ax.axhline(0, color='black', lw=1.5,
                   linestyle='--',
                   label='Halfway line')
        for bar, val in zip(bars, avg_x):
            ax.annotate(f'{val:.2f}',
                xy=(bar.get_x()+bar.get_width()/2,
                    val),
                xytext=(0, 3 if val >= 0 else -12),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

        ax.set_title(title, fontsize=10,
                     fontweight='bold')
        ax.set_xlabel('Player Index', fontsize=10)
        ax.set_ylabel('Avg X Position', fontsize=10)
        ax.set_ylim(-0.8, 0.8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.set_title(f"{title}\nNo data",
                     fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/fig12_player_avg_positions.png",
    dpi=150, bbox_inches='tight',
    facecolor='white')
plt.close()
print("  Saved: fig12_player_avg_positions.png")

# ════════════════════════════════════════════════════
# FIGURE 13 — Opponent Half Presence Per Player
# ════════════════════════════════════════════════════
print("Figure 13: Opponent Half Presence...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Time Each Player Spent in Opponent Half %"
    " — Best Model Per Tactic\n"
    "(Higher = More Aggressive / Attacking Role)",
    fontsize=12, fontweight='bold')

for ax, (title, key), color in zip(
        axes, BEST_MODELS.items(), tactic_colors):
    player_data = load_pkl(
        f"{ANALYSIS_DIR}/{key}/player_positions.pkl")

    if player_data:
        opp_half  = {i: 0 for i in range(5)}
        total_cnt = {i: 0 for i in range(5)}

        for frame in player_data:
            for i, (px, py) in enumerate(frame['left']):
                total_cnt[i] += 1
                if px > 0.0:
                    opp_half[i] += 1

        opp_pct = [opp_half[i] /
                   max(total_cnt[i], 1) * 100
                   for i in range(5)]
        players = [f"P{i}" for i in range(5)]

        bars = ax.bar(players, opp_pct,
                      color=color, alpha=0.85,
                      edgecolor='black')
        for bar, val in zip(bars, opp_pct):
            ax.annotate(f'{val:.1f}%',
                xy=(bar.get_x()+bar.get_width()/2,
                    bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

        ax.set_title(title, fontsize=10,
                     fontweight='bold')
        ax.set_xlabel('Player Index', fontsize=10)
        ax.set_ylabel('Time in Opp Half (%)',
                      fontsize=10)
        ax.set_ylim(0, 70)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.set_title(f"{title}\nNo data",
                     fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/fig13_opp_half_presence.png",
    dpi=150, bbox_inches='tight',
    facecolor='white')
plt.close()
print("  Saved: fig13_opp_half_presence.png")

print(f"\n✅ All figures saved to: {OUTPUT_DIR}/")
print("Figures updated:")
print("  4.  fig4_shoot_xg_map.png    (pitch lines fixed)")
print("  10. fig10_possession_network.png (nodes fixed)")
print("  11. fig11_possession_distribution.png")
print("  12. fig12_player_avg_positions.png")
print("  13. fig13_opp_half_presence.png")