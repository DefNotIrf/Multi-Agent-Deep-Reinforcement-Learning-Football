import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ── CONFIG ──────────────────────────────────────────
ANALYSIS_DIR = "/workspace/analysis"
OUTPUT_DIR   = "/workspace/analysis/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Best models per tactic
BEST_MODELS = {
    "Normal\n(MAPPO v1)"   : "normal_v1",
    "Attack\n(MAPPO v1)"   : "attack_v1",
    "Defense\n(PPO Static)": "defense_ppo_static"
}

# Evaluation results from evaluate.py
EVAL_RESULTS = {
    "normal_ppo_static" : {"GA":1.08, "D%":22, "CS":11, "Shoots":6589,  "GF":0.00},
    "normal_ppo_fuzzy"  : {"GA":1.64, "D%":16, "CS":8,  "Shoots":5,     "GF":0.10},
    "normal_v1"         : {"GA":0.12, "D%":88, "CS":44, "Shoots":0,     "GF":0.00},
    "normal_v2"         : {"GA":0.82, "D%":36, "CS":18, "Shoots":3,     "GF":0.00},
    "attack_ppo_static" : {"GA":2.04, "D%":6,  "CS":3,  "Shoots":5111,  "GF":0.00},
    "attack_ppo_fuzzy"  : {"GA":2.08, "D%":12, "CS":6,  "Shoots":16183, "GF":0.18},
    "attack_v1"         : {"GA":0.24, "D%":78, "CS":39, "Shoots":26,    "GF":0.00},
    "attack_v2"         : {"GA":2.26, "D%":2,  "CS":1,  "Shoots":4339,  "GF":0.00},
    "defense_ppo_static": {"GA":0.94, "D%":48, "CS":24, "Shoots":25,    "GF":0.00},
    "defense2_ppo"      : {"GA":1.58, "D%":38, "CS":19, "Shoots":0,     "GF":0.00},
    "defense3_v1"       : {"GA":2.54, "D%":10, "CS":5,  "Shoots":16057, "GF":0.00},
    "defense3_v2"       : {"GA":2.44, "D%":8,  "CS":4,  "Shoots":1,     "GF":0.00},
    "defense_v3"        : {"GA":2.14, "D%":16, "CS":8,  "Shoots":32,    "GF":0.00},
}

# Groups for comparison figures
groups = {
    "Normal" : [
        "normal_ppo_static",
        "normal_ppo_fuzzy",
        "normal_v1",
        "normal_v2"
    ],
    "Attack" : [
        "attack_ppo_static",
        "attack_ppo_fuzzy",
        "attack_v1",
        "attack_v2"
    ],
    "Defense": [
        "defense_ppo_static",
        "defense2_ppo",
        "defense3_v1",
        "defense3_v2",
        "defense_v3"
    ],
}

group_labels = {
    "Normal" : ["PPO\nStatic","PPO\nFuzzy","MAPPO\nv1","MAPPO\nv2"],
    "Attack" : ["PPO\nStatic","PPO\nFuzzy","MAPPO\nv1","MAPPO\nv2"],
    "Defense": ["PPO\nStatic","PPO\nFuzzy","MAPPO\nv1","MAPPO\nv2","MAPPO\nv3"],
}

tactic_colors = {
    "Normal" : "#3498db",
    "Attack" : "#e74c3c",
    "Defense": "#27ae60"
}

# ── HELPERS ──────────────────────────────────────────
def load_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def compute_xg(bx, by):
    goal_pos = np.array([1.0, 0.0])
    ball_pos = np.array([bx, by])
    distance = np.linalg.norm(ball_pos - goal_pos)
    angle    = np.arctan2(0.05, max(distance, 0.01))
    xg       = (angle * 2.0) / (distance + 1.0)
    return float(np.clip(xg, 0.0, 1.0))

def draw_pitch(ax, bg='#2d6a2d', lc='white'):
    ax.set_facecolor(bg)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.7,   0.7)
    ax.add_patch(patches.Rectangle(
        (-1,-0.65), 2, 1.3,
        lw=2, edgecolor=lc, facecolor='none'))
    ax.axvline(0, color=lc, lw=1.5)
    ax.add_patch(plt.Circle(
        (0,0), 0.15, color=lc, fill=False, lw=1.5))
    ax.plot(0, 0, 'o', color=lc, ms=3)
    for x0,y0,w,h in [
        (-1,-0.22,0.16,0.44),
        (0.84,-0.22,0.16,0.44),
        (-1.04,-0.05,0.04,0.10),
        (1.0,-0.05,0.04,0.10)]:
        ax.add_patch(patches.Rectangle(
            (x0,y0), w, h,
            lw=1.5, edgecolor=lc, facecolor='none'))
    ax.set_aspect('equal')
    ax.axis('off')

# ════════════════════════════════════════════════════
# FIGURE 1 — Ball Position Heatmaps
# ════════════════════════════════════════════════════
print("Figure 1: Ball Position Heatmaps...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Ball Position Heatmaps — Best Model Per Tactic",
    fontsize=14, fontweight='bold')

for ax, (title, key) in zip(axes, BEST_MODELS.items()):
    data = load_pkl(
        f"{ANALYSIS_DIR}/{key}/ball_positions.pkl")
    draw_pitch(ax, bg='#1a4a1a', lc='white')
    if data:
        xs = [p[0] for p in data]
        ys = [p[1] for p in data]
        h  = ax.hist2d(xs, ys, bins=40,
                       range=[[-1,1],[-0.65,0.65]],
                       cmap='hot', alpha=0.75)
        plt.colorbar(h[3], ax=ax, fraction=0.03,
                     label='Frequency')
    r = EVAL_RESULTS.get(key, {})
    ax.set_title(
        f"{title}\n"
        f"GA:{r.get('GA','?')}  "
        f"D%:{r.get('D%','?')}%  "
        f"CS:{r.get('CS','?')}",
        fontsize=10, fontweight='bold', pad=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_ball_heatmaps.png",
            dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved: fig1_ball_heatmaps.png")

# ════════════════════════════════════════════════════
# FIGURE 2 — Player Position Heatmaps
# ════════════════════════════════════════════════════
print("Figure 2: Player Position Heatmaps...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Our Team Player Position Heatmaps — Best Model Per Tactic",
    fontsize=14, fontweight='bold')

for ax, (title, key) in zip(axes, BEST_MODELS.items()):
    data = load_pkl(
        f"{ANALYSIS_DIR}/{key}/player_positions.pkl")
    draw_pitch(ax)
    if data:
        all_xs, all_ys = [], []
        for frame in data:
            for px, py in frame['left']:
                all_xs.append(px)
                all_ys.append(py)
        h = ax.hist2d(all_xs, all_ys, bins=40,
                      range=[[-1,1],[-0.65,0.65]],
                      cmap='hot', alpha=0.85)
        plt.colorbar(h[3], ax=ax, fraction=0.03,
                     label='Frequency')
    ax.set_title(title, fontsize=11,
                 fontweight='bold', pad=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_player_heatmaps.png",
            dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved: fig2_player_heatmaps.png")

# ════════════════════════════════════════════════════
# FIGURE 3 — Time In Each Third
# ════════════════════════════════════════════════════
print("Figure 3: Time in Each Third...")
names, own_t, mid_t, opp_t = [], [], [], []

for title, key in BEST_MODELS.items():
    data = load_pkl(
        f"{ANALYSIS_DIR}/{key}/ball_positions.pkl")
    names.append(title.replace('\n', ' '))
    if data:
        xs    = [p[0] for p in data]
        total = len(xs)
        own_t.append(
            sum(1 for x in xs if x < -0.33)/total*100)
        mid_t.append(
            sum(1 for x in xs
                if -0.33<=x<=0.33)/total*100)
        opp_t.append(
            sum(1 for x in xs if x > 0.33)/total*100)
    else:
        own_t.append(0)
        mid_t.append(0)
        opp_t.append(0)

fig, ax = plt.subplots(figsize=(10, 6))
x, w = np.arange(len(names)), 0.25
b1 = ax.bar(x-w, own_t, w,
            label='Own Third',
            color='#e74c3c', alpha=0.85)
b2 = ax.bar(x,   mid_t, w,
            label='Middle Third',
            color='#f39c12', alpha=0.85)
b3 = ax.bar(x+w, opp_t, w,
            label='Opponent Third',
            color='#27ae60', alpha=0.85)

for bars in [b1, b2, b3]:
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.1f}%',
            xy=(b.get_x()+b.get_width()/2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel('Time (%)', fontsize=12)
ax.set_title(
    'Ball Time in Each Pitch Third — Best Model Per Tactic',
    fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 75)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_time_in_thirds.png",
            dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved: fig3_time_in_thirds.png")

# ════════════════════════════════════════════════════
# FIGURE 4 — Shoot Position + xG Scatter
# ════════════════════════════════════════════════════
print("Figure 4: Shoot Position Maps...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Shooting Position Maps (Color = xG Quality) "
    "— Best Model Per Tactic",
    fontsize=14, fontweight='bold')

for ax, (title, key) in zip(axes, BEST_MODELS.items()):
    data = load_pkl(
        f"{ANALYSIS_DIR}/{key}/shoot_positions.pkl")
    r = EVAL_RESULTS.get(key, {})

    # Background
    ax.set_facecolor('#2d6a2d')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Pitch lines using plot() — always visible
    theta = np.linspace(0, 2*np.pi, 100)

    # Pitch outline
    ax.plot([-1,-1, 1, 1,-1],
            [-0.65,0.65,0.65,-0.65,-0.65],
            'w-', lw=2, zorder=10)
    # Halfway line
    ax.plot([0,0],[-0.65,0.65],
            'w-', lw=1.5, zorder=10)
    # Left penalty box
    ax.plot([-1,-0.84,-0.84,-1],
            [-0.22,-0.22,0.22,0.22],
            'w-', lw=1.5, zorder=10)
    # Right penalty box
    ax.plot([1,0.84,0.84,1],
            [-0.22,-0.22,0.22,0.22],
            'w-', lw=1.5, zorder=10)
    # Left goal
    ax.plot([-1,-1.04,-1.04,-1],
            [-0.05,-0.05,0.05,0.05],
            'w-', lw=1.5, zorder=10)
    # Right goal
    ax.plot([1,1.04,1.04,1],
            [-0.05,-0.05,0.05,0.05],
            'w-', lw=1.5, zorder=10)
    # Centre circle
    ax.plot(0.15*np.cos(theta),
            0.15*np.sin(theta),
            'w-', lw=1.5, zorder=10)
    # Centre dot
    ax.plot(0, 0, 'wo', ms=4, zorder=10)

    if data and len(data) > 0:
        xs  = [p[0] for p in data]
        ys  = [p[1] for p in data]
        xgs = [compute_xg(x, y)
               for x, y in zip(xs, ys)]
        sc  = ax.scatter(xs, ys, c=xgs,
                         cmap='RdYlGn',
                         s=250,
                         alpha=0.95,
                         vmin=0, vmax=0.15,
                         edgecolors='black',
                         linewidths=0.8,
                         zorder=20)
        plt.colorbar(sc, ax=ax,
                     fraction=0.03, label='xG')
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
                zorder=20)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_shoot_xg_map.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_shoot_xg_map.png")

# ════════════════════════════════════════════════════
# FIGURE 5 — xG Distribution Histogram
# ════════════════════════════════════════════════════
print("Figure 5: xG Distribution...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "xG Distribution Per Shot — Best Model Per Tactic",
    fontsize=14, fontweight='bold')

for ax, (title, key) in zip(axes, BEST_MODELS.items()):
    data = load_pkl(
        f"{ANALYSIS_DIR}/{key}/shoot_positions.pkl")

    if data and len(data) > 0:
        xgs = [compute_xg(p[0], p[1]) for p in data]
        ax.hist(xgs, bins=20, range=(0, 0.3),
                color='#3498db', alpha=0.8,
                edgecolor='black')
        ax.axvline(np.mean(xgs), color='red',
                   linestyle='--', lw=2,
                   label=f'Mean={np.mean(xgs):.3f}')
        high = sum(1 for x in xgs if x > 0.08)
        med  = sum(1 for x in xgs if 0.04 < x <= 0.08)
        low  = sum(1 for x in xgs if x <= 0.04)
        ax.set_title(
            f"{title}\n"
            f"High(>0.08):{high} | "
            f"Med:{med} | Low:{low}",
            fontsize=9, fontweight='bold')
        ax.legend(fontsize=9)
    else:
        ax.set_title(f"{title}\nNo shots",
                     fontsize=10, fontweight='bold')
        ax.text(0.5, 0.5, 'No shooting events',
                ha='center', va='center',
                transform=ax.transAxes, fontsize=10)

    ax.set_xlabel('xG per Shot', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_xg_distribution.png",
            dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved: fig5_xg_distribution.png")

# ════════════════════════════════════════════════════
# FIGURE 6 — Goals Conceded All Models
# ════════════════════════════════════════════════════
print("Figure 6: Goals Conceded Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Goals Conceded Per Match — All Architecture Stages",
    fontsize=14, fontweight='bold')

for ax, (tactic, keys) in zip(axes, groups.items()):
    labels = group_labels[tactic]
    gas    = [EVAL_RESULTS.get(k,{}).get('GA', 0)
              for k in keys]
    color  = tactic_colors[tactic]
    bars   = ax.bar(labels, gas,
                    color=color, alpha=0.8,
                    edgecolor='black', lw=0.5)
    for bar, val in zip(bars, gas):
        ax.annotate(f'{val:.2f}',
            xy=(bar.get_x()+bar.get_width()/2,
                bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9)
    ax.set_title(f"{tactic} Tactic",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Goals Conceded/Match', fontsize=10)
    ax.set_ylim(0, max(gas)*1.3 + 0.1 if gas else 1)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig6_goals_conceded.png",
            dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved: fig6_goals_conceded.png")

# ════════════════════════════════════════════════════
# FIGURE 7 — Shoot Events All Models (Log Scale)
# ════════════════════════════════════════════════════
print("Figure 7: Shoot Events Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Shoot Events Per Match — All Architecture Stages "
    "(Log Scale)",
    fontsize=14, fontweight='bold')

for ax, (tactic, keys) in zip(axes, groups.items()):
    labels = group_labels[tactic]
    shoots = [max(EVAL_RESULTS.get(k,{})
                  .get('Shoots', 0)/50, 0.1)
              for k in keys]
    color  = tactic_colors[tactic]
    bars   = ax.bar(labels, shoots,
                    color=color, alpha=0.8,
                    edgecolor='black', lw=0.5)
    ax.set_yscale('log')
    for bar, val in zip(bars, shoots):
        ax.annotate(f'{val:.0f}',
            xy=(bar.get_x()+bar.get_width()/2,
                bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9)
    ax.set_title(f"{tactic} Tactic",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Shoots/Match (log scale)', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig7_shoot_events.png",
            dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved: fig7_shoot_events.png")

# ════════════════════════════════════════════════════
# FIGURE 8 — Draw Rate and Clean Sheets
# ════════════════════════════════════════════════════
print("Figure 8: Draw Rate and Clean Sheets...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Draw Rate % and Clean Sheet % — All Architecture Stages",
    fontsize=14, fontweight='bold')

for ax, (tactic, keys) in zip(axes, groups.items()):
    labels = group_labels[tactic]
    draws  = [EVAL_RESULTS.get(k,{}).get('D%', 0)
              for k in keys]
    css    = [EVAL_RESULTS.get(k,{}).get('CS', 0)*2
              for k in keys]
    color  = tactic_colors[tactic]
    x      = np.arange(len(labels))
    w      = 0.35
    b1     = ax.bar(x-w/2, draws, w,
                    label='Draw %',
                    color=color, alpha=0.85,
                    edgecolor='black')
    b2     = ax.bar(x+w/2, css, w,
                    label='Clean Sheet %',
                    color=color, alpha=0.45,
                    edgecolor='black', hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"{tactic} Tactic",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('%', fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig8_draw_cleansheet.png",
            dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved: fig8_draw_cleansheet.png")

# ════════════════════════════════════════════════════
# FIGURE 9 — Average Ball X All Models
# ════════════════════════════════════════════════════
print("Figure 9: Average Ball X Position...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Average Ball X Position — All Architecture Stages\n"
    "(Positive = Opponent Half, Negative = Own Half)",
    fontsize=14, fontweight='bold')

for ax, (tactic, keys) in zip(axes, groups.items()):
    labels  = group_labels[tactic]
    avg_xs  = []
    for key in keys:
        data = load_pkl(
            f"{ANALYSIS_DIR}/{key}/ball_positions.pkl")
        if data:
            xs = [p[0] for p in data]
            avg_xs.append(np.mean(xs))
        else:
            avg_xs.append(0.0)

    color = tactic_colors[tactic]
    bars  = ax.bar(labels, avg_xs,
                   color=[color if v >= 0
                          else '#e74c3c'
                          for v in avg_xs],
                   alpha=0.85,
                   edgecolor='black')
    ax.axhline(0, color='black', lw=1.5,
               linestyle='--', label='Halfway line')
    for bar, val in zip(bars, avg_xs):
        ax.annotate(f'{val:.3f}',
            xy=(bar.get_x()+bar.get_width()/2, val),
            xytext=(0, 3 if val >= 0 else -12),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9)
    ax.set_title(f"{tactic} Tactic",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Ball X Position', fontsize=10)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig9_avg_ball_x.png",
            dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved: fig9_avg_ball_x.png")

print(f"\n✅ All 9 figures saved to: {OUTPUT_DIR}/")
print("\nFigures:")
for i, n in enumerate([
    "fig1_ball_heatmaps.png",
    "fig2_player_heatmaps.png",
    "fig3_time_in_thirds.png",
    "fig4_shoot_xg_map.png",
    "fig5_xg_distribution.png",
    "fig6_goals_conceded.png",
    "fig7_shoot_events.png",
    "fig8_draw_cleansheet.png",
    "fig9_avg_ball_x.png",
], 1):
    print(f"  {i}. {n}")