import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

# Load Cyber font
cyber_font = fm.FontProperties(fname='fonts/Cyber-Bold.ttf')

# Metallic radar color
RADAR_COLOR = '#C0C0C0'

def adjust_radar_labels(ax):
    """Pushes radar labels outward and adjusts styling."""
    for label in ax.get_xticklabels():
        label.set_fontproperties(cyber_font)
        label.set_color("#C0C0C0")
        label.set_fontsize(14)

        # Move labels further out
        x, y = label.get_position()
        label.set_position((x, y - 0.2))  # Increase value if needed

def plot_emotion_radar(labels, scores):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores = scores.tolist()
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    # Line + fill
    ax.plot(angles, scores, color=RADAR_COLOR, linewidth=2)
    ax.fill(angles, scores, color=RADAR_COLOR, alpha=0.1)

    # Background
    ax.set_facecolor('#0d0d0f')
    fig.patch.set_facecolor('#0d0d0f')
    ax.set_yticklabels([])

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='#C0C0C0', fontproperties=cyber_font)
    adjust_radar_labels(ax)

    # Grid styling
    ax.spines['polar'].set_color(RADAR_COLOR)
    ax.spines['polar'].set_alpha(0.7)
    ax.tick_params(colors='#C0C0C0')
    ax.grid(color='#666', linewidth=1, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.close(fig)
    return fig
