import matplotlib.pyplot as plt
import numpy as np

# Data
names = [
    "Puskesmas A", "Puskesmas B", "Puskesmas C", "Puskesmas D",
    "Puskesmas E", "Puskesmas F", "Puskesmas G", "Puskesmas H",
    "Puskesmas I", "Puskesmas J", "Puskesmas K", "Puskesmas L",
]
X = np.array([15, 45, 48, 50, 52, 70, 75, 72, 78, 74, 47, 95])   # Kelengkapan Alat
Y = np.array([20, 50, 52, 48, 55, 72, 70, 75, 74, 78, 49, 98])   # Kesiapan Tenaga
labels = np.array([-1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, -1])        # Hasil DBSCAN

#  Colour / marker mapping
colour_map = {
    1:  "#3b82f6",   # Klaster 1  – blue
    2:  "#22c55e",   # Klaster 2  – green
    -1: "#ef4444",   # Noise      – red
}
label_text = {1: "Klaster 1", 2: "Klaster 2", -1: "Noise (-1)"}
marker_map = {1: "o", 2: "s", -1: "X"}

# Figure
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#1e1e2e")
ax.set_facecolor("#1e1e2e")

# Plot each group
for lbl in sorted(colour_map.keys()):
    mask = labels == lbl
    ax.scatter(
        X[mask], Y[mask],
        c=colour_map[lbl],
        marker=marker_map[lbl],
        s=140,
        edgecolors="white",
        linewidths=0.8,
        label=label_text[lbl],
        zorder=3,
    )

# Annotate every point
for i, name in enumerate(names):
    short = name.replace("Puskesmas ", "")        # just the letter
    ax.annotate(
        short,
        (X[i], Y[i]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        fontweight="bold",
        color="white",
    )

# Cosmetics
ax.set_title(
    "Scatter Plot DBSCAN – Evaluasi Kesiapan Puskesmas",
    fontsize=15, fontweight="bold", color="white", pad=14,
)
ax.set_xlabel("Kelengkapan Alat (X)", fontsize=12, color="white")
ax.set_ylabel("Kesiapan Tenaga (Y)", fontsize=12, color="white")

ax.tick_params(colors="white", labelsize=10)
for spine in ax.spines.values():
    spine.set_color("#444466")

ax.legend(
    loc="upper left",
    fontsize=10,
    framealpha=0.3,
    labelcolor="white",
    facecolor="#2a2a3e",
    edgecolor="#444466",
)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4, color="gray")

plt.tight_layout()
plt.savefig(
    r"d:\Kuliah\Kuliah Semester 4\Pembelajaran Mesin\Week 11\Code\scatter_dbscan.png",
    dpi=180,
    facecolor=fig.get_facecolor(),
)
plt.show()
