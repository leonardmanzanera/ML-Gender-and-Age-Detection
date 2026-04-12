"""
Chronogramme comparatif — V3 (Sync) vs V3.1 (Async)
Visualise le blocage de la boucle principale frame par frame.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Timings (ms) ─────────────────────────────────────────────────────────────
# V3 Sync — tout bloque sur le thread principal
V3_READ    = 5
V3_YOLO    = 90
V3_VIT     = 120   # ViT ONNX bloquant
V3_CAFFE   = 30    # Caffe gender bloquant
V3_DISPLAY = 5

# V3.1 Async — YOLO bloque encore, ViT en thread background
V31_READ    = 5
V31_YOLO    = 90    # YOLO CPU sans imgsz fix
V31_DISPLAY = 5
V31_VIT_BG  = 120  # ViT dans thread background (non bloquant)

# V3.1 Async + Fix (imgsz=320, device=mps)
V31F_READ    = 5
V31F_YOLO    = 12   # MPS + imgsz=320
V31F_DISPLAY = 5
V31F_VIT_BG  = 120  # toujours en background

N_FRAMES = 4

# ─── Couleurs ─────────────────────────────────────────────────────────────────
C_READ    = "#4A90D9"
C_YOLO    = "#E74C3C"
C_VIT     = "#F39C12"
C_CAFFE   = "#8E44AD"
C_DISPLAY = "#27AE60"
C_VIT_BG  = "#F39C12"
C_IDLE    = "#2C2C2C"

fig, axes = plt.subplots(3, 1, figsize=(16, 10), facecolor="#1A1A2E")
fig.suptitle(
    "Chronogramme comparatif — V3 Sync vs V3.1 Async vs V3.1 Async + Fix (imgsz=320, MPS)",
    color="white", fontsize=13, fontweight="bold", y=0.98
)

def draw_block(ax, y, x_start, width, color, label=None, alpha=1.0):
    rect = mpatches.FancyBboxPatch(
        (x_start, y - 0.35), width, 0.7,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="#1A1A2E",
        linewidth=1.5, alpha=alpha
    )
    ax.add_patch(rect)
    if label and width > 8:
        ax.text(x_start + width / 2, y, label,
                ha="center", va="center",
                fontsize=7, color="white", fontweight="bold")

def setup_ax(ax, title, max_t):
    ax.set_facecolor("#1A1A2E")
    ax.set_xlim(0, max_t)
    ax.set_ylim(-0.2, 2.5)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines["bottom"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Temps (ms)", color="#AAA", fontsize=8)
    for g in np.arange(0, max_t, 50):
        ax.axvline(g, color="#333", linewidth=0.5, zorder=0)

# ─── V3 Synchrone ─────────────────────────────────────────────────────────────
ax = axes[0]
frame_ms = V3_READ + V3_YOLO + V3_VIT + V3_CAFFE + V3_DISPLAY
setup_ax(ax, f"V3 — Synchrone  |  {frame_ms}ms/frame  →  {1000/frame_ms:.0f} FPS", 1100)

ax.set_yticks([1])
ax.set_yticklabels(["Main\nThread"], color="white", fontsize=8)

t = 0
for _ in range(N_FRAMES):
    draw_block(ax, 1, t, V3_READ,    C_READ,    "read")
    t += V3_READ
    draw_block(ax, 1, t, V3_YOLO,    C_YOLO,    "YOLO\n90ms")
    t += V3_YOLO
    draw_block(ax, 1, t, V3_VIT,     C_VIT,     "ViT\n120ms")
    t += V3_VIT
    draw_block(ax, 1, t, V3_CAFFE,   C_CAFFE,   "Caffe\n30ms")
    t += V3_CAFFE
    draw_block(ax, 1, t, V3_DISPLAY, C_DISPLAY, "show")
    t += V3_DISPLAY

# FPS markers
for i in range(N_FRAMES + 1):
    ax.axvline(i * frame_ms, color="#555", linestyle="--", linewidth=0.8)
    if i > 0:
        ax.text(i * frame_ms - frame_ms / 2, 2.1, f"Frame {i}",
                ha="center", color="#888", fontsize=7)

# ─── V3.1 Async (avant fix) ────────────────────────────────────────────────────
ax = axes[1]
frame_main = V31_READ + V31_YOLO + V31_DISPLAY
# ViT BG starts after first submit (frame 1), runs every ~120ms
setup_ax(ax, f"V3.1 — Async ViT (sans fix)  |  ~{frame_main}ms/frame main thread  →  ~{1000/frame_main:.0f} FPS  (mesuré: 11fps car overhead)", 1100)

ax.set_yticks([1, 2])
ax.set_yticklabels(["ViT\nThread", "Main\nThread"], color="white", fontsize=8)

t = 0
vit_t = V31_READ + V31_YOLO  # submit happens after YOLO
for i in range(N_FRAMES):
    draw_block(ax, 2, t, V31_READ,    C_READ,    "read")
    t += V31_READ
    draw_block(ax, 2, t, V31_YOLO,    C_YOLO,    "YOLO\n90ms")
    t += V31_YOLO
    draw_block(ax, 2, t, V31_DISPLAY, C_DISPLAY, "show")
    t += V31_DISPLAY

# ViT background thread
bt = vit_t
for i in range(N_FRAMES + 1):
    draw_block(ax, 1, bt, V31_VIT_BG, C_VIT, "ViT\n120ms", alpha=0.85)
    bt += V31_VIT_BG

# Idle zones (main thread blocked waiting — not applicable here, just YOLO blocks)
ax.annotate("", xy=(V31_READ + V31_YOLO, 1.6), xytext=(V31_READ, 1.6),
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5))
ax.text((V31_READ + V31_READ + V31_YOLO) / 2, 1.72,
        "YOLO bloque encore la boucle principale",
        ha="center", color="#E74C3C", fontsize=7.5)

for i in range(N_FRAMES + 1):
    ax.axvline(i * frame_main, color="#555", linestyle="--", linewidth=0.8)
    if i > 0:
        ax.text(i * frame_main - frame_main / 2, 2.5, f"Frame {i}",
                ha="center", color="#888", fontsize=7)

# ─── V3.1 Async + Fix ─────────────────────────────────────────────────────────
ax = axes[2]
frame_fixed = V31F_READ + V31F_YOLO + V31F_DISPLAY
fps_fixed = 1000 / frame_fixed
setup_ax(ax, f"V3.1 — Async + Fix (imgsz=320 + MPS)  |  ~{frame_fixed}ms/frame  →  ~{fps_fixed:.0f} FPS", 1100)

ax.set_yticks([1, 2])
ax.set_yticklabels(["ViT\nThread", "Main\nThread"], color="white", fontsize=8)

t = 0
for i in range(N_FRAMES):
    draw_block(ax, 2, t, V31F_READ,    C_READ,    "rd")
    t += V31F_READ
    draw_block(ax, 2, t, V31F_YOLO,    C_YOLO,    "YOLO 12ms")
    t += V31F_YOLO
    draw_block(ax, 2, t, V31F_DISPLAY, C_DISPLAY, "sh")
    t += V31F_DISPLAY
    # idle (no YOLO bottleneck)
    if i < N_FRAMES - 1:
        next_frame_start = (i + 1) * frame_fixed
        pass  # frames are back-to-back

bt = V31F_READ + V31F_YOLO
for i in range(N_FRAMES + 1):
    draw_block(ax, 1, bt, V31F_VIT_BG, C_VIT, "ViT 120ms", alpha=0.85)
    bt += V31F_VIT_BG

ax.annotate("", xy=(V31F_READ + V31F_YOLO, 1.6), xytext=(V31F_READ, 1.6),
            arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5))
ax.text((V31F_READ + V31F_READ + V31F_YOLO) / 2, 1.72,
        "YOLO 8x plus rapide → boucle libérée",
        ha="center", color="#27AE60", fontsize=7.5)

for i in range(N_FRAMES + 1):
    ax.axvline(i * frame_fixed, color="#555", linestyle="--", linewidth=0.8)
    if i > 0:
        ax.text(i * frame_fixed - frame_fixed / 2, 2.5, f"Frame {i}",
                ha="center", color="#888", fontsize=7)

# ─── Légende ──────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=C_READ,    label="cap.read()"),
    mpatches.Patch(color=C_YOLO,    label="YOLO predict (bloquant)"),
    mpatches.Patch(color=C_VIT,     label="ViT ONNX inference"),
    mpatches.Patch(color=C_CAFFE,   label="Caffe gender (bloquant)"),
    mpatches.Patch(color=C_DISPLAY, label="cv2.imshow()"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=5,
           framealpha=0.15, labelcolor="white", fontsize=8,
           facecolor="#1A1A2E", edgecolor="#444")

plt.tight_layout(rect=[0, 0.05, 1, 0.97])

out = "figures/chronogram_v3_vs_v31.png"
import os
os.makedirs("figures", exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1A1A2E")
print(f"Saved → {out}")
