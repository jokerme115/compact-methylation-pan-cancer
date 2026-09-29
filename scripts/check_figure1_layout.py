"""Programmatic layout check for the npj-style Fig. 1.

Since the figure cannot be visually inspected in this session, verify via
the matplotlib renderer that (1) every text/patch stays inside the canvas
and (2) no two text elements overlap.  Run after build_figure1_npj_style.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_figure1_npj_style as B


def collect_boxes(fig):
    boxes = []
    for ax in fig.axes:
        for t in ax.texts:
            bb = t.get_window_extent(renderer=RENDERER)
            boxes.append((t.get_text().replace("\n", " ")[:40], bb))
    return boxes


fig = plt.figure(figsize=(7.15, 5.2))
ax_a = fig.add_axes([0.0, 0.455, 1.0, 0.52])
ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.axis("off")
B.panel_a(ax_a)
ax_b = fig.add_axes([0.0, 0.02, 1.0, 0.42])
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis("off")
B.panel_b(fig, ax_b)
fig.canvas.draw()
RENDERER = fig.canvas.get_renderer()

W, H = fig.canvas.get_width_height()
boxes = collect_boxes(fig)
problems = []
for label, bb in boxes:
    if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
        problems.append(f"OUT OF CANVAS: {label!r} bbox=({bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f})")

for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        l1, b1 = boxes[i]
        l2, b2 = boxes[j]
        ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ix > 2 and iy > 2:
            area = ix * iy
            smaller = min(b1.width * b1.height, b2.width * b2.height)
            if smaller > 0 and area / smaller > 0.25:
                problems.append(f"OVERLAP {area/smaller:.0%}: {l1!r} <-> {l2!r}")

if problems:
    print(f"{len(boxes)} texts checked; {len(problems)} issues:")
    for p in problems:
        print("  " + p)
else:
    print(f"{len(boxes)} texts checked; no out-of-canvas or major overlap issues.")
plt.close(fig)
