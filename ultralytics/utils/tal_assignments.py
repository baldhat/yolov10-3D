import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import math
import cv2

def plot(ax, it, color):
    bottom_corners = (it[:4])
    x = bottom_corners[:, 0] 
    y = bottom_corners[:, 2]
    pts = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1)[
        [0, 1, 3, 2]]
    ax.add_artist(Polygon(pts, closed=True, fill=False, edgecolor=color, facecolor=color, zorder=3, linewidth=3))

for i in range(16):
    
    fig, ax = plt.subplots(1, 1,
                        figsize=(36, 18), gridspec_kw={'wspace': 0, 'hspace': 0}, constrained_layout=True)
    
    R = 60
    ax.set_xlim(-R, R)
    ax.set_ylim(0, R)
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('white')
    color = {8: (255, 255, 0), 16: (0, 255, 255), 32: (255, 0, 255)}

    for theta in np.linspace(0, np.pi, 7):
        xs, ys = [R * np.cos(theta), 0], [R * np.sin(theta), 0]
        ax.plot(xs, ys, linewidth=2, color=(0,0,0), zorder=1)

    for radius in np.linspace(0, R, 5):
        if radius == 0:
            continue
        circle = Circle((0, 0), radius, edgecolor=(0,0,0), linewidth=2, fill=False, zorder=1)
        ax.add_artist(circle)
    
    a2d = np.load(f"/home/stud/mijo/dev/2DTAL_{i}.npy")
    a3d = np.load(f"/home/stud/mijo/dev/3DTAL_{i}.npy")
    agt = np.load(f"/home/stud/mijo/dev/GT_{i}.npy")
    
    for assigned2d in a2d:
        plot(ax,assigned2d, "r")
    for assigned3d in a3d:
        plot(ax,assigned3d, "b")
    for gt in agt:
        plot(ax,gt, "g")

    plt.savefig(f"/home/stud/mijo/dev/tal_assignments_{i}.svg", dpi=300, format="svg")
    print(f"Saved to: /home/stud/mijo/dev/tal_assignments_{i}.svg")
print()
