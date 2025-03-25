import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


for i in range(16):
    
    fig, ax = plt.subplots(1, 1,
                        figsize=(36, 18), gridspec_kw={'wspace': 0, 'hspace': 0}, constrained_layout=True)
    color = {8: (255, 255, 0), 16: (0, 255, 255), 32: (255, 0, 255)}

    MAX_DIST = 70
    SCALE = 30

    # Create BEV Space
    R = (MAX_DIST * SCALE)
    space = np.ones((R * 2, R * 2, 3), dtype=np.uint8) * 255

    for theta in np.linspace(0, np.pi, 7):
        space = cv2.line(space, pt1=(int(R - R * np.cos(theta)), int(R - R * np.sin(theta))), pt2=(R, R),
                        color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    for radius in np.linspace(0, R, 5):
        if radius == 0:
            continue
        space = cv2.circle(space, center=(R, R), radius=int(radius), color=(0, 0, 0), thickness=2,
                        lineType=cv2.LINE_AA)
    space = space[:R, :, :]
    
    a2d = np.load(f"/home/stud/mijo/dev/2DTAL_{i}.npy")
    a3d = np.load(f"/home/stud/mijo/dev/3DTAL_{i}.npy")
    agt = np.load(f"/home/stud/mijo/dev/GT_{i}.npy")
    
    for assigned2d in a2d:
        bottom_corners = (assigned2d[:4] * SCALE)
        x = bottom_corners[:, 0] + R
        y = -bottom_corners[:, 2] + R
        pts = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1).astype(np.int32)[
            [0, 1, 3, 2]]
        space = cv2.polylines(space, pts=[pts], isClosed=True, color=(255, 0, 0), thickness=3)
    for assigned3d in a3d:
        bottom_corners = (assigned3d[:4] * SCALE)
        x = bottom_corners[:, 0] + R
        y = -bottom_corners[:, 2] + R
        pts = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1).astype(np.int32)[
            [0, 1, 3, 2]]
        space = cv2.polylines(space, pts=[pts], isClosed=True, color=(0, 0, 255), thickness=3)
    for gt in agt:
        bottom_corners = (gt[:4] * SCALE)
        x = bottom_corners[:, 0] + R
        y = -bottom_corners[:, 2] + R
        pts = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1).astype(np.int32)[
            [0, 1, 3, 2]]
        space = cv2.polylines(space, pts=[pts], isClosed=True, color=(0, 255, 0), thickness=3)

    ax.imshow(space)
    ax.axis("off")
    plt.savefig(f"/home/stud/mijo/dev/tal_assignments_{i}.png", dpi=300)
    print(f"Saved to: /home/stud/mijo/dev/tal_assignments_{i}.png")
print()
