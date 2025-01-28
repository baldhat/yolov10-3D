import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

kitti = pd.read_csv("~/Downloads/labels_kitti_car.csv")
waymo = pd.read_csv("~/Downloads/labels_waymo_car.csv")
rope3d = pd.read_csv("~/Downloads/labels_rope3d_car.csv")
cdrone = pd.read_csv("~/Downloads/labels_cdrone_car.csv")

matplotlib.rcParams.update({'font.size': 22})

pal = sn.color_palette("RdBu_r", n_colors=4)
fig, axes = plt.subplots(1, 3, figsize=(40, 8))
for i, (z, col) in enumerate(zip([kitti["z"], waymo["z"], rope3d["z"], cdrone["z"]], ["blue", "orange", "green", "red"])):
    sn.histplot(z, bins=[0, 200], stat="density", binwidth=5, ax=axes[0], color=pal[i])
axes[0].set_xlim([0, 200])
axes[0].set_xlabel("Depth")
axes[0].yaxis.set_visible(False)

for i, (z, col) in enumerate(zip([kitti["length"], waymo["length"], rope3d["length"], cdrone["length"]], ["blue", "orange", "green", "red"])):
    sn.histplot(z, bins=[0, 8], stat="density", binwidth=0.1, ax=axes[1], color=pal[i])
axes[1].set_xlim([1, 8])
axes[1].set_xlabel("Length")
axes[1].yaxis.set_visible(False)

for i, (z, col) in enumerate(zip([kitti["height"], waymo["height"], rope3d["height"], cdrone["height"]], ["blue", "orange", "green", "red"])):
    sn.histplot(z, bins=[0, 4], stat="density", binwidth=0.2, ax=axes[2], color=pal[i])
axes[2].set_xlim([0, 4])
axes[2].set_xlabel("Height")
axes[2].yaxis.set_visible(False)

fig.legend(labels=["Kitti", "Waymo", "Rope3D", "CDrone"], loc="upper right", ncol=4)

fig.tight_layout()

plt.savefig(f"distributions.svg")
plt.close()

'''
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

kitti = pd.read_csv("~/Downloads/labels_kitti_car.csv")
waymo = pd.read_csv("~/Downloads/labels_waymo_car.csv")
rope3d = pd.read_csv("~/Downloads/labels_rope3d_car.csv")
cdrone = pd.read_csv("~/Downloads/labels_cdrone_car.csv")

fig, ax = plt.subplots()
#for z, col in zip([kitti["z"], waymo["z"]], ["blue", "orange"]):
plt.hist([kitti["z"], waymo["z"], cdrone["z"], rope3d["z"]], bins=range(0, 200, 5), density=True)
ax.set_xlim([0, 200])
ax.yaxis.set_visible(False)
#sn.pairplot(x, corner=True, diag_kind="auto", kind="hist", diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
plt.savefig(f"distributions.svg")
plt.close()
'''