import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path

base_dir = "/home/stud/mijo/experiments/results"

def annot_max(x,y, ax):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= f"epoch={xmax}, AP={ymax}".format(xmax, ymax)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="data",
              arrowprops=arrowprops, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax + 1, ymax + 1), **kw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs='+')
    parser.add_argument("--filename")
    args = parser.parse_args()
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    
    for f in args.runs:
        fullpath = Path(base_dir, f, "results.csv")
        if not os.path.exists(fullpath):
            raise RuntimeError(f"No run with this name: {fullpath}")
        
        df = pd.read_csv(fullpath, header=0)
        df = df.rename(columns=lambda x: x.strip())
        x = df["epoch"]
        y = df["metrics/Car3D@0.7"]
        ax.plot(x, y, label=f)
        annot_max(x, y, ax)
    ax.legend(loc="upper left")
    plt.savefig(Path(base_dir, args.filename))
        
if __name__=="__main__":
    main()