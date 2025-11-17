import numpy as np
import matplotlib.pyplot as plt


data8 = np.load("/home/stud/mijo/iou_entwicklung8.npy")
plt.plot(data8[0])
plt.plot(data8[1])
plt.savefig("fig8.svg")
plt.close()

data1 = np.load("/home/stud/mijo/iou_entwicklung1.npy")
plt.plot(data1[0])
plt.plot(data1[1])
plt.savefig("fig1.svg")
plt.close()

def convolve(x, n=200):
    return np.convolve(x, np.ones(n)/n, mode='valid')

N = 10
convolved0 = np.convolve(data8[0], np.ones(N)/N, mode='valid')
convolved1 = np.convolve(data8[1], np.ones(N)/N, mode='valid')
plt.plot(convolved0)
plt.plot(convolved1)
plt.savefig("convolved8.svg")
plt.close()

N = 10
convolved0 = np.convolve(data1[0], np.ones(N)/N, mode='valid')
convolved1 = np.convolve(data1[1], np.ones(N)/N, mode='valid')
plt.plot(convolved0)
plt.plot(convolved1)
plt.savefig("convolved1.svg")
plt.close()

def running_max_last_n(a: np.ndarray, n: int = 20) -> np.ndarray:
    a = np.ravel(a)
    pad_width = n - 1
    padded = np.concatenate([np.full(pad_width, a[0]), a])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=n)
    return windows.max(axis=1)

max0 = running_max_last_n(data1[0])
max1 = running_max_last_n(data1[1])
plt.plot(max0, c=(0.5, 0, 0))
plt.plot(max1, c=(0, 0.5, 0))
plt.plot(convolve(max0, n=200), c=(1, 0, 0))
plt.plot(convolve(max1, n=200), c=(0, 1, 0))
plt.savefig("max1.svg")
plt.close()

max0 = running_max_last_n(data8[0])
max1 = running_max_last_n(data8[1])
plt.plot(max0, c=(0.5, 0, 0))
plt.plot(max1, c=(0, 0.5, 0))
plt.plot(convolve(max0, n=200), c=(1, 0, 0))
plt.plot(convolve(max1, n=200), c=(0, 1, 0))
plt.savefig("max8.svg")
plt.close()