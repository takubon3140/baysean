# for testing on jupyter notebook
import os
from baysean import BayesianOptimizer

PRJ_ROOT = '/Users/takubon/Desktop/baysean_process'
#dirの移動
os.chdir("%s/" % PRJ_ROOT)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline

sns.set_style('whitegrid')
sns.set(font=['IPAPGothic'])

#%reload_ext autoreload
#%autoreload 2


space_size = value_space.shape
sns.heatmap(value_space, cmap='cubehelix', vmin=np.min(value_space)*1.5, vmax=np.max(value_space)*1.5)
max_i, max_v = np.argmax(value_space), np.max(value_space)
col, row = max_i % space_size[0], max_i // space_size[0]
sns.plt.plot(col, space_size[0] - row-1, 'ro')
print("f(row=%d, col=%d) = %.3f" % (row, col, max_v))


def plot_heatmap(value_space, checked_points):
    sns.heatmap(value_space, cmap='cubehelix', vmin=np.min(value_space)*1.5, vmax=np.max(value_space)*1.5)

    cp = np.array(checked_points)
    rows = value_space.shape[0] - cp[:, 0] - 1
    cols = cp[:, 1]
    plt.plot(cols, rows, 'ro', markersize=5)
    plt.plot(cols[-1], rows[-1], 'o', markersize=10)


params = {
    "col": list(range(50)),
    "row": list(range(50)),
}

bo = BayesianOptimizer(params)
checked_points = []

for i, param in enumerate(bo.supply_next_param()):  # param is dict
    x = [param['row'], param['col']]
    y = unknown_function(x)
    bo.report(y)

    checked_points.append(x)

    if i % 20 == 0:
        plot_heatmap(value_space, checked_points)
        sns.plt.show()
