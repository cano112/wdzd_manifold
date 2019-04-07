# Author: Jaques Grobler <jaques.grobler@inria.fr>
# License: BSD 3 clause
import _thread
import threading
import time
from contextlib import contextmanager


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()
print(__doc__)

from time import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import random

from sklearn import manifold

# Next line to silence pyflakes.
Axes3D

# Variables for manifold learning.
n_neighbors = 10
n_samples = 1000

x = []
y = []
z = []
colors = []

r = 10
R = 100
points_no = 100
layers_no = 60
random_coeff = 0.001

px = 0
py = 0
pz = 0
for i in range(layers_no):
    theta =  np.pi * i / layers_no
    for j in range(points_no):
        phi = 4 * np.pi * j / points_no
        px = (R + r * j / points_no * np.cos(phi)) * np.cos(theta)
        py = (R + r * j / points_no * np.cos(phi)) * np.sin(theta)
        pz = r * j / points_no * np.sin(phi)
        colors.append(r * j / points_no)
        x.append(px + random.uniform(-random_coeff*px, random_coeff*px))
        y.append(py + random.uniform(-random_coeff*py, random_coeff*py))
        z.append(pz + random.uniform(-random_coeff*pz, random_coeff*pz))

# Plot our dataset.
# fig = plt.figure(figsize=(15, 8))
# plt.title("Manifold Learning with %i points, %i neighbors"
#              % (1000, n_neighbors), fontsize=14)
#
# ax = fig.add_subplot(1,1,1, projection='3d')
# ax.scatter(x, y, z, c=colors)
# ax.view_init(40, -10)
#

#
# Perform t-distributed stochastic neighbor embedding.
sphere_data = np.array([x, y, z]).T
perplexities = [1.0,2.0,3.0,5.0,8.0,13.0,21.0,34.0,55.0,89.0,144.0,233.0]
learning_rates = [1000.0, 400.0, 280.0, 230.0, 200.0, 170.0, 100.0, 50.0, 10.0]

print("started drawing")
for learning_rate in learning_rates:
    for perplexity in perplexities:
        with time_limit(300, 'tsne_too_long'):
            fig = plt.figure(figsize=(15, 8))
            t0 = time()

            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity,
                                 learning_rate=learning_rate)
            trans_data = tsne.fit_transform(sphere_data).T
            t1 = time()
            print("t-SNE: %.2g sec" % (t1 - t0))

            ax = fig.add_subplot(111)
            plt.scatter(trans_data[0], trans_data[1], c=colors)
            plt.title("t-SNE ({:2.3g} sec) perplexity {:2.3g} learning rate {:2.3g}".format(t1 - t0, perplexity,
                                                                                            learning_rate))
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
            plt.savefig('tsne_rogal_perp={}_learnrate={}.png'.format(perplexity, learning_rate))
            print("Saved figure for perplexity {} and learning rate {}.".format(perplexity, learning_rate))
            plt.clf()

# plt.show()