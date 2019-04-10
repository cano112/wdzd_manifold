from time import time

import pandas as pd
import numpy as np
from matplotlib.ticker import NullFormatter
from scipy import spatial
# country, sex - metric: same, different
# country (0,1,2...,100), year (range), sex (0,1), age (0-5), suicides/100k pop,
#   gdp_for_year ($) , gdp_per_capita ($), generation
from sklearn import manifold
import matplotlib.pyplot as plt


cols = ['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides/100k pop', 'gdp_for_year ($)' ,
        'gdp_per_capita($)', 'generation']
data = pd.read_csv('../master.csv', usecols=cols).values
countries = np.unique(data[:,0])
age_groups = np.array(['5-14 years', '15-24 years',  '25-34 years',  '35-54 years',  '55-74 years', '75+ years'])
generations = np.unique(data[:, -1])

for piece in data:
    piece[0] = np.where(countries == piece[0])[0][0]
    piece[1] = (piece[1] - 1985)/31.0
    piece[2] = 0 if piece[2] == 'male' else 1
    piece[3] = np.where(age_groups == piece[3])[0][0]/5.0
    piece[4] = piece[4]/22338.0
    piece[5] = piece[5]/43805214.0
    piece[6] = piece[6]/224.97
    piece[7] = int(piece[7].replace(",", ""))/18120714000000.0
    piece[8] = piece[8]/126352.0
    piece[9] = np.where(generations == piece[9])[0][0]/5.0

def distance(first_piece, second_piece):
    second_piece[0] = 0 if first_piece[0] == second_piece[0] else 1
    first_piece[0] = 0

    return 1 - spatial.distance.cosine(first_piece, second_piece)
methods = ['standard', 'ltsa', 'hessian', 'modified']
# methods = ['']
n_neighbors = 10
print("LLE started")
for i, method in enumerate(methods):
    t0 = time()
    num = 1000
    lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=n_neighbors, method=method)
    # lle = manifold.Isomap(n_neighbors, n_components=2)
    trans_data = lle.fit_transform(data[:num]).T
    t1 = time()
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    plt.scatter(trans_data[0], trans_data[1], c=np.random.rand(num))
    plt.title("Isomap ({:2.3g} sec) method {}".format(t1 - t0, method))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.savefig('isomap_defaults_method={}.png'.format(method))
    print("Saved figure for method {} in time {:2.3g}.".format(method, t1-t0))
    # plt.show()
    plt.clf()
