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


print("T-SNE started")
perplexity = 35.0
learning_rate = 200.0
t0 = time()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity,
                     learning_rate=learning_rate, metric=distance, verbose=2)
num = 9000
trans_data = tsne.fit_transform(data[:num]).T
t1 = time()
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
plt.scatter(trans_data[0], trans_data[1], c=np.random.rand(num))
plt.title("t-SNE ({:2.3g} sec) perplexity {:2.3g} learning rate {:2.3g}".format(t1 - t0, perplexity,
                                                                                learning_rate))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.savefig('tsne_rogal_perp={}_learnrate={}.png'.format(perplexity, learning_rate))
print("Saved figure for perplexity {} and learning rate {}.".format(perplexity, learning_rate))
plt.show()
plt.clf()

# codes = pd.read_csv('isocountrycodes.csv')[['name', 'alpha-3']].values
# map_codes = dict(codes)

# print(np.unique(data[:,0]))
# err = []
# for piece in data:
#     try:
#         piece[0] = map_codes[piece[0]]
#     except KeyError:
#         err.append(piece[0])
# print(np.unique(np.array(err)))
# for c in np.unique(data[:,0]):
#     try:
