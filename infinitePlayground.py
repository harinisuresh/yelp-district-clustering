import itertools

from scipy import linalg

from sklearn import mixture
from sklearn.externals.six.moves import xrange
"""Cluster restaurants on map"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

#http://stackoverflow.com/questions/12960516/sklearn-mixture-dpgmm-unexpected-results
#Maybe take a look at this link.

#prints means, weights, covariances.
def pprint(model, data):
    idx = np.unique(model.predict(data))
    m_w_cov = [model.means_, model.weights_, model._get_covars()]
    flattened  = map(lambda x: np.array(x).flatten(), m_w_cov)
    filtered = map(lambda x: x[idx], flattened)
    print np.array(filtered)

def means_filtered(means, idx):
    return [means[i] for i in range(len(means)) if i in idx]

def filter_means(means, thres=1.0):
    new_means = [mean for mean in means if np.linalg.norm(mean) > thres]
    return new_means
    # idx = np.unique(model.predict(data))
    # m_w_cov = [model.means_, model.weights_, model._get_covars()]
    # flattened  = map(lambda x: np.array(x).flatten(), m_w_cov)
    # filtered = map(lambda x: x[idx], flattened)
    # return np.array(filtered)

# def filter_means(means):
#     flattened  = map(lambda x: np.array(x).flatten(), m_w_cov)
#     filtered = map(lambda x: x[idx], flattened)
#     return np.array(filtered)

X = np.zeros((500,2))


meanX = 50.0
meanY = 50.0
var1 = 100.0

for i in range(200):
    X[i] = 15.0 * np.random.rand(1, 2) + np.array([10, 50])

var2 = 150.0
for i in range(200,500):
    X[i] = 12.0 * np.random.rand(1, 2) + np.array([50, 70])


# # Number of samples per component
# n_samples = 100

# # Generate random sample following a sine curve
# np.random.seed(0)
# X = np.zeros((n_samples, 2))
# step = 4 * np.pi / n_samples

# for i in xrange(X.shape[0]):
#     x = i * step - 6
#     X[i, 0] = x + np.random.normal(0, 0.1)
#     X[i, 1] = 3 * (np.sin(x) + np.random.normal(0, .2))
 
#ALPHA = 100.
clf = mixture.DPGMM(n_components=500, covariance_type='full')


clf.fit(X)

classifications = clf.predict(X)
print "classifications", classifications 

print "means"
idx = np.unique(classifications)
print "idx =", idx
new_means = means_filtered(clf.means_, idx)
print "new means"
print new_means
# Number of samples per component
n_samples = 500

np.random.seed(0)
#C = np.array([[0., -0.1], [1.7, .4]])

color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

title = 'Dirichlet Process GMM'
splot = plt.subplot(2, 1, 1)
Y_ = clf.predict(X)
for i, (mean, covar, color) in enumerate(zip(
        clf.means_, clf._get_covars(), color_iter)):
    v, w = linalg.eigh(covar)
    u = w[0] / linalg.norm(w[0])
    # as the DP will not use every component it has access to
    # unless it needs it, we shouldn't plot the redundant
    # components.
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
        
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

plt.show()