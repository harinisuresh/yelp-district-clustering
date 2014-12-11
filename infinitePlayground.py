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



X = np.zeros((500,2))
# X[0, 0] = 10
# X[0, 1] = 10
# X[1, 0] = 10 
# X[1, 1] = 10
# X[2, 0] = 1
# X[2, 1] = 10
# X[3, 0] = 15
# X[3, 1] = 5
# X[4, 0] = 2
# X[4, 1] = 25
# X[5, 0] = 14
# X[5, 1] = 3


meanX = 20.0
meanY = 50.0
var1= 1
for i in range(250):
    X[i,0] = var1 * np.random.rand() + meanX
    X[i,1] = var1 * np.random.rand() + meanY
	#X[i,1] = random.randint(1,10)

var2 = .7
for i in range(250,500):
    X[i,0] = var2 * np.random.rand() + 50
    X[i,1] = var2 * np.random.rand() + 50


print "X1 izz", X
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
clf = mixture.DPGMM(alpha=100.11)


clf.fit(X)

classifications = clf.predict(X)
print "classifications", classifications 
# print clf.means_
# print clf.n_components
# print clf.weights_

print "other"
pprint(clf, X)

print "means"
print clf.means_





# Number of samples per component
n_samples = 250

#np.random.seed(0)
#C = np.array([[0., -0.1], [1.7, .4]])
#X2 = np.r_[np.random.randn(n_samples, 2) + np.array([20, 50]),
          #.7 * np.random.randn(n_samples, 2) + np.array([50, 50])]

# Generate random sample, two components
# np.random.seed(0)
# C = np.array([[0., -0.1], [1.7, .4]])
# X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
#           .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
print "X isss:", X2

# Fit a mixture of Gaussians with EM using five components
#gmm = mixture.GMM(n_components=5, covariance_type='full')
#gmm.fit(X)

# Fit a Dirichlet process mixture of Gaussians using five components
dpgmm = mixture.DPGMM(n_components=5, covariance_type='full')

dpgmm.fit(X2)

print "other"
pprint(dpgmm, X2)

print "means"
print dpgmm.means_

color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

#clf = dpgmm
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