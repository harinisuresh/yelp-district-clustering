import itertools

from scipy import linalg

from sklearn import mixture
from sklearn.externals.six.moves import xrange
"""Cluster restaurants on map"""
import numpy as np
import random

#http://stackoverflow.com/questions/12960516/sklearn-mixture-dpgmm-unexpected-results
#Maybe take a look at this link.

#prints means, weights, covariances.
def pprint(model, data):
    idx = np.unique(model.predict(data))
    m_w_cov = [model.means_, model.weights_, model._get_covars()]
    flattened  = map(lambda x: np.array(x).flatten(), m_w_cov)
    filtered = map(lambda x: x[idx], flattened)
    print np.array(filtered)



X = np.zeros((200,1))
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



for i in range(100):
	X[i,0] = random.randint(1,10)
	#X[i,1] = random.randint(1,10)



for i in range(100,200):
	X[i,0] = random.randint(20,30)
	#X[i,1] = random.randint(30,40)
print X
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
clf = mixture.DPGMM(n_components=2, covariance_type='diag', alpha=.01)


clf.fit(X)

classifications = clf.predict(X)
print classifications 
# print clf.means_
# print clf.n_components
# print clf.weights_

pprint(clf, X)