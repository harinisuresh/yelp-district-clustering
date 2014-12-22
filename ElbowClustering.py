""" Methods to perform the Elbow Method and calculate the gap statistic for data """

import numpy
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import random
import Map 
import DataImporter

def cluster_points(X, mu):
    """
    Clusters the items in a dataset given each item's dimensional array and a 
    set of centeroids on which to cluster.

    Parameters: 
        X - a data matrix where each row is a dimensional array of an item to be clustered
        mu - an array of centroids on which to cluster
    Returns:
        A dictionary of centroid to cluster
    """
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    """
    Given a set of clusters, recalculates the centroids as the means of all points belonging to a cluster.

    Parameters: 
        mu - an array of centroids
        clusters - a dictionary of centroid to cluster
    Returns:
        An array recalculated centroids where each one is the mean of all points belonging to a cluster
    """
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    """
    A boolean indicating whether or not a set of centroids has converged

    Parameters: 
        mu - the latest array of centroids
        oldmu - the array of centroids from the previous iteration
    Returns:
        A boolean indicating whether or not the old and new centroids are the same, 
        representing whether or not the clustering has converged
    """
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
    """
    Implements an iterative process (Lloyd's algorithm) for k-means clustering

    Parameters: 
        X - a data matrix where each row is a dimensional array of an item to be clustered
        K - the number of clusters to use
    Returns:
        A tuple containing (1) an array of centroids and 
        (2) a dictionary of centroid to points beloning to that centroid's cluster
    """
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

def Wk(mu, clusters):
    """
    Given a clustering, calculates Wk, the normalized sum of intra-cluster distances 
    between the points in each cluster

    Parameters: 
        mu - an array of centroids
        clusters - a dictionary of centroid to cluster

    Returns:
        An array containing the Wk for each cluster
    """
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])

def bounding_box(X):
    """
    Calculates the boundaries of a given dataset on all sides

    Parameters: 
        X - a data matrix where each row is a dimensional array of an item to be clustered

    Returns:
        Two tuples, where the first contains the minimum and maximum x values, and the 
        second contains the minimum and maximum y values
    """
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)
 
def gap_statistic(X):
    """
    Calculates values for use in determining the gap statistic 
    for a given number of clusters K

    Parameters: 
        X - a data matrix where each row is a dimensional array of an item to be clustered

    Returns:
        Four values:
        (1) an array of K values that were tested
        (2) an array where each index contains an array of Wk values for the clusters
        formed with that K value (where a Wk value is the normalized sum of intra-cluster 
        distances between the points in each cluster)
        (3) an array of the gap statistic calculated for each K value 
        (4) an array of standard deviations for each K value
    """
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    # Dispersion for real distribution
    print "new Gap"
    ks = np.arange(24,38,2)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    for indk, k in enumerate(ks):
        print k
        mu, clusters = find_centers(X,k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin,xmax),
                          random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk)

def plot_elbow_and_gap(X):
    """
    Displays two plots: (1) The gap statistic across a number of clusters and 
    (2) the elbow graph 

    Parameters: 
        X - a data matrix where each row is a dimensional array of an item to be clustered
    """
    plt.figure(1)
    plt.scatter(X[:,0], X[:,1])

    ks, logWks, logWkbs, sk = gap_statistic(X)

    plt.figure(2)
    plt.xlabel("N_Clusters")
    plt.ylabel("Gap Statistic")
    plt.title("Gap Statistic vs. N_Clusters")
    gaps = [(logWkbs[i] - logWks[i]) for i in range(len(logWks))]
    plt.plot(ks,gaps, marker='o', color='g')
    plt.show()

    plt.figure(3)
    plt.xlabel("N_Clusters")
    plt.ylabel("Log(Wk)")
    plt.title("Log(Wk) vs N_Clusters (Elbow Graph)")
    plt.plot(ks, logWks, marker='o', color='b')
    plt.show()
