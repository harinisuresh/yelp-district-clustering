import itertools

from scipy import linalg
import matplotlib as mpl

from sklearn import mixture
from sklearn.externals.six.moves import xrange
"""Cluster restaurants on map"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from MapUtils import Coordinate, Position, create_n_unique_colors
from Map import Map
from DataImporter import get_pheonix_restaurants, get_vegas_restaurants, get_vegas_reviews
from LDAPredictor import LDAPredictor
import math
import random
import operator
from Utils import make_topic_array_from_tuple_list
from Utils import make_tuple_list_from_topic_array
from math import sqrt

NUM_TOPICS = 50;

def means_filtered(means, idx):
    return [means[i] for i in range(len(means)) if i in idx]

def create_topic_cluster_and_map(restaurants, restaurant_ids_to_topics, my_map, lda, plotCenters=True):
    restaurant_coordinates = []
    restaurant_positions = []
    all_topic_weights = []
    num_restaurants = restaurants.size
    # N_CLUSTERS = int(max(2,math.sqrt(num_restaurants/2.0)))

    LDA_ClUSTER_SCALE_FACTOR =  my_map.image_width()*10
    #LDA_ClUSTER_SCALE_FACTOR = 0.0

    num_topics = 50

    for restaurant in restaurants:
        business_id = restaurant["business_id"]
        coord = Coordinate(restaurant["latitude"],restaurant["longitude"])
        position = my_map.world_coordinate_to_image_position(coord)
        restaurant_coordinates.append(coord)
        restaurant_positions.append(position)
        all_topic_weights_for_restaurant = restaurant_ids_to_topics[business_id]
        all_topic_weights_array_for_restaurant = make_topic_array_from_tuple_list(all_topic_weights_for_restaurant, NUM_TOPICS, LDA_ClUSTER_SCALE_FACTOR)
        all_topic_weights.append(all_topic_weights_array_for_restaurant)

    data_array = []
    pos_array = []
    for i in range(num_restaurants):
        topic_weights = all_topic_weights[i]
        pos = restaurant_positions[i]
        d = [pos.x, pos.y]
        d.extend(topic_weights)
        data_array.append(d)
        pos_array.append([pos.x, pos.y])

    data = np.array(pos_array)
    X = data

    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

    ALPHA = 1000.
    # Fit a Dirichlet process mixture of Gaussians using five components
    dpgmm = mixture.DPGMM(n_components=50, covariance_type='full', alpha=ALPHA)

    dpgmm.fit(X)

    print "other"

    print "means"
    print dpgmm.means_

    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

    clf = dpgmm
    title = 'Dirichlet Process GMM'
    splot = plt.subplot(2, 1, 1)
    Y_ = clf.predict(X)
    classifications = Y_


    print "means"
    print clf.means_
    idx = np.unique(classifications)
    new_means = means_filtered(clf.means_, idx)
    print "new means"
    print new_means


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

def make_label_text_for_cluster(cluster_center, cluster_restaurants, restaurant_ids_to_topics, lda):
    topic_total_weights = {}
    for restaurant in cluster_restaurants:
        business_id = restaurant["business_id"]
        topics = restaurant_ids_to_topics[business_id]
        for topic_id, topic_weight in topics:
            topic_total_weights[topic_id] = topic_total_weights.get(topic_id, 0.0) + topic_weight
    topic_ids = topic_total_weights.keys()
    sorted_topic_total_weights = sorted(topic_total_weights.items(), key=operator.itemgetter(1)) #sort based on values
    print sorted_topic_total_weights
    number_of_best_topics = 2
    
    best_topic_id_pairs = sorted_topic_total_weights[len(sorted_topic_total_weights)-number_of_best_topics:]
    print best_topic_id_pairs

    total_weight = sum([id_weight_pair[1] for id_weight_pair in sorted_topic_total_weights])

    best_topic_ids = [best_topic_id_pair[0] for best_topic_id_pair in best_topic_id_pairs]
    print best_topic_ids

    best_topic_weights = [best_topic_id_pair[1] for best_topic_id_pair in best_topic_id_pairs]

    best_topics_words = [lda.show_topic(best_topic_id) for best_topic_id in best_topic_ids] #list of words for each best topic
    print best_topics_words


    #best_topic_id =  max(topic_ids, key=lambda t_id: topic_total_weights.get(t_id, 0.0)) # get argmax of topic
    #best_topic = lda.show_topic(best_topic_id)

    #list of (weight, word) tuples of the top words in each of the best topics
    best_weights_and_words = [best_topic_words[0] for best_topic_words in best_topics_words]


    #best_weight, best_word = best_topic[0]
    #best_weight_2, best_word_2 = best_topic[1]

    print best_weights_and_words

    best_words = [a[1] for a in best_weights_and_words]
    return best_words, best_topic_weights

def run(my_map, reviews, restaurants):
    predictor = LDAPredictor()
    lda = predictor.lda
    restaurant_ids_to_topics = {}
    for restaurant in restaurants:
        business_id  = restaurant["business_id"]
        review = reviews[business_id]
        prediction = predictor.predict_topics(review)
        #print restaurant["name"], prediction
        restaurant_ids_to_topics[business_id] = make_topic_array_from_tuple_list(prediction, NUM_TOPICS) #topic array of weights for each topic index
    normalized_restaurant_ids_to_topics = normalize_predictions(restaurant_ids_to_topics, restaurants)
    create_topic_cluster_and_map(restaurants, normalized_restaurant_ids_to_topics, my_map, lda)   

def normalize_predictions(predictions, restaurants): 
    all_weights = predictions.values()
    print all_weights[0]
    all_weights_sum = np.sum(all_weights, axis=0)
    print all_weights_sum
    for restaurant in restaurants:
        business_id  = restaurant["business_id"]
        weights = predictions[business_id]
        normalized_weights = np.divide(np.array(weights,dtype=float), np.array(all_weights_sum, dtype=float))
        predictions[business_id] = make_tuple_list_from_topic_array(normalized_weights)
    print predictions.values()[0]
    return predictions


def main():
    my_map = Map.vegas()
    reviews = get_vegas_reviews()
    restaurants = get_vegas_restaurants()
    run(my_map, reviews, restaurants)

if __name__ == '__main__':
    main()