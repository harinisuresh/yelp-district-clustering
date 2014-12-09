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
    for i in range(num_restaurants):
        topic_weights = all_topic_weights[i]
        pos = restaurant_positions[i]
        d = [pos.x, pos.y]
        d.extend(topic_weights)
        data_array.append(d)

    data = np.array(data_array)
    X = data

    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

    ALPHA = 100.
    clf = mixture.DPGMM(n_components=10, covariance_type='diag', alpha=ALPHA,
                       n_iter=100)
    clf.fit(X)
    classifications = clf.predict(X)
    
    im = plt.imread(my_map.image_path)
    implot = plt.imshow(im)

    clusters = np.zeros((clf.n_components,1))
    centers = np.zeros((clf.n_components,1))
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        cluster = data[classifications==i]
        centers[i] = mean
        #plt.scatter(cluster[:, 0], cluster[:, 1], .8, color=color)
        clusters[i] = cluster
   
    N_CLUSTERS = len(clusters)
    print "Infinite mixture model clustering on :", num_restaurants, "restaurants with", N_CLUSTERS, "clusters", "alpha=", ALPHA
    clusters_of_restaurants = [restaurants[classifications==i] for i in range(N_CLUSTERS)]

    colors = create_n_unique_colors(N_CLUSTERS)

    centers_x = [p[0] for p in centers]
    centers_y = [p[1] for p in centers]
    clusters_x = [[p[0] for p in clusters[i]] for i in range(N_CLUSTERS)]
    clusters_y = [[p[1] for p in clusters[i]] for i in range(N_CLUSTERS)]

    # Plot clusters of restaurants with different colors
    for i in range(N_CLUSTERS):
        cluster_x = clusters_x[i]
        cluster_y = clusters_y[i]
        plt.scatter(cluster_x, cluster_y, marker='o', color=colors[i], alpha=0.5)

    if plotCenters:
        plt.scatter(centers_x, centers_y, marker='x', color=[.1,.1,.1], s=60, edgecolor='black',
               alpha=0.9)
        plt.scatter(centers_x, centers_y, marker='o', color=[.1,.1,.1], s=60, facecolors='none',
               alpha=0.9)

    else:

    # Plot labels over map
        for i in range(N_CLUSTERS):
            center_position = Position(centers_x[i], centers_y[i])
            restaurants = clusters_of_restaurants[i]
            label_text, label_weight = make_label_text_for_cluster(center_position, restaurants, restaurant_ids_to_topics, lda)
            #restaurant = restaurants[0]
            #font_size_1 = 7*(1+sqrt(label_weight[0]))**2;
            #font_size_2 = 7*(1+sqrt(label_weight[1]))**2;
            font_size_1 = 10;
            font_size_2 = 10;
            plt.annotate(label_text[0], xy = (centers_x[i], centers_y[i]), xytext = (centers_x[i]-(len(label_text[0])/2.0)*font_size_1, centers_y[i]+font_size_1), fontsize=font_size_1)
            plt.annotate(label_text[1], xy = (centers_x[i], centers_y[i]), xytext = (centers_x[i]-(len(label_text[1])/2.0)*font_size_2, centers_y[i]-font_size_2), fontsize=font_size_2)
            #my_map.add_label_to_image(label_text[0], center_position-8, None, False, 1.0)
            #my_map.add_label_to_image(label_text[1], center_position+8, None, False, 1.0)

    #my_map.image.show()
    plt.show()

    #for i in range(N_CLUSTERS):
     #   plt.annotate(label, xy = (x, y), xytext = (0, 0), textcoords = 'offset points')

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