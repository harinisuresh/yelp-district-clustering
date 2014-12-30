import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from scipy import linalg
from sklearn import mixture
from MapUtils import Coordinate, Position, create_n_unique_colors
from Map import Map
from DataImporter import get_pheonix_restaurants, get_vegas_restaurants, get_vegas_reviews, get_topic_labels, get_words_from_text
from LDAPredictor import LDAPredictor
import math
import random
import operator
from Utils import make_topic_array_from_tuple_list
from Utils import make_tuple_list_from_topic_array, print_median_std_from_clusters
from math import sqrt
from vonoroi import plot_vonoroi_from_points

#constants
NUM_TOPICS = 50
N_CLUSTERS = 30
K = np.sqrt(5.0/6.0)
PIXELS_PER_MILE = 48.684
LDA_ClUSTER_SCALE_FACTOR =  K*PIXELS_PER_MILE
ALPHA = 0.001

def filter_restaurants(restaurants, reviews):
    filtered_restaurants = [restaurant for restaurant in restaurants if restaurant["business_id"] in reviews]
    return filtered_restaurants

def create_data_array(restaurants, restaurant_ids_to_topics, my_map):
    restaurant_coordinates = []
    restaurant_positions = []
    all_topic_weights = []
    num_restaurants = len(restaurants)

    print "Clustering on :", num_restaurants, "restaurants with", N_CLUSTERS, "clusters"

    #create an position and topic arrays for each restaurant
    for restaurant in restaurants:
        business_id = restaurant["business_id"]
        coord = Coordinate(restaurant["latitude"],restaurant["longitude"])
        position = my_map.world_coordinate_to_image_position(coord)
        restaurant_coordinates.append(coord)
        restaurant_positions.append(position)
        all_topic_weights_for_restaurant = restaurant_ids_to_topics[business_id]
        all_topic_weights_array_for_restaurant = make_topic_array_from_tuple_list(all_topic_weights_for_restaurant, NUM_TOPICS, LDA_ClUSTER_SCALE_FACTOR)
        all_topic_weights.append(all_topic_weights_array_for_restaurant)

    #create array containing both position and topics for each restaurant
    data_array = []
    for i in range(num_restaurants):
        topic_weights = all_topic_weights[i]
        pos = restaurant_positions[i]
        d = [pos.x, pos.y]
        d.extend(topic_weights)
        data_array.append(d)
    data = np.array(data_array)
    return data


def k_means_clustering(data):

    #Perform clustering
    centers, center_dist = kmeans(data, N_CLUSTERS, iter=200)
    classifications, classification_dist = vq(data, centers)

    return (centers, center_dist, classifications, classification_dist)


def plot_clusters(my_map, restaurants, restaurant_ids_to_topics, data, lda):
    centers, center_dist, classifications, classification_dist = k_means_clustering(data)

    #Figure 1: Plot clusters of restaurants with different colors, no labels
    plt.figure(1)

    im = plt.imread(my_map.image_path)
    implot = plt.imshow(im)

    clusters = [data[classifications==i] for i in range(N_CLUSTERS)]
    clusters_of_restaurants = [restaurants[classifications==i] for i in range(N_CLUSTERS)]

    colors = create_n_unique_colors(N_CLUSTERS)

    centers_x = [p[0] for p in centers]
    centers_y = [p[1] for p in centers]
    clusters_x = [[p[0] for p in clusters[i]] for i in range(N_CLUSTERS)]
    clusters_y = [[p[1] for p in clusters[i]] for i in range(N_CLUSTERS)]

    for i in range(N_CLUSTERS):
        cluster_x = clusters_x[i]
        cluster_y = clusters_y[i]
        plt.scatter(cluster_x, cluster_y, marker='o', color=colors[i], alpha=0.2)
    
    plt.title("Las Vegas K-Means Clustering")
    plt.show()


    #Figure 2: Plot clusters of restaurants with different colors, with labels
    plt.figure(2)

    im = plt.imread(my_map.image_path)
    implot = plt.imshow(im)

    for i in range(N_CLUSTERS):
        cluster_x = clusters_x[i]
        cluster_y = clusters_y[i]
        plt.scatter(cluster_x, cluster_y, marker='o', color=colors[i], alpha=0.2)

    for i in range(len(centers_x)):
        center_position = Position(centers_x[i], centers_y[i])
        restaurants = clusters_of_restaurants[i]
        label_text, label_weight = make_label_text_for_cluster(center_position, restaurants, restaurant_ids_to_topics, lda)
        font_size_1 = 10;
        font_size_2 = 10;
        plt.annotate(label_text[0], xy = (centers_x[i], centers_y[i]), xytext = (centers_x[i]-(len(label_text[0])/2.0)*font_size_1, centers_y[i]+font_size_1), fontsize=font_size_1)
        plt.annotate(label_text[1], xy = (centers_x[i], centers_y[i]), xytext = (centers_x[i]-(len(label_text[1])/2.0)*font_size_2, centers_y[i]-font_size_2), fontsize=font_size_2)

    plt.title("Las Vegas K-Means Clustering With Labels")
    plt.show()

def make_label_text_for_cluster(cluster_center, cluster_restaurants, restaurant_ids_to_topics, lda, use_human_labels=True):
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

    if use_human_labels:
        topic_labels = get_topic_labels()
        best_words = [topic_labels[pair[0]] for pair in best_topic_id_pairs]

    return best_words, best_topic_weights


def get_predictions(my_map, reviews, restaurants):
    predictor = LDAPredictor()
    lda = predictor.lda
    restaurant_ids_to_topics = {}
    for restaurant in restaurants:
        business_id  = restaurant["business_id"]
        review = reviews[business_id]
        prediction = predictor.predict_topics(review)
        restaurant_ids_to_topics[business_id] = make_topic_array_from_tuple_list(prediction, NUM_TOPICS) #topic array of weights for each topic index
    normalized_restaurant_ids_to_topics = normalize_predictions(restaurant_ids_to_topics, restaurants)
    return normalized_restaurant_ids_to_topics, lda


def normalize_predictions(predictions, restaurants): 
    all_weights = predictions.values()
    all_weights_sum = np.sum(all_weights, axis=0)

    normalized_weights_total = 0

    #vertical normalization
    for restaurant in restaurants:
        business_id  = restaurant["business_id"]
        weights = predictions[business_id]
        normalized_weights = np.divide(np.array(weights,dtype=float), np.array(all_weights_sum, dtype=float))
        normalized_weights_total += sum(normalized_weights)
        predictions[business_id] = normalized_weights
    
    #horizontal normalization
    mean_weights_sum = normalized_weights_total/len(restaurants)
    for restaurant in restaurants:
        business_id = restaurant["business_id"]
        weights = predictions[business_id] 
        weights /= mean_weights_sum
        predictions[business_id] = make_tuple_list_from_topic_array(weights)

    return predictions


def gaussian_clustering(data):
    # Fit a Dirichlet process mixture of Gaussians using five components
    clf = mixture.GMM(n_components=N_CLUSTERS, covariance_type='full')
    clf.fit(data)
    
    # Print means
    print "means"
    print clf.means_
    
    # Classify data
    classifications = clf.predict(data)

    # Print means above a threshold
    idx = np.unique(classifications)
    filtered_means = means_filtered(clf.means_, idx)
    print "new means"
    print filtered_means

    return clf, classifications


def plot_gaussian_clusters(my_map, restaurants, restaurant_ids_to_topics, data, lda):
    # Get GMM
    clf, classifications = gaussian_clustering(data)

    # Get unique colors
    color_iter = itertools.cycle(create_n_unique_colors(N_CLUSTERS))

    # Get clusters
    clusters = [data[classifications==i] for i in range(N_CLUSTERS)]
    clusters_of_restaurants = [restaurants[classifications==i] for i in range(N_CLUSTERS)]
    

    # Figure 1: Gaussian clusters with covariance ellipses, no labels
    plt.figure(1)

    im = plt.imread(my_map.image_path)
    implot = plt.imshow(im)

    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(classifications == i):
            continue
        plt.scatter(data[classifications == i, 0], data[classifications == i, 1], color=color, alpha=0.5)


        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_alpha(0.5)            
        plt.xticks(())
        plt.yticks(())
    
    plt.title('Dirichlet Process GMM')
    plt.show()

    # Figure 2: Gaussian clusters with labels
    plt.figure(2)

    centers_x = [p[0] for p in clf.means_]
    centers_y = [p[1] for p in clf.means_]

    im = plt.imread(my_map.image_path)
    implot = plt.imshow(im)
    angles = np.zeros(N_CLUSTERS)

    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        l_u = len(u)
        angle = np.arctan(u[l_u-2] / u[l_u-1])
        print "ANGLE1", angle
        angle = 180 * angle / np.pi  # convert to degrees
        print "ANGLE2", angle
        print "vars", angle
        angles[i] = angle
        if not np.any(classifications == i):
            continue
        plt.scatter(data[classifications == i, 0], data[classifications == i, 1], color=color, alpha=0.5)

    # Plot labels over map
    for i in range(N_CLUSTERS):
        center_position = Position(centers_x[i], centers_y[i])
        cluster_restaurants = clusters_of_restaurants[i]
        label_text, label_weight = make_label_text_for_cluster(center_position, cluster_restaurants, restaurant_ids_to_topics, lda, use_human_labels)
        print label_text
        text = ""
        if len(label_text) > 1:
            text = label_text[0] + '\n' + label_text[1]
        angle = angles[i]
        if np.isnan(angle):
            angle = 0.0
        print "print", angle
        plt.text(centers_x[i], centers_y[i], text,
        horizontalalignment='center',
        verticalalignment='center',
        rotation=angle, fontsize=9)

    plt.title("Las Vegas Gaussian Clustering With Labels")
    plt.show()
    print_median_std_from_clusters(clusters_of_restaurants)


def means_filtered(means, idx):
    return [means[i] for i in range(len(means)) if i in idx]