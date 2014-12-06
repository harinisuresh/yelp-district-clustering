"""Cluster restaurants on map"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from MapUtils import Coordinate, Position, create_n_unique_colors
from Map import Map
from DataImporter import get_pheonix_restaurants, get_vegas_restaurants, get_reviews_from_restuaraunts
from LDAPredictor import LDAPredictor
import math
import random
import operator

def create_topic_cluster_and_map(restaurants, restaurant_ids_to_topics, my_map, lda):
    restaurant_coordinates = []
    restaurant_positions = []
    all_topic_weights = []
    num_restaurants = restaurants.size
    # N_CLUSTERS = int(max(2,math.sqrt(num_restaurants/2.0)))
    N_CLUSTERS = 60
    LDA_ClUSTER_SCALE_FACTOR =  my_map.image_width() / 2.0
    LDA_ClUSTER_SCALE_FACTOR = 0.0

    num_topics = 50
    print "K-mean clustering on :", num_restaurants, "restaurants with", N_CLUSTERS, "clusters"

    for restaurant in restaurants:
        business_id = restaurant["business_id"]
        coord = Coordinate(restaurant["latitude"],restaurant["longitude"])
        position = my_map.world_coordinate_to_image_position(coord)
        restaurant_coordinates.append(coord)
        restaurant_positions.append(position)
        all_topic_weights_for_restaurant = restaurant_ids_to_topics[business_id]
        all_topic_weights_array_for_restaurant = make_topic_array_from_tuple_list(all_topic_weights_for_restaurant, num_topics, LDA_ClUSTER_SCALE_FACTOR)
        all_topic_weights.append(all_topic_weights_array_for_restaurant)

    data_array = []
    for i in range(num_restaurants):
        topic_weights = all_topic_weights[i]
        pos = restaurant_positions[i]
        d = [pos.x, pos.y]
        d.extend(topic_weights)
        data_array.append(d)

    data = np.array(data_array)
    centers, center_dist = kmeans(data, N_CLUSTERS, iter=200)
    classifications, classification_dist = vq(data, centers)

    im = plt.imread(my_map.image_path)
    implot = plt.imshow(im)

    clusters = [data[classifications==i] for i in range(N_CLUSTERS)]
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
        plt.scatter(cluster_y, cluster_x, marker='o', color=colors[i], alpha=0.8)

    # Plot centers
    plt.scatter(centers_y, centers_x, marker='x', color=[.1,.1,.1], s=60, edgecolor='black',
            alpha=0.9)
    plt.scatter(centers_y, centers_x, marker='o', color=[.1,.1,.1], s=60, facecolors='none',
            alpha=0.9)
    plt.show()

    # Plot labels over map
    for i in range(N_CLUSTERS):
        center_position = Position(centers_x[i], centers_y[i])
        restaurants = clusters_of_restaurants[i]
        label_text = make_label_text_for_cluster(center_position, restaurants, restaurant_ids_to_topics, lda)
        restaurant = restaurants[0]
        my_map.add_label_to_image(label_text, center_position, None, False, 1.0)
    my_map.image.show()

def make_topic_array_from_tuple_list(weight_tuples, num_topics, scale_factor):
    topic_array = [0 for i in range(num_topics)]
    total = float(sum([weight_tuple[1] for weight_tuple in weight_tuples])) # For normalization
    for weight_tuple in weight_tuples:
        topic_index = weight_tuple[0]
        topic_weight = weight_tuple[1]
        topic_array[topic_index] = (topic_weight/total) * scale_factor  # For normalization + scaling
    return topic_array

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
    best_topic_ids = [best_topic_id_pair[1] for best_topic_id_pair in best_topic_id_pairs]
    best_topics_words = [lda.show_topic(best_topic_id) for best_topic_id in best_topic_ids] #list of words for each best topic


    #best_topic_id =  max(topic_ids, key=lambda t_id: topic_total_weights.get(t_id, 0.0)) # get argmax of topic
    #best_topic = lda.show_topic(best_topic_id)

    #list of (weight, word) tuples of the top words in each of the best topics
    best_weights_and_words = [(best_topic_words[0], best_topic_words[1]) for best_topic_words in best_topics_words]


    #best_weight, best_word = best_topic[0]
    #best_weight_2, best_word_2 = best_topic[1]

    print best_weights_and_words

    best_words = [a[0] for a in best_weights_and_words].join(" ")
    return best_words

def run(my_map, reviews, restaurants):
    predictor = LDAPredictor()
    lda = predictor.lda
    restaurant_ids_to_topics = {}
    for restaurant in restaurants:
        business_id  = restaurant["business_id"]
        review = reviews[business_id]
        prediction = predictor.predict_topics(review)
        restaurant_ids_to_topics[business_id] = prediction
    create_topic_cluster_and_map(restaurants, restaurant_ids_to_topics, my_map, lda)   

def main():
    my_map = Map.vegas()
    reviews = get_reviews_from_restuaraunts("Las Vegas")
    restaurants = get_vegas_restaurants()
    run(my_map, reviews, restaurants)

if __name__ == '__main__':
    main()
