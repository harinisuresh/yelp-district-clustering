"""Cluster restaurants on map"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from MapUtils import Coordinate, Position
from Map import Map
from DataImporter import get_pheonix_restaurants, get_vegas_restaurants
import math

def createTopicClusterAndMap(restaurants, restaurants_to_topics_array, my_map):
    restaurant_coordinates = []
    restaurant_positions = []
    all_topic_weights = []
    num_restaurants = restaurants.size
    N_CLUSTERS = int(max(2,math.sqrt(num_restaurants/2.0)))
    print "K-mean clustering on :", num_restaurants, "restaurants with", N_CLUSTERS, "clusters"

    i = 0
    for restaurant in restaurants:
        coord = Coordinate(restaurant["latitude"],restaurant["longitude"])
        position = my_map.world_coordinate_to_image_position(coord)
        restaurant_coordinates.append(coord)
        restaurant_positions.append(position)
        all_topics_for_restaurant = restaurants_to_topics_array[i,:]
        all_topic_weights.append(all_topics_for_restaurant)
        i += 1

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

    colors = np.random.rand(N_CLUSTERS)

    centers_x = [p[0] for p in centers]
    centers_y = [p[1] for p in centers]
    clusters_x = [[p[0] for p in clusters[i]] for i in range(N_CLUSTERS)]
    clusters_y = [[p[1] for p in clusters[i]] for i in range(N_CLUSTERS)]

    # Plot centers
    plt.plot(centers_x, centers_y, 'x', [0,0,0])
    # Plot clusters of restaurants with different colors
    for i in range(N_CLUSTERS):
        cluster_x = clusters_x[i]
        cluster_y = clusters_y[i]
        plt.plot(cluster_x, cluster_y, 'o', colors[i])
    plt.show()

    # Plot labels over map
    for i in range(N_CLUSTERS):
        center_position = Position(centers_x[i], centers_y[i])
        restaurants = clusters_of_restaurants[i]
        restaurant = restaurants[0]
        my_map.add_label_to_image(restaurant["name"], center_position, None, False, 1.0)
    my_map.image.show()
