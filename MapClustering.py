"""Cluster restaurants on map"""
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from MapUtils import Coordinate, Position, create_n_unique_colors
from Map import Map
from DataImporter import get_pheonix_restaurants, get_vegas_restaurants
import math
import random

def create_cluster_and_map():
    my_map = Map.vegas()
    restaurants = get_vegas_restaurants()
    restaurant_coordinates = []
    restaurant_positions = []
    num_restaurants = restaurants.size
    N_CLUSTERS = int(max(2,math.sqrt(num_restaurants/2.0)))
    N_CLUSTERS = 30

    print "K-mean clustering on :", num_restaurants, "restaurants with", N_CLUSTERS, "clusters"

    for restaurant in restaurants:
        coord = Coordinate(restaurant["latitude"],restaurant["longitude"])
        position = my_map.world_coordinate_to_image_position(coord)
        restaurant_coordinates.append(coord)
        restaurant_positions.append(position)

    data = np.array([[pos.y,pos.x] for pos in restaurant_positions])
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
        plt.scatter(cluster_x, cluster_y, marker='o', color=colors[i], alpha=0.8)

    # Plot centers
    plt.scatter(centers_x, centers_y, marker='x', color=[.1,.1,.1], s=60, edgecolor='black',
            alpha=0.9)
    plt.scatter(centers_x, centers_y, marker='o', color=[.1,.1,.1], s=60, facecolors='none',
            alpha=0.9)
    plt.show()

    # Plot labels over map
    for i in range(N_CLUSTERS):
        center_position = Position(centers_x[i], centers_y[i])
        restaurants = clusters_of_restaurants[i]
        restaurant = restaurants[0]
        my_map.add_label_to_image(restaurant["name"], center_position, None, False, 1.0)
    my_map.image.show()

def main():
    create_cluster_and_map()

if __name__ == '__main__':
    main()
