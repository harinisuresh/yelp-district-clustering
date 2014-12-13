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
import ElbowClustering

NUM_TOPICS = 50;

def create_topic_cluster_and_map(restaurants, restaurant_ids_to_topics, my_map, lda):
    restaurant_coordinates = []
    restaurant_positions = []
    all_topic_weights = []
    num_restaurants = restaurants.size
    # N_CLUSTERS = int(max(2,math.sqrt(num_restaurants/2.0)))

    K = np.sqrt(5.0/6.0)
    PIXELS_PER_MILE = 48.684
    LDA_ClUSTER_SCALE_FACTOR =  K*PIXELS_PER_MILE
    #LDA_ClUSTER_SCALE_FACTOR = 0.0

    num_topics = 50
    print "Find K on :", num_restaurants, "restaurants with"

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
    print "starting elbow clustering"
    ElbowClustering.plot_elbow_and_gap(data)


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
    all_weights_sum = np.sum(all_weights, axis=0)
    normalized_weights_total = 0
    for restaurant in restaurants:
        business_id  = restaurant["business_id"]
        weights = predictions[business_id]
        normalized_weights = np.divide(np.array(weights,dtype=float), np.array(all_weights_sum, dtype=float))
        normalized_weights_total += sum(normalized_weights)
        predictions[business_id] = normalized_weights
    mean_weights_sum = normalized_weights_total/len(restaurants)
    for restaurant in restaurants:
        business_id = restaurant["business_id"]
        weights = predictions[business_id] 
        weights /= mean_weights_sum
        predictions[business_id] = make_tuple_list_from_topic_array(weights)

    return predictions


def main():
    my_map = Map.vegas()
    reviews = get_vegas_reviews()
    restaurants = get_vegas_restaurants()
    run(my_map, reviews, restaurants)

if __name__ == '__main__':
    main()
