"""HeatMap of restaurants on map"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial.distance import cdist
from MapUtils import Coordinate, Position, Rectangle, create_n_unique_colors
from Map import Map
from DataImporter import get_pheonix_restaurants, get_vegas_restaurants, get_reviews_from_restuaraunts
from LDAPredictor import LDAPredictor
import math
import random
from Utils import make_topic_array_from_tuple_list


def create_heat_map(restaurants, restaurant_ids_to_topics, novel_restaurant_topics, my_map):
    print "starting heatmap"
    n_topics = 50
    #im = plt.imread(my_map.image_path)
    #implot = plt.imshow(im)

    novel_restaurant_topics_array = make_topic_array_from_tuple_list(novel_restaurant_topics, 50)
    image_width = my_map.image_width()
    image_height = my_map.image_height()
    n_x_bins = 10
    n_y_bins = 10
    distances = np.zeros((10, 10))
    bin_width = image_width/n_x_bins
    bin_height = image_height/n_y_bins

    for xi in range(n_x_bins):
        for yi in range(n_y_bins):
            total_dist = 0.0
            top_left = Position(xi*bin_width, yi*bin_height)
            bottom_right = Position((xi+1)*bin_width, (yi+1)*bin_height)
            print "points:", top_left, bottom_right
            rect = Rectangle(top_left, bottom_right)
            restaurants_in_bucket = [restaurant for restaurant in restaurants if rect.contains(my_map.world_coordinate_to_image_position(Coordinate(restaurant["latitude"], restaurant["longitude"])))]
            print "num rest in bucket", len(restaurants_in_bucket)
            print "num rest in total", len(restaurants)
            for business_id, restaurant_topics in restaurant_ids_to_topics.iteritems():
                restaurant_topics_array = make_topic_array_from_tuple_list(restaurant_topics, 50)
                A = np.array(novel_restaurant_topics_array)
                B = np.array(restaurant_topics_array)
                dist = np.sqrt(np.sum((A - B)**2))
                total_dist += dist
            print total_dist
            distances[xi, yi] = total_dist
    print distances

    plt.pcolor(distances,cmap=plt.cm.Reds,edgecolors='k')
    plt.show()
    print "done"

def run(my_map, reviews, restaurants, novel_review=None, novel_business_id=None):
    if novel_review == None and novel_business_id == None:
        raise Exception("review and business_id can't both be None")
    if novel_business_id != None:
        novel_review = reviews[novel_business_id]
    predictor = LDAPredictor()
    lda = predictor.lda
    novel_topics = predictor.predict_topics(novel_review)
    restaurant_ids_to_topics = {}
    print "starting restaurant id mapping"
    for restaurant in restaurants:
        business_id  = restaurant["business_id"]
        if business_id == novel_business_id:
            continue
        review = reviews[business_id]
        prediction = predictor.predict_topics(review)
        restaurant_ids_to_topics[business_id] = prediction
    create_heat_map(restaurants, restaurant_ids_to_topics, novel_topics, my_map)

def main():
    my_map = Map.vegas()
    reviews = get_reviews_from_restuaraunts("Las Vegas")
    restaurants = get_vegas_restaurants()
    business_id = "l6QcUE8XXLrVH6Ydm4GSNw"
    run(my_map, reviews, restaurants, None, business_id)

if __name__ == '__main__':
    main()
