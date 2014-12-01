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


def create_heat_map(restaurants, restaurant_ids_to_topics, novel_restaurant_topics, my_map, pure_buckets=False):
    print "starting heatmap"
    n_topics = 50

    novel_restaurant_topics_array = make_topic_array_from_tuple_list(novel_restaurant_topics, 50)
    image_width = my_map.image_width()
    image_height = my_map.image_height()
    n_x_bins = 10
    n_y_bins = 10
    distances = np.zeros((n_x_bins, n_y_bins))
    bin_width = image_width/n_x_bins
    bin_height = image_height/n_y_bins

    restuarants_indexed_by_id = {restaurant["business_id"] : restaurant for restaurant in restaurants}

    for xi in range(n_x_bins):
        for yi in range(n_y_bins):
            total_dist = 0.0
            top_left = Position(xi*bin_width, yi*bin_height)
            bottom_right = Position((xi+1)*bin_width, (yi+1)*bin_height)
            print "points:", top_left, bottom_right
            rect = Rectangle(top_left, bottom_right)
            restaurant_ids_in_bucket = {restaurant["business_id"] for restaurant in restaurants if rect.contains(my_map.world_coordinate_to_image_position(Coordinate(restaurant["latitude"], restaurant["longitude"])))}
            print "num in bucket", len(restaurant_ids_in_bucket)
            print "num rest in total", len(restaurants)
            for business_id, restaurant_topics in restaurant_ids_to_topics.iteritems():
                if business_id not in restaurant_ids_in_bucket and pure_buckets:
                    continue
                restaurant = restuarants_indexed_by_id[business_id]
                restaurant_topics_array = make_topic_array_from_tuple_list(restaurant_topics, 50)
                A = np.array(novel_restaurant_topics_array)
                B = np.array(restaurant_topics_array)
                diff = np.sqrt(np.sum((A - B)**2))
                dist = diff
                if pure_buckets == False:
                    rest_pos = my_map.world_coordinate_to_image_position(Coordinate(restaurant["latitude"], restaurant["longitude"]))
                    square_center = rect.center()
                    A1 = np.array([rest_pos.x, rest_pos.y])
                    B1 = np.array([square_center.x, square_center.y])
                    physical_dist = np.sqrt(np.sum((A1 - B1)**2))
                    dist = diff*(physical_dist**2)
                total_dist += dist
            print total_dist
            distances[xi, yi] = total_dist
    print distances


    im = plt.imread(my_map.image_path)
    implot = plt.imshow(im, alpha=0.9, extent=[0,10,0,10])
    plt.pcolor(distances,cmap=plt.cm.cool,edgecolors='k')
    plt.show()

    print "done"


def run(my_map, reviews, restaurants, novel_review=None, novel_business_id=None, restaurant_ids_to_topics=None, pure_buckets=False):
    if novel_review == None and novel_business_id == None:
        raise Exception("review and business_id can't both be None")
    if novel_business_id != None:
        novel_review = reviews[novel_business_id]
    predictor = LDAPredictor()
    lda = predictor.lda
    novel_topics = predictor.predict_topics(novel_review)
    if restaurant_ids_to_topics == None:
        restaurant_ids_to_topics = {}
        print "starting restaurant id mapping"
        for restaurant in restaurants:
            business_id  = restaurant["business_id"]
            if business_id == novel_business_id:
                continue
            review = reviews[business_id]
            prediction = predictor.predict_topics(review)
            restaurant_ids_to_topics[business_id] = prediction
        return restaurant_ids_to_topics
    create_heat_map(restaurants, restaurant_ids_to_topics, novel_topics, my_map)

def main():
    my_map = Map.vegas()
    reviews = get_reviews_from_restuaraunts("Las Vegas")
    restaurants = get_vegas_restaurants()
    business_id = "l6QcUE8XXLrVH6Ydm4GSNw"
    run(my_map, reviews, restaurants, None, business_id)

if __name__ == '__main__':
    main()
