"""HeatMap of restaurants on map"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial.distance import cdist
from MapUtils import Coordinate, Position, Rectangle, create_n_unique_colors
from Map import Map
from DataImporter import get_pheonix_restaurants, get_vegas_restaurants, get_vegas_reviews
from LDAPredictor import LDAPredictor
import math
import random
from Utils import make_topic_array_from_tuple_list, print_median_std_from_clusters


def create_heat_map(restaurants, restaurant_ids_to_topics, novel_restaurant_topics, my_map, pure_buckets=False, novel_business=None):
    print "starting heatmap"
    n_topics = 50

    novel_restaurant_topics_array = make_topic_array_from_tuple_list(novel_restaurant_topics, 50)
    image_width = my_map.image_width()
    image_height = my_map.image_height()
    n_x_bins = 6
    n_y_bins = 6
    distances = np.zeros((n_x_bins, n_y_bins))
    bin_width = image_width/n_x_bins
    bin_height = image_height/n_y_bins
    gaussian_variance = 500 # 50.0 # math.sqrt(bin_width**2+bin_height**2)/2.0

    restuarants_indexed_by_id = {restaurant["business_id"] : restaurant for restaurant in restaurants}
    for xi in range(n_x_bins):
        for yi in range(n_y_bins):
            gauss_weighted_topics = np.array([0.0 for i in range(n_topics)])
            square_center = Position((xi+0.5)*bin_width, (yi+0.5)*bin_height)  
            print "center:", square_center
            square_pos_array = np.array([square_center.x, square_center.y])
            for business_id, restaurant_topics in restaurant_ids_to_topics.iteritems():
                restaurant = restuarants_indexed_by_id[business_id]
                restaurant_topics_array = np.array(make_topic_array_from_tuple_list(restaurant_topics, 50))
                rest_pos = my_map.world_coordinate_to_image_position(Coordinate(restaurant["latitude"], restaurant["longitude"]))
                rest_pos_array = np.array([rest_pos.x, rest_pos.y])
                gaussian_weight = gaussian(rest_pos_array, square_pos_array, gaussian_variance)
                print "gaussian weight", gaussian_weight
                print "rest pos", rest_pos
                print "center pos", square_pos_array
                gauss_weighted_topics += restaurant_topics_array*gaussian_weight
            sum_gauss_weighted_topics = gauss_weighted_topics.sum(axis=0)

            ave_dist_weighted_topics = gauss_weighted_topics/sum_gauss_weighted_topics
            print "sum", sum_gauss_weighted_topics
            print "ave_tops", ave_dist_weighted_topics
            print "novel_tops", novel_restaurant_topics_array
            print "sum2", ave_dist_weighted_topics.sum(axis=0)
            print "sum3", np.array(novel_restaurant_topics_array).sum(axis=0)

            A = np.array(novel_restaurant_topics_array)

            B = ave_dist_weighted_topics
            difference = 2*np.sqrt(2.0) - np.sqrt(np.sum((A - B)**2))
            distances[xi, yi] = difference
            print difference
    print distances


    im = plt.imread(my_map.image_path)
    implot = plt.imshow(im, alpha=0.9, extent=[0,n_x_bins,0,n_y_bins])
    plt.pcolor(distances, cmap=plt.cm.cool, edgecolors='k',  alpha=0.5)
    if novel_business:
        pos = my_map.world_coordinate_to_image_position(Coordinate(novel_business["latitude"], novel_business["longitude"]))
        pos.x /= image_width/n_x_bins
        pos.y /= image_height/n_y_bins
        plt.plot(pos.x, pos.y, marker='x', ms=20)
        plt.plot(pos.x, pos.y, marker='o', color=[.1,.1,.1], ms=20, markerfacecolor='none')
    plt.show()
    print_median_std_from_clusters(clusters_of_restaurants)
    print "done"

def gaussian(x, mean, sigma):
    a = 1.0/(sigma*math.sqrt(2*math.pi))
    b = mean
    c = sigma
    dist_squared = np.sum((x - b)**2)
    return a*math.exp(-1*dist_squared/(2*c*c))

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
        #return restaurant_ids_to_topics
    novel_business = None
    if novel_business_id != None:
        novel_business = [business for business in restaurants if business["business_id"] == novel_business_id][0]
    print "novel topics", novel_topics
    create_heat_map(restaurants, restaurant_ids_to_topics, novel_topics, my_map, pure_buckets, novel_business)

def main():
    print gaussian(np.array([200,200]), np.array([200,200]), 50)
    print gaussian(np.array([150,200]), np.array([200,200]), 50)
    print gaussian(np.array([100,200]), np.array([200,200]), 50)

    my_map = Map.vegas()
    reviews = get_vegas_reviews()
    restaurants = get_vegas_restaurants()
    business_id = "l6QcUE8XXLrVH6Ydm4GSNw"
    run(my_map, reviews, restaurants, None, business_id)

if __name__ == '__main__':
    main()