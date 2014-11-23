"""Helpers to import yelp data"""
import json
from pprint import pprint
import numpy as np

def get_pheonix_restaurants():
    return get_restaurants("Phoenix")

def get_vegas_restaurants():
    return get_restaurants("Las Vegas")

def get_restaurants(city_string):
    f = open('yelp_dataset/yelp_academic_dataset_business.json', "r")
    lines = [line for line in f]
    f.close()

    businesses = [json.loads(line) for line in lines]
    restaurants = [business for business in businesses\
        if business["city"] == city_string\
        and "Restaurants" in business["categories"]]
    return np.array(restaurants)
