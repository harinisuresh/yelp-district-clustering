"""Helpers to import yelp data"""
import json
from pprint import pprint
import numpy as np

class Business:
    def __init__(self, lattitude, longitude, business_id):
        self.coordinate = Coordinate(lattitude,longitude)
        self.business_id = business_id

def get_pheonix_restaurants():
    return get_restaurants("Phoenix")

def get_vegas_restaurants():
    return get_restaurants("Las Vegas")

def get_restaurants(city_string):
    f = open('yelp_dataset/yelp_academic_dataset_business.json', "r")
    print "Reading Restaurant JSON..."
    lines = [line for line in f]
    f.close()

    businesses = [json.loads(line) for line in lines]
    restaurants = [business for business in businesses\
        if business["city"] == city_string\
        and "Restaurants" in business["categories"]]
    return np.array(restaurants)


def get_reviews_from_restuaraunts(city_string):
    restaurants = get_restaurants(city_string)
    relevant_restaurant_ids = {restaurant["business_id"] for restaurant in restaurants}
    f = open('yelp_dataset/yelp_academic_dataset_review.json', "r")
    print "Reading Review JSON..."
    lines = [line for line in f]
    f.close()
    print "Done Reading Review JSON..."
    print "Parsing Review JSON..."
    reviews = [json.loads(line) for line in lines]
    print "Done Parsing Review JSON..."
    restauraunt_id_to_review_text = dict()
    i = 0
    for review in reviews:
        if i > 1000:
            break
        business_id = review["business_id"]
        if business_id in relevant_restaurant_ids:
            print i
            i = i + 1
            val = restauraunt_id_to_review_text.get(business_id, "")
            review_text = review["text"]
            newVal = val + review["text"]
            restauraunt_id_to_review_text[business_id] = newVal
    print restauraunt_id_to_review_text
    return restauraunt_id_to_review_text

def clean_review(text):
    
    
print get_reviews_from_restuaraunts("Phoenix")