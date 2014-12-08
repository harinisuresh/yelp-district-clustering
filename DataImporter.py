"""Helpers to import yelp data"""
import json
from pprint import pprint
import numpy as np
import re
import cPickle as pickle
import os.path

VEGAS_RESTAURANTS_PATH = "pickles/get_restaurants_vegas.p"
PHOENIX_RESTAURANTS_PATH = "pickles/get_restaurants_phoenix.p"

VEGAS_REVIEWS_PATH = "pickles/get_reviews_vegas.p"
PHOENIX_REVIEWS_PATH = "pickles/get_reviews_phoenix.p"

def get_pheonix_restaurants():
    return get_restaurants("Phoenix", PHOENIX_RESTAURANTS_PATH)

def get_vegas_restaurants():
    return get_restaurants("Las Vegas", VEGAS_RESTAURANTS_PATH)

def get_vegas_restaurants_id_to_restaurant():
    return get_restaurants_id_to_restaurant("Las Vegas")

def get_restaurants(city_string, pickle_path):
    if pickle_path and os.path.exists(pickle_path):
        print "Loading pickle"
        return pickle.load( open(pickle_path, "rb" ))
    f = open('yelp_dataset/yelp_academic_dataset_business.json', "r")
    print "Reading Restaurant JSON..."
    lines = [line for line in f]
    f.close()

    businesses = [json.loads(line) for line in lines]
    restaurants = [business for business in businesses\
        if business["city"] == city_string\
        and "Restaurants" in business["categories"]]
    restaurants = np.array(restaurants)
    if pickle_path:
        pickle.dump( restaurants, open( pickle_path, "wb" ))
    return restaurants

def get_restaurants_id_to_restaurant(city_string):
    f = open('yelp_dataset/yelp_academic_dataset_business.json', "r")
    print "Reading Restaurant JSON..."
    lines = [line for line in f]
    f.close()

    businesses = [json.loads(line) for line in lines]
    restaurants = {business["business_id"] : business for business in businesses\
        if business["city"] == city_string\
        and "Restaurants" in business["categories"]}
    return restaurants

def get_vegas_reviews():
    return get_reviews_from_restuaraunts("Las Vegas", VEGAS_REVIEWS_PATH)

def get_phoenix_reviews():
    return get_reviews_from_restuaraunts("Las Vegas", VEGAS_REVIEWS_PATH)

def get_reviews_from_restuaraunts(city_string, pickle_path):
    if pickle_path and os.path.exists(pickle_path):
        print "Loading pickle"
        return pickle.load( open(pickle_path, "rb" ))
    restaurants = get_restaurants(city_string, None)
    relevant_restaurant_ids = {restaurant["business_id"] for restaurant in restaurants}
    print restaurants[0]["name"]
    f = open('yelp_dataset/yelp_academic_dataset_review.json', "r")
    print "Reading Review JSON..."
    lines = [line for line in f]
    f.close()
    print "Done Reading Review JSON..."
    print "Parsing Review JSON..."
    reviews = [json.loads(line) for line in lines]
    print "Done Parsing Review JSON..."
    restauraunt_id_to_review_text = {}
    for review in reviews:
        business_id = review["business_id"]
        if business_id in relevant_restaurant_ids:
            val = restauraunt_id_to_review_text.get(business_id, "")
            review_text = review["text"]
            print review_text
            newVal = val + review["text"]
            restauraunt_id_to_review_text[business_id] = newVal
    pickle.dump(restauraunt_id_to_review_text, open( pickle_path, "wb" ))
    return restauraunt_id_to_review_text


def get_words_from_text(text, stop_words = {}):
    cleaned_text = re.sub('\\n', ' ', text)
    cleaned_text = re.split('[\s,.()!&?/\*\^#@0-9":=\[\]$\\;%]|--', cleaned_text)
    cleaned_text = [x for x in cleaned_text if x!='' and x not in stop_words]
    return cleaned_text
    