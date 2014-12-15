"""Helpers to import yelp data"""
import json
from pprint import pprint
import numpy as np
import re
import cPickle as pickle
import os.path

VEGAS_RESTAURANTS_PATH = "pickles/get_restaurants_vegas.p"
PHOENIX_RESTAURANTS_PATH = "pickles/get_restaurants_phoenix.p"
EDINBURGH_RESTAURANTS_PATH = "pickles/get_restaurants_edinburgh.p"
WATERLOO_RESTAURANTS_PATH = "pickles/get_restaurants_waterloo.p"
MADISON_RESTAURANTS_PATH = "pickles/get_restaurants_madison.p"

VEGAS_REVIEWS_PATH = "pickles/get_reviews_vegas.p"
PHOENIX_REVIEWS_PATH = "pickles/get_reviews_phoenix.p"
EDINBURGH_REVIEWS_PATH = "pickles/get_reviews_edinburgh.p"
WATERLOO_REVIEWS_PATH = "pickles/get_reviews_waterloo.p"
MADISON_REVIEWS_PATH = "pickles/get_reviews_madison.p"

def get_pheonix_restaurants():
    return get_restaurants("Phoenix", PHOENIX_RESTAURANTS_PATH)

def get_vegas_restaurants():
    return get_restaurants("Las Vegas", VEGAS_RESTAURANTS_PATH)

def get_edinburgh_restaurants():
    return get_restaurants("Edinburgh", EDINBURGH_RESTAURANTS_PATH)

def get_waterloo_restaurants():
    return get_restaurants("Waterloo", WATERLOO_RESTAURANTS_PATH)

def get_madison_restaurants():
    return get_restaurants("Madison", MADISON_RESTAURANTS_PATH)

def get_vegas_restaurants_id_to_restaurant():
    return get_restaurants_id_to_restaurant("Las Vegas")

def get_num_restaurants_of_category(restaurants, category):
    return len([restaurant for restaurant in restaurants if category in restaurant["categories"]])

def category_bag_of_words(restaurants):
    bag = {}
    for restaurant in restaurants:
        categories = restaurant["categories"]
        for c in categories:
            bag[c] = bag.get(c,0)+1
    return bag

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
    return get_reviews_from_restuaraunts("Phoenix", PHOENIX_REVIEWS_PATH)

def get_edinburgh_reviews():
    return get_reviews_from_restuaraunts("Edinburgh", EDINBURGH_REVIEWS_PATH)

def get_waterloo_reviews():
    return get_reviews_from_restuaraunts("Waterloo", WATERLOO_REVIEWS_PATH)

def get_madison_reviews():
    return get_reviews_from_restuaraunts("Madison", MADISON_REVIEWS_PATH)

def get_reviews_from_restuaraunts(city_string, pickle_path):
    if pickle_path and os.path.exists(pickle_path):
        print "Loading pickle"
        return pickle.load( open(pickle_path, "rb" ))
    restaurants = get_restaurants(city_string, None)
    relevant_restaurant_ids = {restaurant["business_id"] for restaurant in restaurants}
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
            newVal = val + review["text"]
            restauraunt_id_to_review_text[business_id] = newVal
    pickle.dump(restauraunt_id_to_review_text, open( pickle_path, "wb" ))
    return restauraunt_id_to_review_text


def get_words_from_text(text, stop_words = {}):
    cleaned_text = re.sub('\\n', ' ', text)
    cleaned_text = re.split('[\s,.()!&?/\*\^#@0-9":=\[\]$\\;%]|--', cleaned_text)
    cleaned_text = [x for x in cleaned_text if x!='' and x not in stop_words]
    return cleaned_text

def get_reviews_of_restuarant_from_phoenix(rest_id):
    revs = get_phoenix_reviews()
    return revs[rest_id]

def get_topic_labels():
    labels = [\
    "Buffet/Upscale",
    "Steak & Eggs",
    "Tacos",
    "Mexican",
    "Pho",
    "Seafood/Buffet",
    "Sports Bar",
    "Nightlife",
    "Brunch",
    "Ramen",
    "Chinese",
    "Seafood",
    "Dim Sum",
    "Vegan/Healthy",
    "Tapas",
    "Upscale",
    "Buffet",
    "Indian",
    "Burgers",
    "Greek",
    "Fish & Chips",
    "Street Vendors",
    "Korean BBQ",
    "Mexican/Bar",
    "Italian",
    "Pizza",
    "BBQ",
    "Burgers",
    "Thai",
    "Waffles/Brunch",
    "Bad Service",
    "Luxe",
    "Dessert",
    "Sushi",
    "Deli",
    "Asian/Authentic",
    "Burritos",
    "Steakhouse",
    "Exotic American",
    "Wine/Upscale",
    "Cafe",
    "Upscale",
    "Sushi",
    "Soup",
    "Oysters",
    "Casino/Hotel",
    "Noodles",
    "Chinese",
    "Buffet",
    "Prime Rib",
    ]
    return labels