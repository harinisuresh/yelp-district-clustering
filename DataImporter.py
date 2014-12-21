"""Helper methods to import yelp data"""

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
    """
    Get All Phoenix restaurant.

    Returns:
        All Phoenix restaurants as a list of restaurant dictionaries objects.
    """
    return get_restaurants("Phoenix", PHOENIX_RESTAURANTS_PATH)

def get_vegas_restaurants():
    """
    Get All Vegas restaurant.

    Returns:
        All Vegas restaurants as a list of restaurant dictionaries objects.
    """
    return get_restaurants("Las Vegas", VEGAS_RESTAURANTS_PATH)


def get_edinburgh_restaurants():
    """
    Get All Edinburgh restaurant.

    Returns:
        All Edinburgh restaurants as a list of restaurant dictionaries objects.
    """
    return get_restaurants("Edinburgh", EDINBURGH_RESTAURANTS_PATH)

def get_waterloo_restaurants():
    """
    Get All Waterloo restaurant.

    Returns:
        All Waterloo restaurants as a list of restaurant dictionaries objects.
    """
    return get_restaurants("Waterloo", WATERLOO_RESTAURANTS_PATH)

def get_madison_restaurants():
    """
    Get All Madison restaurant.

    Returns:
        All Madison restaurants as a list of restaurant dictionaries objects.
    """
    return get_restaurants("Madison", MADISON_RESTAURANTS_PATH)

def category_bag_of_words(restaurants):
    """
    Get bag of words representation of restaurant's categories.

    Parameters:
        restaurants - a list of restaurant dictionary objects

    Returns:
        A bag of words dictionary, key-value pairings are category->category count.
    """
    bag = {}
    for restaurant in restaurants:
        categories = restaurant["categories"]
        for c in categories:
            bag[c] = bag.get(c,0)+1
    return bag

def get_restaurants(city_string, pickle_path=None):
    """
    Get all restaurants in a city.

    Parameters:
        city_string - the city
        pickle_path - optional path for storing and retrieving method results from pickle

    Returns:
        All restaurants in the specified city as a 
        list of restaurant dictionary objects.
    """
    if pickle_path and os.path.exists(pickle_path):
        print "Loading pickle..."
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

def get_vegas_reviews():
    """
    Get all Vegas reviews.

    Returns:
        All Vegas reviews as a single string.
    """
    return get_reviews_from_restuaraunts("Las Vegas", VEGAS_REVIEWS_PATH)

def get_phoenix_reviews():
    """
    Get all Phoenix reviews.

    Returns:
        All Phoenix reviews as a single string.
    """
    return get_reviews_from_restuaraunts("Phoenix", PHOENIX_REVIEWS_PATH)

def get_edinburgh_reviews():
    """
    Get all Edinburgh reviews.

    Returns:
        All Edinburgh reviews as a single string.
    """
    return get_reviews_from_restuaraunts("Edinburgh", EDINBURGH_REVIEWS_PATH)

def get_waterloo_reviews():
    """
    Get all Waterloo reviews.

    Returns:
        All Waterloo reviews as a single string.
    """
    return get_reviews_from_restuaraunts("Waterloo", WATERLOO_REVIEWS_PATH)

def get_madison_reviews():
    """
    Get all Madison reviews.

    Returns:
        All Madison reviews as a single string.
    """
    return get_reviews_from_restuaraunts("Madison", MADISON_REVIEWS_PATH)

def get_reviews_from_restuaraunts(city_string, pickle_path):
    """
    Get all reviews for a city.

    Parameters:
        city_string - the city
        pickle_path - optional path for storing and retrieving method results from pickle

    Returns:
        All reviews for a city as a single string.
    """
    total_reviews = 0
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
            total_reviews+=1
    pickle.dump(restauraunt_id_to_review_text, open( pickle_path, "wb" ))
    print total_reviews
    return restauraunt_id_to_review_text

def get_words_from_text(text, stop_words = {}):
    """
    Get a list of words from a given text.

    Parameters:
        text - the text to be parsed into words
        stop_words - optional set of words to be ignored in the text

    Returns:
        A list of words from the text, in order, consisting of only
        non-numeric alphanumeric characters and not containing any
        words in the stop_words set.
    """
    cleaned_text = re.sub('\\n', ' ', text)
    cleaned_text = re.split('[\s,.()!&?/\*\^#@0-9":=\[\]$\\;%]|--', cleaned_text)
    cleaned_text = [x for x in cleaned_text if x!='' and x not in stop_words]
    return cleaned_text

def get_topic_labels():
    """
    Get all topic labels.

    Returns:
        A list of all 50 topic labels as strings.
    """
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