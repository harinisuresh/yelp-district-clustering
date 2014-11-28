"""Helpers to import yelp data"""
import json
from pprint import pprint
import numpy as np

def get_pheonix_restaurants():
    return get_restaurants("Phoenix")

def get_vegas_restaurants():
    return get_restaurants("Las Vegas")

def get_vegas_restaurants_id_to_restaurant():
    return get_restaurants_id_to_restaurant("Las Vegas")

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

def get_reviews_from_restuaraunts(city_string):
    restaurants = get_restaurants(city_string)
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
    i = 0
    for review in reviews:
        business_id = review["business_id"]
        if business_id in relevant_restaurant_ids:
            i = i + 1
            val = restauraunt_id_to_review_text.get(business_id, "")
            review_text = clean_review(review["text"])
            newVal = val + review["text"]
            restauraunt_id_to_review_text[business_id] = newVal

    return restauraunt_id_to_review_text

def clean_review(text):
    cleaned_text = re.split('[\s,.()!&?/\*\^#@0-9":=\[\]$\\;%]|--', text)
    return cleaned_text
    
print get_reviews_from_restuaraunts("Phoenix")
