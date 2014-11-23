import json
from pprint import pprint

def get_pheonix_restaurants():
    f = open('yelp_dataset/yelp_academic_dataset_business.json', "r")
    lines = [line for line in f]
    f.close()

    businesses = [json.loads(line) for line in lines]
    pheonix_businesses = [business for business in businesses\
        if business["city"] == "Phoenix"\
        and "Restaurants" in business["categories"]]
    return pheonix_businesses
