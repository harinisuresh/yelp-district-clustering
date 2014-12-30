"""Cluster restaurants on map"""
import ElbowClustering
import Clustering
import Map 
import DataImporter

def elbow_clustering(restaurants, restaurant_ids_to_topics, my_map):
    data = Clustering.create_data_array(restaurants, restaurant_ids_to_topics, my_map)
    print "starting elbow clustering"
    ElbowClustering.plot_elbow_and_gap(data)


def run(my_map, reviews, restaurants):
    restaurants = Clustering.filter_restaurants(restaurants, reviews)
    normalized_restaurant_ids_to_topics, lda = Clustering.get_predictions(my_map, reviews, restaurants)
    elbow_clustering(restaurants, normalized_restaurant_ids_to_topics, my_map)   


def main():
    my_map = Map.vegas()
    reviews = get_vegas_reviews()
    restaurants = get_vegas_restaurants()
    run(my_map, reviews, restaurants)

if __name__ == '__main__':
    main()
