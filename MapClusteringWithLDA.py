"""Cluster restaurants on map"""
import Clustering
import Map 
import DataImporter


def create_topic_clusters_and_map(restaurants, restaurant_ids_to_topics, my_map, lda, use_human_labels=True):
    data = Clustering.create_data_array(restaurants, restaurant_ids_to_topics, my_map)
    Clustering.plot_clusters(my_map, restaurants, restaurant_ids_to_topics, data, lda)

def run(my_map, reviews, restaurants):
    restaurants = Clustering.filter_restaurants(restaurants)
    normalized_restaurant_ids_to_topics, lda = Clustering.get_predictions(my_map, reviews, restaurants)
    create_topic_clusters_and_map(restaurants, normalized_restaurant_ids_to_topics, my_map, lda)   

def main():
    my_map = Map.Map.vegas()
    reviews = DataImporter.get_vegas_reviews()
    restaurants = DataImporter.get_vegas_restaurants()
    run(my_map, reviews, restaurants)

if __name__ == '__main__':
    main()
