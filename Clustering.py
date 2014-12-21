NUM_TOPICS = 50
N_CLUSTERS = 30
K = np.sqrt(5.0/6.0)
PIXELS_PER_MILE = 48.684
LDA_ClUSTER_SCALE_FACTOR =  K*PIXELS_PER_MILE


def get_clusters(restaurants, restaurant_ids_to_topics, my_map, lda, use_human_labels=True):

	restaurant_coordinates = []
    restaurant_positions = []
    all_topic_weights = []
    num_restaurants = restaurants.size

    print "K-mean clustering on :", num_restaurants, "restaurants with", N_CLUSTERS, "clusters"

    #create an array of position & topic arrays for each restaurant
    for restaurant in restaurants:
        business_id = restaurant["business_id"]
        coord = Coordinate(restaurant["latitude"],restaurant["longitude"])
        position = my_map.world_coordinate_to_image_position(coord)
        restaurant_coordinates.append(coord)
        restaurant_positions.append(position)
        all_topic_weights_for_restaurant = restaurant_ids_to_topics[business_id]
        all_topic_weights_array_for_restaurant = make_topic_array_from_tuple_list(all_topic_weights_for_restaurant, NUM_TOPICS, LDA_ClUSTER_SCALE_FACTOR)
        all_topic_weights.append(all_topic_weights_array_for_restaurant)

    data_array = []
    for i in range(num_restaurants):
        topic_weights = all_topic_weights[i]
        pos = restaurant_positions[i]
        d = [pos.x, pos.y]
        d.extend(topic_weights)
        data_array.append(d)

    print data_array[1:5]    
    data = np.array(data_array)
    centers, center_dist = kmeans(data, N_CLUSTERS, iter=200)
    classifications, classification_dist = vq(data, centers)


    plt.figure(1)
    im = plt.imread(my_map.image_path)
    implot = plt.imshow(im)

    clusters = [data[classifications==i] for i in range(N_CLUSTERS)]
    clusters_of_restaurants = [restaurants[classifications==i] for i in range(N_CLUSTERS)]

    colors = create_n_unique_colors(N_CLUSTERS)

    centers_x = [p[0] for p in centers]
    centers_y = [p[1] for p in centers]
    clusters_x = [[p[0] for p in clusters[i]] for i in range(N_CLUSTERS)]
    clusters_y = [[p[1] for p in clusters[i]] for i in range(N_CLUSTERS)]

    # Plot clusters of restaurants with different colors
    for i in range(N_CLUSTERS):
        cluster_x = clusters_x[i]
        cluster_y = clusters_y[i]
        plt.scatter(cluster_x, cluster_y, marker='o', color=colors[i], alpha=0.5)


    plt.title("Las Vegas K-Means Clustering With Labels")
    plt.show()

    plt.figure(2)

    im = plt.imread(my_map.image_path)
    implot = plt.imshow(im)

    for i in range(N_CLUSTERS):
        cluster_x = clusters_x[i]
        cluster_y = clusters_y[i]
        plt.scatter(cluster_x, cluster_y, marker='o', color=colors[i], alpha=0.5)

    for i in range(len(centers_x)):
        center_position = Position(centers_x[i], centers_y[i])
        restaurants = clusters_of_restaurants[i]
        label_text, label_weight = make_label_text_for_cluster(center_position, restaurants, restaurant_ids_to_topics, lda, use_human_labels)
        font_size_1 = 10;
        font_size_2 = 10;
        plt.annotate(label_text[0], xy = (centers_x[i], centers_y[i]), xytext = (centers_x[i]-(len(label_text[0])/2.0)*font_size_1, centers_y[i]+font_size_1), fontsize=font_size_1)
        plt.annotate(label_text[1], xy = (centers_x[i], centers_y[i]), xytext = (centers_x[i]-(len(label_text[1])/2.0)*font_size_2, centers_y[i]-font_size_2), fontsize=font_size_2)

    plt.title("Las Vegas K-Means Clustering With Labels")

    plt.show()

    print_median_std_from_clusters(clusters_of_restaurants)
