"""Cluster restaurants on map"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from MapUtils import Coordinate, Position
from Map import Map
from DataImporter import get_pheonix_restaurants

N_CLUSTERS = 10
myMap = Map.pheonix()
restaurants = get_pheonix_restaurants()
convert_func = lambda x: myMap.world_coordinate_to_image_position(x)
restaurant_coordinates = []
restaurant_positions = []

for restaurant in restaurants:
    coord = Coordinate(restaurant["latitude"],restaurant["longitude"])
    position = myMap.world_coordinate_to_image_position(coord)
    restaurant_coordinates.append(coord)
    restaurant_positions.append(position)

data = np.array([[pos.y,pos.x] for pos in restaurant_positions if pos])

centers, _ = kmeans(data, N_CLUSTERS, iter=100)
classifications, distance = vq(data,centers)

im = plt.imread(myMap.image_path)
implot = plt.imshow(im)

clusters = [data[classifications==i] for i in range(N_CLUSTERS)]

# coordinates = [Coordinate((33.4279533+33.4618937)/2.0, (-112.1082946 + -112.0371188)/2.0)]
# points = [myMap.world_coordinate_to_image_position(c) for c in coordinates]
# plt.plot([p.x for p in points],[p.y for p in points],'o')
# plt.plot([p.x for p in points],[p.y for p in points],'x')

print centers

colors = np.random.rand(N_CLUSTERS)
print colors

centers_x = [p[1] for p in centers]
centers_y = [p[0] for p in centers]
clusters_x = [[p[1] for p in clusters[i]] for i in range(N_CLUSTERS)]
clusters_y = [[p[0] for p in clusters[i]] for i in range(N_CLUSTERS)]

plt.plot(centers_x, centers_y, 'x', [0,0,0])
for i in range(N_CLUSTERS):
    cluster_x = clusters_x[i]
    cluster_y = clusters_y[i]
    plt.plot(cluster_x, cluster_y, 'o', colors[i])

plt.show()