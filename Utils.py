"""Utils"""
import math
import numpy as np

def make_topic_array_from_tuple_list(weight_tuples, num_topics, scale_factor=1.0):
    topic_array = [0 for i in range(num_topics)]
    total = float(sum([weight_tuple[1] for weight_tuple in weight_tuples])) # For normalization
    for weight_tuple in weight_tuples:
        topic_index = weight_tuple[0]
        topic_weight = weight_tuple[1]
        topic_array[topic_index] = (topic_weight) * scale_factor  # For normalization + scaling
    return topic_array


def make_tuple_list_from_topic_array(topic_array):
	tuple_list = []
	for i in range(len(topic_array)):
		tuple_list.append((i,topic_array[i]))
	return tuple_list
	

def gaussian(x, mean, variance):
    a = math.sqrt(2*math.pi*variance)
    b = mean
    c = variance
    dist_squared = np.sum((x - b)**2)
    return a*math.exp(-1*dist_squared/(2*c*c))


def print_median_std_from_clusters(clusters):
    counts = []
    num_points = sum([len(cluster) for cluster in clusters])
    mean = num_points/len(clusters)
    SD_sum = 0.0
    for i in range(len(clusters)):
        cluster = clusters[i]
        count = len(cluster)
        counts.append(count)
        v = (mean - count)**2
        SD_sum += v
    ave_var = SD_sum/len(clusters)
    SD = math.sqrt(ave_var)
    med = median(counts)
    print "Standard deviation", SD
    print "median", med

def median(mylist):
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]
