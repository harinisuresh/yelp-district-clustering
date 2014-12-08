"""Utils"""
import math
import numpy as np

def make_topic_array_from_tuple_list(weight_tuples, num_topics, scale_factor=1.0):
    topic_array = [0 for i in range(num_topics)]
    total = float(sum([weight_tuple[1] for weight_tuple in weight_tuples])) # For normalization
    for weight_tuple in weight_tuples:
        topic_index = weight_tuple[0]
        topic_weight = weight_tuple[1]
        topic_array[topic_index] = (topic_weight/total) * scale_factor  # For normalization + scaling
    return topic_array

def gaussian(x, mean, variance):
    a = math.sqrt(2*math.pi*variance)
    b = mean
    c = variance
    dist_squared = np.sum((x - b)**2)
    return a*math.exp(-1*dist_squared/(2*c*c))