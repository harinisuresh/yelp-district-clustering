"""Utils"""

def make_topic_array_from_tuple_list(weight_tuples, num_topics, scale_factor=1.0):
    topic_array = [0 for i in range(num_topics)]
    total = float(sum([weight_tuple[1] for weight_tuple in weight_tuples])) # For normalization
    for weight_tuple in weight_tuples:
        topic_index = weight_tuple[0]
        topic_weight = weight_tuple[1]
        topic_array[topic_index] = (topic_weight/total) * scale_factor  # For normalization + scaling
    return topic_array