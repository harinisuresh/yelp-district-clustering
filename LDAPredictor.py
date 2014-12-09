import logging
import operator
from gensim.models import LdaModel
from gensim import corpora
from sklearn.feature_extraction import text as sktext
import re
from DataImporter import get_reviews_from_restuaraunts, get_vegas_restaurants, get_vegas_reviews, get_words_from_text

class LDAPredictor():
    def __init__(self):
        self.dictionary = corpora.Dictionary.load("models/dictionary.dict")
        self.lda = LdaModel.load("models/lda_model_50_topics.lda")
    
    def predict_topics(self, review_text):
        stop_set = sktext.ENGLISH_STOP_WORDS
        words = get_words_from_text(review_text, stop_set)
        review_bow = self.dictionary.doc2bow(words)
        review_topic_predictions = self.lda[review_bow]
        return review_topic_predictions

def main():
    reviews = get_vegas_reviews()
    restaurants = get_vegas_restaurants()
    predictor = LDAPredictor()
    lda = predictor.lda
    for i in range(50):
        print "topic #", i
        print lda.show_topic(i)
    for restaurant in restaurants:
        business_id  = restaurant["business_id"]
        print "Topic Prediction for", restaurant["name"], "is:"
        review = reviews[business_id]
        prediction = predictor.predict_topics(review)
        sorted_prediction = sorted(prediction, key = operator.itemgetter(1))
        print sorted_prediction
        print "Words from best topic:"
        topic = lda.show_topic(sorted_prediction[-1][0])
        sorted_topic = sorted(topic, key = operator.itemgetter(0))
        print sorted_topic

if __name__ == '__main__':
    main()
