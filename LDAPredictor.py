import logging
import operator
from gensim.models import LdaModel
from gensim import corpora
from sklearn.feature_extraction import text as sktext
import re
from DataImporter import get_reviews_from_restuaraunts, get_vegas_restaurants

def get_words_from_text(text, custom_stop_words = []):
    stoplist = sktext.ENGLISH_STOP_WORDS
    # Build tokenized, normalised word vectors for each document
    # We could apply stemming here.
    top_15_percent = set([u"it's", u'like', u'food', u'time', u'really', u'great', u'service', u'just', u'place', u'good', u'chicken'])

    words = [word for word in re.split('[\s,.()!&?/\*\^#@0-9":=\[\]$\\;%]|--', text.lower()) if word not in stoplist and word not in top_15_percent and word != ""]
    return words

class LDAPredictor():
    def __init__(self):
        self.dictionary = corpora.Dictionary.load("models/dictionary.dict")
        self.lda = LdaModel.load("models/lda_model_50_topics.lda")
    
    def predict_topics(self, review_text):
        words = get_words_from_text(review_text)
        review_bow = self.dictionary.doc2bow(words)
        review_topic_predictions = self.lda[review_bow]
        return review_topic_predictions

def main():
    reviews = get_reviews_from_restuaraunts("Las Vegas")
    restaurants = get_vegas_restaurants()
    predictor = LDAPredictor()
    lda = predictor.lda
    for restaurant in restaurants:
        business_id  = restaurant["business_id"]
        print "Prediction for: ", restaurant["name"], "is:"
        review = reviews[business_id]
        prediction = predictor.predict_topics(review)
        print prediction
        print "best topic:"
        sorted_prediction = prediction.sort(key = operator.itemgetter(1))
        print lda.show_topic(sorted_prediction[0][0])
        print lda.show_topic(sorted_prediction[0][0])[0]

if __name__ == '__main__':
    main()
