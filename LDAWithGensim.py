""" Test"""

import logging
import gensim
from gensim.corpora import BleiCorpus
from gensim import corpora
from DataImporter import get_reviews_from_restuaraunts
import re
from sklearn.feature_extraction import text as sktext

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Corpus(object):
    def __init__(self, values, reviews_dictionary, corpus_path):
        self.values = values
        self.reviews_dictionary = reviews_dictionary
        self.corpus_path = corpus_path

    def __iter__(self):
        for business_id, review in self.values.iteritems():
            yield self.reviews_dictionary.doc2bow(review["words"])

    def serialize(self):
        BleiCorpus.serialize(self.corpus_path, self, id2word=self.reviews_dictionary)
        return self


class Dictionary(object):
    def __init__(self, values, dictionary_path):
        self.values = values
        self.dictionary_path = dictionary_path

    def build(self):
        dictionary = corpora.Dictionary(review["words"] for business_id, review in self.values.iteritems())
        dictionary.filter_extremes(keep_n=10000)
        dictionary.compactify()
        corpora.Dictionary.save(dictionary, self.dictionary_path)
        return dictionary


class Train():
    def __init__(self):
        pass

    @staticmethod
    def run(lda_model_path, corpus_path, num_topics, id2word):
        corpus = corpora.BleiCorpus(corpus_path)
        lda = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=id2word, passes=10, eval_every=10, iterations=500)
        lda.save(lda_model_path)
        return lda

def get_words_from_text(text, custom_stop_words = []):
    stoplist = sktext.ENGLISH_STOP_WORDS
    # Build tokenized, normalised word vectors for each document
    # We could apply stemming here.
    top_15_percent = set([u"it's", u'like', u'food', u'time', u'really', u'great', u'service', u'just', u'place', u'good', u'chicken'])

    words = [word for word in re.split('[\s,.()!&?/\*\^#@0-9":=\[\]$\\;%]|--', text.lower()) if word not in stoplist and word not in top_15_percent and word != ""]
    
    return words


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    dictionary_path = "models/dictionary.dict"
    corpus_path = "models/corpus.lda-c"
    lda_num_topics = 50
    lda_model_path = "models/lda_model.lda"
    city = "Las Vegas"
    reviews = get_reviews_from_restuaraunts(city)

    corpus_collection = {business_id : {"review_text" : review_text, "words": get_words_from_text(review_text)} for business_id, review_text in reviews.iteritems()}
    dictionary = Dictionary(corpus_collection, dictionary_path).build()
    Corpus(corpus_collection, dictionary, corpus_path).serialize()
    lda = Train.run(lda_model_path, corpus_path, lda_num_topics, dictionary)
    print "Converged on Topics:"
    lda.print_topics(50)

if __name__ == '__main__':
    main()
