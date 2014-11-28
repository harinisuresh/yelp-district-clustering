""" Play with LDA"""
from DataImporter import get_reviews_from_restuaraunts
from gensim import corpora, models, similarities
import logging
import gensim
from sklearn.feature_extraction import text 
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

city = "Vegas"
documents = get_reviews_from_restuaraunts(city).values()
 
stoplist = text.ENGLISH_STOP_WORDS
# Build tokenized, normalised word vectors for each document
# We could apply stemming here.

texts = [[word for word in re.split('[\s,.()!&?/\*\^#@0-9":=\[\]$\\;%]|--', document.lower()) if word not in stoplist and word != ""]
          for document in documents]
 
# remove words that appear only once
all_tokens = sum(texts, [])

tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)

over_five_percent = [u"it's", u'like', u'food', u'time', u'really', u'great', u'service', u'just', u'-', u'place', u'good', u'chicken']

texts = [[word for word in text if word not in tokens_once and word not in over_five_percent]
         for text in texts]

# now we have normalised texts, we need to turn them into a vector representation
# - here we use bag-of-words. We can use other techniques but it is vital to use
# the same vector space for all computations.
 
# Build a dictionary - a frequency distribution of integer IDs representing words.
# The dictionary object can translate feature id<->word.
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(keep_n=10000)
dictionary.compactify()
dictionary.save('dictionary.dict') 
 
# Build a vector space corpus - use the dictionary to translate
# word vectors into sparse feature vectors
# We will use this corpus to train our models.
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('corpus.mm', corpus)
 
#we could reload with:
#corpus = corpora.MmCorpus('corpus.mm')
n_topics = 10

print "starting LDA"

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics)

topics = sorted([lda.show_topic(t_id, 8) for t_id in range(n_topics)], key=lambda topic_tuple: topic_tuple[0])
print 'done'
lda.print_topics(10)

print lda[dictionary.doc2bow(texts[0])]


