from DataImporter import get_reviews_from_restuaraunts
from gensim import corpora, models, similarities
import logging
import gensim
from sklearn.feature_extraction import text as sktext
import re

def get_top_5_percent(documents):
    city = "Phoenix"
    documents = documents.values()
    #documents = get_reviews_from_restuaraunts(city).values()
     
    stoplist = sktext.ENGLISH_STOP_WORDS
    # Build tokenized, normalised word vectors for each document
    # We could apply stemming here.

    #cleaned_texts = re.split(' ,.', document.lower())

    texts = [[word for word in re.split('[\s,.()!&?/\*\^#@0-9":=\[\]$\\;%]|--', document.lower()) if word not in stoplist and word != ""]
             for document in documents]

    #print texts
    word_counts = {}
    total = sum([len(t) for t in texts])
    for t in texts: 
        for word in t: 
            #print word
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1


    top_5_percent_words = []
    for key in word_counts.keys():
        print key, word_counts[key], total
        if word_counts[key] >= .005*total:
            top_5_percent_words.append(key)

    return top_5_percent_words

     
    # remove words that appear only once
    #all_tokens = sum(texts, [])

   # tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)

    # TODO Remove top 10% of most common words
    #total_words = float(len(all_tokens))
    #over_five_percent = [token for token in all_tokens if all_tokens.count(token)/total_words > 0.05]


    #return over_five_percent

