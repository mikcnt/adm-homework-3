import pandas as pd
from langdetect import detect
import string
import data_collector
import parser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import pickle
import math
import numpy as np
import heapq
import re
import functions


# Utils
def book_title_summary(df):
    n_missing = df[(df['bookTitle'].isna())].shape[0]
    print('There are {} instances that are missing the `bookTitle` column.'.format(n_missing))
    print()
    return df[(df['bookTitle'].isna())].head()

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def rating_value_summary(df):
    print('There are {} books with rating value between 0 and 4'.format(df[(df['ratingValue'] > 0) & (df['ratingValue'] < 4)].shape[0]))
    print('There are {} books with rating value greater than 4'.format(df[(df['ratingValue'] > 4)].shape[0]))

# Exercise 1

# Preprocessing

def remove_punctuation(s):
    reg1 = re.compile(r'\b[^\w\s]\b')
    reg2 = re.compile(r'[^\w\s]')
    first_pass = reg1.sub(' ', s)
    return reg2.sub('', first_pass)

def language_det(s):
    if s == '':
        return 'empty'
    try:
        return detect(s)
    except:
        return 'symbols'
    
def remove_stopwords(s):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(s)
    return ' '.join([w for w in tokens if w not in stop_words])

def stemming(s):
    ps = PorterStemmer()
    tokens = word_tokenize(s)
    return ' '.join([ps.stem(w) for w in tokens])

# Exercise 2

# Indexes

def term_index(documents):
    words = set()
    for s in documents:
        try:
            tokens = set(word_tokenize(s))
            words.update(tokens)
        except:
            continue
        
    term_index = {}
    for i, word in enumerate(words):
        term_index[word] = i
    return term_index

def inverted_index(documents, term_indexes):
    inv_index = defaultdict(list)
    for i, s in enumerate(documents):
        try:
            tokens = set(word_tokenize(s))
            for token in tokens:
                token_index = term_indexes[token]
                inv_index[token_index].append(i)
        except:
            continue
    return inv_index

def tfidf_inv_indexes(documents, term_indexes, inv_indexes):
    tfidf_indexes = defaultdict(dict)
    for doc_id, s in enumerate(documents):
        try:
            tokens = word_tokenize(s)
            tokens_set = set(tokens)
            n_tokens = len(tokens)
            norm = 0
            for token in tokens_set:
                token_index = term_indexes[token]
                # tf = n_times token appears in the document over the number of words of the document
                tf = s.count(token) / n_tokens
                # idf = log of the number of documents in the corpus over the number of documents in which the token appears
                idf = math.log10(len(documents) / (len(inv_indexes[token_index])))
                tf_idf = tf * idf
                # we just computed the tfidf for the a particular token, for the document we're considering
                tfidf_indexes[token_index][doc_id] = tf * idf
                
                # Store also the norm for the document we're considering
                # which is sqrt of the sum of the squares
                norm += tf_idf ** 2
                
            # apply sqrt
            norm = np.sqrt(norm)
            # Normalize each document tfidf
            for token in tokens_set:
                token_index = term_indexes[token]
                tfidf_indexes[token_index][doc_id] /= norm
        except:
            continue
    return tfidf_indexes

# Search engines

## Vanilla search engine

class SimpleSearchEngine:
    def __init__(self, df, term_indexes, inv_indexes):
        self.df = df
        self.term_indexes = term_indexes
        self.inv_indexes = inv_indexes
        
    def search(self, query):
        # Since we performed stemming on the plot column of the dataframe, we need to
        # perform stemming also on the query. Otherwise, our results wouldn't be accurate
        ps = PorterStemmer()
        query_tokens = set([ps.stem(w) for w in word_tokenize(query)])

        # Create term indexes for the query
        # notice: if one of the query element doesn't appear in the term_indexes dictionary
        # we can safely say that the **conjunctive** query has to return nothing
        term_indexes_tokens = []
        for token in query_tokens:
            if token in self.term_indexes.keys():
                term_indexes_tokens.append(self.term_indexes[token])
            else:
                return pd.DataFrame(columns=['bookTitle', 'Plot', 'Url'])

        query_inv_indexes = {}
        for token_index in term_indexes_tokens:
            query_inv_indexes[token_index] = set(self.inv_indexes[token_index])

        # Since it is a conjuntive query, we need to intersect the results of each query token
        documents_id = sorted(set.intersection(*query_inv_indexes.values()))

        return pd.DataFrame(data=self.df[self.df['index'].isin(documents_id)])
    
    def execute_query(self, query):
        return self.search(query)[['bookTitle', 'Plot', 'Url']]

## Scoring functions

class RankCalculator():
    def rank(self, book):
        pass
    
class WeightedRanks(RankCalculator):
    def __init__(self, calculators):
        self.calculators = calculators
        
    def rank(self, book, query, token_ids):
        total_weight = np.sum([weight for weight, _ in self.calculators])
        return np.sum([calculator.rank(book, query, token_ids) * weight / total_weight for weight, calculator in self.calculators])
    
class ByTfidf(RankCalculator):
    def __init__(self, tfidf_indexes):
        self.tfidf_indexes = tfidf_indexes
    
    def rank(self, book, query, token_ids):
        doc = book['index']
        tfidf = 0
        for token_id in token_ids:
            tfidf += self.tfidf_indexes[token_id][doc]
        return tfidf / np.sqrt(len(query.split()))
    
class ByRatingValue(RankCalculator):
    def __init__(self, df):
        self.max_rating = df['ratingValue'].max()
        
    def rank(self, book, query, token_ids):
        return book['ratingValue'] / self.max_rating
    
class ByRatingCount(RankCalculator):
    def __init__(self, df):
        self.max_rating = df['ratingCount'].max()
        
    def rank(self, book, query, token_ids):
        return book['ratingCount'] / self.max_rating
    
class ByTitleMatch(RankCalculator):
    def rank(self, book, query, token_ids):
        title_lenght = len(word_tokenize(book['bookTitle']))
        matches = 0
        for token in set(word_tokenize(query)):
            if token in book['bookTitle'].lower():
                matches += 1
        return matches / title_lenght

## Ranked search engine

class RankedSearchEngine:
    def __init__(self, df, term_indexes, inv_indexes, tfidf_indexes, simple_SE, rank_calculator):
        self.df = df
        self.term_indexes = term_indexes
        self.inv_indexes = inv_indexes
        self.tfidf_indexes = tfidf_indexes
        self.simple_SE = simple_SE
        self.rank_calculator = rank_calculator
        
    def execute_query(self, query, k=10):
        # First stem the query
        ps = PorterStemmer()
        query_tokens = set([ps.stem(w) for w in word_tokenize(query)])
        
        # Extract the token indexes from the vocabulary
        tokens_ids = []
        for token in query_tokens:
            try:
                tokens_ids.append(self.term_indexes[token])
            except:
                return
        
        # Compute the simple conjunctive query to get the books in which the query appears
        conj_query = self.simple_SE.search(query)
        if conj_query.empty:
            return conj_query[['bookTitle', 'Plot', 'Url']]
        
        # Compute the similiarity
        conj_query['Similarity'] = conj_query.apply(lambda t: self.rank_calculator.rank(t, query, tokens_ids), axis=1)

        # Use heaps to extract top k rows
        conj_query_list = conj_query[['bookTitle', 'Plot', 'Url', 'Similarity']].values.tolist()
        heapq.heapify(conj_query_list)
        max_k = heapq.nlargest(k, conj_query_list, key = lambda t: t[3])

        # Convert back to dataframe to show it
        max_k_df = pd.DataFrame(data=max_k, columns=['bookTitle', 'Plot', 'Url', 'Similarity'])

        return max_k_df