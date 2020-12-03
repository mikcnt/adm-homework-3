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
import matplotlib.pyplot as plt 


# Utils
def book_title_summary(df):
    """Show a summary of the missing values for the title column of the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Rows of the dataframe for which the title values are missing.
    """
    n_missing = df[(df['bookTitle'].isna())].shape[0]
    print('There are {} instances that are missing the `bookTitle` column.'.format(n_missing))
    print()
    return df[(df['bookTitle'].isna())].head()

def save_obj(obj, name):
    with open('./indexes/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('./indexes/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def rating_value_summary(df):
    """Print stats of the values of the rating value column.

    Args:
        df (pd.DataFrame): Input dataframe.
    """
    print('There are {} books with rating value between 0 and 4'.format(df[(df['ratingValue'] > 0) & (df['ratingValue'] < 4)].shape[0]))
    print('There are {} books with rating value greater than 4'.format(df[(df['ratingValue'] > 4)].shape[0]))

# Exercise 1

# Preprocessing

def remove_punctuation(s):
    """Remove punctuation from a string. Notice that this works in two different ways:
    Either the punctuation is simply removed from the string (meaning it is just replaced with an empty string),
    or it is replaced with a space. This happens when the punctuation is in the middle of a string.
    
    >>> remove_punctuation('game-changing, life saver!')
    'game changing life saver'

    Args:
        s (str): Input string.

    Returns:
        str: String without punctuation.
    """
    reg1 = re.compile(r'\b[^\w\s]\b')
    reg2 = re.compile(r'[^\w\s]')
    first_pass = reg1.sub(' ', s)
    return reg2.sub('', first_pass)

def language_det(s):
    """Detects the language of a string.

    Args:
        s (str): Input string.

    Returns:
        str: Language of the string; if the program cannot detect the language, it returns 'symbols'.
    """
    if s == '':
        return 'empty'
    try:
        return detect(s)
    except:
        return 'symbols'
    
def remove_stopwords(s):
    """Remove stopwords from a string.

    Args:
        s (str): Input string.

    Returns:
        str: String without the stopwords.
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(s)
    return ' '.join([w for w in tokens if w not in stop_words])

def stemming(s):
    """Stems a given string.

    Args:
        s (str): Input string.

    Returns:
        str: String stemmed.
    """
    ps = PorterStemmer()
    tokens = word_tokenize(s)
    return ' '.join([ps.stem(w) for w in tokens])

# Exercise 2

# Indexes

def term_index(documents):
    """Extract the vocabulary from a given corpus of documents.

    Args:
        documents (list): List of strings.

    Returns:
        dict: Dictionary containing as keys the words that appear in all the documents, as values an integer (each word is mapped with a unique integer).
    """
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
    """Computes the inverted index for a corpus of documents.

    Args:
        documents (list): List of strings.
        term_indexes (dict): Vocabulary dictionary containing the term-index maps.

    Returns:
        dict: Dictionary containing as keys the token ids found in the vocabulary, as values the documents in which each word appears.
    """
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
    """Computes the inverted index with tfidf values for each document.

    Args:
        documents (list): List of strings.
        term_indexes (dict): Vocabulary dictionary containing the term-index maps.
        inv_indexes ([type]): Inverted index dictionary containing the index-documents maps.

    Returns:
        dict: Dictionary similar to the inverted index; each document also presents the tfidf value for that particular doc.
    """
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
    """Class representing a naive search engine, meaning that it will not perform any kind of sorting of the results.
    
    Attributes:
        df (pd.DataFrame): Dataframe containing the data on which the search engine is going to work.
        term_indexes (dict): Dictionary containing the vocabulary with which the search engine is going to use to represent queries.
        inv_indexes (dict): Dictionary containing the information with which the search engine is going to retrieve books on which the query appears.
    
    Methods:
        search: Function which is going to be used just internally, used to retrieve the slice of the dataframe containing the words of the query.
        execute_query: Displays the results of the conjunctive query in the proper way.
    
    """
    def __init__(self, df, term_indexes, inv_indexes):
        self.df = df
        self.term_indexes = term_indexes
        self.inv_indexes = inv_indexes
        
    def search(self, query):
        """Returns slice of dataframe containing the query (use only internally).

        Args:
            query (str): String containing the query words the user wants to search for.

        Returns:
            pd.DataFrame: Resulting dataframe.
        """
        # Since we performed stemming on the plot column of the dataframe, we need to
        # perform stemming also on the query. Otherwise, our results wouldn't be accurate
        query = query.lower()
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
        """Use the search function to retrieve the right documents, then return in the proper way.

        Args:
            query (str): String containing the query words the user wants to search for.

        Returns:
            pd.DataFrame: Dataframe of the conjunctive query with only 3 columns.
        """
        return self.search(query)[['bookTitle', 'Plot', 'Url']]

## Scoring functions

class RankCalculator():
    """Class representing the type of a ranking calculator, that is a scoring function. Every rank calculator is going to have a value between 0 and 1."""
    def rank(self, book):
        pass
    
class WeightedRanks(RankCalculator):
    """Class containing the calculations of the weighted sum of a list of scoring functions and their respective weight.
    This class is going to be used when defining a ranked search engine.
    
    Attributes:
        calculators (list): List of tuples; first element of the tuple consists in the weight we want to give to a certain scoring function, second element represents the scoring function itself.
    Methods:
        rank: Computes the weighted sum of the ranking functions.
    """
    def __init__(self, calculators):
        self.calculators = calculators
        
    def rank(self, book, query, token_ids):
        """Computes the weighted similarity score for a certain document (book), a given query and the token ids of the query.

        Args:
            book (pd.DataFrame row): Row of the dataframe containing the info for a given book.
            query (str): String containing the query that the user types.
            token_ids (list): List containing the integers that we obtain mapping the query tokens with the vocabulary.

        Returns:
            float: Similarity score obtained with the weighted sum of the scoring functions for a certain book and a query. Notice that the returning value is in [0, 1].
        """
        total_weight = np.sum([weight for weight, _ in self.calculators])
        return np.sum([calculator.rank(book, query, token_ids) * weight / total_weight for weight, calculator in self.calculators])
    
class ByTfidf(RankCalculator):
    """Scoring function using the cosin similarity over the tfidf values of a certain book and the query."""
    def __init__(self, tfidf_indexes):
        self.tfidf_indexes = tfidf_indexes
    
    def rank(self, book, query, token_ids):
        doc = book['index']
        tfidf = 0
        for token_id in token_ids:
            tfidf += self.tfidf_indexes[token_id][doc]
        return tfidf / np.sqrt(len(query.split()))
    
class ByRatingValue(RankCalculator):
    """Scoring function using the rating value of a certain book (score = book['ratingValue'] / max['ratingValue'])."""
    def __init__(self, df):
        self.max_rating = df['ratingValue'].max()
        
    def rank(self, book, query, token_ids):
        return book['ratingValue'] / self.max_rating
    
class ByRatingCount(RankCalculator):
    """Scoring function using the rating count of a certain book (score = book['ratingCount'] / max['ratingCount'])."""
    def __init__(self, df):
        self.max_rating = df['ratingCount'].max()
        
    def rank(self, book, query, token_ids):
        return book['ratingCount'] / self.max_rating
    
class ByTitleMatch(RankCalculator):
    """Scoring function using the exact match of query - book title (score = n_matches_title / len(title))."""
    def rank(self, book, query, token_ids):
        title_lenght = len(word_tokenize(book['bookTitle']))
        matches = 0
        for token in set(word_tokenize(query)):
            if token in book['bookTitle'].lower():
                matches += 1
        return matches / title_lenght

## Ranked search engine

class RankedSearchEngine:
    """Class representing a sophisticated search engine, which first executes a conjunctive query and then sorts the value according to a scoring function.
    
    Attributes:
        df (pd.DataFrame): Dataframe containing the data on which the search engine is going to work.
        term_indexes (dict): Dictionary containing the vocabulary with which the search engine is going to use to represent queries.
        inv_indexes (dict): Dictionary containing the information with which the search engine is going to retrieve books on which the query appears.
        simple_SE (class): Vanilla search engine, used to retrieve the results of the conjunctive query.
        rank_calculator (class): Scoring function, used to sort the results of a query.
        
        
    Methods:
        execute_query: Displays the results of the conjunctive query, sorted according to the given rank calculator.
    
    """
    def __init__(self, df, term_indexes, inv_indexes, tfidf_indexes, simple_SE, rank_calculator):
        self.df = df
        self.term_indexes = term_indexes
        self.inv_indexes = inv_indexes
        self.tfidf_indexes = tfidf_indexes
        self.simple_SE = simple_SE
        self.rank_calculator = rank_calculator
        
    def execute_query(self, query, k=10):
        """Retrieve the conjunctive query results, sort them according to the similarity score and extract first k documents.

        Args:
            query (str): Query string given by the user.
            k (int, optional): Number of documents to show with the query. Defaults to 10.

        Returns:
            pd.DataFrame: Dataframe containing the results of the query, with their similarity score.
        """
        # First stem the query
        query = query.lower()
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
    
# Exercise 4

def cumpage_serie(df):
    """Helper function: shows and returns the cumulative pages distribution of the first ten book series in order of appearances in the years of publication.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe containing data along the years for the series, representing when each book serie has published some book with number of pages.
    """
    vis = df[df['bookSeries'].notnull()]
    vis = vis[vis['bookSeries'].str.contains('#')]
    vis = vis[~vis['bookSeries'].str.contains(r'#\d[-–]')]

    def remove_hashtag(s):
        return re.sub(r'\s#\d', '', s)

    vis['bookSeries'] = vis['bookSeries'].apply(remove_hashtag)

    book_series = vis.drop_duplicates(['bookSeries']).head(10)['bookSeries'].to_list()

    vis = vis[vis['bookSeries'].isin(book_series)]

    def find_year(s):
        return re.findall(r'[0-9][0-9][0-9][0-9]', s)[0]

    vis['PublishingDate'] = vis['PublishingDate'].apply(find_year).astype(int)

    vis = vis.groupby(['PublishingDate', 'bookSeries']).sum()['numberOfPages'].astype(int)
    vis = vis.unstack()

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)

    ax.set_ylim(0, 1500)

    plt.xticks(vis.index)

    for year in vis.index:
        plt.vlines(year, 0, vis.max(axis=1)[year], color='silver', linestyles='dashed', zorder=1, label='_nolegend_')

    for col in vis.columns:
        ax.scatter(vis.index, vis.unstack()[col])

    ax.legend(vis.columns, loc='upper center', bbox_to_anchor=(0.5, 1.19), ncol=3, fancybox=True, shadow=True)

    plt.setp(ax, xlabel='Year of publication', ylabel='Cumulative number of pages per book series')

    plt.show()
    
    return vis

def year_pages_visualization(series_pages):
    """Plots the cumulative number of pages for the year-story of the first ten book series (for appearance).

    Args:
        series_pages (pd.DataFrame): Dataframe obtained with the cumpage_serie helper function.
    """
    series = []
    for col in series_pages.columns:
        series.append(series_pages[series_pages[col].notna()][col].copy())

    for serie_n in range(len(series)):
        indexes = [i for i in range(min(series[serie_n].index), max(series[serie_n].index)) if i not in series[serie_n].index]

        for idx in indexes:
            series[serie_n].loc[idx] = 0

        series[serie_n] = series[serie_n].sort_index().reset_index(drop=True)
        series[serie_n] = series[serie_n].cumsum()

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)

    ax.set_ylim(0, 5000)

    plt.xticks(np.arange(max([len(serie) for serie in series])))

    for serie in series:
        ax.plot(serie.index, serie, marker='o')


    ax.legend(series_pages.columns, loc='upper center', bbox_to_anchor=(0.5, 1.19), ncol=3, fancybox=True, shadow=True)

    plt.setp(ax, xlabel='Years since publication of first book of the serie', ylabel='Cumulative series page count')

    plt.show()