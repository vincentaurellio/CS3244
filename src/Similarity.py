import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import string

df_movie_details = pd.read_json('../data/IMDB_movie_details.json', lines = True)
df_reviews = pd.read_pickle('../data/cleaned_reviews.pkl.gz', compression = 'gzip')

# Pre-Processing Movie Synopsis

## Tokenizing
df_movie_details['tokenized_synopsis'] = list(map(word_tokenize, df_movie_details['plot_synopsis']))

## Removing Stop words
stop_words_and_punctuations = set(stopwords.words('english') + list(string.punctuation))
df_movie_details['tokenized_synopsis'] = list(map(lambda x: [word.lower() for word in x if word.lower() not in stop_words_and_punctuations], df_movie_details['tokenized_synopsis']))

## Stemming or Lemmatization
stemmer = PorterStemmer()
df_movie_details['tokenized_synopsis'] = list(map(lambda x: [stemmer.stem(word) for word in x], df_movie_details['tokenized_synopsis']))

## Returning back to text and saving new dataset
df_movie_details['text_tokenized'] = list(map(lambda x: ' '.join(x), df_movie_details['tokenized_synopsis']))
df_movie_details.to_pickle("../data/cleaned_synopsis.pkl.gz", compression = 'gzip')

# Creating new dataframe for Similarity measure
similarity_data = [[], [], []]
for review in df_reviews:
    movie_id = review['movie_id']
    review_tokenized = review['text_tokenized']
    synopsis_tokenized = df_movie_details['movie_id']['text_tokenized'] #get tokenized synopsis of reviewed movie
    review_label = review['is_spoiler']
    similarity_data[0].append(review_tokenized)
    similarity_data[1].append(synopsis_tokenized)
    similarity_data[2].append(review_label)

# TF - IDF
tfidfvectorizer = TfidfVectorizer()
similarity_tfidf = tfidfvectorizer.fit_transform(similarity_data[:2])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(similarity_tfidf[:2], similarity_tfidf[2], test_size=0.3, random_state=42)

# Learn threshold of cosine similarity with training data 

## Final model (Let threshold = p)

# p = some value
# test = [review, synopsis]
cos_similarity = cosine_similarity(test[0], test[1])

# Returns is_spoiler
if cos_similarity > p:
    return True
else:
    return False
