import gensim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import ssl
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn import preprocessing
import streamlit as st
from util import preprocess_data


def tf_idf_vectorization(original_data, data, target):
    n = 1
    ngram_range = (n, n)
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(data)
    tfidf_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    X = preprocessing.normalize(X)
    # Example: Convert a new document into a TF-IDF vector
    new_document = {'headline': [target]}
    new_document = pd.DataFrame(new_document)
    new_preprocessed_document = preprocess_data(new_document['headline'])
    new_tfidf_vector = vectorizer.transform(new_preprocessed_document)
    similarity_scores = cosine_similarity(new_tfidf_vector, tfidf_matrix)
    tf_idf_similarity_scores = similarity_scores.flatten()  # Convert to 1D array
    document_indices = tf_idf_similarity_scores.argsort()[::-1]  # Sort indices in descending order
    df_tf_idf = pd.DataFrame({'Sentence': original_data, 'TF-IDF Similarity Score': tf_idf_similarity_scores})
    df_tf_idf.sort_values('TF-IDF Similarity Score', ascending=False)
    return (X,df_tf_idf)

def word2vec(original_data, data, target):
    story = []
    for sent in data:
        story.append(simple_preprocess(sent))
    model = gensim.models.Word2Vec(window=10,min_count=2)
    model.build_vocab(story)
    model.train(story, total_examples=model.corpus_count, epochs=model.epochs)
    y = model.wv.index_to_key
    sentences = data
    # use n_similarity to compute a cosine similarity (should be reasonably robust)
    sentences_similarity = np.zeros(len(sentences))
    target_sentence_words = [w for w in target.split() if w in y]
    for idx, sentence in enumerate(sentences):
        sentence_words = [w for w in sentence.split() if w in y]
        if (len(target_sentence_words) >=1 & len(sentence_words) >= 1):
            sim = model.wv.n_similarity(target_sentence_words, sentence_words)
            sentences_similarity[idx] = sim
    result = list(zip(sentences_similarity, sentences))
    result.sort(key=lambda item: item[0], reverse=True)
    df_word2vec = pd.DataFrame({'Sentence': original_data, 'Word2Vec Similarity Score': sentences_similarity})
    return df_word2vec

def glove(original_data, data, target):
    nlp = spacy.load("en_core_web_md")
    glove_target = nlp(target).vector
    glove_similarity_scores = [np.dot(glove_target, nlp(sent).vector) /
                               (np.linalg.norm(glove_target) * np.linalg.norm(nlp(sent).vector)) if (np.linalg.norm(glove_target) * np.linalg.norm(nlp(sent).vector))  > 0 else 1
                               for sent in data]
    df_glove = pd.DataFrame({'Sentence': original_data, 'Glove Similarity Score': glove_similarity_scores})
    return df_glove