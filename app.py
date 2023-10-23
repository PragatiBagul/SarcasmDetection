from classification import logistic_regression,svm
from lstm import lstm
from vectorization import word2vec,glove,tf_idf_vectorization
import util
import pandas as pd
import streamlit as st
import nltk
nltk.download("stopwords")

target = st.text_input("So you wanna act sarcastic ?")

# loading in the dataset
address = "./archive/Sarcasm_Headlines_Dataset_v2.json"
json_df = pd.read_json(address,lines = True)
df = pd.DataFrame(json_df)

df.duplicated().sum()
df.drop_duplicates(subset=['headline'], inplace = True)
df.duplicated().sum()

# because the data is too much for the SVM and RNN models to run efficiently, I will split the data to 25% of it's original size
data_shortened = df.sample(frac=.25)

# checking if sample is 0.25 times data or not
if (0.25 * (len(df)) == len(data_shortened)):
    print(len(df), len(data_shortened))

# splitting the dataset
data_unlabeled = df['headline']
data_unlabeled_shortened = data_shortened['headline']

st.set_option('deprecation.showPyplotGlobalUse', False)

# data_unlabeled.head()
data_labels = df['is_sarcastic']
data_labels_shortened = data_shortened['is_sarcastic']

data_clean = util.preprocess_data(data_unlabeled)
data_clean_shortened = util.preprocess_data(data_unlabeled_shortened)

st.title("Original Dataset")
st.dataframe(df)

st.title("Cleaned Headlines")
st.dataframe(data_clean)

if(target):
    st.title("Tf-IDF Values")
    (X,df_tf_idf) = tf_idf_vectorization(data_unlabeled, data_clean, target)
    st.dataframe(df_tf_idf)

    st.title("Word2Vec Values")
    df_word2vec = word2vec(data_unlabeled,data_clean,target)
    st.dataframe(df_word2vec)

    st.title("Glove Values")
    df_glove = glove(data_unlabeled,data_clean,target)
    st.dataframe(df_glove)

    st.title("Logistic Regression")
    logistic_regression(X, data_labels)
    (X_shortened, df_tf_idf_shortened) = tf_idf_vectorization(data_unlabeled_shortened, data_clean_shortened, target)

    st.title("SVM")
    svm(X_shortened,data_labels_shortened)

    st.title("LSTM")
    lstm(data_clean_shortened,data_labels_shortened)

