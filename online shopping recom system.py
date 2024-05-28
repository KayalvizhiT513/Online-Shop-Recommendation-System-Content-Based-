import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')

# Function to tokenize and stem text
stemmer = SnowballStemmer("english")

def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stem = [stemmer.stem(w) for w in tokens]
    return " ".join(stem)

# Function to calculate cosine similarity
tfidvectorizer = TfidfVectorizer(tokenizer=tokenize_stem)

def cosine_sim(txt1,txt2):
    tfid_matrix = tfidvectorizer.fit_transform([txt1,txt2])
    return cosine_similarity(tfid_matrix)[0][1]

# Function to search for products based on query
def search_product(query):
    stemmed_query = tokenize_stem(query)
    # Assuming this is the part of your code that adds the 'stemmed_tokens' column
    amzon_df['stemmed_tokens'] = amzon_df.apply(lambda row: tokenize_stem(row['Title'] + ' ' + row['Description']), axis=1)

    # Calculating cosine similarity between query and stemmed tokens columns
    amzon_df['similarity'] = amzon_df['stemmed_tokens'].apply(lambda x:cosine_sim(stemmed_query,x))
    res = amzon_df.sort_values(by=['similarity'],ascending=False).head(5)[['Title','Description','Category']]
    return res

# Load data
amzon_df = pd.read_csv('amazon_product.csv')

# Main function for Streamlit app
def main():
    st.title('Online Shopping Product Recommender System-Content Based')
    query = st.text_input('Enter your search query:')
    if st.button('Search'):
        if query:
            results = search_product(query)
            st.dataframe(results)
        else:
            st.warning('Please enter a search query.')

if __name__ == '__main__':
    main()
