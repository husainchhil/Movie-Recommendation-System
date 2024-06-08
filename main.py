import streamlit as st
import re
from PIL import Image
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

st.set_page_config(
    'Movie Recommender',
    layout='wide',
    page_icon= Image.open('rsrcs/image.png')
)

## genres filter to be added.

st.header('🍿What do you wanna watch today?')

@st.cache_data(show_spinner="Fetching Data")
def read_data():
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    return movies, ratings

movies, ratings = read_data()

def clean_title(title):
    return re.sub('[^a-zA-Z0-9 ]','', title)

movies['title'] = movies['title'].apply(clean_title)

def clean_genres(genre):
    return genre.replace('|', ' ')

movies['genres'] = movies['genres'].apply(clean_genres)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(movies['title'])

def search(title):
    title = vectorizer.transform([clean_title(title)])
    similarity = cosine_similarity(title,tfidf).flatten()
    indices = np.argpartition(similarity,-5)[-5:]
    return movies.iloc[indices][::-1]

def find_similar_movies(movieId):
    similar_users = ratings[(ratings["movieId"] == movieId) & (ratings["rating"] > 4)]["userId"].unique() # Similar Users who watched same movie as us and recommended it
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"] # Recommendations of our Similar Users
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .10]

    all_users = ratings[ratings['movieId'].isin(similar_user_recs.index) & (ratings['rating']>4)] # Recommendations of All the users who have seen that movie
    all_users_recs = all_users['movieId'].value_counts() / len(all_users['userId'].unique()) # Percentage of All User that recommend that movie

    rec_percentages = pd.concat([similar_user_recs,all_users_recs],axis=1)
    rec_percentages.columns = ['Similar Users Recs', 'All Users Recs']
    rec_percentages['score'] = rec_percentages['Similar Users Recs'] / rec_percentages['All Users Recs']
    rec_percentages.sort_values('score', ascending=False, inplace=True)
    return rec_percentages.head(10).merge(movies, left_index = True, right_on='movieId')[["title", "genres"]].reset_index(drop=True)

@st.cache_data(show_spinner='Searching')
def recommender(title):
    result = search(title)
    return find_similar_movies(result.iloc[0]['movieId'])


title = st.text_input('Enter the movie you like')

if title:
    df = recommender(title)
    if not df.empty:
        st.write("Recommended Movies:")
        st.dataframe(df, use_container_width=True)
    else:
        st.info(
            'No recommended movie found! Here are some similar movies',
            icon= 'ℹ️'
        )
        st.dataframe(search(title)[['title','genres']][:10], use_container_width=True)
        
else:
    st.info(
        'Title field is empty',
        icon= 'ℹ️'
    )
