import streamlit as st
import pickle
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
#  STEP 1: PREPARE DATA
# ==============================

# Check if pickle files exist; if not, create them
if not (os.path.exists('movie_list.pkl') and os.path.exists('similarity.pkl')):
    st.write("Generating movie data files... (this will run once)")

    # Download dataset from Kaggle or use local file
    # You can replace with your own movie dataset if available
    url = "https://raw.githubusercontent.com/themlphdstudent/streamlit-movie-recommender/main/tmdb_5000_movies.csv"
    movies = pd.read_csv(url)

    movies = movies[['id', 'title', 'overview']]
    movies['overview'] = movies['overview'].fillna('')

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['overview']).toarray()

    similarity = cosine_similarity(vectors)

    # Save pickle files
    pickle.dump(movies, open('movie_list.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

# ==============================
#  STEP 2: LOAD DATA
# ==============================
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# ==============================
#  STEP 3: HELPER FUNCTIONS
# ==============================

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except:
        return "https://via.placeholder.com/500x750?text=No+Image"

def recommend(movie):
    if movie not in movies['title'].values:
        st.warning("Movie not found in database.")
        return [], []
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
    return recommended_movie_names, recommended_movie_posters

# ==============================
#  STEP 4: STREAMLIT UI
# ==============================
st.title("🎥 Movie Recommender System")

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button('Show Recommendations'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    if recommended_movie_names:
        cols = st.columns(5)
        for col, name, poster in zip(cols, recommended_movie_names, recommended_movie_posters):
            with col:
                st.text(name)
                st.image(poster)
