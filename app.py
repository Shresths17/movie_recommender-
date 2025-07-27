import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

uploaded_file = st.file_uploader("Upload your movies.csv file", type="csv")

if uploaded_file is not None:
    movies_data = pd.read_csv(uploaded_file)
    movies_data = movies_data.reset_index(drop=True)

    selected_features = ['genre', 'desc', 'rating', 'votes']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    combined_features = movies_data['genre'].astype(str) + ' ' + movies_data['desc'].astype(str) + ' ' + movies_data['rating'].astype(str) + ' ' + movies_data['votes'].astype(str)

    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)

    movie_name = st.text_input("Enter your favourite movie name:").strip().lower()

    if movie_name:
        movies_data['title_clean'] = movies_data['title'].str.lower().str.strip()
        list_of_all_titles = movies_data['title_clean'].tolist()

        close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)

        if close_matches:
            close_match = close_matches[0]
            index_of_movie = movies_data[movies_data.title_clean == close_match].index[0]
            similarity_scores = list(enumerate(similarity[index_of_movie]))
            sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

            st.subheader(f"Top 10 Recommendations for '{movies_data.iloc[index_of_movie]['title']}':")
            for i, (index, score) in enumerate(sorted_similar_movies[1:11]):
                st.write(f"{i + 1}. {movies_data.iloc[index]['title']}")
        else:
            st.warning("Movie not found! Try a different title.")
