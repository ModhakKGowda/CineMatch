"""
CineMatch Recommendation Engine

Phase 2:
1. User-User Collaborative Filtering
2. Item-Item Collaborative Filtering

Similarity:
- Cosine similarity using NumPy
"""

import pandas as pd
import numpy as np


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    return movies, ratings


# --------------------------------------------------
# CREATE UTILITY MATRIX
# --------------------------------------------------

def create_utility_matrix(ratings):
    utility_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    return utility_matrix


# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    # Replace missing values with 0
    vector_a = np.nan_to_num(vector_a)
    vector_b = np.nan_to_num(vector_b)

    dot_product = np.dot(vector_a, vector_b)

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


# --------------------------------------------------
# USER-USER COLLABORATIVE FILTERING
# --------------------------------------------------

def find_similar_users(user_id, utility_matrix, top_n=5):
    """
    Find users most similar to the given user.
    """

    if user_id not in utility_matrix.index:
        return []

    target_user = utility_matrix.loc[user_id].values

    similarities = []

    for other_user_id in utility_matrix.index:

        if other_user_id == user_id:
            continue

        other_user = utility_matrix.loc[other_user_id].values

        similarity = cosine_similarity(
            target_user,
            other_user
        )

        similarities.append(
            (other_user_id, similarity)
        )

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return similarities[:top_n]


# --------------------------------------------------
# USER-BASED RECOMMENDATIONS
# --------------------------------------------------

def recommend_user_based(user_id, top_n=5):
    """
    Recommend movies using User-User Collaborative Filtering.
    """

    movies, ratings = load_data()

    utility_matrix = create_utility_matrix(ratings)

    if user_id not in utility_matrix.index:
        return []

    similar_users = find_similar_users(
        user_id,
        utility_matrix,
        top_n=10
    )

    user_ratings = utility_matrix.loc[user_id]

    recommendations = {}

    for similar_user_id, similarity in similar_users:

        similar_user_ratings = utility_matrix.loc[
            similar_user_id
        ]

        for movie_id, rating in similar_user_ratings.items():

            # Skip movies already rated by the target user
            if pd.notna(user_ratings.get(movie_id)):
                continue

            if pd.isna(rating):
                continue

            if movie_id not in recommendations:
                recommendations[movie_id] = 0

            recommendations[movie_id] += (
                similarity * rating
            )

    ranked_movies = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for movie_id, score in ranked_movies[:top_n]:

        movie = movies[
            movies["movieId"] == movie_id
        ]

        if not movie.empty:

            results.append({
                "movieId": int(movie_id),
                "title": movie.iloc[0]["title"],
                "score": round(score, 2)
            })

    return results


# --------------------------------------------------
# ITEM-ITEM COLLABORATIVE FILTERING
# --------------------------------------------------

def find_similar_movies(movie_id, utility_matrix, top_n=5):
    """
    Find movies similar to a given movie.
    """

    if movie_id not in utility_matrix.columns:
        return []

    target_movie = utility_matrix[movie_id].values

    similarities = []

    for other_movie_id in utility_matrix.columns:

        if other_movie_id == movie_id:
            continue

        other_movie = utility_matrix[
            other_movie_id
        ].values

        similarity = cosine_similarity(
            target_movie,
            other_movie
        )

        similarities.append(
            (other_movie_id, similarity)
        )

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return similarities[:top_n]


# --------------------------------------------------
# ITEM-BASED RECOMMENDATIONS
# --------------------------------------------------

def recommend_item_based(user_id, top_n=5):
    """
    Recommend movies using Item-Item Collaborative Filtering.
    """

    movies, ratings = load_data()

    utility_matrix = create_utility_matrix(ratings)

    if user_id not in utility_matrix.index:
        return []

    user_ratings = utility_matrix.loc[user_id]

    rated_movies = user_ratings[
        user_ratings.notna()
    ].sort_values(
        ascending=False
    )

    recommendations = {}

    # Use the user's highest-rated movies
    for movie_id in rated_movies.head(10).index:

        similar_movies = find_similar_movies(
            movie_id,
            utility_matrix,
            top_n=10
        )

        for similar_movie_id, similarity in similar_movies:

            # Don't recommend movies already watched/rated
            if pd.notna(
                user_ratings.get(similar_movie_id)
            ):
                continue

            if similar_movie_id not in recommendations:
                recommendations[similar_movie_id] = 0

            recommendations[similar_movie_id] += (
                similarity *
                user_ratings[movie_id]
            )

    ranked_movies = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for movie_id, score in ranked_movies[:top_n]:

        movie = movies[
            movies["movieId"] == movie_id
        ]

        if not movie.empty:

            results.append({
                "movieId": int(movie_id),
                "title": movie.iloc[0]["title"],
                "score": round(score, 2)
            })

    return results


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    user_id = 1

    print("\n===== USER-BASED RECOMMENDATIONS =====")

    user_recommendations = recommend_user_based(
        user_id,
        top_n=5
    )

    for recommendation in user_recommendations:
        print(recommendation)

    print("\n===== ITEM-BASED RECOMMENDATIONS =====")

    item_recommendations = recommend_item_based(
        user_id,
        top_n=5
    )

    for recommendation in item_recommendations:
        print(recommendation)
