import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# =====================================================================
# DATA LOADING & UTILITY MATRIX SETUP
# =====================================================================

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

utility_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# =====================================================================
# MEMORY-BASED COLLABORATIVE FILTERING (Cosine Similarity)
# =====================================================================

def compute_cosine_similarity(matrix):
    """Calculates cosine similarity matrix between users."""
    # Rows are users, columns are movies
    user_sim = cosine_similarity(matrix)
    return pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)

def recommend_memory_based(user_id, utility_matrix, user_sim_df, top_n=5):
    """Predicts movies for a user based on top N similar users."""
    if user_id not in utility_matrix.index:
        return []
    
    # Get similarity scores for target user
    sim_scores = user_sim_df[user_id].drop(user_id)
    
    # Unrated movies for the user
    user_ratings = utility_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    
    # Weighted average of ratings from top N similar users
    weighted_ratings = {}
    for movie in unrated_movies:
        other_ratings = utility_matrix[movie].drop(user_id)
        # Filter non-zero ratings
        rated_by = other_ratings[other_ratings > 0]
        if not rated_by.empty:
            weights = sim_scores[rated_by.index]
            if weights.sum() > 0:
                weighted_ratings[movie] = np.dot(weights, rated_by) / weights.sum()
                
    sorted_movies = sorted(weighted_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    movie_ids = [m[0] for m in sorted_movies]
    
    return movies[movies['movieId'].isin(movie_ids)]['title'].tolist()

# =====================================================================
# MODEL-BASED COLLABORATIVE FILTERING (SVD Matrix Factorization)
# =====================================================================

def train_svd():
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    model = SVD(n_factors=50, random_state=42)
    model.fit(trainset)
    return model

def recommend_svd(user_id, svd_model, top_n=5):
    all_movie_ids = movies['movieId'].unique()
    rated_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    
    unwatched_ids = [m for m in all_movie_ids if m not in rated_movie_ids]
    
    predictions = [
        (movie_id, svd_model.predict(user_id, movie_id).est)
        for movie_id in unwatched_ids
    ]
    
    # Sort predictions by estimated rating descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    top_movie_ids = [m[0] for m in predictions[:top_n]]
    top_movies = movies[movies['movieId'].isin(top_movie_ids)]['title'].tolist()
    
    return top_movies

# =====================================================================
# TEST SVD
# =====================================================================

if __name__ == "__main__":

    print("\n====== TRAINING SVD MODEL ======")

    svd_model = train_svd()

    print("SVD model trained successfully!")

    user_id = 1

    print(
        f"\n====== SVD RECOMMENDATIONS FOR USER {user_id} ======"
    )

    recommendations = recommend_svd(
        user_id,
        svd_model,
        top_n=5
    )

    for recommendation in recommendations:
        print(recommendation)
