import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD

def train_svd():
    """Trains and returns the SVD matrix factorization model on MovieLens data."""
    ratings = pd.read_csv("data/ratings.csv")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    model = SVD(n_factors=50, random_state=42)
    model.fit(trainset)
    return model

def recommend_svd(user_id, svd_model, movies_df, ratings_df, top_n=5):
    """Generates top N movie recommendations using the trained SVD model."""
    all_movie_ids = movies_df['movieId'].unique()
    rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    
    unwatched_ids = [m for m in all_movie_ids if m not in rated_movie_ids]
    
    predictions = []
    for m_id in unwatched_ids:
        pred_rating = float(svd_model.predict(user_id, m_id).est)
        predictions.append((m_id, pred_rating))
        
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for movie_id, score in predictions[:top_n]:
        title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
        results.append({
            'movieId': int(movie_id),
            'title': title,
            'score': round(score, 2)
        })
        
    return results
