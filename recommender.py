import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# =====================================================================
# SVD TRAINING & EVALUATION (Deliverable #2)
# =====================================================================

def evaluate_and_train_svd(ratings_df):
    """Evaluates SVD on test split to report RMSE, then builds full model."""
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    # Train-Test Split for RMSE Evaluation
    trainset_split, testset_split = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD(n_factors=50, random_state=42)
    model.fit(trainset_split)
    
    # Calculate RMSE
    predictions = model.test(testset_split)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"SVD Model Test RMSE: {float(rmse):.4f}")
    
    # Retrain on full dataset for live recommendations
    full_trainset = data.build_full_trainset()
    full_model = SVD(n_factors=50, random_state=42)
    full_model.fit(full_trainset)
    
    return full_model

# =====================================================================
# RECOMMENDATION GENERATOR (Deliverable #3)
# =====================================================================

def recommend_svd(user_id, svd_model, movies_df, ratings_df, top_n=5):
    """Returns clean, formatted list of top N movie recommendations."""
    all_movie_ids = movies_df['movieId'].unique()
    rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    
    unwatched_ids = [m for m in all_movie_ids if m not in rated_movie_ids]
    
    # Predict scores and clean numpy scalar types to native floats
    predictions = []
    for m_id in unwatched_ids:
        pred_rating = float(svd_model.predict(user_id, m_id).est)
        predictions.append((m_id, pred_rating))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Map back to movie titles
    results = []
    for movie_id, score in predictions[:top_n]:
        title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
        results.append({
            'movieId': int(movie_id),
            'title': title,
            'score': round(score, 2)
        })
        
    return results

# =====================================================================
# MAIN RUNNER
# =====================================================================

if __name__ == "__main__":
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")

    print("\n====== EVALUATING & TRAINING SVD MODEL ======")
    svd_model = evaluate_and_train_svd(ratings)

    user_id = 1
    print(f"\n====== SVD RECOMMENDATIONS FOR USER {user_id} ======")
    recommendations = recommend_svd(user_id, svd_model, movies, ratings, top_n=5)

    for rec in recommendations:
        print(f"Title: {rec['title']} | Predicted Rating: {rec['score']}")
