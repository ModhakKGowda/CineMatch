# ============================================================
# SVD MATRIX FACTORIZATION
# ============================================================

from surprise import SVD
from surprise import Dataset
from surprise import Reader


def train_svd():

    # Load ratings data
    ratings = pd.read_csv("data/ratings.csv")

    # Surprise expects:
    # userId, movieId, rating
    reader = Reader(
        rating_scale=(0.5, 5.0)
    )

    data = Dataset.load_from_df(
        ratings[["userId", "movieId", "rating"]],
        reader
    )

    # Build the training set
    trainset = data.build_full_trainset()

    # Create SVD model
    model = SVD(
        n_factors=50,
        n_epochs=20,
        random_state=42
    )

    # Train the model
    model.fit(trainset)

    return model


def recommend_svd(user_id, model, top_n=5):

    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    # Movies already rated by the user
    rated_movies = set(
        ratings[
            ratings["userId"] == user_id
        ]["movieId"]
    )

    recommendations = []

    # Predict ratings for movies the user has not rated
    for movie_id in movies["movieId"]:

        if movie_id not in rated_movies:

            prediction = model.predict(
                user_id,
                movie_id
            )

            recommendations.append(
                (
                    movie_id,
                    prediction.est
                )
            )

    # Sort by predicted rating
    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Get movie information
    top_movies = []

    for movie_id, predicted_rating in recommendations[:top_n]:

        title = movies[
            movies["movieId"] == movie_id
        ]["title"].iloc[0]

        top_movies.append(
            {
                "movieId": int(movie_id),
                "title": title,
                "predicted_rating": round(
                    predicted_rating,
                    2
                )
            }
        )

    return top_movies


# ============================================================
# TEST SVD
# ============================================================

if __name__ == "__main__":

    print("\n===== TRAINING SVD MODEL =====")

    svd_model = train_svd()

    print("SVD model trained successfully!")

    user_id = 1

    print(
        f"\n===== SVD RECOMMENDATIONS FOR USER {user_id} ====="
    )

    recommendations = recommend_svd(
        user_id,
        svd_model,
        top_n=5
    )

    for recommendation in recommendations:
        print(recommendation)
