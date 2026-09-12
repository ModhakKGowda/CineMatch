"""
CineMatch Exploratory Data Analysis

Phase 1:
- Load MovieLens data
- Basic data analysis
- Rating distribution
- Movie popularity
- Long-tail distribution
- Utility matrix
- Sparsity
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    return movies, ratings


def basic_analysis(movies, ratings):

    print("===== DATASET INFORMATION =====")

    print("\nMovies shape:", movies.shape)
    print("Ratings shape:", ratings.shape)

    print("\nMovies:")
    print(movies.head())

    print("\nRatings:")
    print(ratings.head())

    print("\nMissing values:")
    print("Movies:")
    print(movies.isnull().sum())

    print("\nRatings:")
    print(ratings.isnull().sum())

    print("\nDuplicate rows:")
    print("Movies:", movies.duplicated().sum())
    print("Ratings:", ratings.duplicated().sum())

    print("\nRating statistics:")
    print(ratings["rating"].describe())


def rating_distribution(ratings):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=ratings,
        x="rating"
    )

    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")

    plt.tight_layout()
    plt.show()


def movie_popularity(movies, ratings):

    movie_rating_counts = (
        ratings.groupby("movieId")
        .size()
        .reset_index(name="rating_count")
    )

    movie_popularity = (
        movie_rating_counts
        .merge(movies, on="movieId")
        .sort_values(
            "rating_count",
            ascending=False
        )
    )

    print("\n===== MOST POPULAR MOVIES =====")

    print(
        movie_popularity[
            ["movieId", "title", "rating_count"]
        ].head(10)
    )

    return movie_popularity


def long_tail_distribution(movie_popularity):

    sorted_counts = (
        movie_popularity["rating_count"]
        .sort_values(ascending=False)
        .values
    )

    plt.figure(figsize=(10, 6))

    plt.plot(sorted_counts)

    plt.title("Long-Tail Distribution of Movie Popularity")
    plt.xlabel("Movies ranked by popularity")
    plt.ylabel("Number of ratings")

    plt.tight_layout()
    plt.show()


def utility_matrix_analysis(ratings):

    utility_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    total_cells = (
        utility_matrix.shape[0]
        * utility_matrix.shape[1]
    )

    number_of_ratings = (
        utility_matrix.notna()
        .sum()
        .sum()
    )

    missing_cells = (
        total_cells - number_of_ratings
    )

    sparsity = (
        missing_cells / total_cells
    )

    print("\n===== UTILITY MATRIX =====")

    print(
        "Utility matrix shape:",
        utility_matrix.shape
    )

    print(
        "Total cells:",
        total_cells
    )

    print(
        "Number of ratings:",
        int(number_of_ratings)
    )

    print(
        "Missing cells:",
        int(missing_cells)
    )

    print(
        "Sparsity:",
        round(sparsity * 100, 2),
        "%"
    )

    return utility_matrix


def main():

    movies, ratings = load_data()

    basic_analysis(
        movies,
        ratings
    )

    rating_distribution(
        ratings
    )

    popularity = movie_popularity(
        movies,
        ratings
    )

    long_tail_distribution(
        popularity
    )

    utility_matrix_analysis(
        ratings
    )


if __name__ == "__main__":
    main()
