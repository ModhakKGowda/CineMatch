import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
ratings = pd.read_csv("data/ratings.csv")  # columns: userId, movieId, rating, timestamp
movies = pd.read_csv("data/movies.csv")   # columns: movieId, title, genres

# Merge ratings with movie titles
df = pd.merge(ratings, movies, on="movieId")

# 2. Sparsity & Utility Matrix Analysis
def analyze_utility_matrix(df):
    utility_matrix = df.pivot(index='userId', columns='movieId', values='rating')
    
    n_users = utility_matrix.shape[0]
    n_movies = utility_matrix.shape[1]
    total_cells = n_users * n_movies
    rated_cells = df['rating'].count()
    
    sparsity = (1 - (rated_cells / total_cells)) * 100
    
    print(f"Users: {n_users} | Movies: {n_movies}")
    print(f"Sparsity Ratio: {sparsity:.2f}%")
    return utility_matrix

# 3. Visualization Plots
def plot_eda(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rating Distribution
    sns.countplot(x='rating', data=df, ax=axes[0], palette='Blues_d')
    axes[0].set_title('Distribution of User Ratings')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Count')
    
    # Long-Tail Distribution (Ratings per Movie)
    movie_counts = df['movieId'].value_counts().values
    axes[1].plot(movie_counts, color='firebrick')
    axes[1].set_title('Long-Tail Distribution: Ratings per Movie')
    axes[1].set_xlabel('Movie Rank')
    axes[1].set_ylabel('Number of Ratings')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    matrix = analyze_utility_matrix(df)
    plot_eda(df)
