"""
CineMatch Exploratory Data Analysis

This file will be used to analyze:
- MovieLens ratings
- Rating distribution
- Movie popularity
- Long-tail distribution
- Utility matrix
- Sparsity
"""

import pandas as pd


def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    return movies, ratings
