from flask import Flask, render_template, request, jsonify
import pandas as pd
from recommender import train_svd, recommend_svd

app = Flask(__name__)

# Load datasets and train SVD model on startup
ratings_df = pd.read_csv("data/ratings.csv")
movies_df = pd.read_csv("data/movies.csv")
svd_model = train_svd()

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    selected_user = 1

    if request.method == "POST":
        selected_user = request.form.get("user_id", default=1, type=int)
        recommendations = recommend_svd(selected_user, svd_model, movies_df, ratings_df, top_n=5)

    return render_template("index.html", user_id=selected_user, recommendations=recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
