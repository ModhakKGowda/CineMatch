from flask import Flask, render_template, request, jsonify
import pandas as pd
from recommender import train_svd, recommend_svd

app = Flask(__name__)

# Load datasets and train SVD model on app startup
print("Loading data and training SVD model...")
ratings_df = pd.read_csv("data/ratings.csv")
movies_df = pd.read_csv("data/movies.csv")
svd_model = train_svd()
print("Model ready!")

@app.route("/")
def home():
    # Fetch a sample of distinct user IDs for the UI dropdown
    sample_users = sorted(ratings_df["userId"].unique()[:20])
    return render_template("index.html", users=sample_users)

@app.route("/api/recommend", methods=["GET"])
def get_recommendations():
    user_id = request.args.get("user_id", default=1, type=int)
    top_n = request.args.get("top_n", default=5, type=int)
    
    recs = recommend_svd(user_id, svd_model, movies_df, ratings_df, top_n=top_n)
    return jsonify({
        "user_id": user_id,
        "recommendations": recs
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
