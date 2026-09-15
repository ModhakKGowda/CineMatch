from flask import Flask, render_template, request, jsonify
import pandas as pd
from recommender import train_svd, recommend_svd

app = Flask(__name__)

# Load data and train SVD model on startup
print("Loading data...")
ratings_df = pd.read_csv("data/ratings.csv")
movies_df = pd.read_csv("data/movies.csv")
print("Training SVD model...")
svd_model = train_svd()
print("Model ready!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/recommend", methods=["GET"])
def get_recommendations():
    user_id = request.args.get("user_id", default=1, type=int)
    top_n = request.args.get("top_n", default=5, type=int)
    
    try:
        recs = recommend_svd(user_id, svd_model, movies_df, ratings_df, top_n=top_n)
        return jsonify({"success": True, "recommendations": recs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
